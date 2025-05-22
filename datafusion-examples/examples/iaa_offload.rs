// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use criterion::{criterion_group, criterion_main, Criterion};
use std::any::Any;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{Array, AsArray, BooleanBuilder, RecordBatch};
use arrow::compute;
use arrow::datatypes::{DataType, UInt32Type};
use datafusion::common::instant::Instant;
use datafusion::common::{internal_err, ScalarValue};
use datafusion::error::Result;
use datafusion::logical_expr::sort_properties::{ExprProperties, SortProperties};
use datafusion::logical_expr::Volatility;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
};
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::physical_plan::{collect, displayable};
use datafusion::prelude::*;

use tracy::*;

fn filter_cpu(input: &arrow::array::PrimitiveArray<UInt32Type>) -> Result<ColumnarValue> {
    zone!("cpu filter");
    let mut b = BooleanBuilder::new();
    for i in input {
        match i {
            Some(v) => {
                // NOTE this must match the predicate used in the offload
                b.append_value(unsafe { FC.limit } <= v && v <= std::u32::MAX);
            }
            None => b.append_value(false),
        }
    }
    let predicates = b.finish();
    let res = compute::filter(input, &predicates).unwrap();

    if unsafe { FC.run_var == RunVariant::Validation } {
        println!("CPU filter {} -> {}", input.len(), res.len());
    }

    let ret = Arc::new(input.clone());
    Ok(ColumnarValue::Array(ret))
}

struct IaaFuture {
    job: usize,
}
impl IaaFuture {
    pub fn submit(
        qpl_job_ptr: *mut qpl_job,
    ) -> impl Future<Output = Result<qpl_status, String>> {
        // submit directly
        let res = unsafe { qpl_submit_job(qpl_job_ptr) };
        if res != qpl_status_QPL_STS_OK {
            panic!("qpl_submit_job failed with {}", res);
        }

        // return a future that polls the result
        IaaFuture {
            job: qpl_job_ptr as usize,
        }
    }
}
impl Future for IaaFuture {
    type Output = Result<qpl_status, String>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match unsafe { qpl_check_job(self.job as *mut qpl_job) } {
            qpl_status_QPL_STS_OK => Poll::Ready(Ok(qpl_status_QPL_STS_OK)),
            qpl_status_QPL_STS_BEING_PROCESSED => {
                cx.waker().wake_by_ref(); // Re-poll later
                Poll::Pending
            }
            res => Poll::Ready(Err(format!("failed to execute job with {}", res))),
        }
    }
}

// TODO this should be a future!
fn qpl_job_runner(qpl_job_ptr: *mut qpl_job) -> qpl_status {
    unsafe {
        // NOTE not sure if auto is already translated to hardware here?
        if FC.var == FilterVariant::IaaOffloadHwAsync {
            // Variant 2: using tokio for async

            // NOTE this segfaults when used with criterion

            let f = IaaFuture::submit(qpl_job_ptr);
            tokio::task::spawn(async {
                let _ = f.await;
            });
        } else {
            if (*qpl_job_ptr).data_ptr.path == qpl_path_t_qpl_path_software {
                // Some optimizations for the software path
                // https://intel.github.io/qpl/documentation/dev_guide_docs/c_use_cases/advanced/c_advanced_topics_filter.html
                // QPL_FLAG_OMIT_AGGREGATES -> we need that for the popcount
                (*qpl_job_ptr).flags = (*qpl_job_ptr).flags | QPL_FLAG_OMIT_CHECKSUMS;
            }

            // This is the default sync way to run the job regardless of execution path
            let res = qpl_execute_job(qpl_job_ptr);
            if res != 0 {
                panic!("qpl_execute_job failed with {}", res);
            }
        }
    }
    qpl_status_QPL_STS_OK
}

fn filter_iaa_offload(
    input: &arrow::array::PrimitiveArray<UInt32Type>,
    execution_path: u32,
) -> Result<ColumnarValue> {
    zone!("iaa_offload");

    let input_size = (input.len() * size_of::<u32>()) as u32; // #of bytes
    let mut res;

    unsafe {
        if FC.job_cfg_size == 0 {
            // TODO pretty sure the job size is runtime constant so we could
            // save the size and just zero the old vec instead of allocating a new one?

            res = qpl_get_job_size(execution_path, &raw mut FC.job_cfg_size);
            if res != 0 {
                panic!("qpl_get_job_size failed with {}", res);
            }

            FC.job_cfg_buf = vec![0u8; FC.job_cfg_size as usize];
            let job_cfg_buf_ptr = FC.job_cfg_buf.as_mut_ptr() as *mut qpl_job;

            // From here on all qpl API calls want the pointer to the struct and nothing else
            res = qpl_init_job(execution_path, job_cfg_buf_ptr);
            if res != 0 {
                panic!("qpl_init_job failed with {}", res);
            }

            // Some tweaks
            (*job_cfg_buf_ptr).numa_id = -1;
        }

        // Pointer to the job config
        let job_cfg_ptr = FC.job_cfg_buf.as_mut_ptr() as *mut qpl_job;

        #[cfg(debug_assertions)]
        if FC.run_var == RunVariant::Validation {
            println!("IAA Init job\n{:?}", (*job_cfg_ptr));
        }

        // 1 bit per input element => 32 bytes per element / input_size
        let mut scan_result_bit_mask = vec![0u8; (input.len() / 8) as usize];

        // Performing a scan operation
        (*job_cfg_ptr).next_in_ptr = input.values().as_ptr() as *mut u8;
        (*job_cfg_ptr).available_in = input_size;
        (*job_cfg_ptr).next_out_ptr = scan_result_bit_mask.as_mut_ptr() as *mut u8;
        (*job_cfg_ptr).available_out = scan_result_bit_mask.len() as u32;
        (*job_cfg_ptr).op = qpl_operation_qpl_op_scan_ge;
        (*job_cfg_ptr).src1_bit_width = 32;
        (*job_cfg_ptr).num_input_elements = input_size / 4;
        (*job_cfg_ptr).out_bit_width = qpl_out_format_qpl_ow_nom;
        // Low Filter Param ≤ element value ≤ High Filter Param
        (*job_cfg_ptr).param_low = FC.limit;

        #[cfg(debug_assertions)]
        if FC.run_var == RunVariant::Validation {
            println!("IAA pre scan\n{:?}", (*job_cfg_ptr));
        }

        res = qpl_job_runner(job_cfg_ptr);
        if res != 0 {
            panic!("qpl_job_runner failed with {}", res);
        }

        #[cfg(debug_assertions)]
        if FC.run_var == RunVariant::Validation {
            println!("IAA post scan\n{:?}", (*job_cfg_ptr));
        }

        let scan_result_bytes = (*job_cfg_ptr).total_out;

        // If pop count is 0 then we dont need to do a select op
        if (*job_cfg_ptr).sum_value != 0 {
            // A scan uses the sum_value aggregate as a pop count (or that should happen according to the docu)
            let mut destination = vec![0u32; (*job_cfg_ptr).sum_value as usize];

            // Performing a select operation
            (*job_cfg_ptr).next_in_ptr = input.values().as_ptr() as *mut u8;
            (*job_cfg_ptr).available_in = input_size;
            (*job_cfg_ptr).next_out_ptr = destination.as_mut_ptr() as *mut u8;
            (*job_cfg_ptr).available_out = (destination.len() * 4) as u32; // #bytes available in output
            (*job_cfg_ptr).op = qpl_operation_qpl_op_select;
            (*job_cfg_ptr).src1_bit_width = 32;
            (*job_cfg_ptr).num_input_elements = input_size / 4;
            (*job_cfg_ptr).out_bit_width = qpl_out_format_qpl_ow_nom;
            (*job_cfg_ptr).next_src2_ptr = scan_result_bit_mask.as_mut_ptr() as *mut u8;
            (*job_cfg_ptr).available_src2 = scan_result_bytes;
            (*job_cfg_ptr).src2_bit_width = 1;

            #[cfg(debug_assertions)]
            if FC.run_var == RunVariant::Validation {
                println!("IAA pre select job\n{:?}", (*job_cfg_ptr));
            }

            res = qpl_job_runner(job_cfg_ptr);
            if res != 0 {
                panic!("qpl_job_runner failed with {}", res);
            }

            #[cfg(debug_assertions)]
            if FC.run_var == RunVariant::Validation {
                println!("IAA post select job\n{:?}", (*job_cfg_ptr));

                println!("IAA filter {} -> {}", input.len(), destination.len());
            }
        } else if FC.run_var == RunVariant::Validation {
            println!("IAA filter {} -> 0", input.len());
        }

        // only finish when less than batchSize elements are give aka the last block has been processed
        if input.len() != FC.batch_size {
            res = qpl_fini_job(job_cfg_ptr);
            if res != 0 {
                panic!("qpl_fini_job failed with {}", res);
            }
            FC.job_cfg_size = 0;
            // TODO dealloc vec?

            #[cfg(debug_assertions)]
            if FC.run_var == RunVariant::Validation {
                println!("IAA job finish");
            }
        }

        //ArrayDataBuilder::
        //let ret = arrow::array::make_array(destination);
        let ret = Arc::new(input.clone());
        return Ok(ColumnarValue::Array(ret));

        // TODO convert back to array
    };
}

mod qpl {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    include!("../bindings.rs");
}
use qpl::*;

#[derive(Debug, Clone, Copy, PartialEq)]
enum FilterVariant {
    NaiveCpu,
    IaaOffload,
    IaaOffloadHw,
    IaaOffloadHwAsync,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum RunVariant {
    Benchmark,
    Validation,
}

struct FilterConfig {
    var: FilterVariant,
    run_var: RunVariant,
    limit: u32,
    batch_size: usize,

    //Offload globals
    job_cfg_size: u32,
    job_cfg_buf: Vec<u8>,
}

static mut FC: FilterConfig = FilterConfig {
    var: FilterVariant::NaiveCpu,
    run_var: RunVariant::Benchmark,
    limit: 2015,
    batch_size: 8192,
    job_cfg_size: 0,
    job_cfg_buf: vec![],
};

#[derive(Debug, Clone)]
struct IAAOffloadUDF {
    signature: Signature,
    aliases: Vec<String>,
}

impl IAAOffloadUDF {
    fn new() -> Self {
        Self {
            signature: Signature::exact(vec![DataType::UInt32], Volatility::Stable),
            aliases: vec!["custom_filter_udf".to_string()],
        }
    }
}

impl ScalarUDFImpl for IAAOffloadUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "custom_filter_udf"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::UInt32)
    }
    fn aliases(&self) -> &[String] {
        &self.aliases
    }
    fn output_ordering(&self, input: &[ExprProperties]) -> Result<SortProperties> {
        Ok(input[0].sort_properties)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        zone!();

        let ScalarFunctionArgs { mut args, .. } = args;
        assert_eq!(args.len(), 1);
        let base = args.pop().unwrap();
        assert_eq!(base.data_type(), DataType::UInt32);

        match base {
            // SINGLE VALUE
            ColumnarValue::Scalar(ScalarValue::UInt32(base)) => {
                // compute the output. Note DataFusion treats `None` as NULL.
                let res = match base {
                    Some(base) => match base > 10 {
                        true => Some(base),
                        false => None,
                    },
                    _ => None,
                };
                Ok(ColumnarValue::Scalar(ScalarValue::from(res)))
            }
            // ARRAY OF VALUES
            ColumnarValue::Array(base_array) => match unsafe { FC.var } {
                FilterVariant::NaiveCpu => {
                    Ok(filter_cpu(base_array.as_primitive::<UInt32Type>()).unwrap())
                }
                FilterVariant::IaaOffload => Ok(filter_iaa_offload(
                    base_array.as_primitive::<UInt32Type>(),
                    qpl_path_t_qpl_path_software,
                )
                .unwrap()),
                _ => Ok(filter_iaa_offload(
                    base_array.as_primitive::<UInt32Type>(),
                    qpl_path_t_qpl_path_hardware,
                )
                .unwrap()),
            },
            _ => {
                internal_err!("Invalid argument types to iaa_offload_filter function")
            }
        }
    }
}

// Borrowed from the TPCH bench
async fn execute_query(ctx: &SessionContext, sql: &str) -> Result<Vec<RecordBatch>> {
    let debug = false;
    let plan = ctx.sql(sql).await?;
    let (state, plan) = plan.into_parts();

    if debug {
        println!("=== Logical plan ===\n{plan}\n");
    }

    let plan = state.optimize(&plan)?;
    if debug {
        println!("=== Optimized logical plan ===\n{plan}\n");
    }
    let physical_plan = state.create_physical_plan(&plan).await?;
    if debug {
        println!(
            "=== Physical plan ===\n{}\n",
            displayable(physical_plan.as_ref()).indent(true)
        );
    }
    let result = collect(physical_plan.clone(), state.task_ctx()).await?;
    if debug {
        println!(
            "=== Physical plan with metrics ===\n{}\n",
            DisplayableExecutionPlan::with_metrics(physical_plan.as_ref()).indent(true)
        );
        if !result.is_empty() {
            // do not call print_batches if there are no batches as the result is confusing
            // and makes it look like there is a batch with no columns
            //pretty::print_batches(&result)?;
        }
    }
    Ok(result)
}

struct QueryResult {
    elapsed: std::time::Duration,
    row_count: usize,
}

async fn run_query(ctx: &SessionContext, fv: FilterVariant) -> Result<Vec<RecordBatch>> {
    unsafe {
        FC.var = fv;
    }

    let sql = "SELECT custom_filter_udf(production_year::INT UNSIGNED) FROM title";

    execute_query(&ctx, sql).await
}

async fn benchmark_query(
    ctx: &SessionContext,
    fv: FilterVariant,
) -> Result<Vec<QueryResult>> {
    let mut millis = vec![];
    // run benchmark
    let mut query_results = vec![];
    for i in 0..7 {
        let start = Instant::now();

        let result = run_query(ctx, fv).await?;

        let elapsed = start.elapsed(); //.as_secs_f64() * 1000.0;
        let ms = elapsed.as_secs_f64() * 1000.0;
        millis.push(ms);
        let row_count = result.iter().map(|b| b.num_rows()).sum();
        //println!("Query {fv:?} iteration {i} took {ms:.1} ms and returned {row_count} rows");
        query_results.push(QueryResult { elapsed, row_count });
    }

    let avg = millis.iter().sum::<f64>() / millis.len() as f64;
    println!("Query {fv:?} avg time: {avg:.2} ms");

    Ok(query_results)
}

// TODO relative
const FILE_PATH: &str =
    "/home/laurin/github/datafusion/benchmarks/data/imdb/title.parquet";

mod test {
    use crate::benchmark_query;
    use crate::IAAOffloadUDF;
    use crate::FILE_PATH;
    use crate::{run_query, FilterVariant, RunVariant, FC};
    use datafusion::logical_expr::ScalarUDF;
    use datafusion::prelude::{SessionConfig, SessionContext};

    #[tokio::test]
    async fn validate() {
        let s = 1_048_576;
        unsafe {
            FC.run_var = RunVariant::Validation;
            FC.batch_size = s;
        }

        let cfg = SessionConfig::new()
            .with_target_partitions(1)
            .with_batch_size(s)
            .with_collect_statistics(true);

        let ctx = SessionContext::new_with_config(cfg);

        ctx.register_udf(ScalarUDF::from(IAAOffloadUDF::new()).clone());

        ctx.register_parquet("title", FILE_PATH, Default::default())
            .await
            .unwrap();

        let _ = run_query(&ctx, FilterVariant::NaiveCpu).await.unwrap();

        let _ = run_query(&ctx, FilterVariant::IaaOffload).await.unwrap();
    }

    #[tokio::test]
    async fn old() {
        /*

        Tracy:
            TRACY_NO_EXIT=1 cargo run --example iaa_offload

        */

        //let batchsizes = vec![8192, 16_384, 32_768, 65_536, 262_144, 524_288, 1_048_576];
        let batchsizes = vec![1_048_576];
        for i in 0..batchsizes.len() {
            let s = batchsizes[i];
            println!("Running with batchsize = {}", s);

            unsafe { FC.batch_size = s };

            let cfg = SessionConfig::new()
                .with_target_partitions(1)
                .with_batch_size(s)
                .with_collect_statistics(true);

            let ctx = SessionContext::new_with_config(cfg);

            // register the UDF with the context so it can be invoked by name and from SQL
            ctx.register_udf(ScalarUDF::from(IAAOffloadUDF::new()).clone());

            // Register the movies CSV file
            ctx.register_parquet("title", FILE_PATH, Default::default())
                .await
                .unwrap();

            // Run 1
            let _ = benchmark_query(&ctx, FilterVariant::NaiveCpu)
                .await
                .unwrap();
            let _ = benchmark_query(&ctx, FilterVariant::IaaOffload)
                .await
                .unwrap();
            //let _ = benchmark_query(&ctx, FilterVariant::IaaOffloadHw).await.unwrap();

            // Run 2
            let _ = benchmark_query(&ctx, FilterVariant::NaiveCpu)
                .await
                .unwrap();
            let _ = benchmark_query(&ctx, FilterVariant::IaaOffload)
                .await
                .unwrap();
            //let _ = benchmark_query(&ctx, FilterVariant::IaaOffloadHw).await.unwrap();
        }

        /*

        TPC-H SF-100
        Alle parallel oder seriell?

        */
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("iaa_offload");
    group.throughput(criterion::Throughput::Bytes(2528312 * 4));
    for b in vec![8192, 16_384, 32_768, 65_536, 262_144, 524_288, 1_048_576] {
        unsafe {
            FC.run_var = RunVariant::Benchmark;
            FC.batch_size = b;
        }

        let cfg = SessionConfig::new()
            .with_target_partitions(1)
            .with_batch_size(b);

        let ctx = SessionContext::new_with_config(cfg);

        ctx.register_udf(ScalarUDF::from(IAAOffloadUDF::new()).clone());

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(ctx.register_parquet("title", FILE_PATH, Default::default()))
            .unwrap();

        // CPU
        group.bench_with_input(format!("cpu/{}", b), &b, |b, s| {
            b.iter(|| {
                criterion::black_box(
                    rt.block_on(run_query(&ctx, FilterVariant::NaiveCpu)),
                )
                .unwrap();
            })
        });
        // QPL
        group.bench_with_input(format!("qpl/{}", b), &b, |b, s| {
            b.iter(|| {
                criterion::black_box(
                    rt.block_on(run_query(&ctx, FilterVariant::IaaOffload)),
                )
                .unwrap();
            })
        });
        // IAA
        // TODO
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
