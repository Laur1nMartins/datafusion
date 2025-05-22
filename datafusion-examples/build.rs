use std::path::PathBuf;

// NOTE make sure this file is called 'build.rs'!
// only then will it be called automatically by the compiler

fn main() {
    println!("cargo:rustc-link-lib=qpl");

    println!("cargo:rustc-flags=-l dylib=stdc++");

    // Generate bindings
    /*
    // NOTE this is only needed to run once
    let bindings = bindgen::Builder::default()
        .header("/usr/local/include/qpl/qpl.h") // Input header file
        .generate_comments(true) // Include comments from the header
        .clang_arg("-I/usr/local/include/qpl/") // Include path for additional headers
        .generate()
        .expect("Unable to generate QPL bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file
    let out_path = PathBuf::from("./");
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
    */
}
