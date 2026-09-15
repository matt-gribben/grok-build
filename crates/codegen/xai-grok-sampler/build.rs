fn main() {
    println!("cargo:rerun-if-changed=proto/agent.proto");

    let protoc = protoc_bin_vendored::protoc_bin_path()
        .expect("locate the vendored protoc compiler for Cursor's pinned schema");
    prost_build::Config::new()
        .protoc_executable(protoc)
        .compile_protos(&["proto/agent.proto"], &["proto"])
        .expect("compile the pinned Cursor agent schema");
}
