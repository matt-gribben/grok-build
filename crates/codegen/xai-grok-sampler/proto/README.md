# Cursor agent protocol schema

`agent.proto` is copied from Rahul Arya's `pi-cursor` reference implementation at commit
`324d2061cac068110b2ac9c783ef419b1fdf0501`. The schema is reverse-engineered from Cursor's
client and is not an official or stable Cursor API. The source project is MIT-licensed; its
license is retained in [`PI_CURSOR_LICENSE`](PI_CURSOR_LICENSE).

The Rust types are generated at build time with `prost-build` and the vendored `protoc` binary.
Update this schema only with a reviewed reference snapshot, and keep the source commit and license
notice in sync.
