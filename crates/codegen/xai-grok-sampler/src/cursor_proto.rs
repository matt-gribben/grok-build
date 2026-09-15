#![allow(clippy::all, warnings)]

pub(crate) mod agent {
    pub(crate) mod v1 {
        include!(concat!(env!("OUT_DIR"), "/agent.v1.rs"));
    }
}
