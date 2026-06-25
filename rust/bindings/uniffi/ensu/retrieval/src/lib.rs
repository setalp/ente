#![allow(unexpected_cfgs)]

mod api;

uniffi::setup_scaffolding!("retrieval");

pub use api::*;
