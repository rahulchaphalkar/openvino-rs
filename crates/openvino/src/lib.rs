//! The [openvino] crate provides high-level, ergonomic, safe Rust bindings to OpenVINO. See the
//! repository [README] for more information, such as build instructions.
//!
//! [openvino]: https://crates.io/crates/openvino
//! [README]: https://github.com/intel/openvino-rs
//!
//! Check the loaded version of OpenVINO:
//! ```
//! assert!(openvino::version().starts_with("2"))
//! ```
//!
//! Most interaction with OpenVINO begins with instantiating a [Core]:
//! ```
//! let _ = openvino::Core::new(None).expect("to instantiate the OpenVINO library");
//! ```

//#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::cargo)]
#![allow(
    clippy::must_use_candidate,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::len_without_is_empty
)]

mod blob;
mod core;
mod error;
mod network;
mod request;
//mod tensor_desc;
mod element_type;
mod layout;
mod port;
mod prepostprocess;
mod preprocess;
mod shape;
mod util;
//mod compiled_model;

pub use crate::core::Core;
pub use blob::Tensor;
pub use element_type::ElementType;
pub use error::{InferenceError, LoadingError, SetupError};
pub use layout::Layout;
pub use network::{CompiledModel, Model};
pub use prepostprocess::PrePostprocess;
pub use prepostprocess::PreprocessInputInfo;
pub use prepostprocess::PreprocessInputModelInfo;
pub use prepostprocess::PreprocessOutputInfo;
pub use prepostprocess::PreprocessSteps;
pub use preprocess::PreprocessInputTensorInfo;
pub use shape::Shape;
// Re-publish some OpenVINO enums with a conventional Rust naming (see
// `crates/openvino-sys/build.rs`).
// pub use openvino_sys::{
//     layout_e as Layout, precision_e as Precision, resize_alg_e as ResizeAlgorithm,
// };
pub use request::InferRequest;
//pub use tensor_desc::TensorDesc;

/// Emit the version string of the OpenVINO C API backing this implementation.
///
/// # Panics
///
/// Panics if no OpenVINO library can be found.
pub fn version() -> String {
    use std::ffi::CStr;
    openvino_sys::load().expect("to have an OpenVINO shared library available");
    let mut ov_version = std::ptr::null_mut();
    unsafe { openvino_sys::ov_get_openvino_version(ov_version) };
    let str_version = unsafe { CStr::from_ptr((*ov_version).buildNumber) }
        .to_string_lossy()
        .into_owned();
    unsafe { openvino_sys::ov_version_free(ov_version) };
    str_version
}
