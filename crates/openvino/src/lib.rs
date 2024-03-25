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

mod tensor;
mod core;
mod error;
mod model;
mod request;
mod element_type;
mod layout;
mod port;
mod prepostprocess;
mod shape;
mod util;

pub use crate::core::Core;
pub use tensor::Tensor;
pub use element_type::ElementType;
pub use error::{InferenceError, LoadingError, SetupError};
pub use layout::Layout;
pub use model::{CompiledModel, Model};
pub use prepostprocess::{PrePostprocess, PreprocessInputInfo, PreprocessInputModelInfo, PreprocessInputTensorInfo, PreprocessOutputInfo, PreprocessOutputTensorInfo, PreprocessSteps};
pub use shape::Shape;
pub use request::InferRequest;
pub use port::Port;

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
