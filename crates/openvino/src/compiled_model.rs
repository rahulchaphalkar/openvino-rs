//! Contains the network representations in OpenVINO:
//!  - [`CNNNetwork`] is the OpenVINO representation of a neural network
//!  - [`ExecutableNetwork`] is the compiled representation of a [`CNNNetwork`] for a device.

use crate::request::InferRequest;
use crate::{cstr, drop_using_function, try_unsafe, util::Result};
//use crate::{Layout, Precision, ResizeAlgorithm};
// use openvino_sys::{
//     ie_exec_network_create_infer_request, ie_exec_network_free, ie_executable_network_t,
//     ie_network_free, ie_network_get_input_name, ie_network_get_inputs_number,
//     ie_network_get_output_name, ie_network_get_outputs_number, ie_network_name_free,
//     ie_network_set_input_layout, ie_network_set_input_precision,
//     ie_network_set_input_resize_algorithm, ie_network_set_output_precision, ie_network_t,
// };

use openvino_sys::{
    ov_compiled_model_create_infer_request, ov_compiled_model_free, ov_compiled_model_t,
    ov_model_free, ov_compiled_model_input_by_name, ov_model_inputs_size,
    ov_model_const_output_by_name, ov_model_outputs_size, ov_compiled_model_inputs_size, ov_compiled_model_outputs_size, 
    
    /*ie_network_name_free,
    ie_network_set_input_layout, ie_network_set_input_precision,
    ie_network_set_input_resize_algorithm, ie_network_set_output_precision,*/ ov_model_t,
};

use std::ffi::CStr;

/// See
/// [`CNNNetwork`](https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1CNNNetwork.html).
pub struct CompiledModel {
    pub(crate) instance: *mut ov_compiled_model_t,
}
drop_using_function!(CompiledModel, ov_model_free);

unsafe impl Send for CompiledModel {}
unsafe impl Sync for CompiledModel {}

impl CompiledModel {
    /// Retrieve the number of network inputs.
    pub fn get_inputs_len(&self) -> Result<usize> {
        let mut num: usize = 0;
        try_unsafe!(ov_compiled_model_inputs_size(self.instance, &mut num))?;
        Ok(num)
    }

    /// Retrieve the number of network outputs.
    pub fn get_outputs_len(&self) -> Result<usize> {
        let mut num: usize = 0;
        try_unsafe!(ov_compiled_model_outputs_size(self.instance, &mut num))?;
        Ok(num)
    }
/*
    /// Retrieve the name identifying the input tensor at `index`.
    pub fn get_input_name(&self, index: usize) -> Result<String> {
        let mut c_name = std::ptr::null_mut();

        try_unsafe!(ov_compiled_model_input_by_name(
            self.instance,
            index,
            std::ptr::addr_of_mut!(c_name)
        ))?;
        let rust_name = unsafe { CStr::from_ptr(c_name) }
            .to_string_lossy()
            .into_owned();
        unsafe { ov_model_free(std::ptr::addr_of_mut!(c_name)) };
        debug_assert!(c_name.is_null());
        Ok(rust_name)
    }
*/
/*
    /// Retrieve the name identifying the output tensor at `index`.
    pub fn get_output_name(&self, index: usize) -> Result<String> {
        let mut c_name = std::ptr::null_mut();
        try_unsafe!(ov_model_const_output_by_name(
            self.instance,
            index,
            std::ptr::addr_of_mut!(c_name)
        ))?;
        let rust_name = unsafe { CStr::from_ptr(c_name) }
            .to_string_lossy()
            .into_owned();
        unsafe { ov_model_free(std::ptr::addr_of_mut!(c_name)) };
        debug_assert!(c_name.is_null());
        Ok(rust_name)
    }
*/
    // /// Configure a resize algorithm for the input tensor at `input_name`.
    // pub fn set_input_resize_algorithm(
    //     &mut self,
    //     input_name: &str,
    //     algorithm: ResizeAlgorithm,
    // ) -> Result<()> {
    //     try_unsafe!(ie_network_set_input_resize_algorithm(
    //         self.instance,
    //         cstr!(input_name),
    //         algorithm
    //     ))
    // }

    // /// Configure a layout for the input tensor at `input_name`.
    // pub fn set_input_layout(&mut self, input_name: &str, layout: Layout) -> Result<()> {
    //     try_unsafe!(ie_network_set_input_layout(
    //         self.instance,
    //         cstr!(input_name),
    //         layout
    //     ))
    // }

    // /// Configure the precision for the input tensor at `input_name`.
    // pub fn set_input_precision(&mut self, input_name: &str, precision: Precision) -> Result<()> {
    //     try_unsafe!(ie_network_set_input_precision(
    //         self.instance,
    //         cstr!(input_name),
    //         precision
    //     ))
    // }

    // /// Configure the precision for the output tensor at `output_name`.
    // pub fn set_output_precision(&mut self, output_name: &str, precision: Precision) -> Result<()> {
    //     try_unsafe!(ie_network_set_output_precision(
    //         self.instance,
    //         cstr!(output_name),
    //         precision
    //     ))
    // }
}

/// See
/// [`ExecutableNetwork`](https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1ExecutableNetwork.html).
pub struct ExecutableNetwork {
    pub(crate) instance: *mut ov_compiled_model_t,
}
drop_using_function!(ExecutableNetwork, ov_compiled_model_free);

unsafe impl Send for ExecutableNetwork {}

impl ExecutableNetwork {
    /// Create an [`InferRequest`].
    pub fn create_infer_request(&mut self) -> Result<InferRequest> {
        let mut instance = std::ptr::null_mut();
        try_unsafe!(ov_compiled_model_create_infer_request(
            self.instance,
            std::ptr::addr_of_mut!(instance)
        ))?;
        Ok(InferRequest { instance })
    }
}
