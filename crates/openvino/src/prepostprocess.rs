/// See [`Pre Post Process`](https://docs.openvino.ai/2023.3/api/c_cpp_api/group__ov__prepostprocess__c__api.html).
use std::ffi::CString;
use openvino_sys::{
    ov_preprocess_input_info_free, ov_preprocess_input_info_get_model_info, ov_preprocess_input_info_get_preprocess_steps, ov_preprocess_input_info_get_tensor_info, ov_preprocess_input_info_t, ov_preprocess_input_model_info_free, ov_preprocess_input_model_info_set_layout, ov_preprocess_input_model_info_t, ov_preprocess_input_tensor_info, ov_preprocess_input_tensor_info_free, ov_preprocess_input_tensor_info_set_from, ov_preprocess_input_tensor_info_set_layout, ov_preprocess_input_tensor_info_t, ov_preprocess_output_info_free, ov_preprocess_output_info_get_tensor_info, ov_preprocess_output_info_t, ov_preprocess_output_set_element_type, ov_preprocess_output_tensor_info_free, ov_preprocess_output_tensor_info_t, ov_preprocess_prepostprocessor_build, ov_preprocess_prepostprocessor_create, ov_preprocess_prepostprocessor_free, ov_preprocess_prepostprocessor_get_input_info, ov_preprocess_prepostprocessor_get_input_info_by_index, ov_preprocess_prepostprocessor_get_input_info_by_name, ov_preprocess_prepostprocessor_get_output_info_by_index, ov_preprocess_prepostprocessor_get_output_info_by_name, ov_preprocess_prepostprocessor_t, ov_preprocess_preprocess_steps_convert_element_type, ov_preprocess_preprocess_steps_convert_layout, ov_preprocess_preprocess_steps_free, ov_preprocess_preprocess_steps_resize, ov_preprocess_preprocess_steps_t, ov_preprocess_resize_algorithm_e_RESIZE_LINEAR
};

use crate::{drop_using_function, layout::Layout, try_unsafe, ElementType, Model, Tensor};

/// The PrePostprocess struct represents pre and post-processing capabilities
#[derive(Debug)]
pub struct PrePostprocess {
    pub(crate) instance: *mut ov_preprocess_prepostprocessor_t,
}
drop_using_function!(PrePostprocess, ov_preprocess_prepostprocessor_free);

/// The PreprocessInputInfo struct represents input information for pre/postprocessing.
pub struct PreprocessInputInfo {
    pub(crate) instance: *mut ov_preprocess_input_info_t,
}
drop_using_function!(PreprocessInputInfo, ov_preprocess_input_info_free);

/// The PreprocessOutputInfo struct represents output information for pre/postprocessing.
pub struct PreprocessOutputInfo {
    pub(crate) instance: *mut ov_preprocess_output_info_t,
}
drop_using_function!(PreprocessOutputInfo, ov_preprocess_output_info_free);

/// The PreprocessSteps struct represents preprocessing steps.
pub struct PreprocessSteps {
    pub(crate) instance: *mut ov_preprocess_preprocess_steps_t,
}
drop_using_function!(PreprocessSteps, ov_preprocess_preprocess_steps_free);

/// The PreprocessInputModelInfo struct represents input model information for pre/postprocessing.
pub struct PreprocessInputModelInfo {
    pub(crate) instance: *mut ov_preprocess_input_model_info_t,
}
drop_using_function!(PreprocessInputModelInfo, ov_preprocess_input_model_info_free);

/// The PreprocessInputTensorInfo struct represents input tensor information for pre/postprocessing.
pub struct PreprocessInputTensorInfo {
    pub(crate) instance: *mut ov_preprocess_input_tensor_info_t,
}
drop_using_function!(
    PreprocessInputTensorInfo,
    ov_preprocess_input_tensor_info_free
);

/// The PreprocessOutputTensorInfo struct represents output tensor information for pre/postprocessing.
pub struct PreprocessOutputTensorInfo {
    pub(crate) instance: *mut ov_preprocess_output_tensor_info_t,
}
drop_using_function!(PreprocessOutputTensorInfo, ov_preprocess_output_tensor_info_free);

impl PreprocessInputModelInfo {
    /// Sets the layout for the model information obj.
    pub fn model_info_set_layout(&self, layout: Layout) -> () {
        let code = try_unsafe!(ov_preprocess_input_model_info_set_layout(
            self.instance,
            layout.instance
        ));
        assert_eq!(code, Ok(()));
    }
}

impl PreprocessInputTensorInfo {
    /// Creates a new PreprocessInputTensorInfo instance.
    pub fn new() -> Self {
        PreprocessInputTensorInfo {
            instance: std::ptr::null_mut(),
        }
    }

    /// Sets the layout for the input tensor.
    pub fn preprocess_input_tensor_set_layout(
        &self,
        layout: &Layout,
    ) -> () {
        let code = try_unsafe!(ov_preprocess_input_tensor_info_set_layout(
            self.instance,
            layout.instance
        ));
        assert_eq!(code, Ok(()));
    }

    /// Sets the input tensor info from an existing tensor.
    pub fn preprocess_input_tensor_set_from(&mut self, tensor: &Tensor) -> () {
        let code = try_unsafe!(ov_preprocess_input_tensor_info_set_from(
            self.instance,
            tensor.instance
        ));
        assert_eq!(code, Ok(()));
    }
}

impl PrePostprocess {
    /// Creates a new PrePostprocess instance for the given model.
    pub fn new(model: &Model) -> Self {
        let mut preprocess = std::ptr::null_mut();
        let code = try_unsafe!(ov_preprocess_prepostprocessor_create(
            model.instance,
            std::ptr::addr_of_mut!(preprocess)
        ));
        assert_eq!(code, Ok(()));
        Self {
            instance: preprocess,
        }
    }

    /// Retrieves the input information by index.
    pub fn get_input_info_by_index(
        &self,
        index: usize,
    ) -> PreprocessInputInfo {
        let mut input_info = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_get_input_info_by_index(
            self.instance,
            index,
            std::ptr::addr_of_mut!(input_info)
        ));
        PreprocessInputInfo {
            instance: input_info,
        }
    }

    /// Retrieves the input information by name.
    pub fn get_input_info_by_name(&self, name: &str) -> PreprocessInputInfo {
        let mut input_info = std::ptr::null_mut();
        let c_layout_desc = CString::new(name).unwrap();
        try_unsafe!(ov_preprocess_prepostprocessor_get_input_info_by_name(
            self.instance,
            c_layout_desc.as_ptr(),
            std::ptr::addr_of_mut!(input_info)
        ));
        PreprocessInputInfo {
            instance: input_info,
        }
    }

    /// Retrieves the output information by name.
    pub fn get_output_info_by_name(
        &self,
        name: &str,
    ) -> PreprocessOutputInfo {
        let mut output_info = std::ptr::null_mut();
        let c_layout_desc = CString::new(name).unwrap();
        try_unsafe!(ov_preprocess_prepostprocessor_get_output_info_by_name(
            self.instance,
            c_layout_desc.as_ptr(),
            std::ptr::addr_of_mut!(output_info)
        ));
        PreprocessOutputInfo {
            instance: output_info,
        }
    }

    /// Retrieves the output information by index.
    pub fn get_output_info_by_index(
        &self,
        index: usize,
    ) -> PreprocessOutputInfo {
        let mut output_info = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_get_output_info_by_index(
            self.instance,
            index,
            std::ptr::addr_of_mut!(output_info)
        ));
        PreprocessOutputInfo {
            instance: output_info,
        }
    }

    /// Retrieves the input information.
    pub fn get_input_info(&self) -> PreprocessInputInfo {
        let mut input_info = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_get_input_info(
            self.instance,
            std::ptr::addr_of_mut!(input_info)
        ));
        assert!(!input_info.is_null());
        PreprocessInputInfo {
            instance: input_info,
        }
    }

    /// Builds model with all steps from pre/postprocessing.
    pub fn build(&self, new_model: &mut Model) -> () {
        let code = try_unsafe!(ov_preprocess_prepostprocessor_build(
            self.instance,
            std::ptr::addr_of_mut!(new_model.instance)
        ));
        assert_eq!(code, Ok(()));
    }
}

impl PreprocessSteps {
    /// Resizes data in tensor
    pub fn preprocess_steps_resize(&mut self, resize_algo: u32) -> () {
        let code = try_unsafe!(ov_preprocess_preprocess_steps_resize(
            self.instance,
            resize_algo,
        ));
        assert_eq!(code, Ok(()));
    }

    /// Converts the layout of data in tensor
    pub fn preprocess_convert_layout(&self, layout: Layout) -> () {
        let code = try_unsafe!(ov_preprocess_preprocess_steps_convert_layout(
            self.instance,
            layout.instance,
        ));
        assert_eq!(code, Ok(()));
    }

    /// Converts the element type of data in tensor
    pub fn preprocess_convert_element_type(&self, element_type: ElementType) -> () {
        let code = try_unsafe!(ov_preprocess_preprocess_steps_convert_element_type(
            self.instance,
            element_type as u32
        ));
        assert_eq!(code, Ok(()));
    }
}

impl PreprocessOutputTensorInfo {
    /// Sets the element type for output tensor info
    pub fn preprocess_set_element_type(&self, element_type: ElementType) -> () {
        let code = try_unsafe!(ov_preprocess_output_set_element_type(
            self.instance,
            element_type as u32
        ));
        assert_eq!(code, Ok(()));
    }
}

impl PreprocessOutputInfo {
    /// Retrieves preprocess output tensor information.
    pub fn get_output_info_get_tensor_info(&self) -> PreprocessOutputTensorInfo {
        let mut preprocess_output_tensor_info: *mut ov_preprocess_output_tensor_info_t = std::ptr::null_mut();
        let code = try_unsafe!(ov_preprocess_output_info_get_tensor_info(
            self.instance,
            std::ptr::addr_of_mut!(preprocess_output_tensor_info)
        ));
        assert_eq!(code, Ok(()));
        PreprocessOutputTensorInfo {
            instance: preprocess_output_tensor_info,
        }
    }
}

impl PreprocessInputInfo {
    /// Retrieves the preprocessing model input information.
    pub fn get_model_info(&self) -> PreprocessInputModelInfo {
        let mut model_info = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_input_info_get_model_info(
            self.instance,
            std::ptr::addr_of_mut!(model_info)
        ));
        PreprocessInputModelInfo {
            instance: model_info,
        }
    }

    /// Retrieves the input tensor information.
    pub fn preprocess_input_info_get_tensor_info(&self) -> PreprocessInputTensorInfo {
        let mut preprocess_input_tensor_info: *mut ov_preprocess_input_tensor_info_t = std::ptr::null_mut();
        let code = try_unsafe!(ov_preprocess_input_info_get_tensor_info(
            self.instance,
            std::ptr::addr_of_mut!(preprocess_input_tensor_info)
        ));
        assert_eq!(code, Ok(()));
        PreprocessInputTensorInfo {
            instance: preprocess_input_tensor_info,
        }
    }

    /// Retrieves preprocessing steps object.
    pub fn get_preprocess_steps(&self) -> PreprocessSteps {
        let mut preprocess_steps = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_input_info_get_preprocess_steps(
            self.instance,
            std::ptr::addr_of_mut!(preprocess_steps)
        ));
        PreprocessSteps {
            instance: preprocess_steps,
        }
    }
}
