use std::ffi::CString;

use openvino_sys::{
    ov_preprocess_input_info_free, ov_preprocess_input_info_get_model_info,
    ov_preprocess_input_info_get_preprocess_steps, ov_preprocess_input_info_t,
    ov_preprocess_input_model_info_free, ov_preprocess_input_model_info_set_layout,
    ov_preprocess_input_model_info_t, ov_preprocess_input_tensor_info_free,
    ov_preprocess_input_tensor_info_t, ov_preprocess_output_info_free, ov_preprocess_output_info_t,
    ov_preprocess_prepostprocessor_build, ov_preprocess_prepostprocessor_create,
    ov_preprocess_prepostprocessor_free, ov_preprocess_prepostprocessor_get_input_info,
    ov_preprocess_prepostprocessor_get_input_info_by_index,
    ov_preprocess_prepostprocessor_get_input_info_by_name,
    ov_preprocess_prepostprocessor_get_output_info_by_name, ov_preprocess_prepostprocessor_t,
    ov_preprocess_preprocess_steps_free, ov_preprocess_preprocess_steps_resize,
    ov_preprocess_preprocess_steps_t,
};

use crate::{drop_using_function, layout::Layout, try_unsafe, Model};

pub struct PrePostprocess {
    pub(crate) instance: *mut ov_preprocess_prepostprocessor_t,
}
//can't find free function in generated functions.rs
drop_using_function!(PrePostprocess, ov_preprocess_prepostprocessor_free);

pub struct PreprocessInputInfo {
    pub(crate) instance: *mut ov_preprocess_input_info_t, //pub instance: *mut ov_preprocess_input_info_t
}
drop_using_function!(PreprocessInputInfo, ov_preprocess_input_info_free);

pub struct PreprocessOutputInfo {
    pub(crate) instance: *mut ov_preprocess_output_info_t, //pub instance: *mut ov_preprocess_input_info_t
}
drop_using_function!(PreprocessOutputInfo, ov_preprocess_output_info_free);

pub struct PreprocessSteps {
    pub(crate) instance: *mut ov_preprocess_preprocess_steps_t,
}
drop_using_function!(PreprocessSteps, ov_preprocess_preprocess_steps_free);

impl PreprocessInputInfo {
    // pub fn new() -> Self {
    //     PreprocessInputInfo {
    //         instance: std::ptr::null_mut()
    //     }
    // }
}

pub struct PreprocessInputModelInfo {
    //pub(crate) instance: *mut ov_preprocess_input_model_info_t
    pub instance: *mut ov_preprocess_input_model_info_t,
}
drop_using_function!(
    PreprocessInputModelInfo,
    ov_preprocess_input_model_info_free
);

// pub struct PreprocessInputTensorInfo {
//     pub(crate) instance: *mut ov_preprocess_input_tensor_info_t
// }
// drop_using_function!(PreprocessInputTensorInfo, ov_preprocess_input_tensor_info_free);

// impl PreprocessInputTensorInfo {

// }

impl PrePostprocess {
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

    pub fn get_input_info_by_index(
        prepostprocess: PrePostprocess,
        index: usize,
    ) -> PreprocessInputInfo {
        let mut input_info = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_get_input_info_by_index(
            //self.instance,
            prepostprocess.instance,
            index,
            std::ptr::addr_of_mut!(input_info)
        ));
        PreprocessInputInfo {
            instance: input_info,
        }
    }

    pub fn get_input_info_by_name(&self, name: &str) -> PreprocessInputInfo {
        let mut input_info = std::ptr::null_mut();
        let c_layout_desc = CString::new(name).unwrap();
        try_unsafe!(ov_preprocess_prepostprocessor_get_input_info_by_name(
            self.instance,
            //name.as_ptr() as *const i8,
            c_layout_desc.as_ptr(),
            std::ptr::addr_of_mut!(input_info)
        ));
        PreprocessInputInfo {
            instance: input_info,
        }
    }

    pub fn get_input_info(prepostprocess: &PrePostprocess) -> PreprocessInputInfo {
        let mut input_info = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_prepostprocessor_get_input_info(
            prepostprocess.instance,
            std::ptr::addr_of_mut!(input_info)
        ));
        assert!(!input_info.is_null());
        PreprocessInputInfo {
            instance: input_info,
        }
    }

    pub fn get_model_info(preprocess_input_info: PreprocessInputInfo) -> PreprocessInputModelInfo {
        let mut model_info = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_input_info_get_model_info(
            preprocess_input_info.instance,
            std::ptr::addr_of_mut!(model_info)
        ));
        PreprocessInputModelInfo {
            instance: model_info,
        }
    }

    pub fn model_info_set_layout(model_info: &PreprocessInputModelInfo, layout: Layout) {
        let code = try_unsafe!(ov_preprocess_input_model_info_set_layout(
            model_info.instance,
            layout.instance
        ));
        assert_eq!(code, Ok(()));
    }

    pub fn build(&self /*, preprocess: &PrePostprocess*/, new_model: &mut Model) -> () {
        let code = try_unsafe!(ov_preprocess_prepostprocessor_build(
            //preprocess.instance,
            self.instance,
            std::ptr::addr_of_mut!(new_model.instance) //new_model.instance
        ));
        assert_eq!(code, Ok(()));
        //new_model
    }
}

impl PreprocessSteps {
    pub fn get_preprocess_steps(input_info: &PreprocessInputInfo) -> Self {
        let mut preprocess_steps = std::ptr::null_mut();
        try_unsafe!(ov_preprocess_input_info_get_preprocess_steps(
            input_info.instance,
            std::ptr::addr_of_mut!(preprocess_steps)
        ));
        assert!(!preprocess_steps.is_null());
        Self {
            instance: preprocess_steps,
        }
        //PreprocessSteps { instance: preprocess_steps }
    }

    pub fn preprocess_steps_resize(preprocess_steps: &PreprocessSteps, resize_algo: u32) -> () {
        let code = try_unsafe!(ov_preprocess_preprocess_steps_resize(
            preprocess_steps.instance,
            resize_algo
        ));
        assert_eq!(code, Ok(()));
    }
}

impl PreprocessOutputInfo {
    pub fn get_output_info_by_name(
        prepostprocess: &PrePostprocess,
        name: &str,
    ) -> PreprocessOutputInfo {
        let mut output_info = std::ptr::null_mut();
        let c_layout_desc = CString::new(name).unwrap();
        try_unsafe!(ov_preprocess_prepostprocessor_get_output_info_by_name(
            //self.instance,
            prepostprocess.instance,
            //name.as_ptr() as *const i8,
            c_layout_desc.as_ptr(),
            std::ptr::addr_of_mut!(output_info)
        ));
        PreprocessOutputInfo {
            instance: output_info,
        }
    }
}
