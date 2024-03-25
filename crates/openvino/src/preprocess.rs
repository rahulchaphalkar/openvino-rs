use crate::{drop_using_function, layout::Layout, try_unsafe, PreprocessInputInfo, Tensor};
use openvino_sys::{
    ov_preprocess_input_info_get_tensor_info, ov_preprocess_input_tensor_info_free,
    ov_preprocess_input_tensor_info_set_from, ov_preprocess_input_tensor_info_set_layout,
    ov_preprocess_input_tensor_info_t,
};
pub struct PreprocessInputTensorInfo {
    pub(crate) instance: *mut ov_preprocess_input_tensor_info_t,
}
drop_using_function!(
    PreprocessInputTensorInfo,
    ov_preprocess_input_tensor_info_free
);

impl PreprocessInputTensorInfo {
    pub fn new() -> Self {
        PreprocessInputTensorInfo {
            instance: std::ptr::null_mut(),
        }
    }

    pub fn preprocess_input_tensor_set_layout(
        preprocess_input_tensor_info: &PreprocessInputTensorInfo,
        layout: &Layout,
    ) -> () {
        let code = try_unsafe!(ov_preprocess_input_tensor_info_set_layout(
            preprocess_input_tensor_info.instance,
            layout.instance
        ));
        assert_eq!(code, Ok(()));
    }

    pub fn preprocess_input_info_get_tensor_info(input_info: &PreprocessInputInfo) -> Self {
        let mut preprocess_input_tensor_info: *mut ov_preprocess_input_tensor_info_t = std::ptr::null_mut();
        let code = try_unsafe!(ov_preprocess_input_info_get_tensor_info(
            input_info.instance,
            std::ptr::addr_of_mut!(preprocess_input_tensor_info)
        ));
        assert_eq!(code, Ok(()));
        Self {
            instance: preprocess_input_tensor_info,
        }
    }

    pub fn preprocess_input_tensor_set_from(&self, tensor: &Tensor) -> () {
        let code = try_unsafe!(ov_preprocess_input_tensor_info_set_from(
            self.instance,
            tensor.instance
        ));
        assert_eq!(code, Ok(()));
    }
}
