use crate::tensor::Tensor;
use crate::{cstr, drop_using_function, try_unsafe, util::Result};
use openvino_sys::{
    ov_infer_request_free, ov_infer_request_get_tensor, ov_infer_request_infer,
    ov_infer_request_set_tensor, ov_infer_request_t, ov_tensor, ov_tensor_t,
};

/// See
/// [`InferRequest`](https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1InferRequest.html).
pub struct InferRequest {
    pub(crate) instance: *mut ov_infer_request_t,
}
drop_using_function!(InferRequest, ov_infer_request_free);

unsafe impl Send for InferRequest {}
unsafe impl Sync for InferRequest {}

impl InferRequest {
    /// Assign a [tensor] to the input (i.e. `name`) on the network.
    pub fn set_tensor(&mut self, name: &str, tensor: &Tensor) -> () {
        try_unsafe!(ov_infer_request_set_tensor(
            self.instance,
            cstr!(name),
            tensor.instance
        ));
    }

    /// Retrieve a [tensor] from the output (i.e. `name`) on the network.
    pub fn get_tensor(&self, name: String) -> Result<Tensor> {
        let mut tensor = std::ptr::null_mut();
        try_unsafe!(ov_infer_request_get_tensor(
            self.instance,
            cstr!(name),
            std::ptr::addr_of_mut!(tensor)
        ))?;
        Ok(Tensor { instance: tensor })
    }

    /// Execute the inference request.
    pub fn infer(&mut self) -> Result<()> {
        try_unsafe!(ov_infer_request_infer(self.instance))
    }
}
