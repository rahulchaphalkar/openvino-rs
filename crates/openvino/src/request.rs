use crate::blob::Tensor;
use crate::{cstr, drop_using_function, try_unsafe, util::Result};
// use openvino_sys::{
//     ie_infer_request_free, ie_infer_request_get_blob, ie_infer_request_infer,
//     /*ie_infer_request_set_batch,*/ ie_infer_request_set_blob, ie_infer_request_t,
// };

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
    /// Set the batch size of the inference requests.
    // pub fn set_batch_size(&mut self, size: usize) -> Result<()> {
    //     try_unsafe!(ie_infer_request_set_batch(self.instance, size))
    // }

    /// Assign a [Blob] to the input (i.e. `name`) on the network.
    pub fn set_tensor(&mut self, name: &str, tensor: &Tensor) -> () {
        //let infer_req = InferRequest{instance: std::ptr::null_mut()};
        try_unsafe!(ov_infer_request_set_tensor(
            //self.instance,
            self.instance,
            cstr!(name),
            tensor.instance
        ));
    }
    // pub fn set_blob(&mut self, name: &str, blob: &Blob) -> Result<()> {
    //     try_unsafe!(ov_infer_request_set_tensor(
    //         self.instance,
    //         cstr!(name),
    //         blob.instance
    //     ))
    // }

    /// Retrieve a [Blob] from the output (i.e. `name`) on the network.
    pub fn get_blob(&mut self, name: &str) -> Result<Tensor> {
        //let mut tensor: Tensor;
        //let mut tensor: *mut Tensor = std::ptr::null_mut();
        let mut init_tensor = ov_tensor_t { _unused: [] };
        let mut tensor = Tensor {
            instance: std::ptr::addr_of_mut!(init_tensor),
        };
        try_unsafe!(ov_infer_request_get_tensor(
            self.instance,
            cstr!(name),
            std::ptr::addr_of_mut!((tensor).instance)
        ))?;
        Ok(tensor)
    }

    pub fn get_tensor(&self, name: String) -> Result<Tensor> {
        let mut tensor = std::ptr::null_mut();
        try_unsafe!(ov_infer_request_get_tensor(
            self.instance,
            cstr!(name),
            std::ptr::addr_of_mut!(tensor)
        ))?;
        Ok(Tensor { instance: tensor })
    }

    // pub fn get_blob(&mut self, name: &str) -> Result<Blob> {
    //     let mut instance = std::ptr::null_mut();
    //     try_unsafe!(ov_infer_request_get_tensor(
    //         self.instance,
    //         cstr!(name),
    //         std::ptr::addr_of_mut!(instance)
    //     ))?;
    //     Ok(unsafe { Blob::from_raw_pointer(instance) })
    // }

    /// Execute the inference request.
    pub fn infer(&mut self) -> Result<()> {
        try_unsafe!(ov_infer_request_infer(self.instance))
    }
}
