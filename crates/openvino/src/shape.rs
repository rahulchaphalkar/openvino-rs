use std::convert::TryInto;
use openvino_sys::{ov_shape_create, ov_shape_free, ov_shape_t};
use crate::{try_unsafe,util::Result};
pub struct Shape {
    pub(crate) instance: ov_shape_t,
}

impl Drop for Shape {
    fn drop(&mut self) {
        let code = unsafe { ov_shape_free(std::ptr::addr_of_mut!(self.instance)) };
        assert_eq!(code, 0);
        debug_assert!(self.instance.dims.is_null());
        debug_assert_eq!(self.instance.rank, 0)
    }
}

impl Shape {
    pub fn new(dimensions: &Vec<i64>) -> Self {
        let mut shape = ov_shape_t {
            rank: 8,
            dims: std::ptr::null_mut(),
        };
        let code = try_unsafe!(ov_shape_create(
            dimensions.len().try_into().unwrap(),
            dimensions.as_ptr(),
            std::ptr::addr_of_mut!(shape)
        ));
        assert_eq!(code, Ok(()));
        Self { instance: shape }
    }

    pub fn get_rank(&self) -> Result<i64> {
        Ok(self.instance.rank)
    }

}
