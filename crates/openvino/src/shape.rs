use std::convert::TryInto;

use openvino_sys::{ov_shape_create, ov_shape_free, ov_shape_t};

use crate::try_unsafe;

//#[derive(Debug)]
pub struct Shape {
    //pub(crate) instance: *mut ov_shape_t
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
        //pub fn new(dimensions: &[usize] )-> Self{
        //let mut shape = std::ptr::null_mut();
        let mut shape = ov_shape_t {
            rank: 8,
            dims: std::ptr::null_mut(),
        };
        let code = try_unsafe!(ov_shape_create(
            dimensions.len().try_into().unwrap(),
            //4,
            dimensions.as_ptr(),
            //dims.as_ptr() as *const i64,
            std::ptr::addr_of_mut!(shape)
        ));
        assert_eq!(code, Ok(()));
        Self { instance: shape }
    }

    pub fn rank(&self) -> i64 {
        self.instance.rank
    }

    //getter setters for rank, dims todo
}
