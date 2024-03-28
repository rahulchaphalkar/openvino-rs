use crate::element_type::ElementType;
use crate::shape::Shape;
use crate::{drop_using_function, try_unsafe, util::Result};
use openvino_sys::{
    self, ov_shape_t, ov_tensor_create, ov_tensor_create_from_host_ptr, ov_tensor_data,
    ov_tensor_free, ov_tensor_get_byte_size, ov_tensor_get_element_type, ov_tensor_get_shape,
    ov_tensor_get_size, ov_tensor_set_shape, ov_tensor_t,
};

/// See [`Blob`](https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1Blob.html).
pub struct Tensor {
    pub instance: *mut ov_tensor_t,
}
drop_using_function!(Tensor, ov_tensor_free);

impl Tensor {
    /// Create a new [`Tensor`] by copying data in to the OpenVINO-allocated memory.
    ///
    /// # Panics
    ///
    /// This function will panic if the number of bytes passed in `data` does not match the expected
    /// size of the tensor `description`.
    pub fn new(data_type: ElementType, shape: Shape) -> Result<Self> {
        let mut tensor = std::ptr::null_mut();
        let element_type = data_type as u32;
        try_unsafe!(ov_tensor_create(
            element_type,
            shape.instance,
            std::ptr::addr_of_mut!(tensor),
        ));
        Ok(Self { instance: tensor })
    }

    pub fn new_from_host_ptr(
        data_type: ElementType,
        shape: Shape,
        data: &[u8],
    ) -> Result<Self> {
        let mut tensor: *mut ov_tensor_t = std::ptr::null_mut();
        let element_type: u32 = data_type as u32;
        let buffer = data.as_ptr() as *mut std::os::raw::c_void;
        try_unsafe!(ov_tensor_create_from_host_ptr(
            element_type,
            shape.instance,
            buffer,
            std::ptr::addr_of_mut!(tensor)
        ));
        Ok(Self { instance: tensor })
    }

    pub fn set_shape(&self, shape: &Shape) -> Result<Self> {
        try_unsafe!(ov_tensor_set_shape(self.instance, shape.instance));
        Ok(Self {
            instance: self.instance,
        })
    }

    pub fn get_shape(&self) -> Result<Shape> {
        let mut instance = ov_shape_t {
            rank: 0,
            dims: std::ptr::null_mut(),
        };
        try_unsafe!(ov_tensor_get_shape(
            self.instance,
            std::ptr::addr_of_mut!(instance),
        ));
        Ok(Shape { instance })
    }

    pub fn get_element_type(&self) -> Result<u32> {
        let mut element_type = ElementType::Undefined as u32;
        try_unsafe!(ov_tensor_get_element_type(
            self.instance,
            std::ptr::addr_of_mut!(element_type),
        ));
        Ok(element_type)
    }

    pub fn get_size(&self) -> Result<usize> {
        let mut elements_size = 0;
        try_unsafe!(ov_tensor_get_size(self.instance, std::ptr::addr_of_mut!(elements_size)));
        Ok(elements_size)
    }

    pub fn get_byte_size(&self) -> Result<usize> {
        let mut byte_size: usize = 0;
        try_unsafe!(ov_tensor_get_byte_size(
            self.instance,
            std::ptr::addr_of_mut!(byte_size),
        ));
        Ok(byte_size)
    }

    pub fn get_data<T>(&mut self) -> Result<&mut [T]> {
        let mut data = std::ptr::null_mut();
        try_unsafe!(ov_tensor_data(
            self.instance,
            //&mut data,
            std::ptr::addr_of_mut!(data),
        ));
        let size = self.get_byte_size()? / std::mem::size_of::<T>();
        let slice = unsafe { std::slice::from_raw_parts_mut(data.cast::<T>(), size) };
        Ok(slice)
    }

    pub fn buffer_mut(&mut self) -> Result<&mut [u8]> {
        let mut buffer = std::ptr::null_mut();
        try_unsafe!(ov_tensor_data(
            self.instance,
            std::ptr::addr_of_mut!(buffer)
        ))?;
        let size = self.get_byte_size()?;
        let slice = unsafe {
            std::slice::from_raw_parts_mut(buffer.cast::<u8>(), size)
        };
        Ok(slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Shape, ElementType};

    #[test]
    fn test_create_tensor() {
        let shape = Shape::new(&vec![1, 3, 227, 227]);
        let tensor = Tensor::new(ElementType::F32, shape).unwrap();
        assert!(!tensor.instance.is_null());
    }

    #[test]
    fn test_get_shape() {
        let tensor = Tensor::new(ElementType::F32, Shape::new(&vec![1, 3, 227, 227])).unwrap();
        let shape = tensor.get_shape().unwrap();
        assert_eq!(shape.get_rank().unwrap(), 4);
    }

    #[test]
    fn test_get_element_type() {
        let tensor = Tensor::new(ElementType::F32, Shape::new(&vec![1, 3, 227, 227])).unwrap();
        let element_type = tensor.get_element_type().unwrap();
        assert_eq!(element_type, ElementType::F32 as u32);
    }

    #[test]
    fn test_get_size() {
        let tensor = Tensor::new(ElementType::F32, Shape::new(&vec![1, 3, 227, 227])).unwrap();
        let size = tensor.get_size().unwrap();
        assert_eq!(size, 1 * 3 * 227 * 227);
    }

    #[test]
    fn test_get_byte_size() {
        let tensor = Tensor::new(ElementType::F32, Shape::new(&vec![1, 3, 227, 227])).unwrap();
        let byte_size = tensor.get_byte_size().unwrap();
        assert_eq!(byte_size, 1 * 3 * 227 * 227 * std::mem::size_of::<f32>() as usize);
    }

}
