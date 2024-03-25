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
    /// Create a new [`Blob`] by copying data in to the OpenVINO-allocated memory.
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

    pub fn set_shape(&self, shape: Shape) -> Result<Self> {
        try_unsafe!(ov_tensor_set_shape(self.instance, shape.instance,));
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
        let elements_size = std::ptr::null_mut();
        try_unsafe!(ov_tensor_get_size(self.instance, elements_size,));
        Ok(elements_size as usize)
    }

    pub fn get_byte_size(&self) -> Result<usize> {
        let mut byte_size: usize = 0;
        try_unsafe!(ov_tensor_get_byte_size(
            self.instance,
            std::ptr::addr_of_mut!(byte_size),
        ));
        Ok(byte_size as usize)
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
/*
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Layout, Precision};

    #[test]
    #[should_panic]
    fn invalid_blob_size() {
        let desc = TensorDesc::new(Layout::NHWC, &[1, 2, 2, 2], Precision::U8);
        // Blob should be 1x2x2x2 = 8 bytes but we pass in 7 bytes:
        let _ = Blob::new(&desc, &[0; 7]).unwrap();
    }

    #[test]
    fn buffer_conversion() {
        // In order to ensure runtime-linked libraries are linked with, we must:
        openvino_sys::library::load().expect("unable to find an OpenVINO shared library");

        const LEN: usize = 200 * 100;
        let desc = TensorDesc::new(Layout::HW, &[200, 100], Precision::U16);

        // Provide a u8 slice to create a u16 blob (twice as many items).
        let mut blob = Blob::new(&desc, &[0; LEN * 2]).unwrap();

        assert_eq!(blob.len().unwrap(), LEN);
        assert_eq!(
            blob.byte_len().unwrap(),
            LEN * 2,
            "we should have twice as many bytes (u16 = u8 * 2)"
        );
        assert_eq!(
            blob.buffer().unwrap().len(),
            LEN * 2,
            "we should have twice as many items (u16 = u8 * 2)"
        );
        assert_eq!(
            unsafe { blob.buffer_mut_as_type::<f32>() }.unwrap().len(),
            LEN / 2,
            "we should have half as many items (u16 = f32 / 2)"
        );
    }

    #[test]
    fn tensor_desc() {
        openvino_sys::library::load().expect("unable to find an OpenVINO shared library");

        let desc = TensorDesc::new(Layout::NHWC, &[1, 2, 2, 2], Precision::U8);
        let blob = Blob::new(&desc, &[0; 8]).unwrap();
        let desc2 = blob.tensor_desc().unwrap();

        // Both TensorDesc's should be equal
        assert_eq!(desc.layout(), desc2.layout());
        assert_eq!(desc.dims(), desc2.dims());
        assert_eq!(desc.precision(), desc2.precision());
    }
}
*/
