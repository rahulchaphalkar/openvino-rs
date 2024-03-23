use std::os::raw::c_void;
use std::ptr::null_mut;

//use crate::element_type::{self, ElementType};
use crate::element_type::{self, ElementType};
use crate::shape::Shape;
//use crate::shape;
//use crate::tensor_desc::TensorDesc;
use crate::{drop_using_function, try_unsafe, util::Result};
// use openvino_sys::{
//     self, ie_blob_buffer__bindgen_ty_1, ie_blob_buffer_t, ie_blob_byte_size, ie_blob_free,
//     ie_blob_get_buffer, ie_blob_get_dims, ie_blob_get_layout, ie_blob_get_precision,
//     ie_blob_make_memory, ie_blob_size, ie_blob_t, tensor_desc_t,
// };
use openvino_sys::{
    self, ov_shape_t, ov_tensor_create, ov_tensor_create_from_host_ptr, ov_tensor_data,
    ov_tensor_free, ov_tensor_get_byte_size, ov_tensor_get_element_type, ov_tensor_get_shape,
    ov_tensor_get_size, ov_tensor_set_shape, ov_tensor_t,
};

/// See [`Blob`](https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1Blob.html).
pub struct Tensor {
    pub instance: *mut ov_tensor_t,
    //host_ptr: *mut std::os::raw::c_void,
}
drop_using_function!(Tensor, ov_tensor_free);

impl Tensor {
    /// Create a new [`Blob`] by copying data in to the OpenVINO-allocated memory.
    ///
    /// # Panics
    ///
    /// This function will panic if the number of bytes passed in `data` does not match the expected
    /// size of the tensor `description`.
    /*  pub fn new(description: &TensorDesc, data: &[u8]) -> Result<Self> {
        let mut blob = Self::allocate(description)?;
        let blob_len = blob.byte_len()?;
        assert_eq!(
            blob_len,
            data.len(),
            "The data to initialize ({} bytes) must be the same as the blob size ({} bytes).",
            data.len(),
            blob_len
        );
    */
    //new tensor

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
        data: &[u8], /*, host_ptr:*mut std::ffi::c_void*/
    ) -> Result<Self> {
        let mut tensor: *mut ov_tensor_t = std::ptr::null_mut();
        let element_type: u32 = data_type as u32;
        //let mut buffer = data;
        //buffer.copy_from_slice(data);
        let buffer = data.as_ptr() as *mut c_void;
        try_unsafe!(ov_tensor_create_from_host_ptr(
            element_type,
            shape.instance,
            //host_ptr,
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
        //let instance = std::ptr::null_mut();
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

    pub fn get_element_type(&self /*data_type: *mut ElementType*/) -> Result<u32> {
        //let mut element_type: u32 = data_type as u32;
        //let mut element_type = std::ptr::null_mut();
        //let mut element_type = ElementType::Undefined as u32;
        let mut element_type = ElementType::Undefined as u32;
        try_unsafe!(ov_tensor_get_element_type(
            self.instance,
            std::ptr::addr_of_mut!(element_type),
        ));
        //Ok(Self { instance: self.instance })
        Ok(element_type)
    }

    pub fn get_size(&self /*elements_size: *mut usize*/) -> Result<usize> {
        let elements_size = std::ptr::null_mut();
        try_unsafe!(ov_tensor_get_size(self.instance, elements_size,));
        //Ok(Self { instance: self.instance })
        Ok(elements_size as usize)
    }

    pub fn get_byte_size(&self) -> Result<usize> {
        let mut byte_size: usize = 0;
        //let mut byte_size = std::ptr::null_mut();
        try_unsafe!(ov_tensor_get_byte_size(
            self.instance,
            std::ptr::addr_of_mut!(byte_size),
        ));
        //Ok(Self { instance: self.instance })
        Ok(byte_size as usize)
    }

    // pub fn get_data(&self, data: *mut *mut std::ffi::c_void) -> Result<Self> {

    //     try_unsafe!(ov_tensor_data(
    //         self.instance,
    //         data,
    //     ));
    //     Ok(Self { instance: self.instance })
    // }
    //pub fn get_data<T>(&self) -> Result<*mut c_void> {
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
            //std::slice::from_raw_parts_mut(buffer.__bindgen_anon_1.buffer.cast::<u8>(), size)
            std::slice::from_raw_parts_mut(buffer.cast::<u8>(), size)
        };
        Ok(slice)
    }
    // Allocate space in OpenVINO for an empty [`Blob`].
    /*
    pub fn allocate(description: &TensorDesc) -> Result<Self> {
        let mut instance = std::ptr::null_mut();
        try_unsafe!(ie_blob_make_memory(
            std::ptr::addr_of!(description.instance),
            std::ptr::addr_of_mut!(instance)
        ))?;
        // try_unsafe!(ov_tensor_create(
        //     std::ptr::addr_of!(description.instance),
        //     std::ptr::addr_of_mut!(instance)
        // ))?;
        Ok(Self { instance })
    }
    */
    /*
        /// Return the tensor description of this [`Blob`].
        ///
        /// # Panics
        ///
        /// May panic in the improbable case where some future version of the OpenVINO library returns a
        /// dimensions array with a size different than the one auto-generated in the bindings; see
        /// `struct dimensions` in `openvino-sys/src/generated/types.rs`.
        // pub fn tensor_desc(&self) -> Result<TensorDesc> {
        //     let blob = self.instance.cast_const();

        //     let mut layout = MaybeUninit::uninit();
        //     try_unsafe!(ie_blob_get_layout(blob, layout.as_mut_ptr()))?;

        //     let mut dimensions = MaybeUninit::uninit();
        //     try_unsafe!(ie_blob_get_dims(blob, dimensions.as_mut_ptr()))?;
        //     // Safety: See `Panics` section in function documentation; itt is not clear to me whether
        //     // this will return the statically-expected size or the dynamic size -- this is not
        //     // effective in the former case.
        //     assert_eq!(unsafe { dimensions.assume_init() }.dims.len(), 8);

        //     let mut precision = MaybeUninit::uninit();
        //     try_unsafe!(ie_blob_get_precision(blob, precision.as_mut_ptr()))?;

        //     Ok(TensorDesc {
        //         // Safety: all reads succeeded so values must be initialized
        //         instance: unsafe {
        //             tensor_desc_t {
        //                 layout: layout.assume_init(),
        //                 dims: dimensions.assume_init(),
        //                 precision: precision.assume_init(),
        //             }
        //         },
        //     })
        // }

        // Get the number of elements contained in the [`Blob`].
        //
        // # Panics
        //
        // Panics if the returned OpenVINO size will not fit in `usize`.

        pub fn len(&self) -> Result<usize> {
            let mut size = 0;
            //try_unsafe!(ie_blob_size(self.instance, std::ptr::addr_of_mut!(size)))?;
            try_unsafe!(ov_tensor_get_size(self.instance, std::ptr::addr_of_mut!(size)))?;
            Ok(usize::try_from(size).unwrap())
        }

        /// Get the size of the current [`Blob`] in bytes.
        ///
        /// # Panics
        ///
        /// Panics if the returned OpenVINO size will not fit in `usize`.
        pub fn byte_len(&self) -> Result<usize> {
            let mut size = 0;
            try_unsafe!(ov_tensor_get_byte_size(
                self.instance,
                std::ptr::addr_of_mut!(size)
            ))?;
            Ok(usize::try_from(size).unwrap())
        }

        /// Retrieve the [`Blob`]'s data as an immutable slice of bytes.
        pub fn buffer(&self) -> Result<&[u8]> {
            let mut buffer = Blob::empty_buffer();
            try_unsafe!(ie_blob_get_buffer(
                self.instance,
                std::ptr::addr_of_mut!(buffer)
            ))?;
            let size = self.byte_len()?;
            let slice = unsafe {
                std::slice::from_raw_parts(buffer.__bindgen_anon_1.buffer as *const u8, size)
            };
            Ok(slice)
        }

        /// Retrieve the [`Blob`]'s data as a mutable slice of bytes.
        pub fn buffer_mut(&mut self) -> Result<&mut [u8]> {
            let mut buffer = Blob::empty_buffer();
            try_unsafe!(ie_blob_get_buffer(
                self.instance,
                std::ptr::addr_of_mut!(buffer)
            ))?;
            let size = self.byte_len()?;
            let slice = unsafe {
                std::slice::from_raw_parts_mut(buffer.__bindgen_anon_1.buffer.cast::<u8>(), size)
            };
            Ok(slice)
        }

        /// Retrieve the [`Blob`]'s data as an immutable slice of type `T`.
        ///
        /// # Safety
        ///
        /// This function is `unsafe`, since the values of `T` may not have been properly initialized;
        /// however, this functionality is provided as an equivalent of what C/C++ users of OpenVINO
        /// currently do to access [`Blob`]s with, e.g., floating point values:
        /// `results.buffer_as_type::<f32>()`.
        pub unsafe fn buffer_as_type<T>(&self) -> Result<&[T]> {
            let mut buffer = Blob::empty_buffer();
            InferenceError::from(ie_blob_get_buffer(
                self.instance,
                std::ptr::addr_of_mut!(buffer),
            ))?;
            // This is very unsafe, but very convenient: by allowing users to specify T, they can
            // retrieve the buffer in whatever shape they prefer. But we must ensure that they cannot
            // read too many bytes, so we manually calculate the resulting slice `size`.
            let size = self.byte_len()? / std::mem::size_of::<T>();
            let slice = std::slice::from_raw_parts(buffer.__bindgen_anon_1.buffer.cast::<T>(), size);
            Ok(slice)
        }

        /// Retrieve the [`Blob`]'s data as a mutable slice of type `T`.
        ///
        /// # Safety
        ///
        /// This function is `unsafe`, since the values of `T` may not have been properly initialized;
        /// however, this functionality is provided as an equivalent of what C/C++ users of OpenVINO
        /// currently do to access [`Blob`]s with, e.g., floating point values:
        /// `results.buffer_mut_as_type::<f32>()`.
        pub unsafe fn buffer_mut_as_type<T>(&mut self) -> Result<&mut [T]> {
            let mut buffer = Blob::empty_buffer();
            InferenceError::from(ie_blob_get_buffer(
                self.instance,
                std::ptr::addr_of_mut!(buffer),
            ))?;
            // This is very unsafe, but very convenient: by allowing users to specify T, they can
            // retrieve the buffer in whatever shape they prefer. But we must ensure that they cannot
            // read too many bytes, so we manually calculate the resulting slice `size`.
            let size = self.byte_len()? / std::mem::size_of::<T>();
            let slice =
                std::slice::from_raw_parts_mut(buffer.__bindgen_anon_1.buffer.cast::<T>(), size);
            Ok(slice)
        }

        /// Construct a Blob from its associated pointer.
        pub(crate) unsafe fn from_raw_pointer(instance: *mut ov_tensor_t) -> Self {
            Self { instance }
        }

        fn empty_buffer() -> ie_blob_buffer_t {
            ie_blob_buffer_t {
                __bindgen_anon_1: ie_blob_buffer__bindgen_ty_1 {
                    buffer: std::ptr::null_mut(),
                },
            }
        }
    */
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
