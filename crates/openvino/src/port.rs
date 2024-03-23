use std::ffi::CStr;

use crate::{try_unsafe, util::Result};
use openvino_sys::{ov_output_const_port_t, ov_port_get_any_name};

pub struct Port {
    pub(crate) instance: *mut ov_output_const_port_t,
}

impl Port {
    pub fn get_name(&self) -> Result<String> {
        let mut c_name = std::ptr::null_mut();
        try_unsafe!(ov_port_get_any_name(
            self.instance,
            std::ptr::addr_of_mut!(c_name)
        ))?;
        let rust_name = unsafe { CStr::from_ptr(c_name) }
            .to_string_lossy()
            .into_owned();
        Ok(rust_name)
    }
}

// ov_port_get_any_name(
//     const ov_output_const_port_t \* port,
//     char \*\* tensor_name
//     );
