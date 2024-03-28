use std::ffi::CString;
use openvino_sys::{ov_layout_create, ov_layout_free, ov_layout_t};
use crate::{drop_using_function, try_unsafe};

pub struct Layout {
    pub(crate) instance: *mut ov_layout_t,
}
drop_using_function!(Layout, ov_layout_free);

impl Layout {
    pub fn new(layout_desc: &str) -> Self {
        let mut layout = std::ptr::null_mut();
        let c_layout_desc = CString::new(layout_desc).unwrap();
        let code = try_unsafe!(ov_layout_create(
            c_layout_desc.as_ptr(),
            std::ptr::addr_of_mut!(layout)
        ));
        assert_eq!(code, Ok(()));
        Self { instance: layout }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_layout() {
        let layout_desc = "NCHW";
        let layout = Layout::new(layout_desc);
        assert_eq!(layout.instance.is_null(), false);
    }
}
