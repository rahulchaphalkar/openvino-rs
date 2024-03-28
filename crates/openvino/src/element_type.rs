#[derive(Debug)]
#[repr(u32)]
pub enum ElementType {
    Undefined = 0,
    Dynamic = 1,
    OvBoolean = 2,
    Bf16 = 3,
    F16 = 4,
    F32 = 5,
    F64 = 6,
    I4 = 7,
    I8 = 8,
    U8 = 14,
}

#[cfg(test)]
mod tests {
    use super::*;
    use openvino_sys::ov_element_type_e_UNDEFINED;

    #[test]
    fn check_discriminant_values() {
        assert_eq!(ov_element_type_e_UNDEFINED, ElementType::Undefined as u32)
    }
}
