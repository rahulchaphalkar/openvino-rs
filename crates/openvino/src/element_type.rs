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
    I16 = 9,
    I32 = 10,
    I64 = 11,
    U1 = 12,
    U4 = 13,
    U8 = 14,
    U16 = 15,
    U32 = 16,
    U64 = 17,
    NF4 = 18,
}

#[cfg(test)]
mod tests {
    use super::*;
    use openvino_sys::{ov_element_type_e_U8, ov_element_type_e_UNDEFINED};

    #[test]
    fn check_discriminant_values() {
        assert_eq!(ov_element_type_e_UNDEFINED, ElementType::Undefined as u32);
        assert_eq!(ov_element_type_e_U8, ElementType::U8 as u32);
    }
}
