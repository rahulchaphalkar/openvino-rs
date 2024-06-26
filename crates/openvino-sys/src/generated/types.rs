/* automatically generated by rust-bindgen 0.68.1 */

#[doc = "!< SUCCESS"]
pub const ov_status_e_OK: ov_status_e = 0;
#[doc = "!< GENERAL_ERROR"]
pub const ov_status_e_GENERAL_ERROR: ov_status_e = -1;
#[doc = "!< NOT_IMPLEMENTED"]
pub const ov_status_e_NOT_IMPLEMENTED: ov_status_e = -2;
#[doc = "!< NETWORK_NOT_LOADED"]
pub const ov_status_e_NETWORK_NOT_LOADED: ov_status_e = -3;
#[doc = "!< PARAMETER_MISMATCH"]
pub const ov_status_e_PARAMETER_MISMATCH: ov_status_e = -4;
#[doc = "!< NOT_FOUND"]
pub const ov_status_e_NOT_FOUND: ov_status_e = -5;
#[doc = "!< OUT_OF_BOUNDS"]
pub const ov_status_e_OUT_OF_BOUNDS: ov_status_e = -6;
#[doc = "!< UNEXPECTED"]
pub const ov_status_e_UNEXPECTED: ov_status_e = -7;
#[doc = "!< REQUEST_BUSY"]
pub const ov_status_e_REQUEST_BUSY: ov_status_e = -8;
#[doc = "!< RESULT_NOT_READY"]
pub const ov_status_e_RESULT_NOT_READY: ov_status_e = -9;
#[doc = "!< NOT_ALLOCATED"]
pub const ov_status_e_NOT_ALLOCATED: ov_status_e = -10;
#[doc = "!< INFER_NOT_STARTED"]
pub const ov_status_e_INFER_NOT_STARTED: ov_status_e = -11;
#[doc = "!< NETWORK_NOT_READ"]
pub const ov_status_e_NETWORK_NOT_READ: ov_status_e = -12;
#[doc = "!< INFER_CANCELLED"]
pub const ov_status_e_INFER_CANCELLED: ov_status_e = -13;
#[doc = "!< INVALID_C_PARAM"]
pub const ov_status_e_INVALID_C_PARAM: ov_status_e = -14;
#[doc = "!< UNKNOWN_C_ERROR"]
pub const ov_status_e_UNKNOWN_C_ERROR: ov_status_e = -15;
#[doc = "!< NOT_IMPLEMENT_C_METHOD"]
pub const ov_status_e_NOT_IMPLEMENT_C_METHOD: ov_status_e = -16;
#[doc = "!< UNKNOW_EXCEPTION"]
pub const ov_status_e_UNKNOW_EXCEPTION: ov_status_e = -17;
#[doc = " @enum ov_status_e\n @ingroup ov_base_c_api\n @brief This enum contains codes for all possible return values of the interface functions"]
pub type ov_status_e = ::std::os::raw::c_int;
#[doc = "!< Undefined element type"]
pub const ov_element_type_e_UNDEFINED: ov_element_type_e = 0;
#[doc = "!< Dynamic element type"]
pub const ov_element_type_e_DYNAMIC: ov_element_type_e = 1;
pub const ov_element_type_e_OV_BOOLEAN: ov_element_type_e = 2;
#[doc = "!< bf16 element type"]
pub const ov_element_type_e_BF16: ov_element_type_e = 3;
#[doc = "!< f16 element type"]
pub const ov_element_type_e_F16: ov_element_type_e = 4;
#[doc = "!< f32 element type"]
pub const ov_element_type_e_F32: ov_element_type_e = 5;
#[doc = "!< f64 element type"]
pub const ov_element_type_e_F64: ov_element_type_e = 6;
#[doc = "!< i4 element type"]
pub const ov_element_type_e_I4: ov_element_type_e = 7;
#[doc = "!< i8 element type"]
pub const ov_element_type_e_I8: ov_element_type_e = 8;
#[doc = "!< i16 element type"]
pub const ov_element_type_e_I16: ov_element_type_e = 9;
#[doc = "!< i32 element type"]
pub const ov_element_type_e_I32: ov_element_type_e = 10;
#[doc = "!< i64 element type"]
pub const ov_element_type_e_I64: ov_element_type_e = 11;
#[doc = "!< binary element type"]
pub const ov_element_type_e_U1: ov_element_type_e = 12;
#[doc = "!< u4 element type"]
pub const ov_element_type_e_U4: ov_element_type_e = 13;
#[doc = "!< u8 element type"]
pub const ov_element_type_e_U8: ov_element_type_e = 14;
#[doc = "!< u16 element type"]
pub const ov_element_type_e_U16: ov_element_type_e = 15;
#[doc = "!< u32 element type"]
pub const ov_element_type_e_U32: ov_element_type_e = 16;
#[doc = "!< u64 element type"]
pub const ov_element_type_e_U64: ov_element_type_e = 17;
#[doc = "!< nf4 element type"]
pub const ov_element_type_e_NF4: ov_element_type_e = 18;
#[doc = "!< f8e4m3 element type"]
pub const ov_element_type_e_F8E4M3: ov_element_type_e = 19;
#[doc = "!< f8e5m2 element type"]
pub const ov_element_type_e_F8E5M3: ov_element_type_e = 20;
#[doc = " @enum ov_element_type_e\n @ingroup ov_base_c_api\n @brief This enum contains codes for element type."]
pub type ov_element_type_e = ::std::os::raw::c_uint;
#[doc = " @struct ov_dimension\n @ingroup ov_dimension_c_api\n @brief This is a structure interface equal to ov::Dimension"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_dimension {
    #[doc = "!< The lower inclusive limit for the dimension."]
    pub min: i64,
    #[doc = "!< The upper inclusive limit for the dimension."]
    pub max: i64,
}
#[test]
fn bindgen_test_layout_ov_dimension() {
    const UNINIT: ::std::mem::MaybeUninit<ov_dimension> = ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<ov_dimension>(),
        16usize,
        concat!("Size of: ", stringify!(ov_dimension))
    );
    assert_eq!(
        ::std::mem::align_of::<ov_dimension>(),
        8usize,
        concat!("Alignment of ", stringify!(ov_dimension))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).min) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_dimension),
            "::",
            stringify!(min)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).max) as usize - ptr as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_dimension),
            "::",
            stringify!(max)
        )
    );
}
#[doc = " @struct ov_dimension\n @ingroup ov_dimension_c_api\n @brief This is a structure interface equal to ov::Dimension"]
pub type ov_dimension_t = ov_dimension;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_layout {
    _unused: [u8; 0],
}
#[doc = " @struct ov_layout_t\n @ingroup ov_layout_c_api\n @brief type define ov_layout_t from ov_layout"]
pub type ov_layout_t = ov_layout;
#[doc = " @struct ov_rank_t\n @ingroup ov_rank_c_api\n @brief type define ov_rank_t from ov_dimension_t"]
pub type ov_rank_t = ov_dimension_t;
#[doc = " @struct ov_shape_t\n @ingroup ov_shape_c_api\n @brief Reprents a static shape."]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_shape_t {
    #[doc = "!< the rank of shape"]
    pub rank: i64,
    #[doc = "!< the dims of shape"]
    pub dims: *mut i64,
}
#[test]
fn bindgen_test_layout_ov_shape_t() {
    const UNINIT: ::std::mem::MaybeUninit<ov_shape_t> = ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<ov_shape_t>(),
        16usize,
        concat!("Size of: ", stringify!(ov_shape_t))
    );
    assert_eq!(
        ::std::mem::align_of::<ov_shape_t>(),
        8usize,
        concat!("Alignment of ", stringify!(ov_shape_t))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).rank) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_shape_t),
            "::",
            stringify!(rank)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).dims) as usize - ptr as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_shape_t),
            "::",
            stringify!(dims)
        )
    );
}
#[doc = " @struct ov_partial_shape\n @ingroup ov_partial_shape_c_api\n @brief It represents a shape that may be partially or totally dynamic.\n A PartialShape may have:\n Dynamic rank. (Informal notation: `?`)\n Static rank, but dynamic dimensions on some or all axes.\n     (Informal notation examples: `{1,2,?,4}`, `{?,?,?}`, `{-1,-1,-1}`)\n Static rank, and static dimensions on all axes.\n     (Informal notation examples: `{1,2,3,4}`, `{6}`, `{}`)\n\n An interface to make user can initialize ov_partial_shape_t"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_partial_shape {
    #[doc = "!< The rank"]
    pub rank: ov_rank_t,
    #[doc = "!< The dimension"]
    pub dims: *mut ov_dimension_t,
}
#[test]
fn bindgen_test_layout_ov_partial_shape() {
    const UNINIT: ::std::mem::MaybeUninit<ov_partial_shape> = ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<ov_partial_shape>(),
        24usize,
        concat!("Size of: ", stringify!(ov_partial_shape))
    );
    assert_eq!(
        ::std::mem::align_of::<ov_partial_shape>(),
        8usize,
        concat!("Alignment of ", stringify!(ov_partial_shape))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).rank) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_partial_shape),
            "::",
            stringify!(rank)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).dims) as usize - ptr as usize },
        16usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_partial_shape),
            "::",
            stringify!(dims)
        )
    );
}
#[doc = " @struct ov_partial_shape\n @ingroup ov_partial_shape_c_api\n @brief It represents a shape that may be partially or totally dynamic.\n A PartialShape may have:\n Dynamic rank. (Informal notation: `?`)\n Static rank, but dynamic dimensions on some or all axes.\n     (Informal notation examples: `{1,2,?,4}`, `{?,?,?}`, `{-1,-1,-1}`)\n Static rank, and static dimensions on all axes.\n     (Informal notation examples: `{1,2,3,4}`, `{6}`, `{}`)\n\n An interface to make user can initialize ov_partial_shape_t"]
pub type ov_partial_shape_t = ov_partial_shape;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_output_const_port {
    _unused: [u8; 0],
}
#[doc = " @struct ov_output_const_port_t\n @ingroup ov_node_c_api\n @brief type define ov_output_const_port_t from ov_output_const_port"]
pub type ov_output_const_port_t = ov_output_const_port;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_output_port {
    _unused: [u8; 0],
}
#[doc = " @struct ov_output_port_t\n @ingroup ov_node_c_api\n @brief type define ov_output_port_t from ov_output_port"]
pub type ov_output_port_t = ov_output_port;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_tensor {
    _unused: [u8; 0],
}
#[doc = " @struct ov_tensor_t\n @ingroup ov_tensor_c_api\n @brief type define ov_tensor_t from ov_tensor"]
pub type ov_tensor_t = ov_tensor;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_infer_request {
    _unused: [u8; 0],
}
#[doc = " @struct ov_infer_request_t\n @ingroup ov_infer_request_c_api\n @brief type define ov_infer_request_t from ov_infer_request"]
pub type ov_infer_request_t = ov_infer_request;
#[doc = " @struct ov_callback_t\n @ingroup ov_infer_request_c_api\n @brief Completion callback definition about the function and args"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_callback_t {
    #[doc = "!< The callback func"]
    pub callback_func:
        ::std::option::Option<unsafe extern "C" fn(args: *mut ::std::os::raw::c_void)>,
    #[doc = "!< The args of callback func"]
    pub args: *mut ::std::os::raw::c_void,
}
#[test]
fn bindgen_test_layout_ov_callback_t() {
    const UNINIT: ::std::mem::MaybeUninit<ov_callback_t> = ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<ov_callback_t>(),
        16usize,
        concat!("Size of: ", stringify!(ov_callback_t))
    );
    assert_eq!(
        ::std::mem::align_of::<ov_callback_t>(),
        8usize,
        concat!("Alignment of ", stringify!(ov_callback_t))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).callback_func) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_callback_t),
            "::",
            stringify!(callback_func)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).args) as usize - ptr as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_callback_t),
            "::",
            stringify!(args)
        )
    );
}
#[doc = " @struct ov_ProfilingInfo_t\n @ingroup ov_infer_request_c_api\n @brief Store profiling info data"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_profiling_info_t {
    #[doc = "!< status"]
    pub status: ov_profiling_info_t_Status,
    #[doc = "!< The absolute time, in microseconds, that the node ran (in total)."]
    pub real_time: i64,
    #[doc = "!< The net host CPU time that the node ran."]
    pub cpu_time: i64,
    #[doc = "!< Name of a node."]
    pub node_name: *const ::std::os::raw::c_char,
    #[doc = "!< Execution type of a unit."]
    pub exec_type: *const ::std::os::raw::c_char,
    #[doc = "!< Node type."]
    pub node_type: *const ::std::os::raw::c_char,
}
#[doc = "!< A node is not executed."]
pub const ov_profiling_info_t_Status_NOT_RUN: ov_profiling_info_t_Status = 0;
#[doc = "!< A node is optimized out during graph optimization phase."]
pub const ov_profiling_info_t_Status_OPTIMIZED_OUT: ov_profiling_info_t_Status = 1;
#[doc = "!< A node is executed."]
pub const ov_profiling_info_t_Status_EXECUTED: ov_profiling_info_t_Status = 2;
pub type ov_profiling_info_t_Status = ::std::os::raw::c_uint;
#[test]
fn bindgen_test_layout_ov_profiling_info_t() {
    const UNINIT: ::std::mem::MaybeUninit<ov_profiling_info_t> = ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<ov_profiling_info_t>(),
        48usize,
        concat!("Size of: ", stringify!(ov_profiling_info_t))
    );
    assert_eq!(
        ::std::mem::align_of::<ov_profiling_info_t>(),
        8usize,
        concat!("Alignment of ", stringify!(ov_profiling_info_t))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).status) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_profiling_info_t),
            "::",
            stringify!(status)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).real_time) as usize - ptr as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_profiling_info_t),
            "::",
            stringify!(real_time)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).cpu_time) as usize - ptr as usize },
        16usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_profiling_info_t),
            "::",
            stringify!(cpu_time)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).node_name) as usize - ptr as usize },
        24usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_profiling_info_t),
            "::",
            stringify!(node_name)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).exec_type) as usize - ptr as usize },
        32usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_profiling_info_t),
            "::",
            stringify!(exec_type)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).node_type) as usize - ptr as usize },
        40usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_profiling_info_t),
            "::",
            stringify!(node_type)
        )
    );
}
#[doc = " @struct ov_profiling_info_list_t\n @ingroup ov_infer_request_c_api\n @brief A list of profiling info data"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_profiling_info_list_t {
    #[doc = "!< The list of ov_profilling_info_t"]
    pub profiling_infos: *mut ov_profiling_info_t,
    #[doc = "!< The list size"]
    pub size: usize,
}
#[test]
fn bindgen_test_layout_ov_profiling_info_list_t() {
    const UNINIT: ::std::mem::MaybeUninit<ov_profiling_info_list_t> =
        ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<ov_profiling_info_list_t>(),
        16usize,
        concat!("Size of: ", stringify!(ov_profiling_info_list_t))
    );
    assert_eq!(
        ::std::mem::align_of::<ov_profiling_info_list_t>(),
        8usize,
        concat!("Alignment of ", stringify!(ov_profiling_info_list_t))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).profiling_infos) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_profiling_info_list_t),
            "::",
            stringify!(profiling_infos)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).size) as usize - ptr as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_profiling_info_list_t),
            "::",
            stringify!(size)
        )
    );
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_model {
    _unused: [u8; 0],
}
#[doc = " @struct ov_model_t\n @ingroup ov_model_c_api\n @brief type define ov_model_t from ov_model"]
pub type ov_model_t = ov_model;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_remote_context {
    _unused: [u8; 0],
}
pub type ov_remote_context_t = ov_remote_context;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_compiled_model {
    _unused: [u8; 0],
}
#[doc = " @struct ov_compiled_model_t\n @ingroup ov_compiled_model_c_api\n @brief type define ov_compiled_model_t from ov_compiled_model"]
pub type ov_compiled_model_t = ov_compiled_model;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_core {
    _unused: [u8; 0],
}
#[doc = " @struct ov_core_t\n @ingroup ov_core_c_api\n @brief type define ov_core_t from ov_core"]
pub type ov_core_t = ov_core;
#[doc = " @struct ov_version\n @ingroup ov_core_c_api\n @brief Represents OpenVINO version information"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_version {
    #[doc = "!< A string representing OpenVINO version"]
    pub buildNumber: *const ::std::os::raw::c_char,
    #[doc = "!< A string representing OpenVINO description"]
    pub description: *const ::std::os::raw::c_char,
}
#[test]
fn bindgen_test_layout_ov_version() {
    const UNINIT: ::std::mem::MaybeUninit<ov_version> = ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<ov_version>(),
        16usize,
        concat!("Size of: ", stringify!(ov_version))
    );
    assert_eq!(
        ::std::mem::align_of::<ov_version>(),
        8usize,
        concat!("Alignment of ", stringify!(ov_version))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).buildNumber) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_version),
            "::",
            stringify!(buildNumber)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).description) as usize - ptr as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_version),
            "::",
            stringify!(description)
        )
    );
}
#[doc = " @struct ov_version\n @ingroup ov_core_c_api\n @brief Represents OpenVINO version information"]
pub type ov_version_t = ov_version;
#[doc = " @struct ov_core_version\n @ingroup ov_core_c_api\n @brief  Represents version information that describes device and ov runtime library"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_core_version_t {
    #[doc = "!< A device name"]
    pub device_name: *const ::std::os::raw::c_char,
    #[doc = "!< Version"]
    pub version: ov_version_t,
}
#[test]
fn bindgen_test_layout_ov_core_version_t() {
    const UNINIT: ::std::mem::MaybeUninit<ov_core_version_t> = ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<ov_core_version_t>(),
        24usize,
        concat!("Size of: ", stringify!(ov_core_version_t))
    );
    assert_eq!(
        ::std::mem::align_of::<ov_core_version_t>(),
        8usize,
        concat!("Alignment of ", stringify!(ov_core_version_t))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).device_name) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_core_version_t),
            "::",
            stringify!(device_name)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).version) as usize - ptr as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_core_version_t),
            "::",
            stringify!(version)
        )
    );
}
#[doc = " @struct ov_core_version_list\n @ingroup ov_core_c_api\n @brief  Represents version information that describes all devices and ov runtime library"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_core_version_list_t {
    #[doc = "!< An array of device versions"]
    pub versions: *mut ov_core_version_t,
    #[doc = "!< A number of versions in the array"]
    pub size: usize,
}
#[test]
fn bindgen_test_layout_ov_core_version_list_t() {
    const UNINIT: ::std::mem::MaybeUninit<ov_core_version_list_t> =
        ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<ov_core_version_list_t>(),
        16usize,
        concat!("Size of: ", stringify!(ov_core_version_list_t))
    );
    assert_eq!(
        ::std::mem::align_of::<ov_core_version_list_t>(),
        8usize,
        concat!("Alignment of ", stringify!(ov_core_version_list_t))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).versions) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_core_version_list_t),
            "::",
            stringify!(versions)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).size) as usize - ptr as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_core_version_list_t),
            "::",
            stringify!(size)
        )
    );
}
#[doc = " @struct ov_available_devices_t\n @ingroup ov_core_c_api\n @brief Represent all available devices."]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_available_devices_t {
    #[doc = "!< devices' name"]
    pub devices: *mut *mut ::std::os::raw::c_char,
    #[doc = "!< devices' number"]
    pub size: usize,
}
#[test]
fn bindgen_test_layout_ov_available_devices_t() {
    const UNINIT: ::std::mem::MaybeUninit<ov_available_devices_t> =
        ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<ov_available_devices_t>(),
        16usize,
        concat!("Size of: ", stringify!(ov_available_devices_t))
    );
    assert_eq!(
        ::std::mem::align_of::<ov_available_devices_t>(),
        8usize,
        concat!("Alignment of ", stringify!(ov_available_devices_t))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).devices) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_available_devices_t),
            "::",
            stringify!(devices)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).size) as usize - ptr as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(ov_available_devices_t),
            "::",
            stringify!(size)
        )
    );
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_preprocess_prepostprocessor {
    _unused: [u8; 0],
}
#[doc = " @struct ov_preprocess_prepostprocessor_t\n @ingroup ov_prepostprocess_c_api\n @brief type define ov_preprocess_prepostprocessor_t from ov_preprocess_prepostprocessor"]
pub type ov_preprocess_prepostprocessor_t = ov_preprocess_prepostprocessor;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_preprocess_input_info {
    _unused: [u8; 0],
}
#[doc = " @struct ov_preprocess_input_info_t\n @ingroup ov_prepostprocess_c_api\n @brief type define ov_preprocess_input_info_t from ov_preprocess_input_info"]
pub type ov_preprocess_input_info_t = ov_preprocess_input_info;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_preprocess_input_tensor_info {
    _unused: [u8; 0],
}
#[doc = " @struct ov_preprocess_input_tensor_info_t\n @ingroup ov_prepostprocess_c_api\n @brief type define ov_preprocess_input_tensor_info_t from ov_preprocess_input_tensor_info"]
pub type ov_preprocess_input_tensor_info_t = ov_preprocess_input_tensor_info;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_preprocess_output_info {
    _unused: [u8; 0],
}
#[doc = " @struct ov_preprocess_output_info_t\n @ingroup ov_prepostprocess_c_api\n @brief type define ov_preprocess_output_info_t from ov_preprocess_output_info"]
pub type ov_preprocess_output_info_t = ov_preprocess_output_info;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_preprocess_output_tensor_info {
    _unused: [u8; 0],
}
#[doc = " @struct ov_preprocess_output_tensor_info_t\n @ingroup ov_prepostprocess_c_api\n @brief type define ov_preprocess_output_tensor_info_t from ov_preprocess_output_tensor_info"]
pub type ov_preprocess_output_tensor_info_t = ov_preprocess_output_tensor_info;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_preprocess_input_model_info {
    _unused: [u8; 0],
}
#[doc = " @struct ov_preprocess_input_model_info_t\n @ingroup ov_prepostprocess_c_api\n @brief type define ov_preprocess_input_model_info_t from ov_preprocess_input_model_info"]
pub type ov_preprocess_input_model_info_t = ov_preprocess_input_model_info;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ov_preprocess_preprocess_steps {
    _unused: [u8; 0],
}
#[doc = " @struct ov_preprocess_preprocess_steps_t\n @ingroup ov_prepostprocess_c_api\n @brief type define ov_preprocess_preprocess_steps_t from ov_preprocess_preprocess_steps"]
pub type ov_preprocess_preprocess_steps_t = ov_preprocess_preprocess_steps;
#[doc = "!< Undefine color format"]
pub const ov_color_format_e_UNDEFINE: ov_color_format_e = 0;
#[doc = "!< Image in NV12 format as single tensor"]
pub const ov_color_format_e_NV12_SINGLE_PLANE: ov_color_format_e = 1;
#[doc = "!< Image in NV12 format represented as separate tensors for Y and UV planes."]
pub const ov_color_format_e_NV12_TWO_PLANES: ov_color_format_e = 2;
#[doc = "!< Image in I420 (YUV) format as single tensor"]
pub const ov_color_format_e_I420_SINGLE_PLANE: ov_color_format_e = 3;
#[doc = "!< Image in I420 format represented as separate tensors for Y, U and V planes."]
pub const ov_color_format_e_I420_THREE_PLANES: ov_color_format_e = 4;
#[doc = "!< Image in RGB interleaved format (3 channels)"]
pub const ov_color_format_e_RGB: ov_color_format_e = 5;
#[doc = "!< Image in BGR interleaved format (3 channels)"]
pub const ov_color_format_e_BGR: ov_color_format_e = 6;
#[doc = "!< Image in GRAY format (1 channel)"]
pub const ov_color_format_e_GRAY: ov_color_format_e = 7;
#[doc = "!< Image in RGBX interleaved format (4 channels)"]
pub const ov_color_format_e_RGBX: ov_color_format_e = 8;
#[doc = "!< Image in BGRX interleaved format (4 channels)"]
pub const ov_color_format_e_BGRX: ov_color_format_e = 9;
#[doc = " @enum ov_color_format_e\n @ingroup ov_prepostprocess_c_api\n @brief This enum contains enumerations for color format."]
pub type ov_color_format_e = ::std::os::raw::c_uint;
#[doc = "!< linear algorithm"]
pub const ov_preprocess_resize_algorithm_e_RESIZE_LINEAR: ov_preprocess_resize_algorithm_e = 0;
#[doc = "!< cubic algorithm"]
pub const ov_preprocess_resize_algorithm_e_RESIZE_CUBIC: ov_preprocess_resize_algorithm_e = 1;
#[doc = "!< nearest algorithm"]
pub const ov_preprocess_resize_algorithm_e_RESIZE_NEAREST: ov_preprocess_resize_algorithm_e = 2;
#[doc = " @enum ov_preprocess_resize_algorithm_e\n @ingroup ov_prepostprocess_c_api\n @brief This enum contains codes for all preprocess resize algorithm."]
pub type ov_preprocess_resize_algorithm_e = ::std::os::raw::c_uint;
