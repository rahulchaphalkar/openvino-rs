//! Define the core interface between Rust and OpenVINO's C
//! [API](https://docs.openvinotoolkit.org/latest/ie_c_api/modules.html).

use crate::{cstr, drop_using_function, try_unsafe, util::Result};
use crate::{
    error::{LoadingError, SetupError},
    network::{CompiledModel, Model},
};

use openvino_sys::{
    self, ov_core_compile_model, ov_core_create, ov_core_free, ov_core_read_model, ov_core_t,
};

/// See [Core](https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1Core.html).
pub struct Core {
    instance: *mut ov_core_t,
}
drop_using_function!(Core, ov_core_free);

unsafe impl Send for Core {}

impl Core {
    /// Construct a new OpenVINO [`Core`]--this is the primary entrypoint for constructing and using
    /// inference networks. Because this function may load OpenVINO's shared libraries at runtime,
    /// there are more ways than usual that this function can fail (e.g., [`LoadingError`]s).
    pub fn new(xml_config_file: Option<&str>) -> std::result::Result<Core, SetupError> {
        openvino_sys::library::load().map_err(LoadingError::SystemFailure)?;

        let file = if let Some(file) = xml_config_file {
            cstr!(file.to_string())
        } else if let Some(file) = openvino_finder::find_plugins_xml() {
            cstr!(file
                .to_str()
                .ok_or(LoadingError::CannotStringifyPath)?
                .to_string())
        } else {
            cstr!(String::new())
        };

        let mut instance = std::ptr::null_mut();
        try_unsafe!(ov_core_create(std::ptr::addr_of_mut!(instance)))?;
        Ok(Core { instance })
    }

    /// Read a [`CNNNetwork`] from a pair of files: `model_path` points to an XML file containing the
    /// OpenVINO network IR and `weights_path` points to the binary weights file.
    pub fn read_network_from_file(
        &mut self,
        model_path: &str,
        weights_path: &str,
    ) -> Result<Model> {
        let mut instance = std::ptr::null_mut();
        try_unsafe!(ov_core_read_model(
            self.instance,
            cstr!(model_path),
            cstr!(weights_path),
            std::ptr::addr_of_mut!(instance)
        ))?;
        Ok(Model { instance })
    }

    /// Instantiate a [`CNNNetwork`] as an [`ExecutableNetwork`] on the specified `device`.
    pub fn compile_model(&mut self, network: Model, device: &str) -> Result<CompiledModel> {
        let mut compiled_model = CompiledModel {
            instance: std::ptr::null_mut(),
        };
        let num_property_args = 0;
        try_unsafe!(ov_core_compile_model(
            self.instance,
            network.instance,
            cstr!(device),
            num_property_args,
            std::ptr::addr_of_mut!(compiled_model.instance)
        ))?;
        Ok(compiled_model)
    }
}
