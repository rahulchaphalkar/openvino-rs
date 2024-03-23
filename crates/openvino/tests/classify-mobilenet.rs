//! Demonstrates using `openvino-rs` to classify an image using an MobileNet model and a prepared
//! input tensor. See [README](fixtures/inception/README.md) for details on how this test fixture
//! was prepared.
mod fixtures;
mod util;

use fixtures::mobilenet::Fixture;
use openvino::{
    Core, ElementType, Layout, PrePostprocess, PreprocessInputTensorInfo, PreprocessSteps,
    Shape, Tensor,
};
use std::fs;
use util::{Prediction, Predictions};

#[test]
fn classify_mobilenet() {
    //initialize openvino runtime core
    let mut core = Core::new(None).unwrap();

    //Read the model
    let mut network = core
        .read_network_from_file(
            &Fixture::graph().to_string_lossy(),
            &Fixture::weights().to_string_lossy(),
        )
        .unwrap();

    //Set up input
    let data = fs::read(Fixture::tensor()).unwrap();
    let input_shape = Shape::new(&vec![1, 224, 224, 3]);
    let element_type = ElementType::F32;
    let tensor = Tensor::new_from_host_ptr(element_type, input_shape, &data).unwrap();

    let pre_post_process = PrePostprocess::new(&network);
    let input_info = pre_post_process.get_input_info_by_name("input");
    let input_tensor_info =
        PreprocessInputTensorInfo::preprocess_input_info_get_tensor_info(&input_info);
    input_tensor_info.preprocess_input_tensor_set_from(&tensor);

    let layout_tensor_string = "NHWC";
    let input_layout = Layout::new(&layout_tensor_string);
    PreprocessInputTensorInfo::preprocess_input_tensor_set_layout(
        &input_tensor_info,
        &input_layout,
    );

    let preprocess_steps = PreprocessSteps::get_preprocess_steps(&input_info);
    PreprocessSteps::preprocess_steps_resize(&preprocess_steps, 0);

    let model_info = PrePostprocess::get_model_info(input_info);
    let layout_string = "NCHW";
    let model_layout = Layout::new(&layout_string);
    PrePostprocess::model_info_set_layout(&model_info, model_layout);

    pre_post_process.build(&mut network);

    let input_port = network.get_input_by_index(0).unwrap(); //got port
    assert_eq!(input_port.get_name().unwrap(), "input");

    //Set up output
    let output_port = network.get_output_by_index(0).unwrap(); //got output port
    assert_eq!(
        output_port.get_name().unwrap(),
        "MobilenetV2/Predictions/Reshape_1"
    );

    // Load the network.
    let mut executable_network = core.compile_model(network, "CPU").unwrap();
    let mut infer_request = executable_network.create_infer_request().unwrap();

    // Read the image.

    // Execute inference.
    infer_request.set_tensor("input", &tensor); /*.unwrap();*/
    infer_request.infer().unwrap();
    let mut results = infer_request
        .get_tensor(output_port.get_name().unwrap())
        .unwrap();

    let buffer = results.get_data::<f32>().unwrap().to_vec();
    // Sort results. It is unclear why the MobileNet output indices are "off by one" but the
    // `.skip(1)` below seems necessary to get results that make sense (e.g. 763 = "revolver" vs 762
    // = "restaurant").
    let mut results: Predictions = buffer
        .iter()
        .skip(1)
        .enumerate()
        .map(|(c, p)| Prediction::new(c, *p))
        .collect();
    results.sort();

    // Compare results using approximate FP comparisons; annotated with classification tags from
    // https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a.
    results[0].assert_approx_eq((963, 0.7134405)); // pizza
    results[1].assert_approx_eq((762, 0.0715866)); // restaurant
    results[2].assert_approx_eq((909, 0.0360171)); // wok
    results[3].assert_approx_eq((926, 0.0160412)); // hot pot
    results[4].assert_approx_eq((567, 0.0152781)); // frying pan

    // This above results almost match (see "off by one" comment above) the output of running
    // OpenVINO's `hello_classification` with the same inputs:
    // $ bin/intel64/Debug/hello_classification /tmp/mobilenet.xml /tmp/val2017/000000062808.jpg CPU
    // Image /tmp/val2017/000000062808.jpg
    // classid probability
    // ------- -----------
    // 964     0.7134405
    // 763     0.0715866
    // 910     0.0360171
    // 927     0.0160412
    // 568     0.0152781
    // 924     0.0148565
    // 500     0.0093886
    // 468     0.0073142
    // 965     0.0058377
    // 545     0.0043731
}
