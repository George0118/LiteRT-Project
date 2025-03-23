#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// Models
#include "model_0_data.h"
#include "model_1_data.h"
#include "model_2_data.h"
#include "model_3_data.h"
#include "model_4_data.h"
#include "model_5_data.h"
#include "model_6_data.h"
#include "model_7_data.h"
#include "model_8_data.h"
#include "model_9_data.h"

#define TFLITE_MINIMAL_CHECK(x)                                  \
    if (!(x))                                                    \
    {                                                            \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }


// Function to run inference on a single image

void RunInference(std::unique_ptr<tflite::Interpreter>& interpreter, const std::string& image_path) {
    // Load image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        printf("Failed to load image: %s\n", image_path.c_str());
        return;
    }

    // Resize to 32x32 and normalize
    cv::resize(image, image, cv::Size(32, 32));
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);

    // Get the input tensor
    int input_index = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input_index);

    float* input_data = input_tensor->data.f;
    std::memcpy(input_data, image.data, 32 * 32 * 3 * sizeof(float));


    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Error running inference" << std::endl;
        return;
    }

    // Get the output tensor
    int output_index = interpreter->outputs()[0];
    TfLiteTensor* output_tensor = interpreter->tensor(output_index);

    int output_size = output_tensor->dims->data[1];
    float* output_data = output_tensor->data.f;
    
    
}


int main(int argc, char *argv[]) {

    if (argc < 2) {
        printf("Usage: %s <model_file>\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const std::string images_directory = "C:\\Users\\gmyst\\Desktop\\Work\\Delft\\LiteRT-Project\\images";

    // Load Model
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_path);
    TFLITE_MINIMAL_CHECK(model != nullptr);

    // Build interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);

    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

    for (int i = 0; i < 10; i++) {
        std::string image_path = images_directory + "\\image_" + std::to_string(i) + ".jpg";
        RunInference(interpreter, image_path);
    }

    return 0;
}
