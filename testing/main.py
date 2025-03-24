import numpy as np
import sys
import cv2
import os
from testing.resource_usage import calculate_mem_usage
from testing.load_model import load_tflite_model
import tracemalloc

def run_inference(interpreter, image):
    """Runs inference on a single image and returns the output."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


if __name__ == "__main__":
    model_name = sys.argv[1]
    model_path_cc = f"../models/model_result/{model_name}/{model_name}_data.cc"
    
    tracemalloc.start()
    
    # Load TFLite model
    interpreter = load_tflite_model(model_path_cc)

    # Get input shape from model
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_type = input_details[0]['dtype']

    # Load and preprocess images
    image_dir = "../images"
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])

    for image_path in image_paths:
        try:
            image = cv2.imread(image_path)
            image = image / 255.0
            image = image.astype(input_type)
            image = np.expand_dims(image, axis=0)
            
            output = run_inference(interpreter, image)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            
    print("--- Inferance Complete ---")
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Peak RAM usage: {peak / 1024 :.2f} KB") 
    
    tracemalloc.stop()
    
    print(f"Model Size: {calculate_mem_usage(interpreter)/1024.0} KB")
