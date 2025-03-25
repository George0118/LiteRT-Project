import numpy as np
import sys
from cv2 import imread
import os
from testing.resource_usage import calculate_mem_usage, calculate_MACs
from testing.load_model import load_tflite_model
import tracemalloc
import time

def run_inference(interpreter, image):
    """Runs inference on a single image and returns the output."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image)

    start_time = time.perf_counter()
    
    # Run inference
    interpreter.invoke()
    
    end_time = time.perf_counter()
    
    latency = end_time - start_time

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return latency, output_data


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
    image_dir = "../converted_images"
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    
    avg_latency = 0

    for image_path in image_paths:
        try:
            image = imread(image_path)
            image = np.expand_dims(image, axis=0)
            
            latency, output = run_inference(interpreter, image)
            avg_latency += latency
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            
    avg_latency /= len(image_paths)
    print("--- Inferance Complete ---")
    
    _, peak = tracemalloc.get_traced_memory()
    print(f"Peak RAM usage: {peak/1024 :.2f} KB") 
    
    tracemalloc.stop()
    
    print(f"Model Size: {calculate_mem_usage(interpreter)/1024.0 :.2f} KB")
    
    print(f"Total MACs: {calculate_MACs(interpreter)/10**6 :.2f} M")
    
    print(f"Average Latency: {avg_latency * 10**3} ms")
