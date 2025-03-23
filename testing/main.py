import tensorflow.lite as tflite
import numpy as np
import sys
import cv2
import os
from testing.resource_usage import calculate_mem_usage
import tracemalloc

def load_tflite_model(model_path):
    """Loads the TensorFlow Lite model and returns the interpreter."""
    print(f"Loading model: {model_path}")
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image_path, input_shape, type):
    """Loads and preprocesses an image to match the model's input shape."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    image = cv2.resize(image, (input_shape[1], input_shape[2]))  # Resize to model input size
    image = image / 255.0  # Normalize
    image = image.astype(type)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

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
    model_path = f"../models/model_result/{model_name}/converted_model.tflite"
    
    tracemalloc.start()
    
    # Load TFLite model
    interpreter = load_tflite_model(model_path)

    # Get input shape from model
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_type = input_details[0]['dtype']

    # Load and preprocess images
    image_dir = "../images"
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])

    for image_path in image_paths:
        try:
            image = preprocess_image(image_path, input_shape, input_type)
            output = run_inference(interpreter, image)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            
    print("--- Inferance Complete ---")
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Peak RAM usage: {peak / 1024 :.2f} KB") 

    tracemalloc.stop()
    
    print(f"Model Memory Usage: {calculate_mem_usage(interpreter)/1024.0} KB")
