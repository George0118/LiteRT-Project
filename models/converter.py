import tensorflow as tf
import numpy as np
import cv2
import os
import subprocess

converted_images_folder = '../converted_images'
input_shape = (32, 32, 3)

def preprocess_image(image):
    """Function to preprocess images"""
    image = cv2.resize(image, (input_shape[0], input_shape[1]))  # Resize to model input size
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def representative_dataset():
    """Function to generate a representative dataset for model quantization"""
    # Load the CIFAR-10 dataset
    (_, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    
    x_test = x_test[:1000]
    
    for image in x_test:
        image = preprocess_image(image)
        yield [image.astype(np.float32)]        
        

model_paths = [f"./model_test/model_{i}" for i in range(10)]
converted_model_paths = [f"./model_result/model_{i}" for i in range(10)]
input_shape = (32,32,3)

for i, path in enumerate(model_paths):
    
    # Converting the TensorFlow model to a TFLite model and applying full quantization (INT8)
    converter = tf.lite.TFLiteConverter.from_saved_model(path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset   # Get the representative dataset for value range
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.target_spec.supported_types = [tf.int8]
    tflite_model = converter.convert()

    os.makedirs(converted_model_paths[i], exist_ok=True)
    tflite_filename = f"{converted_model_paths[i]}/converted_model.tflite"

    # Save the TFLite model
    with open(tflite_filename, "wb") as f:
        f.write(tflite_model)
        
    # Generate the C++ source file for each model
    cc_filename = f"{converted_model_paths[i]}/model_{i}_data.cc"

    # Generate the .cc file using xxd
    with open(cc_filename, "w") as cc_file:
        subprocess.run(
            ["C:/Program Files/Git/usr/bin/xxd.exe", "-i", tflite_filename],
            stdout=cc_file,
            check=True
        )

# Preprocess and save the images so that they can instantly be used in model inference
(_, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    
x_test = x_test[:10]

for i,image in enumerate(x_test):
    image = preprocess_image(image)
    image = (image[0]).astype(np.uint8)
    converted_image_path = converted_images_folder+f"/converted_image_{i}.jpg"
    cv2.imwrite(converted_image_path, image)
    
print("Conversion complete!")
