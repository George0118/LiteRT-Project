import tensorflow as tf
import subprocess
import os

# Paths for input models and output converted models
model_paths = [f"./model_test/model_{i}" for i in range(10)]
converted_model_paths = [f"./model_result/model_{i}" for i in range(10)]

for i, path in enumerate(model_paths):
    # Convert the TensorFlow model to a TFLite model
    converter = tf.lite.TFLiteConverter.from_saved_model(path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Create directory if it doesn't exist
    os.makedirs(converted_model_paths[i], exist_ok=True)

    # Define model file names
    tflite_filename = f"{converted_model_paths[i]}/converted_model.tflite"
    cc_filename = f"{converted_model_paths[i]}/model_{i}_data.cc"
    h_filename = f"{converted_model_paths[i]}/model_{i}_data.h"

    # Save the TFLite model
    with open(tflite_filename, "wb") as f:
        f.write(tflite_model)

    # Generate the .cc file using xxd
    with open(cc_filename, "w") as cc_file:
        subprocess.run(
            ["C:/Program Files/Git/usr/bin/xxd.exe", "-i", tflite_filename],
            stdout=cc_file,
            check=True
        )

    # Read the generated C array name from the .cc file
    with open(cc_filename, "r") as cc_file:
        lines = cc_file.readlines()

    # Extract the array name from the first line (typically "unsigned char converted_model_tflite[]")
    array_name = None
    for line in lines:
        if "unsigned char" in line and "[" in line:
            array_name = line.split()[2]  # Extracts the variable name
            array_name = array_name[:-2]
            break

    if not array_name:
        raise ValueError("Failed to extract model array name from .cc file.")

    # Create the corresponding .h file
    with open(h_filename, "w") as h_file:
        h_file.write(f"""#ifndef MODEL_{i}_DATA_H_
#define MODEL_{i}_DATA_H_

#include <cstddef>

extern const unsigned char {array_name}[];
extern const unsigned int {array_name}_len;

#endif  // MODEL_{i}_DATA_H_
""")

print("Conversion complete: TFLite models, .cc, and .h files saved in the model_result folder.")
