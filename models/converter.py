import tensorflow as tf
import subprocess

model_paths = [f"./model_test/model_{i}" for i in range(10)]
converted_model_paths = [f"./model_result/model_{i}" for i in range(10)]

for i, path in enumerate(model_paths):
    converter = tf.lite.TFLiteConverter.from_saved_model(path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(converted_model_paths[i]+"/converted_model.tflite", "wb") as f:
        f.write(tflite_model)
        
    # Convert the TFLite file into a CC file for usage
    subprocess.run(["C:/Program Files/Git/usr/bin/xxd.exe", "-i", 
                f"{converted_model_paths[i]}/converted_model.tflite"],
                stdout=open(f"{converted_model_paths[i]}/converted_model.cc", "w"),
                check=True)
        
           
print(f"Conversion complete: TFLite models saved at model_result folder.")