import psutil
import tracemalloc
import tensorflow.lite as tflite
import sys

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss  # Memory usage in bytes

def load_tflite_model(model_path):
    print(f"Loading model: {model_path}")
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def check_model_memory_usage(model_path):
    tracemalloc.start()
    
    interpreter = load_tflite_model(model_path)

    peak_memory = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    
    print(f"Peak Memory Usage: {peak_memory / 1024**2:.2f} MB")
    
    return interpreter

if __name__ == "__main__":
    model_path = sys.argv[1]
    model_path = "../models/model_result/" + model_path + "/converted_model.tflite"
    check_model_memory_usage(model_path)