from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.tools.flatbuffer_utils import xxd_output_to_bytes
    
def load_tflite_model(model_path_cc):
    """Loads the TensorFlow Lite model and returns the interpreter."""
    print(f"Loading model: {model_path_cc}")
    model_data = xxd_output_to_bytes(model_path_cc)
    interpreter = Interpreter(model_content=model_data)
    interpreter.allocate_tensors()
    return interpreter