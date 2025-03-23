def calculate_tensor_memory_usage(tensor_details):
    """
    Calculate the memory usage of each tensor based on its details.
    :param tensor_details: Tensor details including shape and data type.
    :return: Memory usage in bytes.
    """

    num_elements = 1
    for dim in tensor_details['shape']:
        num_elements *= dim

    dtype = tensor_details['dtype']
    
    dtype_sizes = {
        'numpy.float32': 4,  # 4 bytes for float32
        'numpy.int8': 1,     # 1 byte for int8
        'numpy.uint8': 1,    # 1 byte for uint8
    }
    
    # Get the size per element (in bytes)
    size_per_element = dtype_sizes.get(dtype, 4)
    
    # Calculate the total memory usage for this tensor
    memory_usage_bytes = num_elements * size_per_element
    return memory_usage_bytes

def calculate_mem_usage(interpreter):
    """Calculates the needed memory for model deployment."""
    memory = 0
    tensor_details = interpreter.get_tensor_details()
    for t in tensor_details:
        memory += calculate_tensor_memory_usage(t)
    
    return memory