def calculate_tensor_memory_usage(tensor_details):
    """
    Calculate the memory usage of each tensor based on its details.
    :param tensor_details: Tensor details including shape and data type.
    :return: Memory usage in bytes.
    """

    num_elements = 1
    for dim in tensor_details['shape']:
        num_elements *= dim

    dtype = tensor_details['dtype'].__name__
    
    dtype_sizes = {
        'int8': 1,
        'uint8': 1, 
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