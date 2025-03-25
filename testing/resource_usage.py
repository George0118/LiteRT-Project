from functools import reduce

def calculate_tensor_memory_usage(tensor):
    """
    Calculate the memory usage of each tensor based on its details.
    :param tensor_details: Tensor details including shape and data type.
    :return: Memory usage in bytes.
    """

    num_elements = 1
    for dim in tensor['shape']:
        num_elements *= dim

    dtype = tensor['dtype'].__name__
    
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

def calculate_tensor_MACs(interpreter, tensor):
    """Calculate the MACs for a tensor."""
    shape = tensor['shape']
    name = tensor['name']
    
    if not "model" in name:   # If the tensor does not concern a layer
        return 0
    
    if "conv2d" in name:  # Conv2D Layer (Batch, H, W, Channels)
        input_channels = shape[3]
        output_channels = shape[3]
        kernel_h, kernel_w = 3,3        # Assume 3x3 Convolution
        
        # MACs Formula for Conv2D: (H * W * C_in) * (K_H * K_W * C_out)
        macs = shape[1] * shape[2] * input_channels * kernel_h * kernel_w * output_channels

    elif "re_lu" in name:  # ReLu Layer
        
        # Apply Relu to each data inside the tensor
        macs = shape[1] * shape[2] * shape[3]
    
    elif "add" in name:     # Add Layer
        
        # Apply Add Operator to each data of the input tensors
        macs = shape[1] * shape[2] * shape[3]
        
    elif "global_average_pooling2d" in name:    # 2D Global Average Pooling
        input_shape = interpreter.get_tensor_details()[tensor['index']-1]['shape']
        
        # Height x Width for each Channel
        macs = input_shape[1] * input_shape[2] * input_shape[3]
        
    elif "average_pooling2d" in name:   # 2D Pooling Layer
        input_shape = interpreter.get_tensor_details()[tensor['index']-1]['shape']
        input_channels = input_shape[3]
        output_channels = shape[3]
        pool_h, pool_w = 2,2        # Assume 2x2 Pooling Window
        
        # MACs Formula for Conv2D: (H * W * C_in) * (K_H * K_W * C_out)
        macs = input_shape[1] * input_shape[2] * input_channels * pool_h * pool_w * output_channels
        
    elif "multiply" in name:   # Multiplication Layer
        
        # Apply Mul Operator to each data of the input tensors
        macs = shape[1] * shape[2] * shape[3]
        
    elif "dense" in name:       # Dense Layer
        input_shape = interpreter.get_tensor_details()[tensor['index']-1]['shape']
        input_neurons = reduce(lambda x, y: x * y, input_shape[1:])     # Flatten input
        output_neurons = shape[1]
        
        # MACs Formula for FC Layer: input_neurons * output_neurons
        macs = input_neurons * output_neurons
        
    else:
        print("Unsupported Layer!")
        macs = 0  # Skip unsupported layers
    
    return macs

def calculate_MACs(interpreter):
    """Calculates the number of MACs for the current model."""
    macs = 0
    tensor_details = interpreter.get_tensor_details()
    for t in tensor_details:
        macs += calculate_tensor_MACs(interpreter, t)
        
    return macs