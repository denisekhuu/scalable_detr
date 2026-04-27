# taken from detectron2 / fvcore with a few modifications
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/analysis.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import typing
from collections import Counter, OrderedDict
import numpy as np
from numpy import prod
from itertools import zip_longest


def get_shape(val: object) -> typing.List[int]:
    """
    Get the shapes from a jit value object.
    Args:
        val (torch._C.Value): jit value object.
    Returns:
        list(int): return a list of ints.
    """
    if val.isCompleteTensor():  # pyre-ignore
        r = val.type().sizes()  # pyre-ignore
        if not r:
            r = [1]
        return r
    elif val.type().kind() in ("IntType", "FloatType"):
        return [1]
    else:
        raise ValueError()


def addmm_flop_jit(
    inputs: typing.List[object], outputs: typing.List[object]
) -> typing.Counter[str]:
    """
    This method counts the flops for fully connected layers with torch script.
    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object.
    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    # Count flop for nn.Linear
    # inputs is a list of length 3.
    input_shapes = [get_shape(v) for v in inputs[1:3]]
    # input_shapes[0]: [batch size, input feature dimension]
    # input_shapes[1]: [batch size, output feature dimension]
    assert len(input_shapes[0]) == 2
    assert len(input_shapes[1]) == 2
    batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][1]
    flop = batch_size * input_dim * output_dim
    flop_counter = Counter({"addmm": flop})
    return flop_counter


def bmm_flop_jit(inputs, outputs):
    """Flop counter for aten::bmm and aten::matmul with 3-D inputs."""
    input_shapes = [get_shape(v) for v in inputs]
    # shapes: (B, M, K) and (B, K, N)
    assert len(input_shapes) == 2
    a, b = input_shapes[0], input_shapes[1]
    # support both 2-D and 3-D (batch) matmul
    if len(a) == 3 and len(b) == 3:
        B, M, K = a
        _, K2, N = b
    elif len(a) == 2 and len(b) == 2:
        M, K = a
        K2, N = b
        B = 1
    else:
        raise ValueError(f"Unexpected matmul shapes: {a}, {b}")
    assert K == K2
    flop = B * M * K * N
    return Counter({"matmul": flop})



def basic_binary_op_flop_jit(inputs, outputs, name):
    input_shapes = []
    for v in inputs:
        try:
            input_shapes.append(get_shape(v))
        except ValueError:
            continue
    
    if not input_shapes:
        # If no valid inputs, return zero flops
        return Counter({name: 0})

    input_shapes = [s[::-1] for s in input_shapes]
    max_shape = np.array(list(zip_longest(*input_shapes, fillvalue=1))).max(1)
    flop = prod(max_shape)
    flop_counter = Counter({name: flop})
    return flop_counter


def rsqrt_flop_jit(inputs, outputs):
    input_shapes = [get_shape(v) for v in inputs]
    flop = prod(input_shapes[0]) * 2
    flop_counter = Counter({"rsqrt": flop})
    return flop_counter

def dropout_flop_jit(inputs, outputs):
    input_shapes = [get_shape(v) for v in inputs[:1]]
    flop = prod(input_shapes[0])
    flop_counter = Counter({"dropout": flop})
    return flop_counter

def softmax_flop_jit(inputs, outputs):
    # from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/profiler/internal/flops_registry.py
    input_shapes = [get_shape(v) for v in inputs[:1]]
    flop = prod(input_shapes[0]) * 5
    flop_counter = Counter({'softmax': flop})
    return flop_counter

def _reduction_op_flop_jit(inputs, outputs, reduce_flops=1, finalize_flops=0):
    input_shapes = [get_shape(v) for v in inputs]
    output_shapes = [get_shape(v) for v in outputs]

    in_elements = prod(input_shapes[0])
    out_elements = prod(output_shapes[0])

    num_flops = (in_elements * reduce_flops
        + out_elements * (finalize_flops - reduce_flops))

    return num_flops


def conv_flop_count(
    x_shape: typing.List[int],
    w_shape: typing.List[int],
    out_shape: typing.List[int],
) -> typing.Counter[str]:
    """
    This method counts the flops for convolution. Note only multiplication is
    counted. Computation for addition and bias is ignored.
    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    batch_size, Cin_dim, Cout_dim = x_shape[0], w_shape[1], out_shape[1]
    out_size = prod(out_shape[2:])
    kernel_size = prod(w_shape[2:])
    flop = batch_size * out_size * Cout_dim * Cin_dim * kernel_size
    flop_counter = Counter({"conv": flop})
    return flop_counter


def conv_flop_jit(
    inputs: typing.List[object], outputs: typing.List[object]
) -> typing.Counter[str]:
    """
    This method counts the flops for convolution using torch script.
    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object before convolution.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object after convolution.
    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    assert len(inputs) >= 2, f"Expected at least 2 inputs for convolution, got {len(inputs)}"
    x, w = inputs[:2]
    x_shape, w_shape, out_shape = (
        get_shape(x),
        get_shape(w),
        get_shape(outputs[0]),
    )
    return conv_flop_count(x_shape, w_shape, out_shape)


def einsum_flop_jit(
    inputs: typing.List[object], outputs: typing.List[object]
) -> typing.Counter[str]:
    """
    This method counts the flops for the einsum operation. We currently support
    two einsum operations: "nct,ncp->ntp" and "ntg,ncg->nct".
    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object before einsum.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object after einsum.
    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    assert len(inputs) >= 2
    equation = inputs[0].toIValue()  # pyre-ignore
    equation = equation.replace(" ", "")
    letter_order = OrderedDict((k, 0) for k in equation if k.isalpha()).keys()
    mapping = {ord(x): 97 + i for i, x in enumerate(letter_order)}
    equation = equation.translate(mapping)
    if len(inputs) == 2:
        input_shapes_jit = inputs[1].node().inputs()  # pyre-ignore
        input_shapes = [get_shape(v) for v in input_shapes_jit]
    else:
        input_shapes = []
        for v in inputs[1:]:
            try:
                input_shapes.append(get_shape(v))
            except ValueError:
                input_shapes.append(None)

    if equation == "abc,abd->acd":
        n, c, t = input_shapes[0]
        p = input_shapes[-1][-1]
        flop = n * c * t * p
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    elif equation == "abc,adc->adb":
        n, t, g = input_shapes[0]
        c = input_shapes[-1][1]
        flop = n * t * g * c
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    elif equation == "...ab,acb->...ac":
        out_shape = get_shape(outputs[0])
        b_dim = None
        if input_shapes[0] is not None:
            b_dim = input_shapes[0][-1]       # last dim of input1 = 'b'
        elif input_shapes[1] is not None:
            b_dim = input_shapes[1][-1]       # last dim of input2 = 'b'
        if b_dim is None:
            b_dim = 1
        flop = prod(out_shape) * b_dim
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    elif equation == "...ab,abc->...ac":
        out_shape = get_shape(outputs[0])
        b_dim = None
        if input_shapes[0] is not None:
            b_dim = input_shapes[0][-1]       # last dim of input1 = 'b'
        elif input_shapes[1] is not None:
            b_dim = input_shapes[1][-2]       # second-to-last of input2 = 'b'
        if b_dim is None:
            b_dim = 1
        flop = prod(out_shape) * b_dim
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    elif equation == "abc,bdc->abd":
        out_shape = get_shape(outputs[0])
        c_dim = None
        if input_shapes[0] is not None:
            c_dim = input_shapes[0][-1]       # last dim of input1 = 'c'
        elif input_shapes[1] is not None:
            c_dim = input_shapes[1][-1]       # last dim of input2 = 'c'
        if c_dim is None:
            c_dim = 1
        flop = prod(out_shape) * c_dim
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    else:
        raise NotImplementedError("Unsupported einsum operation.")


def matmul_flop_jit(
    inputs: typing.List[object], outputs: typing.List[object]
) -> typing.Counter[str]:
    """
    This method counts the flops for matmul.
    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object before matmul.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object after matmul.
    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    input_shapes = [get_shape(v) for v in inputs]
    assert len(input_shapes) == 2
    a, b = input_shapes[0], input_shapes[1]
    assert a[-1] == b[-2] if len(b) >= 2 else a[-1] == b[0]
    if len(a) == 3 and len(b) == 3:
        # Batched matmul: (B, M, K) x (B, K, N) -> (B, M, N)
        B, M, K = a
        _, _, N = b
        flop = B * M * K * N
    elif len(a) == 2 and len(b) == 2:
        # Standard 2D matmul: (M, K) x (K, N) -> (M, N)
        M, K = a
        K, N = b
        flop = M * K * N
    else:
        # General case: batch dims are all but last two
        flop = prod(a) * b[-1]
    flop_counter = Counter({"matmul": flop})
    return flop_counter


def batchnorm_flop_jit(
    inputs: typing.List[object], outputs: typing.List[object]
) -> typing.Counter[str]:
    """
    This method counts the flops for batch norm.
    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object before batch norm.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object after batch norm.
    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    # Inputs[0] contains the shape of the input.
    input_shape = get_shape(inputs[0])
    assert 2 <= len(input_shape) <= 5
    flop = prod(input_shape) * 4
    flop_counter = Counter({"batchnorm": flop})
    return flop_counter


def linear_flop_jit(
    inputs: typing.List[object], outputs: typing.List[object]
) -> typing.Counter[str]:
    """
    This method counts the flops for linear (fully connected) layers.
    Args:
        inputs (list(torch._C.Value)): The input shape.
        outputs (list(torch._C.Value)): The output shape.
    Returns:
        Counter: A Counter dictionary that records the number of flops.
    """
    # aten::linear has inputs: input, weight, bias
    input_shape = get_shape(inputs[0])
    weight_shape = get_shape(inputs[1])
    
    # input_shape: [..., in_features]
    # weight_shape: [out_features, in_features]
    # output_shape: [..., out_features]
    
    # Number of elements is all dimensions except the last one of input
    num_instances = prod(input_shape[:-1])
    in_features = input_shape[-1]
    out_features = weight_shape[0]
    
    # Each output element is a dot product: in_features multiplications
    flop = num_instances * in_features * out_features
    flop_counter = Counter({"linear": flop})
    return flop_counter


def baddbmm_flop_jit(
    inputs: typing.List[object], outputs: typing.List[object]
) -> typing.Counter[str]:
    """
    Count flops for baddbmm: output = input + beta * (mat1 @ mat2)
    Args:
        inputs (list(torch._C.Value)): JIT values for baddbmm operation.
        outputs (list(torch._C.Value)): The output shape.
    Returns:
        Counter: A Counter dictionary that records the number of flops.
    """
    # baddbmm inputs: self, mat1, mat2, beta, alpha
    try:
        input_shape = get_shape(inputs[0])  # input
        mat1_shape = get_shape(inputs[1])   # mat1
        mat2_shape = get_shape(inputs[2])   # mat2
        
        # Expected shapes:
        # input: [b, n, m]
        # mat1: [b, n, p]
        # mat2: [b, p, m]
        # output: [b, n, m]
        
        if len(mat1_shape) == 3 and len(mat2_shape) == 3:
            b, n, p = mat1_shape
            m = mat2_shape[-1]
            # Matrix multiplication: b * n * p * m
            # Addition: b * n * m
            flop = b * n * p * m + b * n * m
            flop_counter = Counter({"baddbmm": flop})
            return flop_counter
    except:
        pass
    
    return Counter({"baddbmm": 0})


def layer_norm_flop_jit(
    inputs: typing.List[object], outputs: typing.List[object]
) -> typing.Counter[str]:
    """
    Count flops for layer normalization.
    Layer norm: y = (x - mean) / sqrt(var + eps) * weight + bias
    Roughly: 4 * number_of_elements operations (mean, var, normalize, scale)
    Args:
        inputs (list(torch._C.Value)): JIT values.
        outputs (list(torch._C.Value)): The output shape.
    Returns:
        Counter: A Counter dictionary that records the number of flops.
    """
    try:
        input_shape = get_shape(inputs[0])
        # For layer norm, we count: 4 ops per element (mean, var, norm, scale)
        flop = prod(input_shape) * 4
        flop_counter = Counter({"layer_norm": flop})
        return flop_counter
    except:
        return Counter({"layer_norm": 0})

def baddbmm_flop_jit(
    inputs: typing.List[object], outputs: typing.List[object]
) -> typing.Counter[str]:
    """
    Count flops for baddbmm: output = input + beta * (mat1 @ mat2)
    Args:
        inputs (list(torch._C.Value)): JIT values for baddbmm operation.
        outputs (list(torch._C.Value)): The output shape.
    Returns:
        Counter: A Counter dictionary that records the number of flops.
    """
    try:
        input_shape = get_shape(inputs[0])  # input
        mat1_shape = get_shape(inputs[1])   # mat1
        mat2_shape = get_shape(inputs[2])   # mat2
        
        if len(mat1_shape) == 3 and len(mat2_shape) == 3:
            b, n, p = mat1_shape
            m = mat2_shape[-1]
            flop = b * n * p * m + b * n * m
            flop_counter = Counter({"baddbmm": flop})
            return flop_counter
    except:
        pass
    
    return Counter({"baddbmm": 0})


def layer_norm_flop_jit(
    inputs: typing.List[object], outputs: typing.List[object]
) -> typing.Counter[str]:
    """
    Count flops for layer normalization.
    Layer norm: y = (x - mean) / sqrt(var + eps) * weight + bias
    Roughly: 4 * number_of_elements operations (mean, var, normalize, scale)
    Args:
        inputs (list(torch._C.Value)): JIT values.
        outputs (list(torch._C.Value)): The output shape.
    Returns:
        Counter: A Counter dictionary that records the number of flops.
    """
    try:
        input_shape = get_shape(inputs[0])
        # For layer norm, we count: 4 ops per element (mean, var, norm, scale)
        flop = prod(input_shape) * 4
        flop_counter = Counter({"layer_norm": flop})
        return flop_counter
    except:
        return Counter({"layer_norm": 0})
