# taken from detectron2 with a few modifications
# to include bmm and a few other ops
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/analysis.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import typing
from collections import Counter, defaultdict
import torch
import torch.nn as nn
from functools import partial

from .jit_handles import (
    addmm_flop_jit,
    batchnorm_flop_jit,
    conv_flop_jit,
    einsum_flop_jit,
    matmul_flop_jit,
    bmm_flop_jit,
    basic_binary_op_flop_jit,
    rsqrt_flop_jit,
    softmax_flop_jit,
    dropout_flop_jit,
    linear_flop_jit,
    baddbmm_flop_jit,
    layer_norm_flop_jit,
)

# A dictionary that maps supported operations to their flop count jit handles.
_SUPPORTED_OPS: typing.Dict[str, typing.Callable] = {
    "aten::addmm": addmm_flop_jit,
    "aten::_convolution": conv_flop_jit,
    "aten::einsum": einsum_flop_jit,
    "aten::matmul": matmul_flop_jit,
    "aten::batch_norm": batchnorm_flop_jit,
    "aten::bmm": bmm_flop_jit,
    "aten::add": partial(basic_binary_op_flop_jit, name='aten::add'),
    "aten::add_": partial(basic_binary_op_flop_jit, name='aten::add_'),
    "aten::mul": partial(basic_binary_op_flop_jit, name='aten::mul'),
    "aten::sub": partial(basic_binary_op_flop_jit, name='aten::sub'),
    "aten::div": partial(basic_binary_op_flop_jit, name='aten::div'),
    "aten::floor_divide": partial(basic_binary_op_flop_jit, name='aten::floor_divide'),
    "aten::relu": partial(basic_binary_op_flop_jit, name='aten::relu'),
    "aten::relu_": partial(basic_binary_op_flop_jit, name='aten::relu_'),
    "aten::rsqrt": rsqrt_flop_jit,
    "aten::softmax": softmax_flop_jit,
    "aten::dropout": dropout_flop_jit,
    "aten::linear": linear_flop_jit,
    "aten::baddbmm": baddbmm_flop_jit,
    "aten::layer_norm": layer_norm_flop_jit,
}

# A list that contains ignored operations.
_IGNORED_OPS: typing.List[str] = [
    "aten::Int",
    "aten::__and__",
    "aten::arange",
    "aten::cat",
    "aten::clamp",
    "aten::clamp_",
    "aten::contiguous",
    "aten::copy_",
    "aten::detach",
    "aten::empty",
    "aten::eq",
    "aten::expand",
    "aten::flatten",
    "aten::floor",
    "aten::full",
    "aten::gt",
    "aten::index",
    "aten::index_put_",
    "aten::max",
    "aten::nonzero",
    "aten::permute",
    "aten::remainder",
    "aten::reshape",
    "aten::select",
    "aten::size",
    "aten::slice",
    "aten::split_with_sizes",
    "aten::squeeze",
    "aten::t",
    "aten::to",
    "aten::transpose",
    "aten::unsqueeze",
    "aten::view",
    "aten::zeros",
    "aten::zeros_like",
    "prim::Constant",
    "prim::Int",
    "prim::ListConstruct",
    "prim::ListUnpack",
    "prim::NumToTensor",
    "prim::TupleConstruct",
]

_HAS_ALREADY_SKIPPED = False


def _infer_module_from_node(node, module_names, model, inputs_list):
    """
    Try to determine which named module a JIT node belongs to by examining
    the node's scope name.
    
    Args:
        node: JIT node
        module_names: dict of module names to module objects
        model: the full model
        inputs_list: list of inputs to the node
    
    Returns:
        str: module name if found, None otherwise
    """
    try:
        # Try to get scope name from the node
        scope_names = node.scopeName()
        if scope_names:
            # scope_names is something like "__module.backbone/...__module.transformer/..."
            # Extract module names
            parts = scope_names.split('/')
            for part in parts:
                # Look for patterns like "__module.backbone" or "__module.transformer"
                if '__module.' in part:
                    module_part = part.split('__module.')[-1]
                    # Get the first module name component
                    for name in module_names.keys():
                        if name.startswith(module_part.split('[')[0]):
                            # Return the top-level module (first part before '.')
                            top_level = name.split('.')[0] if '.' in name else name
                            return top_level
    except:
        pass
    
    return None


def flop_count(
    model: nn.Module,
    inputs: typing.Tuple[object, ...],
    whitelist: typing.Union[typing.List[str], None] = None,
    customized_ops: typing.Union[
        typing.Dict[str, typing.Callable], None
    ] = None,
    module_tracking: bool = False,
) -> typing.Union[typing.DefaultDict[str, float], typing.DefaultDict[str, typing.DefaultDict[str, float]]]:
    """
    Given a model and an input to the model, compute the Gflops of the given
    model. Note the input should have a batch size of 1.
    Args:
        model (nn.Module): The model to compute flop counts.
        inputs (tuple): Inputs that are passed to `model` to count flops.
            Inputs need to be in a tuple.
        whitelist (list(str)): Whitelist of operations that will be counted. It
            needs to be a subset of _SUPPORTED_OPS. By default, the function
            computes flops for all supported operations.
        customized_ops (dict(str,Callable)) : A dictionary contains customized
            operations and their flop handles. If customized_ops contains an
            operation in _SUPPORTED_OPS, then the default handle in
             _SUPPORTED_OPS will be overwritten.
        module_tracking (bool): If True, returns a nested dictionary with flops
            broken down by named modules (e.g., 'transformer', 'backbone', etc).
            If False, returns a flat dictionary with flops by operation type.
    Returns:
        defaultdict: If module_tracking=False, a dictionary that records the 
            number of gflops for each operation. If module_tracking=True, a 
            dictionary with module names as keys and operation flop dicts as values.
    """
    # Copy _SUPPORTED_OPS to flop_count_ops.
    # If customized_ops is provided, update _SUPPORTED_OPS.
    flop_count_ops = _SUPPORTED_OPS.copy()
    if customized_ops:
        flop_count_ops.update(customized_ops)

    # If whitelist is None, count flops for all suported operations.
    if whitelist is None:
        whitelist_set = set(flop_count_ops.keys())
    else:
        whitelist_set = set(whitelist)

    # Torch script does not support parallell torch models.
    if isinstance(
        model,
        (nn.parallel.distributed.DistributedDataParallel, nn.DataParallel),
    ):
        model = model.module  # pyre-ignore

    assert set(whitelist_set).issubset(
        flop_count_ops
    ), "whitelist needs to be a subset of _SUPPORTED_OPS and customized_ops."
    assert isinstance(inputs, tuple), "Inputs need to be in a tuple."

    # Build a mapping of module names for module-level tracking
    module_names = {}
    module_params = {}
    if module_tracking:
        for name, module in model.named_modules():
            module_names[name] = module
            for param_name, param in module.named_parameters(recurse=False):
                module_params[id(param)] = name

    # Compatibility with torch.jit.
    if hasattr(torch.jit, "get_trace_graph"):
        trace, _ = torch.jit.get_trace_graph(model, inputs)
        trace_nodes = trace.graph().nodes()
    else:
        trace, _ = torch.jit._get_trace_graph(model, inputs)
        trace_nodes = trace.nodes()

    skipped_ops = Counter()
    total_flop_counter = Counter()
    module_flop_counter = defaultdict(lambda: Counter()) if module_tracking else None

    for node in trace_nodes:
        kind = node.kind()
        if kind not in whitelist_set:
            # If the operation is not in _IGNORED_OPS, count skipped operations.
            if kind not in _IGNORED_OPS:
                skipped_ops[kind] += 1
            continue

        handle_count = flop_count_ops.get(kind, None)
        if handle_count is None:
            continue

        inputs_list, outputs = list(node.inputs()), list(node.outputs())
        flops_counter = handle_count(inputs_list, outputs)
        total_flop_counter += flops_counter
        
        # Track by module if enabled
        if module_tracking:
            # Try to determine which module this operation belongs to
            module_name = _infer_module_from_node(node, module_names, model, inputs_list)
            if module_name:
                module_flop_counter[module_name] += flops_counter
            else:
                # Put unattributed flops in 'other'
                module_flop_counter['other'] += flops_counter

    global _HAS_ALREADY_SKIPPED
    if len(skipped_ops) > 0 and not _HAS_ALREADY_SKIPPED:
        _HAS_ALREADY_SKIPPED = True
        for op, freq in skipped_ops.items():
            logging.warning("Skipped operation {} {} time(s)".format(op, freq))

    # Convert flop count to gigaflops.
    final_count = defaultdict(float)
    for op in total_flop_counter:
        final_count[op] = total_flop_counter[op] / 1e9

    if module_tracking:
        # Convert module-level flop counts to gigaflops
        final_module_count = defaultdict(lambda: defaultdict(float))
        for module_name, op_counts in module_flop_counter.items():
            for op, count in op_counts.items():
                final_module_count[module_name][op] = count / 1e9
        return final_module_count
    
    return final_count