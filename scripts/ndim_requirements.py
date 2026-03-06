#!/usr/bin/env python3
"""
Comprehensive script to discover NDim requirements for NumPy functions
implemented in numpy-ts.

For each function, tests various input shapes/ndims to determine:
- Minimum ndim required
- Maximum ndim supported
- Special ndim behaviors (e.g., matmul 1D@1D -> scalar)
"""
import numpy as np
import json
import sys
import traceback

def make_array(shape, dtype='float64'):
    """Create a test array of given shape."""
    if dtype == 'int32':
        return np.arange(1, np.prod(shape) + 1, dtype=np.int32).reshape(shape)
    elif dtype == 'bool':
        return np.ones(shape, dtype=bool)
    return np.arange(1.0, np.prod(shape) + 1.0).reshape(shape)

def test_fn(fn_name, fn, test_cases):
    """
    Test a function with various cases.
    test_cases: list of dicts with keys 'args', 'kwargs', 'desc'
    Returns list of {desc, shape_in, ndim_in, shape_out, ndim_out, error}
    """
    results = []
    for case in test_cases:
        desc = case.get('desc', str(case))
        args = case.get('args', [])
        kwargs = case.get('kwargs', {})
        try:
            out = fn(*args, **kwargs)
            if isinstance(out, np.ndarray):
                result = {
                    'desc': desc,
                    'input_shapes': [a.shape if isinstance(a, np.ndarray) else None for a in args],
                    'input_ndims': [a.ndim if isinstance(a, np.ndarray) else None for a in args],
                    'output_shape': list(out.shape),
                    'output_ndim': out.ndim,
                    'status': 'ok'
                }
            elif isinstance(out, (int, float, complex, np.number, np.bool_)):
                result = {
                    'desc': desc,
                    'input_shapes': [a.shape if isinstance(a, np.ndarray) else None for a in args],
                    'input_ndims': [a.ndim if isinstance(a, np.ndarray) else None for a in args],
                    'output_shape': [],
                    'output_ndim': 0,
                    'status': 'ok'
                }
            elif isinstance(out, tuple):
                result = {
                    'desc': desc,
                    'input_shapes': [a.shape if isinstance(a, np.ndarray) else None for a in args],
                    'input_ndims': [a.ndim if isinstance(a, np.ndarray) else None for a in args],
                    'output_shape': [o.shape if isinstance(o, np.ndarray) else None for o in out],
                    'output_ndim': None,
                    'status': 'ok'
                }
            else:
                result = {
                    'desc': desc,
                    'input_shapes': [a.shape if isinstance(a, np.ndarray) else None for a in args],
                    'input_ndims': [a.ndim if isinstance(a, np.ndarray) else None for a in args],
                    'output_type': type(out).__name__,
                    'status': 'ok'
                }
        except Exception as e:
            result = {
                'desc': desc,
                'input_shapes': [a.shape if isinstance(a, np.ndarray) else None for a in args],
                'input_ndims': [a.ndim if isinstance(a, np.ndarray) else None for a in args],
                'status': 'error',
                'error': str(e)
            }
        results.append(result)
    return results


# ============================================================
# Define test cases for each function
# ============================================================
all_results = {}

# Helper arrays
s0 = np.array(5.0)              # 0D scalar
v3 = make_array((3,))           # 1D, 3 elements
v4 = make_array((4,))           # 1D, 4 elements
m23 = make_array((2, 3))        # 2D, 2x3
m33 = make_array((3, 3))        # 2D, 3x3
m34 = make_array((3, 4))        # 2D, 3x4
t234 = make_array((2, 3, 4))    # 3D
t2234 = make_array((2, 2, 3, 4)) # 4D
t22 = make_array((2, 2))        # 2D square
t222 = make_array((2, 2, 2))    # 3D batch
t2222 = make_array((2, 2, 2, 2)) # 4D batch
bool_v3 = make_array((3,), 'bool')
bool_m23 = make_array((2,3), 'bool')
int_v3 = make_array((3,), 'int32')
int_m23 = make_array((2,3), 'int32')

# ============================================================
# matmul
# ============================================================
all_results['matmul'] = test_fn('matmul', np.matmul, [
    {'desc': '1D@1D (dot)', 'args': [v3, v3]},
    {'desc': '2D@1D (matvec)', 'args': [m23, v3]},
    {'desc': '1D@2D (vecmat)', 'args': [v3, m33]},
    {'desc': '2D@2D', 'args': [m23, m34]},
    {'desc': '3D@3D (batched)', 'args': [make_array((2,3,4)), make_array((2,4,5))]},
    {'desc': '4D@4D (batched)', 'args': [make_array((2,2,3,4)), make_array((2,2,4,5))]},
    {'desc': '3D@2D (broadcast)', 'args': [make_array((2,3,4)), make_array((4,5))]},
    {'desc': '1D@3D', 'args': [v3, make_array((2,3,4))]},
    {'desc': '0D fail', 'args': [s0, s0]},
])

# ============================================================
# dot
# ============================================================
all_results['dot'] = test_fn('dot', np.dot, [
    {'desc': '0D scalar', 'args': [s0, s0]},
    {'desc': '1D@1D inner', 'args': [v3, v3]},
    {'desc': '2D@1D', 'args': [m23, v3]},
    {'desc': '1D@2D', 'args': [v3, m33]},
    {'desc': '2D@2D', 'args': [m23, m34]},
    {'desc': '3D general', 'args': [t234, make_array((4,5))]},
    {'desc': '4D general', 'args': [t2234, make_array((4,5))]},
])

# ============================================================
# inner
# ============================================================
all_results['inner'] = test_fn('inner', np.inner, [
    {'desc': '0D', 'args': [s0, s0]},
    {'desc': '1D@1D', 'args': [v3, v3]},
    {'desc': '2D@1D', 'args': [m23, v3]},
    {'desc': '2D@2D', 'args': [m23, m23]},
    {'desc': '3D@1D', 'args': [t234, v4]},
    {'desc': '3D@2D', 'args': [t234, make_array((2,4))]},
])

# ============================================================
# outer
# ============================================================
all_results['outer'] = test_fn('outer', np.outer, [
    {'desc': '1D@1D', 'args': [v3, v4]},
    {'desc': '2D@2D (flattened)', 'args': [m23, m23]},
    {'desc': '3D@3D', 'args': [t234, t234]},
])

# ============================================================
# tensordot
# ============================================================
all_results['tensordot'] = test_fn('tensordot', np.tensordot, [
    {'desc': '1D scalar (axes=1)', 'args': [v3, v3], 'kwargs': {'axes': 1}},
    {'desc': '2D (axes=1)', 'args': [m23, m34], 'kwargs': {'axes': 1}},
    {'desc': '2D (axes=2)', 'args': [m23, make_array((3,4))], 'kwargs': {'axes': [[1],[0]]}},
    {'desc': '3D (axes=1)', 'args': [t234, make_array((4,5))], 'kwargs': {'axes': 1}},
    {'desc': '3D outer (axes=0)', 'args': [v3, v4], 'kwargs': {'axes': 0}},
])

# ============================================================
# einsum
# ============================================================
all_results['einsum'] = test_fn('einsum', np.einsum, [
    {'desc': 'i,i-> (dot)', 'args': ['i,i->', v3, v3]},
    {'desc': 'ij,jk->ik (matmul)', 'args': ['ij,jk->ik', m23, m34]},
    {'desc': 'ijk->ij (sum last)', 'args': ['ijk->ij', t234]},
    {'desc': 'ijkl->ij (sum last 2)', 'args': ['ijkl->ij', t2234]},
    {'desc': 'i->i (trace-like)', 'args': ['ii->', m33]},
])

# ============================================================
# vecdot
# ============================================================
all_results['vecdot'] = test_fn('vecdot', np.linalg.vecdot, [
    {'desc': '1D', 'args': [v3, v3]},
    {'desc': '2D (batch)', 'args': [m23, m23]},
    {'desc': '3D (batch)', 'args': [t234, t234]},
    {'desc': '2D axis=0', 'args': [m23, m23], 'kwargs': {'axis': 0}},
])

# ============================================================
# cross
# ============================================================
all_results['cross'] = test_fn('cross', np.cross, [
    {'desc': '1D-3 x 1D-3', 'args': [make_array((3,)), make_array((3,))]},
    {'desc': '1D-2 x 1D-2', 'args': [make_array((2,)), make_array((2,))]},
    {'desc': '2D batched', 'args': [make_array((3,3)), make_array((3,3))]},
    {'desc': '3D batched', 'args': [make_array((2,3,3)), make_array((2,3,3))]},
])

# ============================================================
# diagonal
# ============================================================
all_results['diagonal'] = test_fn('diagonal', np.diagonal, [
    {'desc': '2D', 'args': [m33]},
    {'desc': '3D', 'args': [t234]},
    {'desc': '4D', 'args': [t2234]},
    {'desc': '2D offset=1', 'args': [m33], 'kwargs': {'offset': 1}},
])

# ============================================================
# trace
# ============================================================
all_results['trace'] = test_fn('trace', np.trace, [
    {'desc': '2D', 'args': [m33]},
    {'desc': '3D', 'args': [t234]},
    {'desc': '4D', 'args': [t2234]},
])

# ============================================================
# transpose
# ============================================================
all_results['transpose'] = test_fn('transpose', np.transpose, [
    {'desc': '1D', 'args': [v3]},
    {'desc': '2D', 'args': [m23]},
    {'desc': '3D', 'args': [t234]},
    {'desc': '4D', 'args': [t2234]},
    {'desc': '3D with axes', 'args': [t234], 'kwargs': {'axes': [2,0,1]}},
])

# ============================================================
# sum, prod, mean, std, var (axis-based reductions)
# ============================================================
for fn_name, fn in [('sum', np.sum), ('prod', np.prod), ('mean', np.mean),
                     ('std', np.std), ('var', np.var)]:
    all_results[fn_name] = test_fn(fn_name, fn, [
        {'desc': '0D', 'args': [s0]},
        {'desc': '1D', 'args': [v3]},
        {'desc': '2D', 'args': [m23]},
        {'desc': '2D axis=0', 'args': [m23], 'kwargs': {'axis': 0}},
        {'desc': '3D', 'args': [t234]},
        {'desc': '3D axis=1', 'args': [t234], 'kwargs': {'axis': 1}},
        {'desc': '4D', 'args': [t2234]},
    ])

# ============================================================
# amax, amin, argmax, argmin
# ============================================================
for fn_name, fn in [('amax', np.amax), ('amin', np.amin),
                     ('argmax', np.argmax), ('argmin', np.argmin)]:
    all_results[fn_name] = test_fn(fn_name, fn, [
        {'desc': '0D', 'args': [s0]},
        {'desc': '1D', 'args': [v3]},
        {'desc': '2D', 'args': [m23]},
        {'desc': '2D axis=0', 'args': [m23], 'kwargs': {'axis': 0}},
        {'desc': '3D', 'args': [t234]},
        {'desc': '4D', 'args': [t2234]},
    ])

# ============================================================
# cumsum, cumprod
# ============================================================
for fn_name, fn in [('cumsum', np.cumsum), ('cumprod', np.cumprod)]:
    all_results[fn_name] = test_fn(fn_name, fn, [
        {'desc': '1D', 'args': [v3]},
        {'desc': '2D', 'args': [m23]},
        {'desc': '2D axis=0', 'args': [m23], 'kwargs': {'axis': 0}},
        {'desc': '3D', 'args': [t234]},
    ])

# ============================================================
# sort, argsort
# ============================================================
for fn_name, fn in [('sort', np.sort), ('argsort', np.argsort)]:
    all_results[fn_name] = test_fn(fn_name, fn, [
        {'desc': '1D', 'args': [v3]},
        {'desc': '2D', 'args': [m23]},
        {'desc': '2D axis=0', 'args': [m23], 'kwargs': {'axis': 0}},
        {'desc': '3D', 'args': [t234]},
        {'desc': '4D', 'args': [t2234]},
    ])

# ============================================================
# reshape
# ============================================================
all_results['reshape'] = test_fn('reshape', np.reshape, [
    {'desc': '1D->2D', 'args': [make_array((6,)), (2,3)]},
    {'desc': '2D->3D', 'args': [m23, (2,1,3)]},
    {'desc': '3D->2D', 'args': [t234, (2,12)]},
    {'desc': '3D->4D', 'args': [t234, (2,3,2,2)]},
])

# ============================================================
# concatenate / stack / hstack / vstack
# ============================================================
all_results['concatenate'] = test_fn('concatenate', np.concatenate, [
    {'desc': '1D', 'args': [[v3, v3]]},
    {'desc': '2D axis=0', 'args': [[m23, m23]]},
    {'desc': '2D axis=1', 'args': [[m23, m23]], 'kwargs': {'axis': 1}},
    {'desc': '3D axis=0', 'args': [[t234, t234]]},
    {'desc': '4D axis=0', 'args': [[t2234, t2234]]},
])

all_results['stack'] = test_fn('stack', np.stack, [
    {'desc': '1D arrays', 'args': [[v3, v3, v3]]},
    {'desc': '2D arrays', 'args': [[m23, m23]]},
    {'desc': '3D arrays', 'args': [[t234, t234]]},
])

# ============================================================
# flip, rot90, roll
# ============================================================
all_results['flip'] = test_fn('flip', np.flip, [
    {'desc': '1D', 'args': [v3]},
    {'desc': '2D', 'args': [m23]},
    {'desc': '3D', 'args': [t234]},
    {'desc': '4D', 'args': [t2234]},
    {'desc': '2D axis=0', 'args': [m23], 'kwargs': {'axis': 0}},
])

all_results['rot90'] = test_fn('rot90', np.rot90, [
    {'desc': '2D', 'args': [m23]},
    {'desc': '3D', 'args': [t234]},
    {'desc': '4D', 'args': [t2234]},
])

all_results['roll'] = test_fn('roll', np.roll, [
    {'desc': '1D', 'args': [v3, 1]},
    {'desc': '2D', 'args': [m23, 1]},
    {'desc': '3D', 'args': [t234, 1]},
    {'desc': '2D axis=0', 'args': [m23, 1], 'kwargs': {'axis': 0}},
])

# ============================================================
# squeeze, expand_dims
# ============================================================
all_results['squeeze'] = test_fn('squeeze', np.squeeze, [
    {'desc': '1D no-op', 'args': [v3]},
    {'desc': '2D with size-1', 'args': [make_array((1,3))]},
    {'desc': '3D with size-1', 'args': [make_array((1,3,1))]},
    {'desc': '4D with size-1', 'args': [make_array((1,2,1,3))]},
])

all_results['expand_dims'] = test_fn('expand_dims', np.expand_dims, [
    {'desc': '1D', 'args': [v3, 0]},
    {'desc': '2D', 'args': [m23, 0]},
    {'desc': '3D', 'args': [t234, -1]},
])

# ============================================================
# swapaxes, moveaxis, rollaxis
# ============================================================
all_results['swapaxes'] = test_fn('swapaxes', np.swapaxes, [
    {'desc': '2D', 'args': [m23, 0, 1]},
    {'desc': '3D', 'args': [t234, 0, 2]},
    {'desc': '4D', 'args': [t2234, 1, 3]},
])

all_results['moveaxis'] = test_fn('moveaxis', np.moveaxis, [
    {'desc': '2D', 'args': [m23, 0, 1]},
    {'desc': '3D', 'args': [t234, 0, 2]},
    {'desc': '4D', 'args': [t2234, [0,1], [2,3]]},
])

# ============================================================
# where
# ============================================================
all_results['where'] = test_fn('where', np.where, [
    {'desc': '1D', 'args': [bool_v3, v3, v3]},
    {'desc': '2D', 'args': [bool_m23, m23, m23]},
    {'desc': '3D', 'args': [make_array((2,3,4), 'bool'), t234, t234]},
])

# ============================================================
# nonzero, argwhere
# ============================================================
all_results['nonzero'] = test_fn('nonzero', np.nonzero, [
    {'desc': '1D', 'args': [v3]},
    {'desc': '2D', 'args': [m23]},
    {'desc': '3D', 'args': [t234]},
])

all_results['argwhere'] = test_fn('argwhere', np.argwhere, [
    {'desc': '1D', 'args': [v3]},
    {'desc': '2D', 'args': [m23]},
    {'desc': '3D', 'args': [t234]},
])

# ============================================================
# broadcast_to
# ============================================================
all_results['broadcast_to'] = test_fn('broadcast_to', np.broadcast_to, [
    {'desc': '1D->2D', 'args': [v3, (2,3)]},
    {'desc': '2D->3D', 'args': [m23, (4,2,3)]},
    {'desc': '1D->3D', 'args': [make_array((3,)), (2,4,3)]},
])

# ============================================================
# take
# ============================================================
all_results['take'] = test_fn('take', np.take, [
    {'desc': '1D', 'args': [v3, np.array([0,1])]},
    {'desc': '2D axis=0', 'args': [m23, np.array([0,1])], 'kwargs': {'axis': 0}},
    {'desc': '3D axis=1', 'args': [t234, np.array([0,1])], 'kwargs': {'axis': 1}},
])

# ============================================================
# clip
# ============================================================
all_results['clip'] = test_fn('clip', np.clip, [
    {'desc': '1D', 'args': [v3, 2.0, 4.0]},
    {'desc': '2D', 'args': [m23, 2.0, 4.0]},
    {'desc': '3D', 'args': [t234, 2.0, 6.0]},
    {'desc': '4D', 'args': [t2234, 2.0, 8.0]},
])

# ============================================================
# tile, repeat
# ============================================================
all_results['tile'] = test_fn('tile', np.tile, [
    {'desc': '1D', 'args': [v3, 2]},
    {'desc': '2D', 'args': [m23, (2,3)]},
    {'desc': '3D', 'args': [t234, (2,1,1)]},
])

all_results['repeat'] = test_fn('repeat', np.repeat, [
    {'desc': '1D', 'args': [v3, 2]},
    {'desc': '2D', 'args': [m23, 2]},
    {'desc': '2D axis=0', 'args': [m23, 2], 'kwargs': {'axis': 0}},
    {'desc': '3D', 'args': [t234, 2], 'kwargs': {'axis': 1}},
])

# ============================================================
# unary math functions (elementwise - all ndims)
# ============================================================
for fn_name, fn in [
    ('sin', np.sin), ('cos', np.cos), ('tan', np.tan),
    ('arcsin', np.arcsin), ('arccos', np.arccos), ('arctan', np.arctan),
    ('sinh', np.sinh), ('cosh', np.cosh), ('tanh', np.tanh),
    ('exp', np.exp), ('log', np.log), ('sqrt', np.sqrt),
    ('abs', np.abs), ('negative', np.negative), ('sign', np.sign),
    ('floor', np.floor), ('ceil', np.ceil), ('round', np.round),
]:
    all_results[fn_name] = test_fn(fn_name, fn, [
        {'desc': '0D', 'args': [np.array(0.5)]},
        {'desc': '1D', 'args': [make_array((3,)) * 0.1]},
        {'desc': '2D', 'args': [make_array((2,3)) * 0.1]},
        {'desc': '3D', 'args': [make_array((2,3,4)) * 0.1]},
        {'desc': '4D', 'args': [make_array((2,2,3,4)) * 0.1]},
    ])

# ============================================================
# Binary math (add, subtract, multiply, divide)
# ============================================================
for fn_name, fn in [
    ('add', np.add), ('subtract', np.subtract),
    ('multiply', np.multiply), ('divide', np.divide),
    ('power', np.power), ('maximum', np.maximum), ('minimum', np.minimum),
]:
    all_results[fn_name] = test_fn(fn_name, fn, [
        {'desc': '0D', 'args': [s0, s0]},
        {'desc': '1D', 'args': [v3, v3]},
        {'desc': '2D', 'args': [m23, m23]},
        {'desc': '3D', 'args': [t234, t234]},
        {'desc': '4D', 'args': [t2234, t2234]},
        {'desc': 'broadcast 1D+2D', 'args': [v3, m23]},
        {'desc': 'broadcast 1D+3D', 'args': [make_array((4,)), t234]},
    ])

# ============================================================
# linalg.norm
# ============================================================
all_results['linalg_norm'] = test_fn('linalg.norm', np.linalg.norm, [
    {'desc': '1D vector', 'args': [v3]},
    {'desc': '2D matrix', 'args': [m23]},
    {'desc': '2D axis=0', 'args': [m23], 'kwargs': {'axis': 0}},
    {'desc': '3D axis=-1', 'args': [t234], 'kwargs': {'axis': -1}},
    {'desc': '3D axis=(0,1)', 'args': [t234], 'kwargs': {'axis': (0,1)}},
    {'desc': '4D', 'args': [t2234], 'kwargs': {'axis': -1}},
])

# ============================================================
# linalg.det, inv, solve (2D only, or batch ND)
# ============================================================
all_results['linalg_det'] = test_fn('linalg.det', np.linalg.det, [
    {'desc': '2D', 'args': [t22]},
    {'desc': '3D batch', 'args': [t222]},
    {'desc': '4D batch', 'args': [t2222]},
])

all_results['linalg_inv'] = test_fn('linalg.inv', np.linalg.inv, [
    {'desc': '2D', 'args': [t22]},
    {'desc': '3D batch', 'args': [t222]},
    {'desc': '4D batch', 'args': [t2222]},
])

all_results['linalg_solve'] = test_fn('linalg.solve', np.linalg.solve, [
    {'desc': '2D solve', 'args': [t22, make_array((2,))]},
    {'desc': '3D batch solve', 'args': [t222, make_array((2,2))]},
])

# ============================================================
# logical functions
# ============================================================
for fn_name, fn in [
    ('logical_and', np.logical_and), ('logical_or', np.logical_or),
    ('logical_not', np.logical_not),
]:
    cases = [
        {'desc': '1D', 'args': [bool_v3, bool_v3] if fn_name != 'logical_not' else [bool_v3]},
        {'desc': '2D', 'args': [bool_m23, bool_m23] if fn_name != 'logical_not' else [bool_m23]},
        {'desc': '3D', 'args': [make_array((2,3,4),'bool'), make_array((2,3,4),'bool')] if fn_name != 'logical_not' else [make_array((2,3,4),'bool')]},
    ]
    all_results[fn_name] = test_fn(fn_name, fn, cases)

# ============================================================
# all, any
# ============================================================
for fn_name, fn in [('all', np.all), ('any', np.any)]:
    all_results[fn_name] = test_fn(fn_name, fn, [
        {'desc': '1D', 'args': [bool_v3]},
        {'desc': '2D', 'args': [bool_m23]},
        {'desc': '2D axis=0', 'args': [bool_m23], 'kwargs': {'axis': 0}},
        {'desc': '3D', 'args': [make_array((2,3,4),'bool')]},
        {'desc': '4D', 'args': [make_array((2,2,3,4),'bool')]},
    ])

# ============================================================
# diff, gradient
# ============================================================
all_results['diff'] = test_fn('diff', np.diff, [
    {'desc': '1D', 'args': [v3]},
    {'desc': '2D', 'args': [m23]},
    {'desc': '2D axis=0', 'args': [m23], 'kwargs': {'axis': 0}},
    {'desc': '3D', 'args': [t234]},
])

all_results['gradient'] = test_fn('gradient', np.gradient, [
    {'desc': '1D', 'args': [v3]},
    {'desc': '2D', 'args': [m23]},
    {'desc': '3D', 'args': [t234]},
])

# ============================================================
# median, percentile, quantile
# ============================================================
for fn_name, fn in [('median', np.median), ('percentile', np.percentile), ('quantile', np.quantile)]:
    q_arg = [50] if fn_name == 'percentile' else ([0.5] if fn_name == 'quantile' else [])
    all_results[fn_name] = test_fn(fn_name, fn, [
        {'desc': '1D', 'args': [v3] + q_arg},
        {'desc': '2D', 'args': [m23] + q_arg},
        {'desc': '2D axis=0', 'args': [m23] + q_arg, 'kwargs': {'axis': 0}},
        {'desc': '3D', 'args': [t234] + q_arg},
        {'desc': '4D', 'args': [t2234] + q_arg},
    ])

# ============================================================
# average
# ============================================================
all_results['average'] = test_fn('average', np.average, [
    {'desc': '1D', 'args': [v3]},
    {'desc': '2D', 'args': [m23]},
    {'desc': '2D axis=0', 'args': [m23], 'kwargs': {'axis': 0}},
    {'desc': '3D', 'args': [t234]},
])

# ============================================================
# unique
# ============================================================
all_results['unique'] = test_fn('unique', np.unique, [
    {'desc': '1D', 'args': [np.array([3,1,2,1,3])]},
    {'desc': '2D', 'args': [np.array([[3,1],[2,1]])]},
])

# ============================================================
# FFT functions (1D axis, can take ND)
# ============================================================
all_results['fft'] = test_fn('fft.fft', np.fft.fft, [
    {'desc': '1D', 'args': [v3]},
    {'desc': '2D', 'args': [m23]},
    {'desc': '3D', 'args': [t234]},
])

all_results['fft2'] = test_fn('fft.fft2', np.fft.fft2, [
    {'desc': '2D', 'args': [m23]},
    {'desc': '3D', 'args': [t234]},
    {'desc': '4D', 'args': [t2234]},
])

all_results['fftn'] = test_fn('fft.fftn', np.fft.fftn, [
    {'desc': '1D', 'args': [v3]},
    {'desc': '2D', 'args': [m23]},
    {'desc': '3D', 'args': [t234]},
    {'desc': '4D', 'args': [t2234]},
])

# ============================================================
# max ndim test
# ============================================================
print("# Testing NumPy maximum ndim")
for n in [32, 33, 48, 64]:
    try:
        shape = [2] * n
        a = np.zeros(shape)
        print(f"ndim={n}: shape {a.shape[:3]}... OK")
    except Exception as e:
        print(f"ndim={n}: FAIL - {e}")

print()
print("=" * 60)
print("NDIM REQUIREMENTS REPORT")
print("=" * 60)

for fn_name, results in all_results.items():
    print(f"\n## {fn_name}")
    ok_cases = [r for r in results if r['status'] == 'ok']
    error_cases = [r for r in results if r['status'] == 'error']
    if ok_cases:
        print(f"  OK cases:")
        for r in ok_cases:
            in_ndims = [str(n) for n in r.get('input_ndims', []) if n is not None]
            out = r.get('output_ndim', r.get('output_shape', '?'))
            print(f"    [{','.join(in_ndims)}] -> ndim={out} : {r['desc']}")
    if error_cases:
        print(f"  ERRORS:")
        for r in error_cases:
            in_ndims = [str(n) for n in r.get('input_ndims', []) if n is not None]
            print(f"    [{','.join(in_ndims)}] ERROR: {r['error'][:80]} : {r['desc']}")
