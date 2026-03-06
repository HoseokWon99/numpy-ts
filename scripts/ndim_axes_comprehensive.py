#!/usr/bin/env python3
"""
Comprehensive NDim + Axes discovery for all implemented numpy-ts functions.

Tests every function category with:
- All meaningful input dimensionalities (0D through 5D)
- All valid axis values (positive, negative, multi-axis)
- keepdims parameter where applicable
- Error cases (invalid ndim, invalid axis)

Run with: python scripts/ndim_axes_comprehensive.py
"""
import numpy as np
import sys

def h(shape, dtype='float64', scale=1.0):
    """Create ascending array of given shape."""
    if not shape:
        return np.array(0.5, dtype=dtype)
    n = int(np.prod(shape))
    if dtype == 'bool':
        return np.array([i % 2 == 0 for i in range(n)], dtype=bool).reshape(shape)
    if dtype in ('int32', 'int64'):
        return np.arange(1, n + 1, dtype=dtype).reshape(shape)
    return (np.arange(1.0, n + 1.0, dtype=dtype) * scale).reshape(shape)

PASS = '\033[32mOK\033[0m'
FAIL = '\033[31mFAIL\033[0m'

results = {}

def check(fn_name, desc, fn, *args, **kwargs):
    try:
        out = fn(*args, **kwargs)
        shape = out.shape if isinstance(out, np.ndarray) else ()
        results.setdefault(fn_name, []).append({'desc': desc, 'status': 'ok', 'out_shape': list(shape)})
        return out
    except Exception as e:
        results.setdefault(fn_name, []).append({'desc': desc, 'status': 'error', 'error': str(e)[:120]})
        return None

def expect_error(fn_name, desc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        results.setdefault(fn_name, []).append({'desc': desc, 'status': 'unexpected_ok'})
    except Exception as e:
        results.setdefault(fn_name, []).append({'desc': desc, 'status': 'expected_error', 'error': str(e)[:80]})

# ============================================================
# 1. ELEMENTWISE UNARY: 0D through 5D
# ============================================================
print("Testing elementwise unary functions...")
UNARY_FNS = [
    ('sin', np.sin, {}), ('cos', np.cos, {}), ('tan', np.tan, {}),
    ('arcsin', np.arcsin, {}), ('arccos', np.arccos, {}), ('arctan', np.arctan, {}),
    ('sinh', np.sinh, {}), ('cosh', np.cosh, {}), ('tanh', np.tanh, {}),
    ('exp', np.exp, {}), ('exp2', np.exp2, {}), ('expm1', np.expm1, {}),
    ('log', np.log, {}), ('log2', np.log2, {}), ('log10', np.log10, {}), ('log1p', np.log1p, {}),
    ('sqrt', np.sqrt, {}), ('cbrt', np.cbrt, {}), ('square', np.square, {}),
    ('abs', np.abs, {}), ('negative', np.negative, {}), ('sign', np.sign, {}),
    ('floor', np.floor, {}), ('ceil', np.ceil, {}), ('trunc', np.trunc, {}), ('rint', np.rint, {}),
    ('isfinite', np.isfinite, {}), ('isinf', np.isinf, {}), ('isnan', np.isnan, {}),
]
SHAPES = [(), (3,), (2, 3), (2, 3, 4), (2, 2, 3, 4), (2, 2, 2, 3, 4)]
for fn_name, fn, kw in UNARY_FNS:
    for shape in SHAPES:
        a = h(shape, scale=0.1) if fn_name in ('arcsin', 'arccos') else h(shape, scale=0.5)
        check(fn_name, f'{shape}', fn, a, **kw)

# ============================================================
# 2. ELEMENTWISE BINARY: same-shape 0D-5D + broadcasting
# ============================================================
print("Testing elementwise binary functions...")
BINARY_FNS = [
    ('add', np.add), ('subtract', np.subtract),
    ('multiply', np.multiply), ('divide', np.divide),
    ('power', np.power), ('maximum', np.maximum), ('minimum', np.minimum),
    ('mod', np.mod), ('floor_divide', np.floor_divide),
    ('greater', np.greater), ('less', np.less), ('equal', np.equal),
]
for fn_name, fn in BINARY_FNS:
    for shape in [(), (3,), (2, 3), (2, 3, 4), (2, 2, 3, 4), (2, 2, 2, 3, 4)]:
        a = h(shape, scale=0.5); b = h(shape, scale=0.3) + 0.1
        check(fn_name, f'same {shape}', fn, a, b)
    # Broadcasting cases
    a1 = h((4,)); b2 = h((2, 4)); b3 = h((2, 3, 4)); b4 = h((2, 2, 3, 4))
    check(fn_name, 'broadcast 1D+2D', fn, a1, b2)
    check(fn_name, 'broadcast 1D+3D', fn, a1, b3)
    check(fn_name, 'broadcast 1D+4D', fn, a1, b4)
    check(fn_name, 'broadcast 2D+3D', fn, h((3, 4)), b3)
    check(fn_name, 'broadcast scalar+3D', fn, np.array(2.0), b3)

# ============================================================
# 3. REDUCTIONS: sum, prod, mean, std, var
#    ALL axes, keepdims, multi-axis
# ============================================================
print("Testing reduction functions (all axes, keepdims, multi-axis)...")
REDUCTION_FNS = [
    ('sum', np.sum), ('prod', np.prod), ('mean', np.mean),
    ('std', np.std), ('var', np.var),
]
# 0D
for fn_name, fn in REDUCTION_FNS:
    check(fn_name, '0D no-axis', fn, h(()))
# 1D
for fn_name, fn in REDUCTION_FNS:
    a = h((6,))
    check(fn_name, '1D axis=None', fn, a)
    check(fn_name, '1D axis=0', fn, a, axis=0)
    check(fn_name, '1D axis=-1', fn, a, axis=-1)
    check(fn_name, '1D axis=0 keepdims', fn, a, axis=0, keepdims=True)
# 2D
for fn_name, fn in REDUCTION_FNS:
    a = h((3, 4))
    for ax in [0, 1, -1, -2]:
        check(fn_name, f'2D axis={ax}', fn, a, axis=ax)
        check(fn_name, f'2D axis={ax} keepdims', fn, a, axis=ax, keepdims=True)
    check(fn_name, '2D axis=(0,1)', fn, a, axis=(0, 1))
    check(fn_name, '2D axis=(0,1) keepdims', fn, a, axis=(0, 1), keepdims=True)
# 3D
for fn_name, fn in REDUCTION_FNS:
    a = h((2, 3, 4))
    for ax in [0, 1, 2, -1, -2, -3]:
        check(fn_name, f'3D axis={ax}', fn, a, axis=ax)
        check(fn_name, f'3D axis={ax} keepdims', fn, a, axis=ax, keepdims=True)
    check(fn_name, '3D axis=(0,1)', fn, a, axis=(0, 1))
    check(fn_name, '3D axis=(1,2)', fn, a, axis=(1, 2))
    check(fn_name, '3D axis=(0,1,2)', fn, a, axis=(0, 1, 2))
    check(fn_name, '3D axis=(0,2) keepdims', fn, a, axis=(0, 2), keepdims=True)
# 4D
for fn_name, fn in REDUCTION_FNS:
    a = h((2, 3, 4, 5))
    for ax in [0, 1, 2, 3, -1, -4]:
        check(fn_name, f'4D axis={ax}', fn, a, axis=ax)
    check(fn_name, '4D axis=(0,1,2,3)', fn, a, axis=(0, 1, 2, 3))
    check(fn_name, '4D axis=(1,3) keepdims', fn, a, axis=(1, 3), keepdims=True)

# ============================================================
# 4. NAN VARIANTS: nansum, nanmean, nanstd, nanvar, nanprod
# ============================================================
print("Testing nan reduction functions...")
NAN_FNS = [
    ('nansum', np.nansum), ('nanprod', np.nanprod), ('nanmean', np.nanmean),
    ('nanstd', np.nanstd), ('nanvar', np.nanvar),
]
for fn_name, fn in NAN_FNS:
    a2 = h((3, 4))
    a2_nan = a2.copy(); a2_nan[0, 0] = np.nan
    a3 = h((2, 3, 4))
    a3_nan = a3.copy(); a3_nan[0, :, 0] = np.nan
    check(fn_name, '2D no-axis with NaN', fn, a2_nan)
    for ax in [0, 1, -1]:
        check(fn_name, f'2D axis={ax} with NaN', fn, a2_nan, axis=ax)
        check(fn_name, f'2D axis={ax} keepdims with NaN', fn, a2_nan, axis=ax, keepdims=True)
    for ax in [0, 1, 2, -1]:
        check(fn_name, f'3D axis={ax} with NaN', fn, a3_nan, axis=ax)
    check(fn_name, '3D axis=(0,1) with NaN', fn, a3_nan, axis=(0, 1))

# nanmin, nanmax, nanargmin, nanargmax
for fn_name, fn in [('nanmin', np.nanmin), ('nanmax', np.nanmax),
                     ('nanargmin', np.nanargmin), ('nanargmax', np.nanargmax)]:
    a = h((3, 4)); a_nan = a.copy(); a_nan[0, 0] = np.nan
    check(fn_name, '2D no-axis', fn, a_nan)
    for ax in [0, 1, -1]:
        check(fn_name, f'2D axis={ax}', fn, a_nan, axis=ax)
    a3 = h((2, 3, 4)); a3[0, 0, 0] = np.nan
    for ax in [0, 1, 2]:
        check(fn_name, f'3D axis={ax}', fn, a3, axis=ax)

# ============================================================
# 5. AMAX/AMIN/ARGMAX/ARGMIN: all axes
# ============================================================
print("Testing amax/amin/argmax/argmin...")
for fn_name, fn in [('amax', np.amax), ('amin', np.amin),
                     ('argmax', np.argmax), ('argmin', np.argmin)]:
    check(fn_name, '0D', fn, h(()))
    for shape in [(4,), (3, 4), (2, 3, 4), (2, 2, 3, 4)]:
        ndim = len(shape)
        check(fn_name, f'{shape} no-axis', fn, h(shape))
        for ax in range(ndim):
            check(fn_name, f'{shape} axis={ax}', fn, h(shape), axis=ax)
            check(fn_name, f'{shape} axis={-ax-1}', fn, h(shape), axis=-(ax + 1))
        if fn_name in ('amax', 'amin'):
            check(fn_name, f'{shape} keepdims', fn, h(shape), axis=0, keepdims=True)

# ============================================================
# 6. ALL / ANY: all axes, keepdims
# ============================================================
print("Testing all/any...")
for fn_name, fn in [('all', np.all), ('any', np.any)]:
    check(fn_name, '0D', fn, np.array(True))
    for shape in [(4,), (3, 4), (2, 3, 4), (2, 2, 3, 4)]:
        a = h(shape, 'bool')
        ndim = len(shape)
        check(fn_name, f'{shape} no-axis', fn, a)
        for ax in range(ndim):
            check(fn_name, f'{shape} axis={ax}', fn, a, axis=ax)
            check(fn_name, f'{shape} axis={-ax-1}', fn, a, axis=-(ax + 1))
            check(fn_name, f'{shape} axis={ax} keepdims', fn, a, axis=ax, keepdims=True)
        if ndim >= 2:
            check(fn_name, f'{shape} axis=(0,1)', fn, a, axis=(0, 1))

# ============================================================
# 7. LOGICAL FUNCTIONS
# ============================================================
print("Testing logical functions...")
for fn_name, fn in [('logical_and', np.logical_and), ('logical_or', np.logical_or),
                     ('logical_xor', np.logical_xor)]:
    for shape in [(4,), (3, 4), (2, 3, 4), (2, 2, 3, 4)]:
        a = h(shape, 'bool'); b = h(shape, 'bool')
        check(fn_name, f'{shape}', fn, a, b)

for shape in [(4,), (3, 4), (2, 3, 4), (2, 2, 3, 4)]:
    check('logical_not', f'{shape}', np.logical_not, h(shape, 'bool'))

# ============================================================
# 8. CUMULATIVE: cumsum, cumprod — all axes
# ============================================================
print("Testing cumsum/cumprod (all axes)...")
for fn_name, fn in [('cumsum', np.cumsum), ('cumprod', np.cumprod)]:
    a1 = h((5,))
    check(fn_name, '1D no-axis', fn, a1)
    check(fn_name, '1D axis=0', fn, a1, axis=0)
    for shape in [(3, 4), (2, 3, 4), (2, 2, 3, 4)]:
        a = h(shape)
        check(fn_name, f'{shape} no-axis (flattened)', fn, a)
        for ax in range(len(shape)):
            check(fn_name, f'{shape} axis={ax}', fn, a, axis=ax)
            check(fn_name, f'{shape} axis={-ax-1}', fn, a, axis=-(ax + 1))

# ============================================================
# 9. SORT / ARGSORT — all axes
# ============================================================
print("Testing sort/argsort (all axes)...")
for fn_name, fn in [('sort', np.sort), ('argsort', np.argsort)]:
    for shape in [(5,), (3, 4), (2, 3, 4), (2, 2, 3, 4)]:
        ndim = len(shape)
        for ax in range(ndim):
            check(fn_name, f'{shape} axis={ax}', fn, h(shape), axis=ax)
            check(fn_name, f'{shape} axis={-ax-1}', fn, h(shape), axis=-(ax + 1))
    # Stable sort
    check('sort', '2D stable', np.sort, h((3, 4)), axis=0, kind='stable')

# ============================================================
# 10. DIFF — all axes
# ============================================================
print("Testing diff (all axes)...")
for shape in [(5,), (3, 4), (2, 3, 4), (2, 2, 3, 4)]:
    ndim = len(shape)
    for ax in range(ndim):
        check('diff', f'{shape} axis={ax}', np.diff, h(shape), axis=ax)
        check('diff', f'{shape} n=2 axis={ax}', np.diff, h(shape), n=2, axis=ax)

# ============================================================
# 11. MEDIAN / PERCENTILE / QUANTILE — all axes
# ============================================================
print("Testing median/percentile/quantile (all axes)...")
for shape in [(5,), (3, 4), (2, 3, 4), (2, 2, 3, 4)]:
    a = h(shape)
    ndim = len(shape)
    check('median', f'{shape} no-axis', np.median, a)
    for ax in range(ndim):
        check('median', f'{shape} axis={ax}', np.median, a, axis=ax)
        check('median', f'{shape} axis={-ax-1}', np.median, a, axis=-(ax + 1))
        check('median', f'{shape} axis={ax} keepdims', np.median, a, axis=ax, keepdims=True)
    check('percentile', f'{shape}', np.percentile, a, 75)
    for ax in range(ndim):
        check('percentile', f'{shape} axis={ax}', np.percentile, a, 75, axis=ax)
    check('quantile', f'{shape}', np.quantile, a, 0.75)
    for ax in range(ndim):
        check('quantile', f'{shape} axis={ax}', np.quantile, a, 0.75, axis=ax)

# nanmedian
for shape in [(5,), (3, 4), (2, 3, 4)]:
    a = h(shape); a.flat[0] = np.nan
    check('nanmedian', f'{shape} no-axis', np.nanmedian, a)
    for ax in range(len(shape)):
        check('nanmedian', f'{shape} axis={ax}', np.nanmedian, a, axis=ax)

# ============================================================
# 12. AVERAGE — all axes
# ============================================================
print("Testing average (all axes)...")
for shape in [(5,), (3, 4), (2, 3, 4), (2, 2, 3, 4)]:
    a = h(shape)
    check('average', f'{shape} no-axis', np.average, a)
    for ax in range(len(shape)):
        check('average', f'{shape} axis={ax}', np.average, a, axis=ax)

# ============================================================
# 13. FLIP — axis and no-axis
# ============================================================
print("Testing flip (all axes)...")
for shape in [(5,), (3, 4), (2, 3, 4), (2, 2, 3, 4)]:
    a = h(shape)
    check('flip', f'{shape} no-axis', np.flip, a)
    ndim = len(shape)
    for ax in range(ndim):
        check('flip', f'{shape} axis={ax}', np.flip, a, axis=ax)
        check('flip', f'{shape} axis={-ax-1}', np.flip, a, axis=-(ax + 1))

# ============================================================
# 14. ROLL — with and without axis
# ============================================================
print("Testing roll (all axes)...")
for shape in [(5,), (3, 4), (2, 3, 4), (2, 2, 3, 4)]:
    a = h(shape)
    check('roll', f'{shape} no-axis', np.roll, a, 2)
    for ax in range(len(shape)):
        check('roll', f'{shape} axis={ax}', np.roll, a, 2, axis=ax)
        check('roll', f'{shape} axis={-ax-1}', np.roll, a, -1, axis=-(ax + 1))

# ============================================================
# 15. CONCATENATE — all axes
# ============================================================
print("Testing concatenate (all axes)...")
for shape in [(3,), (3, 4), (2, 3, 4), (2, 2, 3, 4)]:
    a = h(shape)
    for ax in range(len(shape)):
        check('concatenate', f'{shape} axis={ax}', np.concatenate, [a, a], axis=ax)

# ============================================================
# 16. STACK — all axes
# ============================================================
print("Testing stack (all axes)...")
for shape in [(3,), (2, 3), (2, 3, 4), (2, 2, 3, 4)]:
    a = h(shape)
    ndim = len(shape)
    for ax in range(ndim + 1):  # stack creates new axis
        check('stack', f'{shape} axis={ax}', np.stack, [a, a], axis=ax)

# ============================================================
# 17. SPLIT / ARRAY_SPLIT — all axes
# ============================================================
print("Testing split/array_split (all axes)...")
for fn_name, fn in [('split', np.split), ('array_split', np.array_split)]:
    for shape, n in [((6,), 3), ((4, 6), 2), ((2, 3, 6), 2)]:
        a = h(shape)
        for ax in range(len(shape)):
            check(fn_name, f'{shape} axis={ax}', fn, a, n, axis=ax)

# ============================================================
# 18. SWAPAXES — all pairs
# ============================================================
print("Testing swapaxes...")
for shape in [(3, 4), (2, 3, 4), (2, 3, 4, 5)]:
    a = h(shape)
    ndim = len(shape)
    for i in range(ndim):
        for j in range(ndim):
            if i != j:
                check('swapaxes', f'{shape} ({i},{j})', np.swapaxes, a, i, j)

# ============================================================
# 19. MOVEAXIS — single and multiple
# ============================================================
print("Testing moveaxis...")
for shape in [(2, 3, 4), (2, 3, 4, 5)]:
    a = h(shape)
    ndim = len(shape)
    for src in range(ndim):
        for dst in range(ndim):
            check('moveaxis', f'{shape} {src}->{dst}', np.moveaxis, a, src, dst)
    # Multi-axis
    check('moveaxis', f'{shape} [0,1]->[2,3]', np.moveaxis, a, [0, 1], [ndim-2, ndim-1])

# ============================================================
# 20. EXPAND_DIMS — all positions
# ============================================================
print("Testing expand_dims...")
for shape in [(), (3,), (2, 3), (2, 3, 4)]:
    a = h(shape)
    ndim = len(shape)
    for ax in range(ndim + 1):
        check('expand_dims', f'{shape} axis={ax}', np.expand_dims, a, axis=ax)
    check('expand_dims', f'{shape} axis=-1', np.expand_dims, a, axis=-1)

# ============================================================
# 21. SQUEEZE
# ============================================================
print("Testing squeeze...")
for shape in [(1, 3), (1, 3, 1), (2, 1, 3, 1, 4)]:
    a = h(shape)
    check('squeeze', f'{shape} all', np.squeeze, a)
    for ax, s in enumerate(shape):
        if s == 1:
            check('squeeze', f'{shape} axis={ax}', np.squeeze, a, axis=ax)

# ============================================================
# 22. REPEAT — all axes
# ============================================================
print("Testing repeat...")
for shape in [(3,), (2, 3), (2, 3, 4)]:
    a = h(shape)
    check('repeat', f'{shape} no-axis', np.repeat, a, 2)
    for ax in range(len(shape)):
        check('repeat', f'{shape} axis={ax}', np.repeat, a, 2, axis=ax)

# ============================================================
# 23. TILE
# ============================================================
print("Testing tile...")
for shape in [(3,), (2, 3), (2, 3, 4)]:
    a = h(shape)
    check('tile', f'{shape} scalar reps', np.tile, a, 2)
    check('tile', f'{shape} ndim reps', np.tile, a, [2] * len(shape))
    check('tile', f'{shape} extra dim', np.tile, a, [2] * (len(shape) + 1))

# ============================================================
# 24. RESHAPE — ND -> MD
# ============================================================
print("Testing reshape...")
check('reshape', '1D->2D', np.reshape, h((6,)), (2, 3))
check('reshape', '1D->3D', np.reshape, h((24,)), (2, 3, 4))
check('reshape', '2D->1D', np.reshape, h((2, 3)), (6,))
check('reshape', '2D->3D', np.reshape, h((2, 3)), (2, 1, 3))
check('reshape', '3D->2D', np.reshape, h((2, 3, 4)), (2, 12))
check('reshape', '3D->4D', np.reshape, h((2, 3, 4)), (2, 3, 2, 2))
check('reshape', '4D->2D', np.reshape, h((2, 2, 3, 4)), (4, 12))
check('reshape', '4D->5D', np.reshape, h((2, 2, 3, 4)), (2, 2, 3, 2, 2))
check('reshape', '-1 inference', np.reshape, h((2, 3, 4)), (-1,))
check('reshape', '-1 in 2D', np.reshape, h((2, 3, 4)), (-1, 4))

# ============================================================
# 25. TRANSPOSE — all permutations
# ============================================================
print("Testing transpose...")
for shape in [(3,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]:
    a = h(shape)
    check('transpose', f'{shape} default', np.transpose, a)
    ndim = len(shape)
    if ndim >= 2:
        # Try a few permutations
        import itertools
        for perm in list(itertools.permutations(range(ndim)))[:6]:
            check('transpose', f'{shape} perm={perm}', np.transpose, a, perm)

# ============================================================
# 26. ROT90
# ============================================================
print("Testing rot90...")
for shape in [(3, 4), (2, 3, 4), (2, 3, 4, 5)]:
    a = h(shape)
    ndim = len(shape)
    for k in [1, 2, 3]:
        check('rot90', f'{shape} k={k}', np.rot90, a, k=k)
        if ndim >= 3:
            for ax_pair in [(0, 1), (0, 2), (1, 2)]:
                check('rot90', f'{shape} k={k} axes={ax_pair}', np.rot90, a, k=k, axes=ax_pair)

# ============================================================
# 27. BROADCAST_TO — ND
# ============================================================
print("Testing broadcast_to...")
check('broadcast_to', '1D->2D', np.broadcast_to, h((3,)), (4, 3))
check('broadcast_to', '1D->3D', np.broadcast_to, h((3,)), (2, 4, 3))
check('broadcast_to', '1D->4D', np.broadcast_to, h((3,)), (2, 2, 4, 3))
check('broadcast_to', '2D->3D', np.broadcast_to, h((1, 3)), (4, 2, 3))
check('broadcast_to', '2D->4D', np.broadcast_to, h((1, 3)), (2, 4, 2, 3))
check('broadcast_to', '3D->5D', np.broadcast_to, h((1, 1, 3)), (2, 3, 4, 2, 3))

# ============================================================
# 28. TAKE — all axes
# ============================================================
print("Testing take...")
for shape in [(5,), (3, 4), (2, 3, 4)]:
    a = h(shape)
    idx = np.array([0, 1])
    check('take', f'{shape} no-axis', np.take, a, idx)
    for ax in range(len(shape)):
        check('take', f'{shape} axis={ax}', np.take, a, idx, axis=ax)

# ============================================================
# 29. CLIP — ND
# ============================================================
print("Testing clip...")
for shape in [(), (5,), (3, 4), (2, 3, 4), (2, 2, 3, 4)]:
    a = h(shape)
    check('clip', f'{shape}', np.clip, a, 2.0, 10.0)

# ============================================================
# 30. WHERE — broadcasting
# ============================================================
print("Testing where...")
for shape in [(5,), (3, 4), (2, 3, 4), (2, 2, 3, 4)]:
    a = h(shape); b = -a; c = a > 5
    check('where', f'{shape}', np.where, c, a, b)
# Broadcasting
check('where', 'broadcast cond+values', np.where, h((4,), 'bool'), h((3, 4)), h((3, 4)))

# ============================================================
# 31. NONZERO / ARGWHERE — ND
# ============================================================
print("Testing nonzero/argwhere...")
for fn_name, fn in [('nonzero', np.nonzero), ('argwhere', np.argwhere)]:
    for shape in [(5,), (3, 4), (2, 3, 4)]:
        check(fn_name, f'{shape}', fn, h(shape, 'bool'))

# ============================================================
# 32. MATMUL — all valid combinations
# ============================================================
print("Testing matmul (all combinations)...")
check('matmul', '1D@1D->scalar', np.matmul, h((4,)), h((4,)))
check('matmul', '2D@1D->1D', np.matmul, h((3, 4)), h((4,)))
check('matmul', '1D@2D->1D', np.matmul, h((3,)), h((3, 4)))
check('matmul', '2D@2D->2D', np.matmul, h((2, 3)), h((3, 4)))
check('matmul', '3D@3D->3D (batch)', np.matmul, h((2, 3, 4)), h((2, 4, 5)))
check('matmul', '4D@4D->4D (batch)', np.matmul, h((2, 2, 3, 4)), h((2, 2, 4, 5)))
check('matmul', '5D@5D->5D (batch)', np.matmul, h((2, 2, 2, 3, 4)), h((2, 2, 2, 4, 5)))
check('matmul', '3D@2D->3D (broadcast)', np.matmul, h((2, 3, 4)), h((4, 5)))
check('matmul', '4D@3D (broadcast batch)', np.matmul, h((2, 2, 3, 4)), h((2, 4, 5)))
check('matmul', '1D@3D', np.matmul, h((3,)), h((2, 3, 4)))
expect_error('matmul', '0D should fail', np.matmul, np.array(2.0), np.array(2.0))

# ============================================================
# 33. DOT — ND cases
# ============================================================
print("Testing dot (ND)...")
check('dot', '0D', np.dot, h(()), h(()))
check('dot', '1D@1D', np.dot, h((3,)), h((3,)))
check('dot', '2D@1D', np.dot, h((3, 4)), h((4,)))
check('dot', '1D@2D', np.dot, h((3,)), h((3, 4)))
check('dot', '2D@2D', np.dot, h((2, 3)), h((3, 4)))
check('dot', '3D@2D', np.dot, h((2, 3, 4)), h((4, 5)))
check('dot', '4D@2D', np.dot, h((2, 2, 3, 4)), h((4, 5)))
check('dot', '3D@3D', np.dot, h((2, 3, 4)), h((5, 4, 6)))  # sum-product rule

# ============================================================
# 34. INNER / OUTER — ND
# ============================================================
print("Testing inner/outer...")
check('inner', '0D', np.inner, h(()), h(()))
check('inner', '1D@1D', np.inner, h((4,)), h((4,)))
check('inner', '2D@1D', np.inner, h((3, 4)), h((4,)))
check('inner', '2D@2D', np.inner, h((3, 4)), h((2, 4)))
check('inner', '3D@1D', np.inner, h((2, 3, 4)), h((4,)))
check('inner', '3D@2D', np.inner, h((2, 3, 4)), h((2, 4)))

check('outer', '1D@1D', np.outer, h((3,)), h((4,)))
check('outer', '2D@2D (flattened)', np.outer, h((2, 3)), h((3, 4)))
check('outer', '3D@3D (flattened)', np.outer, h((2, 3, 4)), h((2, 3, 4)))

# ============================================================
# 35. TENSORDOT — axes combos
# ============================================================
print("Testing tensordot...")
check('tensordot', '1D axes=1', np.tensordot, h((4,)), h((4,)), axes=1)
check('tensordot', '2D axes=1', np.tensordot, h((2, 3)), h((3, 4)), axes=1)
check('tensordot', '2D axes=2', np.tensordot, h((2, 3)), h((2, 3)), axes=2)
check('tensordot', '3D axes=1', np.tensordot, h((2, 3, 4)), h((4, 5)), axes=1)
check('tensordot', '3D axes=0 (outer)', np.tensordot, h((3,)), h((4,)), axes=0)
check('tensordot', '3D explicit axes', np.tensordot, h((2, 3, 4)), h((2, 4)), axes=[[1, 2], [1, 0]])

# ============================================================
# 36. TRACE — ND
# ============================================================
print("Testing trace (ND)...")
check('trace', '2D', np.trace, h((3, 3)))
check('trace', '2D non-square', np.trace, h((3, 4)))
check('trace', '2D offset=1', np.trace, h((3, 3)), offset=1)
check('trace', '2D offset=-1', np.trace, h((3, 3)), offset=-1)
check('trace', '3D (batch)', np.trace, h((2, 3, 3)))
check('trace', '4D (batch)', np.trace, h((2, 2, 3, 3)))
check('trace', '5D (batch)', np.trace, h((2, 2, 2, 3, 3)))
# Custom axis1/axis2
check('trace', '3D axis1=0,axis2=2', np.trace, h((3, 2, 3)), axis1=0, axis2=2)
check('trace', '4D axis1=1,axis2=3', np.trace, h((2, 3, 2, 3)), axis1=1, axis2=3)

# ============================================================
# 37. DIAGONAL — ND, all axis pairs
# ============================================================
print("Testing diagonal (ND)...")
check('diagonal', '2D', np.diagonal, h((3, 4)))
check('diagonal', '3D default', np.diagonal, h((2, 3, 4)))
check('diagonal', '4D default', np.diagonal, h((2, 3, 4, 4)))
check('diagonal', '3D axis1=0,axis2=2', np.diagonal, h((3, 2, 3)), axis1=0, axis2=2)
check('diagonal', '4D axis1=1,axis2=3', np.diagonal, h((2, 3, 4, 4)), axis1=1, axis2=3)
check('diagonal', '2D offset=1', np.diagonal, h((4, 4)), offset=1)
check('diagonal', '2D offset=-1', np.diagonal, h((4, 4)), offset=-1)

# ============================================================
# 38. CROSS — batched ND
# ============================================================
print("Testing cross...")
check('cross', '1D-3 x 1D-3', np.cross, h((3,)), h((3,)))
check('cross', '1D-2 x 1D-2', np.cross, h((2,)), h((2,)))
check('cross', '2D batched', np.cross, h((4, 3)), h((4, 3)))
check('cross', '3D batched', np.cross, h((2, 4, 3)), h((2, 4, 3)))
check('cross', '4D batched', np.cross, h((2, 2, 4, 3)), h((2, 2, 4, 3)))

# ============================================================
# 39. EINSUM — various subscripts
# ============================================================
print("Testing einsum...")
check('einsum', '1D dot', np.einsum, 'i,i->', h((4,)), h((4,)))
check('einsum', '2D matmul', np.einsum, 'ij,jk->ik', h((2, 3)), h((3, 4)))
check('einsum', '2D trace', np.einsum, 'ii->', h((3, 3)))
check('einsum', '3D batch matmul', np.einsum, 'bij,bjk->bik', h((2, 3, 4)), h((2, 4, 5)))
check('einsum', '3D sum last axis', np.einsum, 'ijk->ij', h((2, 3, 4)))
check('einsum', '4D sum', np.einsum, 'ijkl->ij', h((2, 3, 4, 5)))
check('einsum', 'outer product', np.einsum, 'i,j->ij', h((3,)), h((4,)))

# ============================================================
# 40. LINALG BATCH OPS: det, inv, solve — 2D through 5D
# ============================================================
print("Testing linalg batch ops...")
for ndim_batch in [0, 1, 2, 3]:
    batch = (2,) * ndim_batch
    A = h(batch + (3, 3)) + np.eye(3) * 5  # Well-conditioned
    check('linalg_det', f'{A.shape}', np.linalg.det, A)
    check('linalg_inv', f'{A.shape}', np.linalg.inv, A)

# solve: batch
for ndim_batch in [0, 1, 2]:
    batch = (2,) * ndim_batch
    A = h(batch + (3, 3)) + np.eye(3) * 5
    b = h(batch + (3,))
    check('linalg_solve', f'A={A.shape} b={b.shape}', np.linalg.solve, A, b)
    b2 = h(batch + (3, 2))
    check('linalg_solve', f'A={A.shape} b={b2.shape}', np.linalg.solve, A, b2)

# norm — all axes
for shape in [(4,), (3, 4), (2, 3, 4)]:
    a = h(shape)
    check('linalg_norm', f'{shape} no-axis', np.linalg.norm, a)
    for ax in range(len(shape)):
        check('linalg_norm', f'{shape} axis={ax}', np.linalg.norm, a, axis=ax)
    if len(shape) >= 2:
        check('linalg_norm', f'{shape} fro', np.linalg.norm, a, ord='fro')
        check('linalg_norm', f'{shape} axis tuple', np.linalg.norm, a, axis=(0, 1))

# svd batch
for ndim_batch in [0, 1, 2]:
    batch = (2,) * ndim_batch
    A = h(batch + (3, 4))
    check('linalg_svd', f'{A.shape} full', np.linalg.svd, A, full_matrices=True)
    check('linalg_svd', f'{A.shape} reduced', np.linalg.svd, A, full_matrices=False)

# qr batch
for ndim_batch in [0, 1]:
    batch = (2,) * ndim_batch
    A = h(batch + (4, 3))
    check('linalg_qr', f'{A.shape}', np.linalg.qr, A)

# eig/eigh
check('linalg_eig', '2D', np.linalg.eig, h((3, 3)) + np.eye(3) * 3)
check('linalg_eigh', '2D', np.linalg.eigh, h((3, 3)) + np.eye(3) * 3)
# batch eig (numpy 2.0+)
try:
    check('linalg_eig', '3D batch', np.linalg.eig, h((2, 3, 3)) + np.eye(3) * 3)
except Exception as e:
    pass

# ============================================================
# 41. FFT — with explicit axis parameter
# ============================================================
print("Testing fft with axis param...")
for shape in [(8,), (4, 8), (2, 4, 8), (2, 2, 4, 8)]:
    a = h(shape)
    ndim = len(shape)
    # fft along each axis
    for ax in range(ndim):
        check('fft', f'{shape} axis={ax}', np.fft.fft, a, axis=ax)
    # ifft
    check('fft', f'{shape} ifft axis=-1', np.fft.ifft, a, axis=-1)

# fft2
for shape in [(4, 8), (2, 4, 8), (2, 2, 4, 8)]:
    a = h(shape)
    ndim = len(shape)
    for i in range(ndim):
        for j in range(ndim):
            if i != j:
                check('fft2', f'{shape} axes=({i},{j})', np.fft.fft2, a, axes=(i, j))

# fftn
for shape in [(4, 8), (2, 4, 8)]:
    a = h(shape)
    check('fftn', f'{shape} all axes', np.fft.fftn, a)
    check('fftn', f'{shape} last 2 axes', np.fft.fftn, a, axes=list(range(-2, 0)))

# fftshift/ifftshift
for shape in [(8,), (4, 8), (2, 4, 8)]:
    a = h(shape)
    check('fftshift', f'{shape}', np.fft.fftshift, a)
    check('ifftshift', f'{shape}', np.fft.ifftshift, a)
    for ax in range(len(shape)):
        check('fftshift', f'{shape} axis={ax}', np.fft.fftshift, a, axes=ax)

# ============================================================
# 42. 0D SPECIAL CASES
# ============================================================
print("Testing 0D special cases...")
s = np.array(2.5)
check('sum_0D', '0D->scalar', np.sum, s)
check('mean_0D', '0D->scalar', np.mean, s)
check('sin_0D', '0D->0D', np.sin, s)
check('exp_0D', '0D->0D', np.exp, s)
check('abs_0D', '0D->0D', np.abs, s)
check('add_0D', '0D+0D', np.add, s, s)
check('multiply_0D', '0D*0D', np.multiply, s, s)
check('squeeze_0D', '0D no-op', np.squeeze, s)
# 0D reshape -> 1D
check('reshape_0D', '0D->1D', np.reshape, s, (1,))

# ============================================================
# 43. 5D / HIGH DIM GENERAL TESTS
# ============================================================
print("Testing 5D+ inputs...")
a5 = h((2, 2, 2, 3, 4))
check('sin_5D', '5D', np.sin, a5 * 0.1)
check('sum_5D', '5D no-axis', np.sum, a5)
check('sum_5D', '5D axis=2', np.sum, a5, axis=2)
check('mean_5D', '5D axis=-1', np.mean, a5, axis=-1)
check('flip_5D', '5D', np.flip, a5)
check('transpose_5D', '5D', np.transpose, a5)
check('concatenate_5D', '5D axis=0', np.concatenate, [a5, a5], axis=0)

# ============================================================
# 44. MAX NDIM
# ============================================================
print("Testing MAX_NDIM...")
for n in [32, 48, 63, 64]:
    try:
        a = np.zeros([1] * n)
        check('max_ndim', f'ndim={n} OK', np.sum, a)
    except Exception as e:
        check('max_ndim', f'ndim={n} FAIL: {e}', np.zeros, [1] * n)
expect_error('max_ndim', 'ndim=65 should fail', np.zeros, [1] * 65)

# ============================================================
# REPORT
# ============================================================
print("\n" + "=" * 70)
print("COMPREHENSIVE NDIM + AXES DISCOVERY REPORT")
print("=" * 70)

total_ok = 0
total_err = 0
total_unexpected = 0

for fn_name, cases in results.items():
    ok = [c for c in cases if c['status'] in ('ok', 'expected_error')]
    err = [c for c in cases if c['status'] == 'error']
    unexp = [c for c in cases if c['status'] == 'unexpected_ok']
    total_ok += len(ok)
    total_err += len(err)
    total_unexpected += len(unexp)
    if err or unexp:
        print(f"\n{fn_name}: {len(ok)} OK, {len(err)} ERROR, {len(unexp)} UNEXPECTED_OK")
        for c in err:
            print(f"  ERROR [{c['desc']}]: {c.get('error', '')}")
        for c in unexp:
            print(f"  UNEXPECTED OK [{c['desc']}]")

print(f"\n{'='*70}")
print(f"TOTALS: {total_ok} OK, {total_err} ERRORS, {total_unexpected} UNEXPECTED")
print(f"{'='*70}")
