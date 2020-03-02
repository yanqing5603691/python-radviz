from pathlib import Path
import ctypes
from sys import platform
import numpy as np
#
__debug = False
#
#
#
__lib_folder = Path(Path(__file__).resolve().parent).joinpath("kendall_tau_bin")
__lib = None
if platform == "linux" or platform == "linux2": # linux
    f = __lib_folder.joinpath("linux_kendall_tau.so")
    if f.exists(): 
        __lib = ctypes.CDLL(f.resolve())
        if __debug: print(f"Found {f.stem} library.")
elif platform == "darwin": # OS X
    f = __lib_folder.joinpath("osx_kendall_tau.so")
    if f.exists(): 
        __lib = ctypes.CDLL(f.resolve())
        if __debug: print(f"Found {f.stem} library.")
elif platform == "win32": # Windows
    f = __lib_folder.joinpath("win_kendall_tau.so")
    if f.exists(): 
        __lib = ctypes.CDLL(f.resolve())
        if __debug: print(f"Found {f.stem} library.")

if __lib is not None:
    __lib.kendall_tau_identity_distance.restype = ctypes.c_double
    __lib.kendall_tau_weighted_identity_distance.restype = ctypes.c_double
#
#
#
#
#
#
def __py_identity_distance_arr(arr, normalize=True):
    x = np.asarray(arr)
    n = arr.shape[0]
    norm_factor = (n*(n-1)/2.0)
    res = 0.0
    for i in range(n):
        for j in range(i, n):
            if x[i] > x[j]:
                res += 1
    if normalize:
        return res / norm_factor
    return res

def __lib_identity_distance_arr(arr, normalize=True):
    c_arr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    #c_arr = (ctypes.c_double * arr.shape[0])(*arr)
    c_length = ctypes.c_int(arr.shape[0])
    c_normalize = ctypes.c_int(1) if normalize else ctypes.c_int(0)
    c_return = __lib.kendall_tau_identity_distance(c_arr, c_length, c_normalize)
    return c_return

def __py_identity_weighted_distance_arr(arr, normalize=True):
    x = np.asarray(arr)
    n = arr.shape[0]
    norm_factor = 0.0
    res = 0.0
    for i in range(n):
        for j in range(n):
            diff = np.abs(x[i] - x[j])
            norm_factor += diff
            if j>i and x[i] > x[j]:
                res += diff
    if normalize:
        if (res == 0) or (norm_factor == 0): return 0.0
        return res / (norm_factor / 2)
    return res

def __lib_identity_weighted_distance_arr(arr, normalize=True):
    c_arr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    #c_arr = (ctypes.c_double * arr.shape[0])(*arr)
    c_length = ctypes.c_int(arr.shape[0])
    c_normalize = ctypes.c_int(1) if normalize else ctypes.c_int(0)
    c_return = __lib.kendall_tau_weighted_identity_distance(c_arr, c_length, c_normalize)
    return c_return
#
#
#
def __prepare_data(x, y):
    _x = np.asarray(x).copy()
    _y = None if y is None else np.asarray(y)
    if (_y is not None) and (_x.shape != _y.shape):
        raise RuntimeError(f"Array x and y must have the same shape. x.shape={_x.shape} and y.shape={_y.shape}.")
    if len(_x.shape) > 2:
        raise RuntimeError(f"Array x and y must have 1D or 2D shape, not {_x.shape}.")
    #
    #
    if _y is not None:
        if len(_x.shape) == 2:
            def reorder_fn(e): ##order _x rows according idx
                # pylint: disable=unbalanced-tuple-unpacking
                z, i = np.array_split(e,2)
                return z[np.array(i, dtype=int)]
            idx = np.argsort(_y, axis=1) #index sorting _y
            _x = np.apply_along_axis(reorder_fn, 1, np.c_[_x, idx])
        else: #1D array
            idx = np.argsort(_y) #index sorting _y
            _x = _x[idx]
    
    return _x
#
#
#
def distance(x, y=None, normalize=True, use_python=False):
    arr = __prepare_data(x, y)
    if __lib is None or use_python:
        if len(arr.shape) == 2:
            return np.apply_along_axis(__py_identity_distance_arr, 1, arr, normalize)
        else:
            return __py_identity_distance_arr(arr, normalize)
    else:
        if len(arr.shape) == 2:
            return np.apply_along_axis(__lib_identity_distance_arr, 1, arr, normalize)
        else:
            return __lib_identity_distance_arr(arr, normalize)
#
#
#
def weighted_distance(x, y=None, normalize=True, use_python=False):
    arr = __prepare_data(x, y)
    if __lib is None or use_python:
        if len(arr.shape) == 2:
            return np.apply_along_axis(__py_identity_weighted_distance_arr, 1, arr, normalize)
        else:
            return __py_identity_weighted_distance_arr(arr, normalize)
    else:
        if len(arr.shape) == 2:
            return np.apply_along_axis(__lib_identity_weighted_distance_arr, 1, arr, normalize)
        else:
            return __lib_identity_weighted_distance_arr(arr, normalize)
#
#
#  
#