#!/usr/bin/env python3
"""
Refactored GPUStruct:
- Simplify pointer handling
- Use explicit struct formats
- Improved error handling and docstrings
"""
import numpy as np
import struct
import pycuda.driver as cuda

class GPUStruct:
    """
    Packs a Python-side value list into a GPU struct, manages
    host/device pointer allocations, and supports copy_to/from_gpu.

    Usage:
        fields = [
            ("pixels_per_thread", np.uint64, 1),
            ("NODATA", float, -9999.0),
            ("ncols", np.uint64, 1024),
            ("nrows", np.uint64, 1024),
            ("cellSize", np.float32, 30.0),
        ]
        st = GPUStruct(fields)
        st.copy_to_gpu()
        kernel(..., st.get_device_ptr())
        st.copy_from_gpu()
        print(st.NODATA)
    """
    def __init__(self, field_defs):
        # field_defs: list of (name, dtype, value)
        self._fields = []  # list of (name, dtype, device_ptr?)
        self._fmt = ''     # struct format string
        self._values = []  # local values
        self._device_ptr = None
        for name, dtype, val in field_defs:
            self._fields.append((name, dtype))
            self._values.append(val)
            # map dtype to struct char
            if np.issubdtype(dtype, np.integer):
                self._fmt += 'Q'  # interpret all ints as unsigned long long
            elif np.issubdtype(dtype, np.floating):
                self._fmt += 'd' if dtype == np.float64 else 'f'
            else:
                raise TypeError(f"Unsupported dtype {dtype}")
        # precompute pack size
        self._size = struct.calcsize(self._fmt)

    def copy_to_gpu(self):
        # pack local values
        packed = struct.pack(self._fmt, *self._values)
        # allocate/push to GPU
        if self._device_ptr is None:
            self._device_ptr = cuda.mem_alloc(self._size)
        cuda.memcpy_htod(self._device_ptr, packed)

    def copy_from_gpu(self):
        if self._device_ptr is None:
            raise RuntimeError("copy_to_gpu must be called first")
        buf = bytearray(self._size)
        cuda.memcpy_dtoh(buf, self._device_ptr)
        unpacked = struct.unpack(self._fmt, buf)
        for (name, _), val in zip(self._fields, unpacked):
            setattr(self, name, val)

    def get_device_ptr(self):
        if self._device_ptr is None:
            raise RuntimeError("copy_to_gpu must be called first")
        return int(self._device_ptr)

    def __del__(self):
        try:
            if self._device_ptr:
                self._device_ptr.free()
                self._device_ptr = None
        except Exception:
            pass
