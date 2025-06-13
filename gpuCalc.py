#!/usr/bin/env python3
"""
Optimized gpuCalc.py leveraging full VRAM, dynamic grid sizing,
stream overlap, pinned-memory pooling, and full 3×3 kernels.
"""

from multiprocessing import Process
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

def _ceildiv(a: int, b: int) -> int:
    return -(-a // b)


# ─────────────────────────────────────────────────────────────────────────────
# Local GPUStruct definition (replaces gpustruct.py)
# ─────────────────────────────────────────────────────────────────────────────

class GPUStruct:
    """
    Packs a small C‐struct of scalars onto the GPU.

    Initialize with a list of (dtype, name, value) tuples, where dtype is a
    NumPy scalar type (e.g. np.uint64, np.float32), name is the C‐field name,
    and value is the initial scalar.
    """

    def __init__(self, objs):
        # objs: list of (dtype, name, value)
        self._fields = []
        self._values = {}
        for dtype, name, val in objs:
            self._fields.append((dtype, name))
            # store as a 0‐d NumPy array
            self._values[name] = np.array(val, dtype=dtype)

        self._ptr     = None
        self._packstr = None

    def copy_to_gpu(self):
        # Build the packed byte string
        parts = []
        for dtype, name in self._fields:
            arr = self._values[name]
            parts.append(arr.tobytes())
        self._packstr = b"".join(parts)

        # Upload or update on GPU
        if self._ptr is None:
            self._ptr = cuda.mem_alloc(len(self._packstr))
        cuda.memcpy_htod(self._ptr, self._packstr)

    def get_ptr(self):
        if self._ptr is None:
            raise RuntimeError("GPUStruct: copy_to_gpu() not called")
        return self._ptr


# ─────────────────────────────────────────────────────────────────────────────
# The GPUCalculator itself
# ─────────────────────────────────────────────────────────────────────────────

class GPUCalculator(Process):
    def __init__(self, input_pipe, output_pipes, functions):
        super().__init__()
        self.input_pipe   = input_pipe
        self.output_pipes = output_pipes
        self.functions    = functions

    def _unpack_header(self):
        try:
            hdr = self.input_pipe.recv()
        except EOFError:
            raise RuntimeError("GPUCalculator: no header received")
        # hdr = (ncols, nrows, cellsize, NODATA, xll, yll, GeoT, prj)
        self.ncols, self.nrows, self.cellsize, self.NODATA, *_ = hdr

    def _gpu_alloc(self):
        total, free = cuda.mem_get_info()
        use_bytes   = int(free * 0.7)
        # compute how many rows fit in ~70% VRAM
        self.page_rows = max(3, min(self.nrows, use_bytes // (self.ncols * 4)))

        # host‐pinned buffers
        self.h_buf = cuda.pagelocked_empty((self.page_rows, self.ncols), np.float32)
        self.d_buf = cuda.pagelocked_empty((self.page_rows, self.ncols), np.float32)

        # device buffers
        self.d_in  = cuda.mem_alloc(self.h_buf.nbytes)
        self.d_out = cuda.mem_alloc(self.d_buf.nbytes)

        # dynamic grid to exactly cover data
        bx, by = 32, 8
        gx = _ceildiv(self.ncols, bx)
        gy = _ceildiv(self.page_rows, by)
        self.block = (bx, by, 1)
        self.grid  = (gx, gy, 1)

        # two streams for overlap
        self.streams = [cuda.Stream(), cuda.Stream()]

    def run(self):
        import pycuda.autoinit

        # 1) header & buffers
        self._unpack_header()
        self._gpu_alloc()

        # 2) compile kernels once
        mods  = [self._get_kernel(f) for f in self.functions]
        funcs = [m.get_function("raster_function") for m in mods]

        # 3) build & upload constant struct
        num_threads = self.block[0]*self.block[1]*self.grid[0]*self.grid[1]
        ppth = _ceildiv(self.page_rows * self.ncols, num_threads)
        stc = GPUStruct([
            (np.uint64, "pixels_per_thread", np.uint64(ppth)),
            (np.float64, "NODATA",            self.NODATA),
            (np.uint64, "ncols",              np.uint64(self.ncols)),
            (np.uint64, "nrows",              np.uint64(self.page_rows)),
            (np.uint64, "npixels",            np.uint64(self.page_rows * self.ncols)),
            (np.float32,"cellSize",           np.float32(self.cellsize)),
        ])
        stc.copy_to_gpu()
        ptr = stc.get_ptr()

        # 4) initial carry rows
        self.carry_rows = [
            np.full(self.ncols, self.NODATA, np.float32),
            np.zeros(self.ncols,             np.float32),
        ]

        done, idx = 0, 0
        while True:
            more = self._recv_page(done)
            s = self.streams[idx % 2]

            # async H2D
            cuda.memcpy_htod_async(self.d_in, self.h_buf, stream=s)
            # launch kernels
            for fn in funcs:
                fn(self.d_in, self.d_out, ptr,
                   block=self.block, grid=self.grid, stream=s)
            # async D2H
            cuda.memcpy_dtoh_async(self.d_buf, self.d_out, stream=s)
            s.synchronize()

            # send out results
            self._write_page(done)

            # advance
            rows = min(self.page_rows - 2, self.nrows - done)
            done += rows
            idx  += 1
            if not more:
                break

        # close downstream
        for p in self.output_pipes:
            p.close()
        # free device
        self.d_in.free()
        self.d_out.free()

    def _recv_page(self, done_rows: int) -> bool:
        tgt = self.h_buf
        if done_rows == 0:
            tgt[0, :] = self.NODATA
            row = 1
        else:
            tgt[0, :], tgt[1, :] = self.carry_rows
            row = 2

        last  = done_rows + self.page_rows >= self.nrows + 2
        limit = self.nrows - done_rows if last else self.page_rows

        try:
            while row < limit:
                tgt[row, :] = self.input_pipe.recv()
                row += 1
        except EOFError:
            pass

        if last:
            tgt[row:, :] = self.NODATA
        else:
            self.carry_rows = [
                tgt[self.page_rows-2].copy(),
                tgt[self.page_rows-1].copy()
            ]

        return not last

    def _write_page(self, done_rows: int):
        """
        Send out the computed rows for this page:
          - For full pages: rows 1..(page_rows-1)
          - For the last page: rows 1..rows_remaining
        """
        rows_remaining = self.nrows - done_rows

        if rows_remaining < (self.page_rows - 1):
            # final page: send exactly rows_remaining rows
            limit = rows_remaining + 1
        else:
            # full page: send page_rows-2 rows (r=1..page_rows-2)
            limit = self.page_rows - 1

        for r in range(1, limit):
            row = self.d_buf[r]
            for p in self.output_pipes:
                p.send(row)


    def _getRasterFunc(self, func: str):
        if func == "slope":
            return ("""
                __device__ float slope(float dx, float dy) {
                    return atan(sqrt(dx*dx + dy*dy));
                }
            """, "slope(dx, dy)")
        elif func == "aspect":
            return ("""
                __device__ float aspect(float dx, float dy, double NODATA) {
                    float a = atan2(dy, -dx) * (180.0f/M_PI);
                    if (dx==NODATA||dy==NODATA||(dx==0&&dy==0)) return NODATA;
                    if (a<0) a+=360.0f;
                    return a*(M_PI/180.0f);
                }
            """, "aspect(dx, dy, f->NODATA)")
        elif func == "hillshade":
            base, _ = self._getRasterFunc("slope")
            return (base + """
                __device__ float hillshade(float dx, float dy) {
                    float sl = slope(dx, dy);
                    float zen = (90.0f-45.0f)*(M_PI/180.0f);
                    float azi = ((360.0f-315.0f)+90.0f)*(M_PI/180.0f);
                    float hs = 255.0f*((cos(zen)*cos(sl)) +
                                     (sin(zen)*sin(sl)*cos(azi-sl)));
                    return hs<0?0:hs;
                }
            """, "hillshade(dx, dy)")
        else:
            raise NotImplementedError(f"Function '{func}' not supported")

    def _getKernel(self, funcType: str):
        func_def, func_call = self._getRasterFunc(funcType)
        src = f"""
        #include <math.h>
        #ifndef M_PI
        #define M_PI 3.14159265358979323846
        #endif
        typedef struct {{
            unsigned long long pixels_per_thread;
            double NODATA;
            unsigned long long ncols,nrows,npixels;
            float cellSize;
        }} passed_in;

        __device__ int getKernel(float *nbhd, const float *data,
                                 unsigned long off, passed_in *f) {{
            if(off<f->ncols||off>=f->npixels-f->ncols) return 1;
            unsigned long y = off%f->ncols;
            if(y==0||y==f->ncols-1) return 1;
            nbhd[1]=data[off-f->ncols]; nbhd[7]=data[off+f->ncols];
            nbhd[0]=data[off-f->ncols-1]; nbhd[3]=data[off-1];
            nbhd[6]=data[off+f->ncols-1];
            nbhd[2]=data[off-f->ncols+1]; nbhd[5]=data[off+1];
            nbhd[8]=data[off+f->ncols+1];
            return 0;
        }}

        {func_def}

        __global__ void raster_function(const float *data,
                                        float *res,
                                        passed_in *f) {{
            unsigned long x = blockIdx.x*blockDim.x + threadIdx.x;
            unsigned long y = blockIdx.y*blockDim.y + threadIdx.y;
            unsigned long off = y*f->ncols + x;
            unsigned long step = (gridDim.x*blockDim.x)*(gridDim.y*blockDim.y);
            float nbhd[9];
            for(unsigned long i=0;i<f->pixels_per_thread&&off<f->npixels;
                ++i,off+=step) {{
                if(data[off]==f->NODATA) {{
                    res[off]=f->NODATA;
                }} else if(getKernel(nbhd,data,off,f)) {{
                    res[off]=f->NODATA;
                }} else {{
                    for(int q=0;q<9;++q) if(nbhd[q]==f->NODATA) nbhd[q]=data[off];
                    float dx=(nbhd[2]+2*nbhd[5]+nbhd[8] - nbhd[0]-2*nbhd[3]-nbhd[6])/(8*f->cellSize);
                    float dy=(nbhd[6]+2*nbhd[7]+nbhd[8] - nbhd[0]-2*nbhd[1]-nbhd[2])/(8*f->cellSize);
                    res[off]={func_call};
                }}
            }}
        }}
        """
        return SourceModule(src)

    # alias _get_kernel → _getKernel
    _get_kernel = _getKernel


if __name__ == "__main__":
    print("Run via scheduler.py, not standalone.")
