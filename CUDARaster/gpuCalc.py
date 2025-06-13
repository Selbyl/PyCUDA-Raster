from multiprocessing import Process,Pipe
import numpy as np
from .gpustruct import GPUStruct
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

TOTALROWCOUNT = 0

"""
GPUCalculator

Class that takes and sends data from pipes and goes GPU calculations on it
designed to run as a separate process and inherits from Process module
currently supported functions: slope, aspect, hillshade

copyright            : (C) 2016 by Alex Feurst, Charles Kazer, William Hoffman
email                : fuersta1@xavier.edu, ckazer1@swarthmore.edu, whoffman1@gulls.salisbury.edu

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
class GPUCalculator(Process):
  
    """
    __init__

    paramaters:
        header - six-tuple header expected to be in this order: (ncols, nrows, cellsize, NODATA, xllcorner, yllcorner)
        _input_pipe - a Pipe object to read information from
        _outputPipe - a Pipe object to send information to
        function_types - list of strings that are supported function names as strings

    creates empty instance variables needed later
    """
    def __init__(self, header, _input_pipe, _output_pipes, function_types):
        Process.__init__(self)

        self.input_pipe = _input_pipe
        self.output_pipes = _output_pipes 
        self.functions = function_types
        self.header = header

    #--------------------------------------------------------------------------#

    """
    _unpackInfo

    gets data from header and creates carry_over_rows needed in _processData
    """
    def _unpackInfo(self):
        #unpack header info
        self.totalCols = self.header[0]
        self.totalRows = self.header[1]
        self.cellsize = self.header[2]
        self.NODATA = self.header[3]

        #carry over rows used to insert last two lines of data from one page
        #as first two lines in next page
        self.carry_over_rows = [np.full(shape=self.totalCols, fill_value=self.NODATA, dtype=np.float32) \
        , np.empty(shape=self.totalCols, dtype=np.float32)] # second array does not need to be filled with NODATA
        self.np_copy_arr = [i for i in range(self.totalCols)]


    def __del__(self):
        pass

    #--------------------------------------------------------------------------#
        
    """
    run

    Overrides default Process.run()
    Given a kernel type, retrieves the C code for that kernel, and runs the
    data processing loop

    does CUDA initialization and sets local device and context
    """
    def run(self):
        import pycuda.autoinit
        self._unpackInfo()
        self._gpuAlloc()

        compiled_kernels = []
        for function in self.functions:
            kernel = self._getKernel(function)
            compiled_kernels.append(kernel.get_function("raster_function"))

        #Process data while we continue to receive input
        count = 0
        while self._recvData(count):
            #Copy input data to GPU
            cuda.memcpy_htod(self.data_gpu, self.to_gpu_buffer)
            for i in range(len(compiled_kernels)):
                self._processData(compiled_kernels[i])
                #Get data back from GPU
                cuda.memcpy_dtoh(self.from_gpu_buffer, self.result_gpu)
                self._writeData(count, self.output_pipes[i])

            count += (self.maxPossRows-2)  # -2 because of buffer rows
            print("Page done... %.3f %% completed" % ((float(count) / float(self.totalRows)) * 100))
        #Process remaining data in buffer
        cuda.memcpy_htod(self.data_gpu, self.to_gpu_buffer)
        for i in range(len(self.functions)):
            self._processData(compiled_kernels[i])
            cuda.memcpy_dtoh(self.from_gpu_buffer, self.result_gpu) 
            self._writeData(count, self.output_pipes[i])

        for pipe in self.output_pipes:
            pipe.close()

        print("GPU calculations finished")
        # clean up on GPU
        self.data_gpu.free()
        self.result_gpu.free()
        cuda.Context.pop()

    #--------------------------------------------------------------------------#

    """
    _gpuAlloc

    determines how much free memory is on the GPU and allocates as much as needed
    creates pagelocked buffers of equal size to GPU memory
    """
    def _gpuAlloc(self):
        #Get GPU information
        self.freeMem = cuda.mem_get_info()[0] * .5 * .8
        self.maxPossRows = np.int(np.floor(self.freeMem / (4 * self.totalCols)))
        # set max rows to smaller number to save memory usage
        if self.totalRows < self.maxPossRows:
            print("reducing max rows to reduce memory use on GPU")
            self.maxPossRows = self.totalRows
            #self.maxPossRows = 100

        # create pagelocked buffers and GPU arrays
        self.to_gpu_buffer = cuda.pagelocked_empty((self.maxPossRows , self.totalCols), np.float32)
        self.from_gpu_buffer = cuda.pagelocked_empty((self.maxPossRows , self.totalCols), np.float32)
        self.data_gpu = cuda.mem_alloc(self.to_gpu_buffer.nbytes)
        self.result_gpu = cuda.mem_alloc(self.from_gpu_buffer.nbytes)

    #--------------------------------------------------------------------------#

    """
    _recvData

    Receives a page worth of data from the input pipe. The input pipe comes
    from dataLoader.py. Copies over 2 rows from the previous page so the GPU 
    kernel computation works correctly.
    If the pipe closes, fill the rest of the page with NODATA, and return false
    to indicate that we should break out of the processing loop.
    """
    def _recvData(self, count):
        if count == 0:
            #If this is the first page, insert a buffer row
            np.put(self.to_gpu_buffer[0], self.np_copy_arr, self.carry_over_rows[0])    
            row_count = 1
        else:
            #otherwise, insert carry over rows from last page
            np.put(self.to_gpu_buffer[0], self.np_copy_arr, self.carry_over_rows[0])
            np.put(self.to_gpu_buffer[1], self.np_copy_arr, self.carry_over_rows[1])
            row_count = 2

        if count + row_count + self.maxPossRows > self.totalRows and row_count > 1: # this is the last page
            try:
                while row_count + count < self.totalRows: # get rest of rows from pipe
                    np.put(self.to_gpu_buffer[row_count], self.np_copy_arr, self.input_pipe.recv())
                    row_count += 1
            #Pipe was closed unexpectedly
            except EOFError:
                print("Pipe closed unexpectedly.")
                self.stop()
            self.to_gpu_buffer[row_count].fill(self.NODATA)
            return False # finished receiving data, tell run to end

        else: # this is not the last page
            try:        
                while row_count < self.maxPossRows: # get max poss rows from pipe
                    np.put(self.to_gpu_buffer[row_count], self.np_copy_arr, self.input_pipe.recv())
                    row_count += 1
            #Pipe was closed unexpectedly
            except EOFError:
                print("Pipe closed unexpectedly.")
                self.stop()              
            #Update carry over rows
            np.put(self.carry_over_rows[0], self.np_copy_arr, self.to_gpu_buffer[self.maxPossRows-2])
            np.put(self.carry_over_rows[1], self.np_copy_arr, self.to_gpu_buffer[self.maxPossRows-1])

            return True # not finished reveiving data, tell run to keep looping

    #--------------------------------------------------------------------------#

    """
    _processData

    Using the given kernel code packed in mod, allocates memory on the GPU,
    and runs the kernel.
    """
    def _processData(self, func):
        #GPU layout information
        grid = (256,256)
        block = (32,32,1)
        num_blocks = grid[0] * grid[1]
        threads_per_block = block[0]*block[1]*block[2]
        pixels_per_thread = (self.maxPossRows * self.totalCols) / (threads_per_block * num_blocks)    
        # minimize work by each thread while makeing sure each pixel is calculated   
        while pixels_per_thread < 1:
            grid = (grid[0] - 16,grid[1] - 16)
            num_blocks = grid[0] * grid[1]
            pixels_per_thread = (self.maxPossRows * self.totalCols) / (threads_per_block * num_blocks)
        pixels_per_thread = np.ceil(pixels_per_thread)

        #information struct passed to GPU
        stc = GPUStruct([
            (np.uint64, 'pixels_per_thread', pixels_per_thread),
            (np.float64, 'NODATA', self.NODATA),
            (np.uint64, 'ncols', self.totalCols),
            (np.uint64, 'nrows', self.maxPossRows),
            (np.uint64, 'npixels', self.maxPossRows*self.totalCols),
            (np.float32, 'cellSize', self.cellsize)
            ])

        stc.copy_to_gpu()

        #Call GPU kernel
        func(self.data_gpu, self.result_gpu, stc.get_ptr(), block=block, grid=grid)

    #--------------------------------------------------------------------------#

    """
    _writeData

    Writes results to output pipe. This pipe goes to dataSaver.py
    """
    def _writeData(self, count, out_pipe):
        if count + (self.maxPossRows-1) > self.totalRows:
            r = self.totalRows - (count-1)
        else:
            r = self.maxPossRows-1
        for row in range(1, r):
            out_pipe.send(self.from_gpu_buffer[row])

    #--------------------------------------------------------------------------#

    """
    stop 

    Alerts the thread that it needs to quit
    Cleans up CUDA and pipes
    """
    def stop(self):
        print("Stopping gpuCalc...")
        self.data_gpu.free()
        self.result_gpu.free()
        cuda.Context.pop()
        for pipe in self.output_pipes:
            pipe.close()
        self.input_pipe.close()
        exit(1)

    #--------------------------------------------------------------------------#

    """
    _getRasterFunc

    Given a string representing the raster function to calculate,
    returns the required code to append to the CUDA kernel.
    """
    # NOTE: The kernel code in _getKernel only supports functions that are based
    # on a 3x3 grid of neighbors.
    #
    # To add your own computation, add an if statement looking for it, and return
    # a tuple containg the C code for your function surrounded by triple quotes,
    # and how that function should be called within the code in _getKernel.
    # 
    # Possible parameters you can use from _getKernel:
    # float *nbhd      /* this is the 3x3 grid of neighbors */
    # float dz_dx
    # float dz_dy
    # unsigned long long file_info->pixels_per_thread
    # double file_info->NODATA
    # unsigned long long file_info->ncols
    # unsigned long long file_info->nrows
    # unsigned long long file_info->npixels
    # float file_info->cellSize


    def _getRasterFunc(self, func):
        if func == "slope":
            return (\
                """
                /*
                    GPU only function that calculates slope for a pixel
                */
                __device__ float slope(float dz_dx, float dz_dy){
                    return atan(sqrt(pow(dz_dx, 2) + pow(dz_dy, 2)));
                }
                """,\
                """
                slope(dz_dx, dz_dy)
                """)

        elif func == "aspect":
            return (\
                """
                /*
                    GPU only function that calculates aspect for a pixel
                */
                __device__ float aspect(float dz_dx, float dz_dy, double NODATA){
                    float aspect = 57.29578 * (atan2(dz_dy, -(dz_dx)));
                    if(dz_dx == NODATA || dz_dy == NODATA || (dz_dx == 0.0 && dz_dy == 0.0)){
                        return NODATA;
                    } else{
                        if(aspect > 90.0){
                            aspect = 360.0 - aspect + 90.0;
                        } else {
                            aspect = 90.0 - aspect;
                        }
                            aspect = aspect * (M_PI / 180.0);
                            return aspect;
                        }
                }
                """,\
                """
                aspect(dz_dx, dz_dy, file_info->NODATA)
                """)

        elif func == "hillshade":
            return (self._getRasterFunc('slope')[0] + \
                """
                /*
                    GPU only function that calculates aspect for a pixel
                    to be ONLY used by hillshade
                */    
                __device__ float hillshade_aspect(float dz_dx, float dz_dy){
                    float aspect;
                            if(dz_dx != 0){
                                aspect = atan2(dz_dy, -(dz_dx));
                                if(aspect < 0){
                                    aspect = ((2 * M_PI) + aspect);
                            }
                        } else if(dz_dx == 0){
                            if(dz_dy > 0){
                                    aspect = (M_PI / 2);
                                }else if(dz_dy < 0){
                                    aspect = ((2 * M_PI) - (M_PI / 2));
                                }else{
                                    aspect = atan2(dz_dy, -(dz_dx));
                            }
                        }
                    return aspect;
                }

                /*
                    GPU only function that calculates hillshade for a pixel
                */
                __device__ float hillshade(float dz_dx, float dz_dy){
                    /* calc slope and aspect */
                    float slp = slope(dz_dx, dz_dy);
                    float asp = hillshade_aspect(dz_dx, dz_dy);

                    /* calc zenith */
                        float altitude = 45;
                        float zenith_deg = 90 - altitude;
                        float zenith_rad = zenith_deg * (M_PI / 180.0);
    
                    /* calc azimuth */
                        float azimuth = 315;
                        float azimuth_math = (360 - azimuth + 90);
                        if(azimuth_math >= 360.0){
                                azimuth_math = azimuth_math - 360;
                    }	
                    float azimuth_rad = (azimuth_math * M_PI / 180.0);

                    float hs = 255.0 * ( ( cos(zenith_rad) * cos(slp) ) + ( sin(zenith_rad) * sin(slp) * cos(azimuth_rad - asp) ) );

                        if(hs < 0){
                                return 0;
                    } else {
                        return hs;
                    }
                }
                """,\
                """
                hillshade(dz_dx, dz_dy)
                """)
        else:
            print("Function %s not implemented" % func)
            raise NotImplementedError

    #--------------------------------------------------------------------------#

    """
    _getKernel

    Packages the kernel module. This kernel assumes that the raster calculations
    will be based on the dx_dz and dy_dz values, which are calculated from a 3x3
    grid surrounding the current pixel.
    """
    # NOTE: To create another raster function, you must create an additional
    # entry in _getRasterCalc. Currently only supports calculations based on a
    # 3x3 grid surrounding a pixel.
    # The GPUCalculator class is set up to automatically insert buffer rows at
    # the beginning and end of the file so that all rows are calculated correctly.
    def _getKernel(self, funcType):
        func_def, func_call = self._getRasterFunc(funcType)
        mod = SourceModule("""
                    #include <math.h>
                    #include <stdio.h>
                    #ifndef M_PI
                    #define M_PI 3.14159625
                    #endif
                    typedef struct{
                            /* struct representing the relevant data passed in by host */
                            unsigned long long pixels_per_thread;
                            double NODATA;
                            unsigned long long ncols;
                            unsigned long long nrows;
                            unsigned long long npixels;
                            float cellSize;
                    } passed_in;

                    /************************************************************************************************
                            GPU only function that gets the neighbors of the pixel at offset
                            stores them in the passed-by-reference array 'store'
                    ************************************************************************************************/
                    __device__ int getKernel(float *store, float *data, unsigned long offset, passed_in *file_info){
                            //NOTE: This is more or less appropriated from Liam's code. Treats edge rows and columns
                            // as buffers, they will be dropped.
                            if (offset < file_info->ncols || offset >= (file_info->npixels - file_info->ncols)){
                                    return 1;
                            }
                            unsigned long y = offset % file_info->ncols; //FIXME: I'm not sure why this works...
                            if (y == (file_info->ncols - 1) || y == 0){
                                    return 1;
                            }
                            // Grab neighbors above and below.
                            store[1] = data[offset - file_info->ncols];
                            store[7] = data[offset + file_info->ncols];
                            // Grab right side neighbors.
                            store[2] = data[offset - file_info->ncols + 1];
                            store[5] = data[offset + 1];
                            store[8] = data[offset + file_info->ncols + 1];
                            // Grab left side neighbors.
                            store[0] = data[offset - file_info->ncols - 1];
                            store[3] = data[offset - 1];
                            store[6] = data[offset + file_info->ncols - 1];
                            /* return a value otherwise it throws a warning expression not having effect */
                            return 0;
                    }
                    """\
                            + func_def + \
                    """
                    /************************************************************************************************
                            CUDA Kernel function to calculate the slope of pixels in 'data' and stores them in 'result'
                            handles a variable number of calculations based on its thread/block location 
                            and the size of pixels_per_thread in file_info
                    ************************************************************************************************/
                    __global__ void raster_function(float *data, float *result, passed_in *file_info){
                            /* get individual thread x,y values */
                            unsigned long long x = blockIdx.x * blockDim.x + threadIdx.x;
                            unsigned long long y = blockIdx.y * blockDim.y + threadIdx.y; 
                            unsigned long long offset = (gridDim.x*blockDim.x) * y + x; 
                            //gridDim.x * blockDim.x is the width of the grid in threads. This moves us to the correct
                            //block and thread.
                            unsigned long long i;
                            /* list to store 3x3 kernel each pixel needs to calc slope */
                            float nbhd[9];
                            /* iterate over assigned pixels and calculate slope for all of them */
                            /* do npixels + 1 to make last row(s) get done */
                            for(i=0; i < file_info -> pixels_per_thread + 1 && offset < file_info -> npixels; ++i){	    
                                    if(data[offset] == file_info -> NODATA){
                                        result[offset] = file_info -> NODATA;
                                    } else {
                                        int q = getKernel(nbhd, data, offset, file_info);
                                        if (q) {
                                            result[offset] = file_info->NODATA;
                                        } else {
                                            for(q = 0; q < 9; ++q){
                                                if(nbhd[q] == file_info -> NODATA){
                                                    nbhd[q] = data[offset];
                                                }
                                            }
                                            float dz_dx = (nbhd[2] + (2*nbhd[5]) + nbhd[8] - (nbhd[0] + (2*nbhd[3]) + nbhd[6])) / (8 * file_info -> cellSize);
                                            float dz_dy = (nbhd[6] + (2*nbhd[7]) + nbhd[8] - (nbhd[0] + (2*nbhd[1]) + nbhd[2])) / (8 * file_info -> cellSize);

                                            result[offset] =""" + func_call + """;
                                        }
                                    }
                                    offset += (gridDim.x*blockDim.x) * (gridDim.y*blockDim.y);
                                    //Jump to next row
                            }
                    }
                    """)
        return mod

if __name__=="__main__":
    pass
