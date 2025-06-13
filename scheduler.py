#!/usr/bin/env python3
import sys
from multiprocessing import Pipe
from dataLoader import dataLoader
from gpuCalc   import GPUCalculator
from dataSaver import dataSaver

# Must match dataLoader default or override via constructor
ROWS_DEFAULT = 15

def main(input_file: str, output_file: str, function: str):
    # ─── Create unidirectional pipes ───────────────────
    # loader → GPU
    loader_recv, loader_send = Pipe(duplex=False)  
    # GPU    → saver
    gpu_recv,    gpu_send    = Pipe(duplex=False)

    # ─── Instantiate the loader but do NOT start yet ───
    loader = dataLoader(input_file, loader_send, ROWS_DEFAULT)

    # ─── Read header immediately (constructor already did GDAL open) ───
    header = loader.getHeaderInfo()

    # ─── Pre-inject header into each pipe ─────────────
    loader_send.send(header)   # for GPUCalculator
    gpu_send.send(header)      # for dataSaver

    # ─── Instantiate GPU and saver processes ───────────
    gpu_proc   = GPUCalculator(loader_recv, [gpu_send], [function])
    saver_proc = dataSaver(output_file, gpu_recv)

    # ─── Start loader, GPU, and saver ───────────────────
    loader.start()
    gpu_proc.start()
    saver_proc.start()

    # ─── Close parent’s copies of pipe ends ────────────
    loader_send.close()
    loader_recv.close()
    gpu_send.close()
    gpu_recv.close()

    # ─── Wait for all to finish ────────────────────────
    loader.join()
    gpu_proc.join()
    saver_proc.join()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: scheduler.py <in.tif> <out.tif> <func>", file=sys.stderr)
        sys.exit(1)
    inp, out, func = sys.argv[1], sys.argv[2], sys.argv[3].lower()
    main(inp, out, func)
