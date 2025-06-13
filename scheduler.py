#!/usr/bin/env python3
"""
scheduler.py
Starts and manages processes which load data, do raster calculations on GPU,
and save data back to disk.
"""
import sys
import time
from multiprocessing import Pipe, Process, active_children
from time import sleep
from dataLoader import main as data_loader_main
from dataSaver import main as data_saver_main
from gpuCalc import GPUCalculator


def main(input_file, output_file, functions):
    """
    Unidirectional loader → GPU → saver pipeline.
    Closes parent pipe ends immediately after starting children
    to ensure EOF is signaled and no recv() blocks.
    """
    # Create pipes
    loader_conn, gpu_conn    = Pipe(duplex=False)  # loader → GPU
    gpu_out_conn, saver_conn = Pipe(duplex=False)  # GPU → saver

    # Spawn processes
    loader_proc = Process(
        target=data_loader_main,
        args=(input_file, loader_conn)
    )
    gpu_proc = GPUCalculator(
        gpu_conn,
        [gpu_out_conn],
        functions
    )
    saver_proc = Process(
        target=data_saver_main,
        args=(output_file, saver_conn)
    )

    loader_proc.start()
    gpu_proc.start()
    saver_proc.start()

    # Forward header from loader → GPU into saver
    header_tuple = loader_conn.recv()
    gpu_conn.send(header_tuple)
    saver_conn.send(header_tuple)

    # Close parent’s copies of pipe ends to signal EOF
    loader_conn.close()
    gpu_conn.close()
    gpu_out_conn.close()
    saver_conn.close()

    # Wait for clean exit
    loader_proc.join()
    gpu_proc.join()
    saver_proc.join()


if __name__ == '__main__':
    if len(sys.argv) < 5 or len(sys.argv) % 2 != 1:
        print("Usage: scheduler.py input output1 func1 [output2 func2 ...]", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    out_files = sys.argv[2::2]
    funcs = [f.lower() for f in sys.argv[3::2]]
    main(input_file, out_files[0], funcs)
    if len(sys.argv) < 5 or len(sys.argv) % 2 != 1:
        print("Usage: scheduler.py input output1 func1 [output2 func2 ...]", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    out_files = sys.argv[2::2]
    funcs = [f.lower() for f in sys.argv[3::2]]
    header = None  # will be populated by loader
    main(input_file, out_files[0], header, funcs)
