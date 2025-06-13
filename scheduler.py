#!/usr/bin/env python3
import sys
from multiprocessing import Pipe, Process, active_children
from time import sleep
from dataLoader import dataLoader          # <— import the class
from dataSaver import dataSaver            # <— import the class
from gpuCalc   import GPUCalculator

def main(input_file, output_file, functions):
    # Create unidirectional pipes
    loader_conn, gpu_conn    = Pipe(duplex=False)
    gpu_out_conn, saver_conn = Pipe(duplex=False)

    # Spawn your loader as its own Process subclass
    loader_proc = dataLoader(input_file, loader_conn)

    # GPUCalculator is already a Process subclass
    gpu_proc    = GPUCalculator(gpu_conn, [gpu_out_conn], functions)

    # Spawn your saver as its own Process subclass
    saver_proc  = dataSaver(output_file, saver_conn)

    # Start them
    loader_proc.start()
    gpu_proc.start()
    saver_proc.start()

    # Forward header from loader→GPU→saver
    header = loader_conn.recv()
    gpu_conn.send(header)
    saver_conn.send(header)

    # Close parent’s copies to signal EOF
    loader_conn.close()
    gpu_conn.close()
    gpu_out_conn.close()
    saver_conn.close()

    # Wait for clean exit
    loader_proc.join()
    gpu_proc.join()
    saver_proc.join()

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: scheduler.py <in> <out> <func>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], [sys.argv[3].lower()])