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


def main(input_file, output_file, header, functions):
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
        header,
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
    header_tuple = gpu_conn.recv()
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


def run(input_files, output_files, functions, disk_rows=15):
    """
    Batch-driven CLI entry point: processes multiple input files/outputs.
    """
    start = time.time()
    # Create per-file pipes
    input_pipe, gpu_pipe = Pipe(duplex=False)
    output_pipes = [Pipe(duplex=False) for _ in output_files]

    # Start loader
    loader_proc = Process(
        target=data_loader_main,
        args=(input_files, input_pipe)
    )
    loader_proc.start()
    header = loader_proc.get_header_info()  # example method

    # Start GPU
    gpu_conn_in = gpu_pipe
    gpu_conn_out = [p[0] for p in output_pipes]
    calc_proc = GPUCalculator(header, gpu_conn_in, gpu_conn_out, functions)
    calc_proc.start()

    # Close unused ends in parent
    input_pipe.close()
    gpu_pipe.close()
    for _, sender in output_pipes:
        sender.close()

    # Start savers
    saver_procs = []
    for out_file, (recv_conn, _) in zip(output_files, output_pipes):
        p = Process(target=data_saver_main, args=(out_file, recv_conn))
        p.start()
        saver_procs.append(p)

    # Monitor processes
    try:
        while any(p.is_alive() for p in [loader_proc, calc_proc] + saver_procs):
            if loader_proc.exitcode not in (None, 0):
                calc_proc.terminate()
                for p in saver_procs:
                    p.terminate()
                break
            if calc_proc.exitcode not in (None, 0):
                loader_proc.terminate()
                for p in saver_procs:
                    p.terminate()
                break
            sleep(1)
    finally:
        # Ensure cleanup
        for p in [loader_proc, calc_proc] + saver_procs:
            if p.is_alive():
                p.terminate()

        total = time.time() - start
        mins, secs = divmod(total, 60)
        print(f"Total time: {int(mins)} mins, {secs:.2f} secs")


if __name__ == '__main__':
    if len(sys.argv) < 5 or len(sys.argv) % 2 != 1:
        print("Usage: scheduler.py input output1 func1 [output2 func2 ...]", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    out_files = sys.argv[2::2]
    funcs = [f.lower() for f in sys.argv[3::2]]
    header = None  # will be populated by loader
    main(input_file, out_files[0], header, funcs)
