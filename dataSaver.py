#!/usr/bin/env python3
import sys
import numpy as np
from os.path import exists
from os import remove
from multiprocessing import Process
from osgeo import gdal

gdal.UseExceptions()

class dataSaver(Process):
    """
    Reads an 8-tuple header and streamed rows from a pipe,
    writes them out to a GeoTIFF, then exits cleanly.
    """

    def __init__(self, output_file: str, input_pipe, write_rows: int = 15):
        super().__init__()
        self.output_file = output_file
        self.input_pipe = input_pipe
        self.write_rows = write_rows

    def run(self):
        # ─── Receive and unpack header ───────────────────────
        try:
            hdr = self.input_pipe.recv()
        except EOFError:
            print("No header received; exiting.", file=sys.stderr)
            return

        # hdr = (ncols, nrows, cellsize, NODATA, xll, yll, geot, proj)
        ncols, nrows, cellsize, nodata, xll, yll, geot, proj = hdr

        # ─── Prepare output GeoTIFF ─────────────────────────
        if exists(self.output_file):
            remove(self.output_file)
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(
            self.output_file,
            ncols, nrows, 1,
            gdal.GDT_Float32,
            options=["COMPRESS=DEFLATE", "BIGTIFF=YES"]
        )
        band = ds.GetRasterBand(1)
        band.SetNoDataValue(nodata)
        ds.SetGeoTransform(geot)
        try:
            ds.SetProjection(proj)
        except RuntimeError:
            print("Warning: invalid projection", file=sys.stderr)

        # ─── Stream in and write row-blocks ──────────────────
        written = 0
        try:
            while written < nrows:
                chunk = min(self.write_rows, nrows - written)
                block = np.empty((chunk, ncols), dtype=np.float32)
                for i in range(chunk):
                    block[i, :] = self.input_pipe.recv()
                band.WriteArray(block, 0, written)
                written += chunk
                print(f"Written {written}/{nrows} rows", end="\r")
        except EOFError:
            # upstream closed early; finish up
            pass
        finally:
            ds.FlushCache()
            ds = None
            self.input_pipe.close()
            print(f"\nOutput written: {self.output_file}")

if __name__ == "__main__":
    print("dataSaver must be run via scheduler.py", file=sys.stderr)
    sys.exit(1)
