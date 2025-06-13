#!/usr/bin/env python3
import sys
import numpy as np
from multiprocessing import Process
from osgeo import gdal, gdalconst
from numpy import frombuffer, float32

gdal.UseExceptions()

class dataLoader(Process):
    """
    Reads a GeoTIFF or ASCII grid in row-chunks and sends them
    (as float32 numpy arrays) down a pipe to the GPU process.
    """

    def __init__(self, input_file: str, output_pipe, read_rows: int = 15):
        super().__init__()
        self.input_file = input_file
        self.output_pipe = output_pipe
        self.read_rows = read_rows

        # open file & read header immediately so scheduler can query it
        ds = gdal.Open(self.input_file, gdalconst.GA_ReadOnly)
        band = ds.GetRasterBand(1)
        gt = ds.GetGeoTransform()
        nodata = band.GetNoDataValue()
        if nodata is None:
            nodata = -9999.0

        self.header = (
            band.XSize,         # ncols
            band.YSize,         # nrows
            abs(gt[1]),         # cellsize
            float(nodata),      # NODATA
            gt[0],              # xllcorner
            gt[3],              # yllcorner
            gt,                 # full GeoTransform
            ds.GetProjection()  # projection WKT
        )
        ds = None  # close file

    def getHeaderInfo(self):
        return self.header

    def run(self):
        # Re-open for reading in chunks
        ds = gdal.Open(self.input_file, gdalconst.GA_ReadOnly)
        band = ds.GetRasterBand(1)
        total_rows = band.YSize
        total_cols = band.XSize
        data_type = band.DataType

        row = 0
        try:
            while row < total_rows:
                to_read = min(self.read_rows, total_rows - row)
                raw = band.ReadRaster(
                    0, row,
                    total_cols, to_read,
                    buf_type=data_type
                )
                # split into individual rows
                arr = frombuffer(raw, dtype=float32).reshape(to_read, total_cols)
                for r in arr:
                    self.output_pipe.send(r)
                row += to_read
        except RuntimeError as e:
            print(f"Error reading {self.input_file}: {e}", file=sys.stderr)
        finally:
            # signal EOF
            self.output_pipe.close()
            ds = None
            print(f"Input loaded: {self.input_file}")

if __name__ == "__main__":
    print("dataLoader is a module, not a standalone script", file=sys.stderr)
    sys.exit(1)
