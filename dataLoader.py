from osgeo import gdal, gdalconst as gdc
from multiprocessing import Process, Pipe
import struct
from numpy import float32, frombuffer

gdal.UseExceptions()
fmttypes = {'Byte':'B', 'UInt16':'H', 'Int16':'h', 'UInt32':'I', 'Int32':'i', 'Float32':'f', 'Float64':'d'}

"""
dataLoader

Class that reads data from a given input file and passes it to a Pipe object
designed to run as a separate process and inherits from Process module
currently supported input file types: [GEOTiff (.tif), ESRI ASCII format (.asc)]

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
class dataLoader(Process):

    """
    __init__

    paramaters:
        _input_file - must be a valid file path as a string
        _output_pipe - a Pipe object to pass read information into
   
    opens the input file and grabs the header information
    sets several instance variables
    """
    def __init__(self, _input_file, _output_pipe, _disk_rows):
        Process.__init__(self)
        self.read_rows = _disk_rows
        self.output_pipe = _output_pipe
        self.file_name = _input_file
        self.file_type = None
        self._openFile()
        self._readHeaderInfo()
        self._closeFile()

    #--------------------------------------------------------------------------#

    """
    _closeFile

    closes open GDAL file
    """ 
    def _closeFile(self):
        self.open_file = None
        self.open_raster_band = None

    #--------------------------------------------------------------------------#

    """
    getHeaderInfo
    
    Returns header information as a six-tuple in this order:
    (ncols, nrows, cellsize, NODATA, xllcorner, yllcorner)
    (ncols, nrows, cellsize, NODATA, xllcorner, yllcorner, GeoT, prj)
    """
    def getHeaderInfo(self):
        return self.totalCols, self.totalRows, self.cellsize, self.NODATA, self.xllcorner, self.yllcorner, self.GeoT, self.prj

    #--------------------------------------------------------------------------#

    """
    _readHeaderInfo

    requires file to be opened already, gets header info from file and puts it in instance variables
    """
    def _readHeaderInfo(self):
        self.GeoT = self.open_file.GetGeoTransform()
        self.prj = self.open_file.GetProjection()
        self.NODATA = self.open_raster_band.GetNoDataValue()
        self.xllcorner = self.GeoT[0]
        self.yllcorner = self.GeoT[3]
        if self.file_type == "tif":
            self.cellsize = self.open_raster_band.GetScale()
        elif self.file_type == "asc":
            self.cellsize = self.GeoT[1]
        self.totalRows = self.open_raster_band.YSize
        self.totalCols = self.open_raster_band.XSize

    #--------------------------------------------------------------------------#
  
    """
    _openFile

    opens file_name and sets it to open_file, supports '.tif' and '.asc' files
    """
    def _openFile(self):
        self.open_file = gdal.Open(self.file_name, GA_ReadOnly)
        self.open_raster_band = self.open_file.GetRasterBand(1)
        self.dataType = self.open_raster_band.DataType
        #self.unpackVal = fmttypes[gdal.GetDataTypeName(self.dataType)]*self.open_raster_band.XSize
        if ".asc" in self.file_name:
            self.file_type = "asc"
        elif ".tif" in self.file_name:
            self.file_type = "tif"
        else:
            print("Unsupported file type")
            self.stop()

    #--------------------------------------------------------------------------#

    """
    getFileType

    Returns the extension of the input file type as a string.
    """
    def getFileType(self):
        return self.file_type

    #--------------------------------------------------------------------------#

    """
    stop 

    Alerts the thread that it needs to quit
    Closes file and pipe
    """
    def stop(self):
        print("Stopping loader...")
        self.open_file = None
        self.open_raster_band = None
        self.output_pipe.close()
        exit(1)

    #--------------------------------------------------------------------------#

    """
    run

    calls functions needed to read all data from file_name
    """
    def run(self):
        self._openFile()
        self._loadFunc()

    #--------------------------------------------------------------------------#

    """
    _getLines

    Each _getLines function reads in a file of a different type and sends data
    to the GPUCalculator class
    """
    def _getLines(self):
        line_num = 0
        while line_num <= self.totalRows:
            # no need to read more, ended exactly at EOF
            if line_num == self.totalRows:
                return
            try:
                # remaining rows < read_rows, only read in as many as are extra
                if line_num + self.read_rows >= self.totalRows:
                    remaining = self.totalRows - line_num
                    line_tup = self.open_raster_band.ReadRaster(0,line_num,self.totalCols,remaining,buf_type=self.dataType)
                    for line in range(remaining-1):
                        self.output_pipe.send(frombuffer(line_tup[(line*self.totalCols*4): (line*self.totalCols + self.totalCols)*4], dtype=float32))
                    return
                # read in as many rows as read_rows indicates
                else:
                    line_tup = self.open_raster_band.ReadRaster(0, line_num, self.totalCols, self.read_rows, buf_type=self.dataType)
                    for line in range(self.read_rows):
                        self.output_pipe.send(frombuffer(line_tup[(line*self.totalCols*4): (line*self.totalCols + self.totalCols)*4], dtype=float32))
            # EOF
            except RuntimeError as e:
                print(e)
                return
            line_num += self.read_rows
        print("finished reading rows")

    #--------------------------------------------------------------------------#

    """
    _loadFunc

    sends data one row at a time to output_pipe, sends exactly the number of rows as are in the input file
    """
    def _loadFunc(self):
        self._getLines()
        self.output_pipe.close()
        print("Input file loaded from disk")

if __name__=="__main__":
    pass

