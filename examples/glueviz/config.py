import numpy as np
from pandas import read_csv
from glue.config import data_factory
from glue.core import Data

import pandas as pd
import gzip

from pyxlma.lmalib.io.read import lmafile

# class lmafile:
#     def __init__(self,filename):
#         self.file = filename
#
#     def datastart(self):
#         with gzip.open(self.file,'rt') as f:
#             for line_no, line in enumerate(f):
#                     if line.rstrip() == "*** data ***":
#                         break
#             f.close()
#             return line_no
#
#     def readfile(self):
#         lmad = pd.read_csv(self.file,delim_whitespace=True,header=None,
#                                           compression='gzip',skiprows=self.datastart()+1)
#         columns = ['time','lat','lon','alt','chi','p','mask']
#         lmad.columns = columns
#         return lmad

def is_lma_dataset(filename, **kwargs):
    return '.dat' in filename


@data_factory('LMA data file', priority=10000, 
              identifier=is_lma_dataset)
def read_lma_data(filename):
    lma = lmafile(filename)
    df = lma.readfile()
    data = Data()
    for column in df.columns:
        data[column] = df[column]
    # data['distance'] = np.hypot(data['x'], data['y'])
    return data

# def is_glm_dataset(filename, **kwargs):
#     return 'OR_GLM' in filename
#
# @data_factory('GLM gridded image', priority=10000,
#               identifier=is_glm_dataset)
# def read_glm_image(filename):
#     # Load the GLM data
#     glm = open_glm_time_series([filename])
#     x_1d = glm.x
#     y_1d = glm.y
#
#     # Convert the 1D fixed grid coordinates to 2D lon, lat
#
#     from lmatools.grid.fixed import get_GOESR_coordsys
#     x,y = np.meshgrid(x_1d, y_1d) # Two 2D arrays of fixed grid coordinates
#     nadir = -75.0 # Really should come from file.
#     geofixCS, grs80lla = get_GOESR_coordsys(nadir)
#     z=np.zeros_like(x)
#     lon,lat,alt=grs80lla.fromECEF(*geofixCS.toECEF(x,y,z))
#     lon.shape = x.shape
#     lat.shape = y.shape
#
#     # Add the 2D arrays back to the original dataset and save to disk.
#     # This doesn't have the CF standard name metadata or unit information,
#     # but that isn't too hard to add.
#
#     import xarray as xr
#     glm['longitude'] = xr.DataArray(lon, dims=('y', 'x'))
#     glm['latitude'] = xr.DataArray(lat, dims=('y', 'x'))



