"""Test functionality of pyxlma.io"""

from os import listdir
from datetime import datetime as dt
import xarray as xr
from pyxlma.lmalib.io import read as lma_read


def test_read_mf_dataset():
    """Test reading of multiple .dat.gz files into a single lma dataset"""
    files_to_read = listdir('examples/data/lma/')
    files_to_read = ['examples/data/lma/'+file for file in files_to_read]
    dataset, start_date = lma_read.dataset(files_to_read)
    assert start_date == dt(2023, 12, 24, 0, 57, 1)
    assert dataset == xr.open_dataset('examples/data/lma_netcdf/lma.nc')
