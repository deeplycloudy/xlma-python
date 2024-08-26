"""Test functionality of pyxlma.lmalib.io"""

from os import listdir
from datetime import datetime as dt
import xarray as xr
import pandas as pd
import numpy as np
from pyxlma.lmalib.io import read as lma_read
from tests.test_flash import compare_dataarrays


def test_read_mf_dataset():
    """Test reading of multiple .dat.gz files into a single lma dataset"""
    files_to_read = listdir('examples/data/')
    files_to_read = ['examples/data/'+file for file in files_to_read]
    dataset, start_date = lma_read.dataset(files_to_read)
    assert start_date == dt(2023, 12, 24, 0, 57, 1)
    truth = xr.open_dataset('tests/truth/lma_netcdf/lma.nc')
    for var in truth.data_vars:
        compare_dataarrays(dataset, truth, var)


def test_read_onefile_dataset():
    dataset, start_date = lma_read.dataset('examples/data/WTLMA_231224_005701_0001.dat.gz')
    assert start_date == dt(2023, 12, 24, 0, 57, 1)
    truth = xr.open_dataset('tests/truth/lma_netcdf/small_lma.nc')
    for var in truth.data_vars:
        compare_dataarrays(dataset, truth, var)


def test_read_nldn():
    dataset = lma_read.nldn('examples/network_samples/gld360enldnns_20231224_daily_v1_lit.raw')
    truth = pd.DataFrame({
        'latitude' : [33.582, 33.590, 33.585],
        'longitude' : [-101.881, -102.032, -101.872],
        'peak_current_kA' : [12.0, -9.0, -15.0],
        'multiplicity' : [0, 0, 0],
        'semimajor': [0.4, 0.2, 2.2],
        'semiminor': [0.2, 0.2, 1.5],
        'majorminorratio': [1.8, 1.0, 1.5],
        'ellipseangle' : [75, 3, 26],
        'chi2' : [0.8, 1.0, 2.1],
        'num_stations': [4, 5, 6],
        'type': ['IC', 'CG', 'IC'],
        'datetime' : [np.datetime64('2023-12-24T00:57:01.123456789'), np.datetime64('2023-12-24T00:57:31.987654321'), np.datetime64('2023-12-24T00:57:58.135792468')]
    })
    assert dataset.equals(truth)


def test_read_entln():
    dataset = lma_read.entln('examples/network_samples/lxarchive_pulse20231224.csv')
    truth = pd.DataFrame({
        'type': ['IC', 'CG', 'IC'],
        'datetime' : [np.datetime64('2023-12-24T00:57:04.123456789'), np.datetime64('2023-12-24T00:57:26.987654321'), np.datetime64('2023-12-24T00:57:47.246813579')],
        'latitude' : [33.581914, 33.590077, 33.584480],
        'longitude' : [-101.880986, -102.032033, -101.871498],
        'peak_current_kA' : [12.345, 9.876, 15.79],
        'icheight' : [3014, 6028, 13591],
        'num_stations': [8, 7, 11],
        'ellipseangle' : [104., 99., 102.],
        'semimajor': [0.1855, 0.3465, 0.089],
        'semiminor': [0.029, 0.054, 0.031]
    })
    assert dataset.equals(truth)
