import pytest
from pyxlma.plot.xlma import *
import xarray as xr
import datetime as dt
import pandas as pd

@pytest.mark.mpl_image_compare
def test_xlma_plot():
    dataset = xr.open_dataset('tests/truth/lma_netcdf/lma.nc')
    start_time = dt.datetime(2023, 12, 24, 0, 57, 0, 0)
    print(pd.Series(dataset.event_time.data.flatten()))
    xlma = XlmaPlot(dataset, start_time, True, True, 'xarray', False, xlim=[-103.5, -99.5], ylim=[31.5, 35.5], zlim=[0, 20], cmap='rainbow', chi2=1)
    return xlma.fig