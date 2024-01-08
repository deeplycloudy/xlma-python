import pytest
import xarray as xr
import datetime as dt
from pyxlma.plot.interactive import *

@pytest.mark.mpl_image_compare
def test_interactive():
    dataset = xr.open_dataset('tests/truth/lma_netcdf/lma.nc')
    buffer = 2.5
    xlim = (dataset.network_center_longitude.data-buffer, dataset.network_center_longitude.data+buffer)
    ylim = (dataset.network_center_latitude.data-buffer, dataset.network_center_latitude.data+buffer)
    zlim = (0, 20)
    interact = InteractiveLMAPlot(dataset, clon=dataset.network_center_longitude, clat=dataset.network_center_latitude, xlim=xlim, ylim=ylim, zlim=zlim, plot_cmap='rainbow')
    return interact.lma_plot.fig
