from pyxlma.plot.xlma_plot_feature import *
import xarray as xr
import numpy as np
from datetime import datetime as dt

def test_subset():
    lma = xr.open_dataset('tests/truth/lma_netcdf/lma.nc')
    lon_subset, lat_subset, alt_subset, time_subset, selection = subset(lma.event_longitude.data, lma.event_latitude.data, lma.event_altitude.data,
                                                                    lma.event_time.data, lma.event_chi2.data, lma.event_stations.data,
                                                                    (-101.7, -101.4), (33.4, 34.8), (0, 3000), 
                                                                    (np.datetime64('2023-12-24T00:57:00'), np.datetime64('2023-12-24T00:57:10')), 1, 8)
    assert np.allclose(lon_subset, np.array([-101.59106, -101.598236, -101.59939, -101.59875, -101.601425, -101.60555, -101.60554, -101.60838, -101.60368, -101.62052]))
    assert np.allclose(lat_subset, np.array([33.68953, 33.684334, 33.68242, 33.67924, 33.678104, 33.676983, 33.675335, 33.677456, 33.67102, 33.666958]))
    assert np.allclose(alt_subset, np.array([2974.67, 2986.99, 2936.03, 2920.17, 2797.01, 2933.09, 2659.97, 2886.72, 2716.22, 2943.72]))
    assert np.allclose(time_subset.astype(float), np.array(['2023-12-24T00:57:07.731986515', '2023-12-24T00:57:07.800978747', '2023-12-24T00:57:07.803362858', '2023-12-24T00:57:07.805963963',
                                              '2023-12-24T00:57:07.806720943', '2023-12-24T00:57:07.809493631', '2023-12-24T00:57:07.810448100', '2023-12-24T00:57:07.811465266',
                                              '2023-12-24T00:57:07.814960674', '2023-12-24T00:57:07.826344209']).astype(np.datetime64).astype(float))
    assert np.sum(selection) == 10

def test_color_by_time_datetime_nolimit():
    some_datetimes = np.array([dt(2021, 4, 9, 1, 51, 0), dt(2021, 4, 9, 1, 52, 0), dt(2021, 4, 9, 1, 53, 0), dt(2021, 4, 9, 1, 54, 0), dt(2021, 4, 9, 1, 59, 0)])
    vmin, vmax, colors = color_by_time(some_datetimes)
    assert vmin == 0
    assert vmax == 480
    assert np.allclose(colors, np.array([0, 60, 120, 180, 480]))


def test_color_by_time_datetime_limit():
    some_datetimes = np.array([dt(2021, 4, 9, 1, 51, 0), dt(2021, 4, 9, 1, 52, 0), dt(2021, 4, 9, 1, 53, 0), dt(2021, 4, 9, 1, 54, 0), dt(2021, 4, 9, 1, 59, 0)])
    limits = [dt(2021, 4, 9, 1, 50, 0), dt(2021, 4, 9, 2, 0, 0)]
    vmin, vmax, colors = color_by_time(some_datetimes, limits)
    assert vmin == 0
    assert vmax == 600
    assert np.allclose(colors, np.array([60, 120, 180, 240, 540]))

def test_color_by_time_xarray():
    dataset = xr.open_dataset('tests/truth/lma_netcdf/lma.nc')
    vmin, vmax, colors = color_by_time(dataset.event_time)
    assert vmin == 0
    assert np.isclose(vmax, 57.943683385849)
    assert np.isclose(np.mean(colors), 30.483982899376258)
    assert np.isclose(np.std(colors), 17.25687093241869)

def test_setup_hist():
    lma = xr.open_dataset('tests/truth/lma_netcdf/lma.nc')
    alt_lon, alt_lat, alt_time, lat_lon = setup_hist(lma.event_longitude.data, lma.event_latitude.data, lma.event_altitude.data,
                                                     lma.event_time.data, 2, 2, 2, 2)
    
    assert np.allclose(alt_lon, np.array([[21082, 1], [0, 1]]))
    assert np.allclose(alt_lat, np.array([[5, 21077], [1, 1]]))
    assert np.allclose(alt_time, np.array([[9991, 1], [11091, 1]]))
    assert np.allclose(lat_lon, np.array([[6, 21077], [0, 1]]))
