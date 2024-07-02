"""Test functionality of pyxlma.lmalib.flash"""

import xarray as xr
import numpy as np
import pytest
from pyxlma.lmalib.flash.cluster import cluster_flashes
from pyxlma.lmalib.flash.properties import *


def compare_dataarrays(tocheck, truth, var, rtol=1.e-5, atol=1.e-8):
    """Compare two dataarrays"""
    if truth[var].data.dtype == 'datetime64[ns]' or truth[var].data.dtype == 'timedelta64[ns]':
        if tocheck[var].data.dtype == 'float64':
            truth[var].data = truth[var].data.astype(float)/1e9
        np.testing.assert_allclose(tocheck[var].data.astype(float), truth[var].data.astype(float), rtol=rtol, atol=atol, equal_nan=True)
    else:
        np.testing.assert_allclose(tocheck[var].data, truth[var].data, rtol=rtol, atol=atol, equal_nan=True)


def test_cluster_flashes():
    """Test clustering of flashes"""
    dataset = xr.open_dataset('tests/truth/lma_netcdf/lma.nc')
    clustered = cluster_flashes(dataset)
    truth = xr.open_dataset('tests/truth/lma_netcdf/lma_clustered.nc')
    for var in truth.data_vars:
        compare_dataarrays(clustered, truth, var)


def test_flash_stats():
    """Test calculation of flash statistics"""
    dataset = xr.open_dataset('tests/truth/lma_netcdf/lma.nc')
    stats = flash_stats(cluster_flashes(dataset))
    truth = xr.open_dataset('tests/truth/lma_netcdf/lma_stats.nc')
    for var in truth.data_vars:
        compare_dataarrays(stats, truth, var)


def test_filter_flashes():
    """Test filtering of flashes"""
    dataset = xr.open_dataset('tests/truth/lma_netcdf/lma_stats.nc')
    filtered = filter_flashes(dataset, flash_event_count=(100, 500))
    assert np.min(filtered.flash_event_count.data) >= 100
    assert np.max(filtered.flash_event_count.data) <= 500

    truth = xr.open_dataset('tests/truth/lma_netcdf/lma_filtered.nc')

    for var in truth.data_vars:
        compare_dataarrays(filtered, truth, var)


def test_flilter_flashes_no_prune():
    dataset = xr.open_dataset('tests/truth/lma_netcdf/lma_stats.nc')
    filtered = filter_flashes(dataset, flash_event_count=(100, 500), prune=False)
    assert np.all(filtered.event_id.data == dataset.event_id.data)
    assert np.min(filtered.flash_event_count.data) >= 100
    assert np.max(filtered.flash_event_count.data) <= 500


def test_filter_no_stats():
    """"Test filtering of flashes without flash statistics"""
    dataset = xr.open_dataset('tests/truth/lma_netcdf/lma.nc')
    dataset = cluster_flashes(dataset)
    with pytest.raises(ValueError, match='Before filtering a dataset by flash properties, call flash_stats on the dataset to compute flash properties.'):
        filter_flashes(dataset, flash_event_count=(100, 500))


def test_event_area():
    dataset = xr.open_dataset('tests/truth/lma_netcdf/lma.nc')
    x, y, z = local_cartesian(dataset.event_longitude.data, dataset.event_latitude.data, dataset.event_altitude.data, 
                              dataset.network_center_longitude.data, dataset.network_center_latitude.data, dataset.network_center_altitude.data)
    assert np.isclose(event_hull_area(x, y, z), 5491450433206.501)


def test_event_volume():
    dataset = xr.open_dataset('tests/truth/lma_netcdf/lma.nc')
    x, y, z = local_cartesian(dataset.event_longitude.data, dataset.event_latitude.data, dataset.event_altitude.data, 
                              dataset.network_center_longitude.data, dataset.network_center_latitude.data, dataset.network_center_altitude.data)
    assert np.isclose(event_hull_volume(x[0:10], y[0:10], z[0:10]), 56753729942.624825)
    