from glob import glob
import xarray as xr
from pyxlma.xarray_util import *

def test_get_1d_dims():
    lma = xr.open_dataset('tests/truth/lma_netcdf/lma_stats.nc')
    dims_1d = get_1d_dims(lma)
    print(lma)
    assert dims_1d == ['number_of_flashes']


def test_gen_1d_datasets():
    lma = xr.open_dataset('tests/truth/lma_netcdf/lma_stats.nc')
    datasets_1d = list(gen_1d_datasets(lma))
    assert len(datasets_1d) == 1
    assert list(datasets_1d[0].dims.keys()) == ['number_of_flashes']


def test_get_1d_datasets():
    lma = xr.open_dataset('tests/truth/lma_netcdf/lma_stats.nc')
    datasets_1d = get_1d_datasets(lma)
    assert len(datasets_1d) == 1
    assert list(datasets_1d[0].dims.keys()) == ['number_of_flashes']


def test_get_scalar_vars():
    lma = xr.open_dataset('tests/truth/lma_netcdf/lma_stats.nc')
    scalar_vars = get_scalar_vars(lma)
    assert scalar_vars == ['network_center_latitude', 'network_center_longitude', 'network_center_altitude',
                           'flash_distance_separation_threshold', 'flash_time_separation_threshold']


def test_concat_1d_dims_no_scalars():
    glm_datasets = [xr.open_dataset(path) for path in glob('examples/network_samples/OR_GLM*.nc')]
    concatenated = concat_1d_dims(glm_datasets)
    assert dict(concatenated.dims) == {'number_of_events': 29664, 'number_of_groups': 10420, 'number_of_flashes': 606,
                                       'number_of_time_bounds': 6, 'number_of_wavelength_bounds': 6, 'number_of_field_of_view_bounds': 6}


def test_concat_1d_dims_scalars():
    glm_datasets = [xr.open_dataset(path) for path in glob('examples/network_samples/OR_GLM*.nc')]
    concatenated = concat_1d_dims(glm_datasets, stack_scalars='scalars')
    assert dict(concatenated.dims) == {'number_of_events': 29664, 'number_of_groups': 10420, 'number_of_flashes': 606, 'number_of_time_bounds': 6,
                                       'number_of_wavelength_bounds': 6, 'number_of_field_of_view_bounds': 6, 'scalars': 3}


if __name__ == "__main__":
    test_get_1d_dims()
    test_gen_1d_datasets()
    test_get_1d_datasets()
    test_get_scalar_vars()
    test_concat_1d_dims_no_scalars()
    test_concat_1d_dims_scalars()