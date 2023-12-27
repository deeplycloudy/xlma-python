import xarray as xr
from pyxlma.lmalib.grid import *
from tests.test_flash import compare_dataarrays


def test_create_regular_grid():
    """Test creation of regular grid"""
    dataset = xr.open_dataset('tests/truth/lma_netcdf/lma_stats.nc')
    grid_range = 0.5
    grid_h_res = 0.1
    grid_height = 20
    grid_v_res = 1
    lon_range = (dataset.network_center_longitude - grid_range, dataset.network_center_longitude + grid_range, grid_h_res)
    lat_range = (dataset.network_center_latitude - grid_range, dataset.network_center_latitude + grid_range, grid_h_res)
    alt_range = (0, grid_height, grid_v_res)
    time_range = (dataset.event_time.data.min(), dataset.event_time.data.max(), np.timedelta64(1, 'm'))
    grid_edge_ranges ={
        'grid_longitude_edge':lon_range,
        'grid_latitude_edge':lat_range,
        'grid_altitude_edge':alt_range,
        'grid_time_edge':time_range,
    }
    grid_center_names ={
        'grid_longitude_edge':'grid_longitude',
        'grid_latitude_edge':'grid_latitude',
        'grid_altitude_edge':'grid_altitude',
        'grid_time_edge':'grid_time',
    }
    empty_grid = create_regular_grid(grid_edge_ranges, grid_center_names)
    xr.testing.assert_equal(empty_grid, xr.open_dataset('tests/truth/lma_netcdf/empty_grid.nc'))

def test_assign_regular_bins():
    """Test assigning lightning data to regular bins"""
    empty_grid = xr.open_dataset('tests/truth/lma_netcdf/empty_grid.nc')
    dataset = xr.open_dataset('tests/truth/lma_netcdf/lma_stats.nc')
    event_coord_names = {
                'event_longitude':'grid_longitude_edge',
                'event_latitude':'grid_latitude_edge',
                'event_altitude':'grid_altitude_edge',
                'event_time':'grid_time_edge',
            }
    binned_grid = assign_regular_bins(empty_grid, dataset, event_coord_names, append_indices=True)
    truth = xr.open_dataset('tests/truth/lma_netcdf/binned_grid.nc')

    for var in truth.data_vars:
        compare_dataarrays(binned_grid, truth, var)


def test_events_to_grid():
    """Test gridding lightning data"""
    empty_grid = xr.open_dataset('tests/truth/lma_netcdf/empty_grid.nc')
    dataset = xr.open_dataset('tests/truth/lma_netcdf/lma_stats.nc')
    event_coord_names = {
                'event_longitude':'grid_longitude_edge',
                'event_latitude':'grid_latitude_edge',
                'event_altitude':'grid_altitude_edge',
                'event_time':'grid_time_edge',
            }
    binned_grid = assign_regular_bins(empty_grid, dataset, event_coord_names, append_indices=True)

    grid_spatial_coords=('grid_time', 'grid_altitude', 'grid_latitude', 'grid_longitude')
    event_spatial_vars = ('event_time', 'event_altitude', 'event_latitude', 'event_longitude')

    gridded_lma = events_to_grid(binned_grid, empty_grid, pixel_id_var='pixel_id', event_spatial_vars=event_spatial_vars, grid_spatial_coords=grid_spatial_coords)

    truth = xr.open_dataset('tests/truth/lma_netcdf/gridded_lma.nc')

    for var in truth.data_vars:
        compare_dataarrays(gridded_lma, truth, var)