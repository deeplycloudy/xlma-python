#!/usr/bin/env python3
# Example script for clustering and gridding flash data with pyxlma
# Created 5 September 2023 by Sam Gardner <samuel.gardner@ttu.edu>

import sys
import glob
from os import path
from datetime import datetime as dt
from pathlib import Path
import xarray as xr
import numpy as np
from pyxlma.lmalib.io import read as lma_read
from pyxlma.lmalib.flash.cluster import cluster_flashes
from pyxlma.lmalib.flash.properties import flash_stats, filter_flashes
from pyxlma.lmalib.grid import create_regular_grid, assign_regular_bins, events_to_grid


# Parameters
grid_range = 200 # km from LMA center to edge of grid
grid_height = 20 # km from LMA center to top of grid
grid_h_res = 0.5 # km, horizontal grid resolution
grid_v_res = 1 # km, vertical grid resolution
grid_time_res = 1 # minutes, time resolution of grid
max_chi2 = 5 # all VHF sources with a chi2 greater than this will be discarded
min_events_per_flash = 10 # all flashes with fewer VHF sources will be discarded


def flash_sort_grid(paths_to_read, output_path):

    print('Reading data...')
    dataset, start_time = lma_read.dataset(paths_to_read)
    end_time = (np.max(dataset.event_time.data).astype('datetime64[m]')+np.timedelta64(1,'m')).astype('datetime64[ns]')

    # We want to bin the data into 10-minute intervals of the hour
    bin_start = np.datetime64(start_time.replace(minute=(start_time.minute // 10)*10, second=0, microsecond=0))

    while bin_start < end_time:
        print('Processing: '+bin_start.item().strftime('%Y-%m-%d %H:%M'))
        variables_to_filter = [var for var in dataset.data_vars if 'number_of_events' in dataset[var].dims]
        filtered = dataset[variables_to_filter].where(dataset.event_chi2 <= max_chi2, drop=True)
        dataset_filtered = xr.merge([dataset.drop(variables_to_filter), filtered])

        clustered_lma = filter_flashes(flash_stats(cluster_flashes(dataset_filtered)), flash_event_count=(min_events_per_flash, None))
        x_range = (grid_range*-1000, grid_range*1000, grid_h_res*1000)
        y_range = (grid_range*-1000, grid_range*1000, grid_h_res*1000)
        z_range = (0, grid_height*1000, grid_v_res*1000)
        time_range = (bin_start, end_time, np.timedelta64(grid_time_res, 'm'))

        grid_edge_ranges ={
            'grid_x_edge':x_range,
            'grid_y_edge':y_range,
            'grid_altitude_edge':z_range,
            'grid_time_edge':time_range,
        }
        grid_center_names ={
            'grid_x_edge':'grid_x',
            'grid_y_edge':'grid_y',
            'grid_altitude_edge':'grid_altitude',
            'grid_time_edge':'grid_time',
        }
        event_coord_names = {
            'event_x':'grid_x_edge',
            'event_y':'grid_y_edge',
            'event_altitude':'grid_altitude_edge',
            'event_time':'grid_time_edge',
        }
        empty_grid = create_regular_grid(grid_edge_ranges, grid_center_names)
        binned_events = assign_regular_bins(empty_grid, clustered_lma, event_coord_names, append_indices=True)
        grid_spatial_coords=('grid_time', 'grid_altitude', 'grid_y', 'grid_x')
        event_spatial_vars = ('event_time', 'event_altitude', 'event_y', 'event_x')
        gridded_lma = events_to_grid(binned_events, empty_grid, pixel_id_var='pixel_id', event_spatial_vars=event_spatial_vars, grid_spatial_coords=grid_spatial_coords)

        combined = xr.combine_by_coords((gridded_lma, clustered_lma))
        Re = 6378137 # m, radius of Earth
        lon_center = combined.network_center_longitude
        lat_center = combined.network_center_latitude
        combined['grid_latitude_edge'] = np.rad2deg(combined.grid_y_edge/(Re)+np.deg2rad(lat_center))
        combined['grid_longitude_edge'] = np.rad2deg((combined.grid_x_edge)/(Re*np.cos(np.deg2rad(combined.grid_latitude_edge.data)))+np.deg2rad(lon_center))
        combined['grid_latitude'] = np.rad2deg(combined.grid_y/(Re)+np.deg2rad(lat_center))
        combined['grid_longitude'] = np.rad2deg((combined.grid_x)/(Re*np.cos(np.deg2rad(combined.grid_latitude.data)))+np.deg2rad(lon_center))
        combined['flash_center_x'] = Re*(np.deg2rad(combined.flash_center_longitude)-np.deg2rad(lon_center))*np.cos(np.deg2rad(combined.flash_center_latitude))
        combined['flash_center_y'] = Re*(np.deg2rad(combined.flash_center_latitude)-np.deg2rad(lat_center))
        combined = combined.swap_dims({'grid_x_edge':'grid_longitude_edge', 'grid_y_edge':'grid_latitude_edge', 'grid_x':'grid_longitude', 'grid_y':'grid_latitude'})

        filename = path.basename(paths_to_read[0]).split('_')[0] + '_' + bin_start.item().strftime('%Y%m%d_%H%M') + '00_0600_map' + str(int(grid_h_res*1000)) + 'm.nc'
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in combined.data_vars}
        combined.to_netcdf(path.join(output_path, filename), encoding=encoding)
        bin_start = bin_start + np.timedelta64(10, 'm')



if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python3 pyxlma_flash_sort_grid.py <input_paths> <output_path>')
        exit(1)
    input_paths = sys.argv[1:-1]
    if len(input_paths) == 1:
        input_paths = glob.glob(input_paths[0])
    out_path = sys.argv[-1]
    if not path.exists(out_path):
        Path(out_path).mkdir(parents=True, exist_ok=True)
    if not path.isdir(out_path):
        print('Error: output path exists and is not a directory')
        exit(1)
    griddedData = flash_sort_grid(input_paths, out_path)
