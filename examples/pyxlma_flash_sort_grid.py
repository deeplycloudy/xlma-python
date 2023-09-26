#!/usr/bin/env python3
# Example script for clustering and gridding flash data with pyxlma
# Created 5 September 2023 by Sam Gardner <samuel.gardner@ttu.edu>

import argparse
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


def flash_sort_grid(paths_to_read, grid_proj, grid_range, grid_height, grid_h_res, grid_v_res, grid_time_res, max_chi2, min_events_per_flash, output_path):

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
        time_range = (bin_start, end_time, np.timedelta64(grid_time_res, 'm'))
        if grid_proj == 'xyz':
            x_range = (grid_range*-1000, grid_range*1000, grid_h_res*1000)
            y_range = (grid_range*-1000, grid_range*1000, grid_h_res*1000)
            z_range = (0, grid_height*1000, grid_v_res*1000)
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
            grid_spatial_coords=('grid_time', 'grid_altitude', 'grid_y', 'grid_x')
            event_spatial_vars = ('event_time', 'event_altitude', 'event_y', 'event_x')
        elif grid_proj == 'lla':
            lon_range = (dataset.network_center_longitude - grid_range, dataset.network_center_longitude + grid_range, grid_h_res)
            lat_range = (dataset.network_center_latitude - grid_range, dataset.network_center_latitude + grid_range, grid_h_res)
            alt_range = (0, grid_height, grid_v_res)
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
            event_coord_names = {
                'event_longitude':'grid_longitude_edge',
                'event_latitude':'grid_latitude_edge',
                'event_altitude':'grid_altitude_edge',
                'event_time':'grid_time_edge',
            }
            grid_spatial_coords=('grid_time', 'grid_altitude', 'grid_latitude', 'grid_longitude')
            event_spatial_vars = ('event_time', 'event_altitude', 'event_latitude', 'event_longitude')
        else:
            raise ValueError('grid_proj must be \'xyz\' or \'lla\'')
        empty_grid = create_regular_grid(grid_edge_ranges, grid_center_names)
        binned_events = assign_regular_bins(empty_grid, clustered_lma, event_coord_names, append_indices=True)
        gridded_lma = events_to_grid(binned_events, empty_grid, pixel_id_var='pixel_id', event_spatial_vars=event_spatial_vars, grid_spatial_coords=grid_spatial_coords)
        combined = xr.combine_by_coords((gridded_lma, clustered_lma))

        filename = path.basename(paths_to_read[0]).split('_')[0] + '_' + bin_start.item().strftime('%Y%m%d_%H%M') + '00_0600_map' + str(int(grid_h_res*1000)) + 'm.nc'
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in combined.data_vars}
        combined.to_netcdf(path.join(output_path, filename), encoding=encoding)
        bin_start = bin_start + np.timedelta64(10, 'm')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster VHF source data into flashes and create gridded data products, similar to lmatools flash_sort_and_grid.py\nOutputs to netCDF files, one per 10-minute interval.')
    parser.add_argument('-i', '--input', nargs='+', required=True, help='Path or paths to LMA .dat files to process.')
    parser.add_argument('--grid-projection', help='Whether to use a grid based on meters from the LMA or lat/lon degrees. Specify \'xyz\' (meters from center) or \'lla\' (lat/lon/alt). Default is xyz.', choices=['xyz', 'lla'], default='xyz')
    parser.add_argument('--horizontal-range', help='Distance from the LMA center to the edge of the grid, in units of kilometers or degrees. Default is 0.25km or 0.25°.', type=float, default=0.25)
    parser.add_argument('--vertical-range', help='Distance from the LMA center to the top of the grid, in units of kilometers. Default is 20km.', type=float, default=20)
    parser.add_argument('--horizontal-res', help='Horizontal gridded product resolution, in units of kilometers or degrees. Default is 0.5km.', type=float, default=0.5)
    parser.add_argument('--vertical-res', help='Vertical gridded product resolution, in units of kilometers. Default is 1km.', type=float, default=1)
    parser.add_argument('--time-res', help='Time resolution of gridded products, in units of minutes. Default is 1 minute.', type=float, default=1)
    parser.add_argument('--max-chi2', help='Maximum chi2 value for a VHF source to be included in a flash. Default is 5.', type=float, default=5)
    parser.add_argument('--min-events', help='Minimum number of VHF sources per flash. Flashes with fewer sources will be discarded. Default is 10.', type=int, default=10)
    parser.add_argument('--output', help='Path to write 10-minute gridded data files to.', required=True)
    args = parser.parse_args()
    input_paths = args.input
    if len(input_paths) == 1:
        input_paths = glob.glob(input_paths[0])
    else:
        input_paths = sorted(input_paths)
    grid_proj = args.grid_projection
    grid_range = args.horizontal_range
    grid_height = args.vertical_range
    grid_h_res = args.horizontal_res
    grid_v_res = args.vertical_res
    grid_time_res = args.time_res
    max_chi2 = args.max_chi2
    min_events_per_flash = args.min_events
    out_path = args.output
    if not path.exists(out_path):
        Path(out_path).mkdir(parents=True, exist_ok=True)
    if not path.isdir(out_path):
        print('Error: output path exists and is not a directory')
        exit(1)
    griddedData = flash_sort_grid(input_paths, grid_proj, grid_range, grid_height, grid_h_res, grid_v_res, grid_time_res, max_chi2, min_events_per_flash, out_path)
