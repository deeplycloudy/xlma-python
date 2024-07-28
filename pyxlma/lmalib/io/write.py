import numpy as np
import datetime as dt
import warnings

def dataset(dataset, path):
    "Write an LMA dataset to a netCDF file"

    if 'flash_event_count' in dataset.data_vars:
        if np.all(dataset.flash_event_count == np.iinfo(np.uint32).max):
            dataset = dataset.drop_vars(['flash_init_latitude', 'flash_init_longitude', 'flash_init_altitude', 'flash_area', 'flash_volume', 'flash_energy', 'flash_center_latitude', 'flash_center_longitude', 'flash_center_altitude', 'flash_power', 'flash_event_count', 'flash_duration_threshold', 'flash_time_start', 'flash_time_end', 'flash_duration'])
    
    for var in dataset.data_vars:
        if np.all(dataset[var].data == np.nan):
            dataset = dataset.drop_vars(var)
    for attr in dataset.attrs:
        if type(dataset.attrs[attr]) == dt.datetime:
            dataset.attrs[attr] = dataset.attrs[attr].strftime('%Y-%m-%d %H:%M:%S.%f')
    dataset.attrs.pop('config_times', None)
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in dataset.data_vars}
    dataset = dataset.chunk('auto')
    dataset.to_netcdf(path, encoding=encoding)

def lma_dat_file(dataset, path, use_gzip=True):
    """
    Write an LMA dataset to a legacy .dat(.gz) file for use in IDL XLMA.

    Parameters
    ----------
    dataset : xarray.Dataset
        The LMA dataset to write.
    path : str
        The target path to the file to write.
    use_gzip : bool
        Whether to compress the file using gzip.
    """

    if 'network_configurations' in dataset.dims:
        warnings.warn('Attempting to write dataset with multiple network configurations to a single file. This is highly experimental.\n'
                      'Caveats:\n-Location names will be appended and separated by semicolon\n- The center of the network is derived from the mean of the locations of the stations')
        center_lon = dataset.station_longitude.data.mean()
        center_lat = dataset.station_latitude.data.mean()
        center_alt = dataset.station_altitude.data.mean()
        num_active = np.sum(np.clip(np.sum(dataset.station_active.data, axis=0), 0, 1))
    else:
        center_lon = dataset.network_center_longitude.data.item()
        center_lat = dataset.network_center_latitude.data.item()
        center_alt = dataset.network_center_altitude.data.item()
        num_active = np.sum(dataset.station_active.data)
    lma_header = (f'Lightning Mapping Array analyzed data\n'
                  f'Analysis program: {dataset.attrs["event_algorithm_name"]}\n'
                  f'Analysis program version: {dataset.attrs["event_algorithm_version"]}\n'
                  f'File created: {dt.datetime.utcnow().strftime("%a %b %d %H:%M:%S %Y")}\n'
                  f'Data start time: {dataset.attrs["analysis_start_time"].strftime("%M/%d/%y %H:%M:%S")}\n'
                  f'Number of seconds analyzed: {dataset.attrs["number_of_seconds_analyzed"]}\n'
                  f'Location: {"; ".join(np.unique(dataset.station_network.data).astype(str).tolist())}\n'
                  f'Coordinate center (lat,lon,alt): {center_lat:.7f}, {center_lon:.7f}, {center_alt:.2f}\n'
                  f'Coordinate frame: cartesian\n'
                #   f'Maximum diameter of LMA (km): '
                #   f'Maximum light-time across LMA (ns): '
                  f'Number of stations: {dataset.sizes['number_of_stations']}'
                  f'Number of active stations: {num_active}\n'
                #   f'Active stations:'
                #   f'Minimum number of stations per solution: {}\n'
                #   f'Maximum reduced chi-squared: {}\n'
                #   f'Maximum number of chi-squared iterations: {}\n'
                  ### STATION INFO TABLE
                  ### STATION DATA TABLE
                  f'Metric file version: 4\n'
                  f'Data: time (UT sec of day), lat, lon, alt(m), reduced chi^2, P(dBW), mask\n'
                  f'Data format: 15.9f 12.8f 13.8f 9.2f 6.2f 5.1f 7x\n'
                  f'Number of events: {dataset.sizes["number_of_events"]}\n'
                  f'*** data ***\n'
    )