import importlib.metadata
import numpy as np
import datetime as dt
import warnings
import importlib
from functools import reduce
import gzip

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


def count_decimals_in_array(a):
    a = np.array(a).astype(str)
    # Find the position of the decimal point in each string
    decimal_positions = np.char.find(a, '.')

    # Handle the case where there are no decimal points
    decimal_positions[decimal_positions == -1] = np.char.str_len(a[decimal_positions == -1])

    # Calculate the number of characters after the decimal point
    decimal_counts = np.char.str_len(a) - decimal_positions - 1

    return np.max(decimal_counts)


def create_station_info_string(dataset):
    """
    Create a string representation of the station information table for an LMA dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        The LMA dataset to create the station information table for.

    Returns
    -------
    str
        The string representation of the station information table.
    """
    if 'number_of_stations' not in dataset.dims:
        raise ValueError('Dataset does not contain station information.')
    header_line = 'Station information: '
    info_to_include = []
    if 'station_code' in dataset.data_vars:
        header_line += 'id, '
        info_to_include.append('station_code')
    else:
        raise ValueError('Stations do not have letter codes. This is incompatible with the XLMA dat format, please assign station codes to all stations.')
    
    if 'station_name' in dataset.data_vars:
        header_line += 'name, '
        info_to_include.append('station_name')
    else:
        raise ValueError('Stations do not have names. This is incompatible with the XLMA dat format, please assign names to all stations.')
    
    if 'station_latitude' in dataset.data_vars:
        header_line += 'lat(d), '
        info_to_include.append('station_latitude')
    else:
        raise ValueError('Stations do not have latitudes. This is incompatible with the XLMA dat format, please assign latitudes to all stations.')
    
    if 'station_longitude' in dataset.data_vars:
        header_line += 'lon(d), '
        info_to_include.append('station_longitude')
    else:
        raise ValueError('Stations do not have longitudes. This is incompatible with the XLMA dat format, please assign longitudes to all stations.')
    
    if 'station_altitude' in dataset.data_vars:
        header_line += 'alt(m), '
        info_to_include.append('station_altitude')
    else:
        raise ValueError('Stations do not have altitudes. This is incompatible with the XLMA dat format, please assign altitudes to all stations.')
    
    if 'station_delay' in dataset.data_vars:
        header_line += 'delay(ns), '
        info_to_include.append('station_delay')
        # This is optional
    
    if 'station_board_revision' in dataset.data_vars:
        header_line += 'board_rev, '
        info_to_include.append('station_board_revision')
    else:
        raise ValueError('Stations do not have board revisions. This is incompatible with the XLMA dat format, please assign board revisions to all stations.')
    
    if 'station_receive_channels' in dataset.data_vars:
        header_line += 'rec_ch, '
        info_to_include.append('station_receive_channels')
        # This is optional
    header_line = header_line[:-2]
    stations_strings = []
    for station_num in range(dataset.sizes['number_of_stations']):
        this_station = dataset.isel(number_of_stations=station_num)
        station_string = 'Sta_info: '
        for var in info_to_include:
            station_string += f'{this_station[var].data.astype(str).item()}  '
        stations_strings.append(station_string)
    return header_line+'\n'+'\n'.join(stations_strings)


def create_station_data_string(dataset):
    pass


def create_event_data_string(dataset):
    if 'analysis_start_time' in dataset.attrs:
        start_date = dataset.attrs['analysis_start_time']
        start_date = np.array([start_date]).astype('datetime64[D]')
    else:
        start_date = dataset.event_time.data.min().astype('datetime64[D]')
    columns_line = 'Data: '
    format_line = 'Data format: '
    data_to_combine = []

    day_sec = (dataset.event_time - start_date).data.astype('timedelta64[ns]').astype(np.float64)/1e9
    day_sec_str = day_sec.astype(str)
    day_sec_length = np.max(np.char.str_len(day_sec_str))
    day_sec_str = np.char.ljust(day_sec_str, day_sec_length, fillchar='0')
    columns_line += 'time (UT sec of day), '
    format_line += f'{day_sec_length}.9f, ' #time is always in nanoseconds
    data_to_combine.append(day_sec_str.T)

    lat = dataset.event_latitude.data.astype(str)
    lat_length = np.max(np.char.str_len(lat))
    lat = np.char.ljust(lat, lat_length, fillchar='0')
    columns_line += 'lat, '
    lat_dec_count = count_decimals_in_array(lat)
    format_line += f'{lat_length}.{lat_dec_count}f, '
    data_to_combine.append(lat.T)

    lon = dataset.event_longitude.data.astype(str)
    lon_length = np.max(np.char.str_len(lon))
    lon = np.char.ljust(lon, lon_length, fillchar='0')
    columns_line += 'lon, '
    lon_dec_count = count_decimals_in_array(lon)
    format_line += f'{lon_length}.{lon_dec_count}f, '
    data_to_combine.append(lon.T)

    alt = dataset.event_altitude.data.astype(str)
    alt_length = np.max(np.char.str_len(alt))
    alt = np.char.ljust(alt, alt_length, fillchar='0')
    columns_line += 'alt(m), '
    alt_dec_count = count_decimals_in_array(alt)
    format_line += f'{alt_length}.{alt_dec_count}f, '
    data_to_combine.append(alt.T)

    chi2 = dataset.event_chi2.data.astype(str)
    chi2_length = np.max(np.char.str_len(chi2))
    chi2 = np.char.ljust(chi2, chi2_length, fillchar='0')
    columns_line += 'reduced chi^2, '
    chi2_dec_count = count_decimals_in_array(chi2)
    format_line += f'{chi2_length}.{chi2_dec_count}f, '
    data_to_combine.append(chi2.T)

    power = dataset.event_power.data.astype(str)
    power_length = np.max(np.char.str_len(power))
    power = np.char.ljust(power, power_length, fillchar='0')
    columns_line += 'P(dBW), '
    power_dec_count = count_decimals_in_array(power)
    format_line += f'{power_length}.{power_dec_count}f, '
    data_to_combine.append(power.T)

    masks = np.vectorize(np.base_repr)(dataset.event_mask.data, base=16)
    mask_len = np.max(np.char.str_len(masks))
    masks = np.char.rjust(masks, mask_len, fillchar='0')
    masks = '0x' + masks
    mask_len += 2
    columns_line += 'mask, '
    format_line += f'{mask_len}x, '
    data_to_combine.append(masks.T)

    columns_line = columns_line[:-2]
    format_line = format_line[:-2]
    N_ev_line = f'Number of events: {dataset.sizes["number_of_events"]}'
    data_line = '*** data ***'
    data_str = '\n'.join(reduce(lambda x, y:np.char.add(np.char.add(x, '  '), y), data_to_combine))
    return columns_line+'\n'+format_line+'\n'+N_ev_line+'\n'+data_line+'\n'+data_str


    
    

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
        stations_active = np.sum(dataset.station_active.data, axis=0)
        num_active = np.sum(np.clip(stations_active, 0, 1))
        stations_active = ' '.join(np.extract(stations_active, dataset.station_code.data).astype(str).tolist())
    else:
        num_active = np.sum(dataset.station_active.data)
        stations_active = ' '.join(np.extract(dataset.station_active.data, dataset.station_code.data).astype(str).tolist())
    
    center_lon = dataset.network_center_longitude.data.item()
    center_lat = dataset.network_center_latitude.data.item()
    center_alt = dataset.network_center_altitude.data.item()
    
    unique_codes, code_counts = np.unique(dataset.station_code.data, return_counts=True)
    if np.max(code_counts) > 1:
        raise ValueError('Duplicate station codes found in dataset. This is incompatible with the XLMA dat format, please rename conflicting station letters.')
    if b'' in unique_codes:
        raise ValueError('Empty station codes found in dataset. This is incompatible with the XLMA dat format, please assign station codes to all stations.')
    
    if dataset.attrs["event_algorithm_name"].startswith('pyxlma'):
        analysis_program = dataset.attrs["event_algorithm_name"]
    else:
        analysis_program = 'pyxlma; '+dataset.attrs["event_algorithm_name"]
    if dataset.attrs["event_algorithm_version"].startswith('pyxlma'):
        analysis_program_version = dataset.attrs["event_algorithm_version"]
    else:
        analysis_program_version = f'pyxlma-{importlib.metadata.version("pyxlma")}; {dataset.attrs["event_algorithm_version"]}'

    lma_file = (f'pyxlma exported data -- https://github.com/deeplycloudy/xlma-python\n'
                  f'Analysis program: {analysis_program}\n'
                  f'Analysis program version: {analysis_program_version}\n'
                  f'File created: {dt.datetime.utcnow().strftime("%a %b %d %H:%M:%S %Y")}\n'
                  f'Data start time: {dataset.attrs["analysis_start_time"].strftime("%m/%d/%y %H:%M:%S")}\n'
                  f'Number of seconds analyzed: {(dataset.attrs["analysis_end_time"] - dataset.attrs["analysis_start_time"]).total_seconds()}\n'
                  f'Location: {"; ".join(np.unique(dataset.station_network.data).astype(str).tolist())}\n'
                  f'Coordinate center (lat,lon,alt): {center_lat:.7f}, {center_lon:.7f}, {center_alt:.2f}\n'
                  f'Coordinate frame: cartesian\n'
                  f'Number of stations: {dataset.sizes['number_of_stations']}\n'
                  f'Number of active stations: {num_active}\n'
                  f'Active stations: {stations_active}\n'
                  f'Minimum number of stations per solution: {dataset.attrs["min_stations"]}\n'
                  f'Maximum reduced chi-squared: {dataset.attrs["max_chi2"]}\n'
                  f'Maximum number of chi-squared iterations: {dataset.attrs["max_chi2_iterations"]}\n'
                  f'{create_station_info_string(dataset)}\n'
                  f'Station mask order: {dataset.attrs["station_mask_order"]}\n'
                  ### STATION DATA TABLE
                  f'{create_event_data_string(dataset)}'
    )
    if use_gzip:
        with gzip.open(path, 'wb') as f:
            f.write(lma_file.encode('utf-8'))
    else:
        with open(path, 'w') as f:
            f.write(lma_file)