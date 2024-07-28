import xarray as xr
import pandas as pd
import numpy as np
import gzip
import datetime as dt
from os import path
import collections
import string
import warnings
from pyxlma.lmalib.io.cf_netcdf import new_dataset, new_template_dataset

class open_gzip_or_dat:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        if self.filename.endswith('.gz'):
            self.file = gzip.open(self.filename)
        else:
            self.file = open(self.filename, 'rb')
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

def mask_to_int(mask):
    """ Convert object array of mask strings to integers"""
    if len(mask.shape) == 0:
        mask_int = np.asarray([], dtype=int)
    else:
        try:
            # mask is a plain integer
            mask_int = np.fromiter((int(v) for v in mask), int)
        except ValueError:
            # mask is a string representing a base-16 (hex) number
            mask_int = np.fromiter((int(v,16) for v in mask), int)
    return mask_int

def combine_datasets(lma_data):
    """ lma_data is a list of xarray datasets of the type returned by
        pyxlma.lmalib.io.cf_netcdf.new_dataset or
        pyxlma.lmalib.io.read.to_dataset
    """
    def reorder_stations(dataset, index_mapping):
        for var in dataset.data_vars:
            if 'number_of_stations' in dataset[var].dims:
                if var == 'station_code':
                    new_station_codes = dataset[var].data.copy()
                    new_station_codes = new_station_codes[index_mapping]
                    dataset = dataset.assign(station_code=('number_of_stations', new_station_codes))
                else:
                    reordered_data = dataset[var].isel(number_of_stations=index_mapping)
                    dataset[var].data = reordered_data.values
        return dataset
    # Get a list of all the global attributes from each dataset
    attrs = [d.attrs for d in lma_data]
    # Create a dict of {attr_name: [list of values from each dataset]}
    # Will be None if that attribute is not present in one of the lma_data
    all_attrs = {
        k: [d.get(k) for d in attrs]
        for k in set().union(*attrs)
    }
    final_attrs = {}
    for k in all_attrs:
        if k.startswith('analysis_') and k.endswith('_time'):
            continue
        attr_vals =  all_attrs[k]
        set_of_values = set(attr_vals)
        if len(set_of_values) == 1:
            final_attrs[k] = tuple(set_of_values)[0]
        else:
            final_attrs[k] = '; '.join(attr_vals)
    # Get the station data from the first dataset and assign each station a unique index
    lma_data[0]['station_code'] = lma_data[0].number_of_stations
    lma_data[0]['number_of_stations'] = np.arange(len(lma_data[0].number_of_stations))
    lma_data[0].attrs['config_times'] = [[[lma_data[0].attrs['analysis_start_time'], lma_data[0].attrs['analysis_end_time']]]]
    all_data = lma_data[0]
    # Get the attributes attached to each variable in the dataset
    dv_attrs = {}
    new_ds = new_template_dataset()
    for var in new_ds['data_vars']:
        dv_attrs[var] = new_ds['data_vars'][var]['attrs']
    # Define list of 'properties', things which identify a station and are not expected to change
    property_vars = ('station_latitude', 'station_longitude', 'station_altitude', 'station_code', 'station_network')
    # Define list of variables to be recalculated for each station after the datasets are combined
    recalc_vars = ('station_event_fraction', 'station_power_ratio', 'network_center_latitude', 'network_center_longitude', 'network_center_altitude')
    # Will be set to True if network_center location needs to be recalculated
    recalc_center = False
    # Check each subsequent dataset for new stations
    for new_file_num in range(1, len(lma_data)):
        new_file = lma_data[new_file_num]
        if (np.all(new_file.network_center_latitude.data != all_data.network_center_latitude.data)
            or np.all(new_file.network_center_longitude.data.item() != all_data.network_center_longitude.data)
            or np.all(new_file.network_center_altitude.data != all_data.network_center_altitude.data)):
            recalc_center = True
        # Demote station code to a data variable and assign an index to each station in the new file
        new_file['station_code'] = new_file.number_of_stations
        new_file['number_of_stations'] = np.arange(len(new_file.number_of_stations))
        stations_in_file = []
        # Check each station in the new file against all known stations
        station_is_new = True
        old_ids_to_check_for_missing = all_data.number_of_stations.data.tolist()
        for station_num in range(len(new_file.number_of_stations.data)):
            station = new_file.isel(number_of_stations=station_num)
            old_ids_to_search = collections.deque(range(len(all_data.number_of_stations.data)))
            old_ids_to_search.rotate(-1*station_num) # start with the same index because stations USUALLY come in the same order
            for old_station_num in old_ids_to_search:
                old_station = all_data.isel(number_of_stations=old_station_num)
                all_props_match = True
                for prop in property_vars:
                    if station.data_vars[prop].data.item() != old_station.data_vars[prop].data.item():
                        all_props_match = False
                        break
                if all_props_match:
                    station_is_new = False
                    stations_in_file.append(old_station_num)
                    old_ids_to_check_for_missing.remove(old_station_num)
                    break
            if station_is_new:
                stations_in_file.append(-1)
        # Find the indices of any newly-discovered stations
        indices_of_new_stations = np.where(np.array(stations_in_file) == -1)[0]
        # Add the new stations to the known stations
        for idx_of_new_station in indices_of_new_stations:
            new_station = new_file.isel(number_of_stations=idx_of_new_station)
            # The new station is appended to the end of the known stations
            new_station_index = len(all_data.number_of_stations)
            # Expand all of the previous data to contain the new station. Fill with station properties.
            fill_vals_dict = {}
            for var in new_station.data_vars:
                if 'number_of_stations' in all_data[var].dims:
                    if var in property_vars:
                        fill_vals_dict[var] = new_station[var].data.item()
                    else:
                        fill_vals_dict[var] = 0
            all_data = all_data.reindex({'number_of_stations': np.arange(new_station_index+1)}, fill_value=fill_vals_dict)
            # Update the station index for the new station
            stations_in_file[idx_of_new_station] = new_station_index
        # Check for any previously-known stations that are no longer in the new file
        for missing_station_id in old_ids_to_check_for_missing:
            dead_station_data = all_data.isel(number_of_stations=missing_station_id)
            # a station has been removed, create a row of nan values for this file to indicate that the station is no longer present
            # first create a temporary index for the dead station
            temp_station_id = len(new_file.number_of_stations)
            fill_vals_dict = {}
            for var in dead_station_data.data_vars:
                if 'number_of_stations' in all_data[var].dims:
                    if var in property_vars:
                        fill_vals_dict[var] = dead_station_data[var].data.item()
                    else:
                        fill_vals_dict[var] = 0
            new_file = new_file.reindex({'number_of_stations': np.arange(temp_station_id+1)}, fill_value=fill_vals_dict)
            # Update the station index for the dead station
            stations_in_file.append(missing_station_id)
        # Re-order the station data to match the order of the stations in the new file
        # This can happen if lma_analysis decides to change the order of the stations, or if new stations are added or removed
        new_file = reorder_stations(new_file, np.argsort(stations_in_file))
        # Concatenate the new file's station information with the previously-known station information
        station_to_merge = [d.drop_dims(['number_of_events']) for d in [all_data, new_file]]
        lma_station_data = xr.concat(station_to_merge, dim='network_configurations').drop_vars(
            ['network_center_latitude', 'network_center_longitude', 'network_center_altitude'])
        for var_name in lma_station_data.data_vars:
            da = lma_station_data[var_name]
            if 'network_configurations' in da.dims:
                # black magic from openAI that somehow determines if a data array is identical across the 'network_configurations' variable
                if np.all((da == da.isel({'network_configurations': 0})).all('network_configurations')) or var_name in recalc_vars:
                    # Remove the 'network_configurations' dimension if the data is identical across it
                    lma_station_data[var_name] = da.isel(network_configurations=0)
        if 'network_configurations' in lma_station_data.dims:
            unique_netw_configs = np.unique(lma_station_data.station_active.data, return_index=True, axis=0)[1]
            lma_station_data = lma_station_data.isel(network_configurations=sorted(unique_netw_configs))
        # Rebuild the event_contributing_stations array
        event_contributing_stations = xr.concat(
                [d['event_contributing_stations'] for d in [all_data, new_file]],
                dim='number_of_events'
            )
        # Attach the event_contributing_stations array to the station dataset
        lma_station_data['event_contributing_stations'] = event_contributing_stations

        # Combining the events is a simple concatenation. If only the stations had been this easy...
        lma_event_data = xr.concat(
            [d.drop_dims(['number_of_stations']) for d in [all_data, new_file]],
            dim='number_of_events'
        )
        # merge all the data from this file with the previously-known data
        combined_event_data = xr.merge([lma_station_data, lma_event_data])
        all_data = combined_event_data
        # Log continuous intervals of data coverage
        if 'network_configurations' in all_data.dims:
            this_netw_conf = np.where((all_data.station_active.data == new_file.station_active.data).all(axis=1))[0][0]
        else:
            this_netw_conf = 0
        if len(all_data.attrs['config_times']) > this_netw_conf:
            times_of_this_config = all_data.attrs['config_times'][this_netw_conf]
            for gap in times_of_this_config:
                if new_file.attrs['analysis_start_time'] == gap[1]:
                    gap[1] = new_file.attrs['analysis_end_time']
                    break
                elif new_file.attrs['analysis_end_time'] == gap[0]:
                    gap[0] = new_file.attrs['analysis_start_time']
                    break
            else:
                times_of_this_config.append([new_file.attrs['analysis_start_time'], new_file.attrs['analysis_end_time']])
        else:
            all_data.attrs['config_times'].append([[new_file.attrs['analysis_start_time'], new_file.attrs['analysis_end_time']]])
        all_data.attrs['analysis_start_time'] = min(all_data.attrs['analysis_start_time'], new_file.attrs['analysis_start_time'])
        all_data.attrs['analysis_end_time'] = max(all_data.attrs['analysis_end_time'], new_file.attrs['analysis_end_time'])
        all_data['network_center_longitude'] = all_data.network_center_longitude.isel(number_of_events=0)
        all_data['network_center_latitude'] = all_data.network_center_latitude.isel(number_of_events=0)
        all_data['network_center_altitude'] = all_data.network_center_altitude.isel(number_of_events=0)
    # Update the global attributes
    all_data.attrs.update(final_attrs)
    # To reduce complexity and resource usage, if the 'network_configurations' dimension is the same for all variables, then the dimension is unnecessary
    if 'network_configurations' in all_data.dims:
        # Identify unique network configurations
        unique_configs = np.unique(all_data.station_active.data, return_index=True, axis=0)[1]
        if len(unique_configs) == 1:
            unique_configs = unique_configs[0]
        else:
            all_data['station_event_fraction'] = 100*all_data.event_contributing_stations.sum(dim='number_of_events')/all_data.number_of_events.shape[0]
            all_data['station_power_ratio'] = (all_data.event_contributing_stations * all_data.event_power).sum(dim='number_of_events')/all_data.event_power.sum(dim='number_of_events')
    # recalculate variables that depend on the station data
    if recalc_center:
        all_data['network_center_longitude'] = all_data.station_longitude.mean(dim='number_of_stations')
        all_data['network_center_latitude'] = all_data.station_latitude.mean(dim='number_of_stations')
        all_data['network_center_altitude'] = all_data.station_altitude.mean(dim='number_of_stations')
    # Make sure all station codes are unique
    if np.max(np.unique_counts(all_data.station_code.data).counts) > 1:
        unique_stat = []
        the_alphabet = string.ascii_letters
        for i in range(len(all_data.station_code.data)):
            stat = all_data.station_code.data[i]
            if stat in unique_stat:
                warnings.warn(f'Conflicting station code discovered at index {i}')
                stat_str = stat.decode('utf-8')
                if stat_str.isupper():
                    lowc = stat_str.lower()
                    if bytes(lowc, 'utf-8') not in unique_stat:
                        warnings.warn(f'Renaming station {stat_str} to {lowc}')
                        unique_stat.append(bytes(lowc, 'utf-8'))
                    else:
                        for new_letter in the_alphabet:
                            if bytes(new_letter, 'utf-8') not in unique_stat:
                                warnings.warn(f'Renaming station {stat_str} to {new_letter}')
                                unique_stat.append(bytes(new_letter, 'utf-8'))
                                break
                        else:
                            warnings.warn(f'Assigning station a blank ID. This station will not be included in files generated by pyxlma.lmalib.io.write.lma_dat_file')
                elif stat_str.islower():
                    upc = stat_str.upper()
                    if bytes(upc, 'utf-8') not in unique_stat:
                        warnings.warn(f'Renaming station {stat_str} to {upc}')
                        unique_stat.append(bytes(upc, 'utf-8'))
                    else:
                        for new_letter in the_alphabet:
                            if bytes(new_letter, 'utf-8') not in unique_stat:
                                warnings.warn(f'Renaming station {stat_str} to {new_letter}')
                                unique_stat.append(bytes(new_letter, 'utf-8'))
                                break
                        else:
                            warnings.warn(f'Assigning station a blank ID. This station will not be included in files generated by pyxlma.lmalib.io.write.lma_dat_file')
            else:    
                unique_stat.append(stat)
        all_data.station_code.data = np.array(unique_stat, dtype='S1')
    bin_to_dec = 2**np.flip(np.arange(all_data.event_contributing_stations.data.shape[1]))
    all_data.event_mask.data = np.sum((bin_to_dec * all_data.event_contributing_stations.data), axis=1) # convert bin to dec
    all_data.attrs['station_mask_order'] = np.apply_along_axis(lambda x: ''.join(x), 0, all_data.station_code.astype(str).data).item()
    # restore previously cached data var attributes
    for var_name in all_data.data_vars:
        if var_name in dv_attrs:
            all_data[var_name].attrs = dv_attrs[var_name]
    all_data.station_active.attrs['_FillValue'] = 255
    return all_data

def dataset(filenames, sort_time=True):
    """ Create an xarray dataset of the type returned by
        pyxlma.lmalib.io.cf_netcdf.new_dataset for each filename in filenames
    """
    if type(filenames) == str:
        filenames = [filenames]
    lma_data = []
    starttime = None
    if sort_time:
        all_starttimes = []
    next_event_id = 0
    for filename in sorted(filenames):
        lma_file = lmafile(filename)
        if starttime is None:
            starttime = lma_file.starttime
        else:
            starttime = min(lma_file.starttime, starttime)
        ds = to_dataset(lma_file, event_id_start=next_event_id).set_index(
            {'number_of_stations':'station_code', 'number_of_events':'event_id'})
        ds.attrs['analysis_start_time'] = lma_file.starttime
        ds.attrs['analysis_end_time'] = lma_file.starttime + dt.timedelta(seconds=lma_file.analyzed_sec)
        lma_data.append(ds)
        next_event_id += ds.sizes['number_of_events']
        if sort_time:
            all_starttimes.append(lma_file.starttime)
    if sort_time:
        sorting = np.argsort(all_starttimes)
        lma_data = [lma_data[i] for i in sorting]
    ds = combine_datasets(lma_data)
    if sort_time:
        ds = ds.sortby('event_time')
    ds = ds.reset_index(('number_of_events'))
    if 'number_of_events_' in ds.coords:
        # Older xarray versions appended a trailing underscore. reset_coords then dropped
        # converted the coordinate variables into regular variables while not touching
        # the original dimension name, allowing us to rename. In newer versions, the variable
        # names are never changed at the reset_index step, so the renaming step modifies
        # the dimension name, too.
        resetted = ('number_of_events_', 'number_of_stations_')
        ds = ds.rename({resetted[0]:'event_id'})
    else:
        # The approach for newer xarray versions requires we explicitly rename only the variables.
        # The generic "rename" in the block above renames vars, coords, and dims.
        resetted = ('number_of_events', 'number_of_stations')
        ds = ds.rename_vars({resetted[0]:'event_id'})
    return ds, starttime

def to_dataset(lma_file, event_id_start=0):
    """ lma_file: an instance of an lmafile object

    returns an xarray dataset of the type returned by
        pyxlma.lmalib.io.cf_netcdf.new_dataset
    """
    
    lma_data = lma_file.readfile()
    starttime = lma_file.starttime
    stations = lma_file.stations

    N_events = lma_data.shape[0]
    N_stations = lma_file.stations.shape[0]
    ds = new_dataset(events=N_events, stations=N_stations)

    # Index from dataset variable to lma_data column name
    station_mapping = {
        'station_code':'ID',
        'station_latitude':'Lat',
        'station_longitude':'Long',
        'station_altitude':'Alt',
        'station_event_fraction':'sources',
        'station_power_ratio':'<P/P_m>',
        'station_active':'active',
    }
    event_mapping = {
        'event_latitude':'lat',
        'event_longitude':'lon',
        'event_altitude':'alt(m)',
        'event_power':'P(dBW)',
        'event_stations':'Station Count',
        'event_chi2':'reduced chi^2',
    }

    for var, col in event_mapping.items():
        ds[var][:] = lma_data[col]
    for var, col in station_mapping.items():
        ds[var][:] = stations[col]

    ds['event_id'][:] = (event_id_start
                         + np.arange(N_events, dtype=ds['event_id'].dtype))
    ds['event_mask'][:] = lma_file.mask_ints
    ds.event_chi2.attrs['valid_range'][1] = lma_file.maximum_chi2
    ds.event_stations.attrs['valid_range'][0] = lma_file.minimum_stations

    time_units = lma_file.startday.strftime(
        "seconds since %Y-%m-%d 00:00:00 +00:00")
    ds['event_time'].data = lma_data.Datetime
    # ds['event_time'].attrs.pop('units')
    # ds['event_time'].encoding['units'] = time_units

    # Assign to the data attribute to not overwrite units metadata
    ds['network_center_latitude'].data = lma_file.center_lat
    ds['network_center_longitude'].data = lma_file.center_lon
    ds['network_center_altitude'].data = lma_file.center_alt
    ds['station_network'][:] = lma_file.network_location

    # Global attrs
    ds.attrs['title'] = "Lightning Mapping Array Dataset, L1b events and station information."
    # production_date:          1970-01-01 00:00:00 +00:00
    ds.attrs['history'] = "LMA source file created "+lma_file.file_created
    ds.attrs['event_algorithm_name'] = lma_file.analysis_program
    ds.attrs['event_algorithm_version'] = lma_file.analysis_program_version
    ds.attrs['original_filename'] =  path.basename(lma_file.file)
    # -- Populate the station mask information --
    # int, because NetCDF doesn't have booleans
    station_mask_bools = np.zeros((N_events, N_stations), dtype='int8')
    # Don't presume stations are in the correct order. Construct a lookup
    # using the order already present in the station_code variable so that
    # everything lines up along the number_of_stations dimension in ds.
    stncode_to_index = {}
    for i, stn in enumerate(ds['station_code'].data):
        stncode_to_index[stn.decode()] = i
    for col in lma_file.station_contrib_cols:
        i = stncode_to_index[col[0]]
        # col_name = col[2:]
        station_mask_bools[:, i] = lma_data[col]
    ds['event_contributing_stations'][:] = station_mask_bools

    # -- Convert the station_active flag to a psuedo-boolean --
    station_active_data = np.zeros(N_stations, dtype='uint8')
    station_active_data[ds.station_active.data == b'A'] = 1
    station_active_data[ds.station_active.data == b'NA'] = 0
    ds.station_active.data = station_active_data

    # Convert station event count to station event fraction
    if N_events != 0:
        ds.station_event_fraction.data = 100*ds.station_event_fraction.data/N_events
    return ds


def nldn(filenames):
    """
    Read Viasala NLDN data
     
    Reads in one or multiple NLDN files and and returns a pandas dataframe with appropriate column names


    Parameters
    ----------
    filenames : str or list of str
        The file or files to read in
    
    
    Returns
    -------
    full_df : `pandas.DataFrame`
        A pandas dataframe of entln data, with columns:
        'latitude' - the latitude of the event
        'longitude' - the longitude of the event
        'peak_current_kA' - the peak current in kA
        'multiplicity' - the number of strokes in the event
        'semimajor' - the semimajor axis length in km of the 50% confidence ellipse
        'semiminor' - the semiminor axis length in km of the 50% confidence ellipse
        'ellipseangle' - the angle of the 50% confidence ellipse
        'chi2' - the reduced chi-squared value of the event
        'num_stations' - the number of stations contributing to the event
        'type' - 'IC' or 'CG' for intracloud or cloud-to-ground
        'datetime' - the time of the event

        
    Notes
    -----
    This function is designed to read NLDN files in the format that matches the format of the file located
    in examples/network_samples/gld360enldnns_20231224_daily_v1_lit.raw (This is not real NLDN data, but provides a correct sample of the format)

    Other file formats may exist and may not be read in correctly. If you have a file that is not read in correctly,
    please open an issue on the pyxlma GitHub page.
    """
    if type(filenames) is str:
        filenames = [filenames]
    full_df = pd.DataFrame({})
    for filename in filenames:
        this_file = pd.read_csv(filename, sep='\\s+', header=None, 
                    names=[
                        'date', 'time', 'latitude', 'longitude', 'peak_current_kA', 'curr_unit', 'multiplicity', 'semimajor',
                        'semiminor', 'majorminorratio', 'ellipseangle', 'chi2', 'num_stations', 'type'
                        ])
        if len(this_file['curr_unit'].drop_duplicates()) == 1:
            this_file.drop(columns=['curr_unit'], inplace=True)
        else:
            raise ValueError('Multiple current units in file')
        this_file['datetime'] = pd.to_datetime(this_file['date']+' '+this_file['time'], format='%m/%d/%y %H:%M:%S.%f')
        this_file.drop(columns=['date','time'], inplace=True)
        this_file['type'] = this_file['type'].map({'G':'CG','C':'IC'})
        full_df = pd.concat([full_df, this_file])
    return full_df


def entln(filenames):
    """
    Read Earth Networks Total Lightning Network data
     
    Reads in one or multiple ENTLN files and and returns a pandas dataframe with appropriate column names


    Parameters
    ----------
    filenames : str or list of str
        The file or files to read in
    
    
    Returns
    -------
    full_df : `pandas.DataFrame`
        A pandas dataframe of entln data, with columns:
        'type' - 'IC' or 'CG' for intracloud or cloud-to-ground
        'datetime' - the time of the event
        'latitude' - the latitude of the event
        'longitude' - the longitude of the event
        'peak_current_kA' - the peak current in kA
        'icheight' - the height of the IC event in meters
        'num_stations' - the number of stations contributing to the event
        'ellipseangle' - the angle of the 50% confidence ellipse
        'semimajor' - the semimajor axis length in km of the 50% confidence ellipse
        'semiminor' - the semiminor axis length in km of the 50% confidence ellipse

    Notes
    -----
    This function is designed to read ENTLN files in the format that matches the format of the file located
    in examples/network_samples/lxarchive_pulse20231224.csv (This is not real ENTLN data, but provides a correct sample of the format)

    Other file formats may exist and may not be read in correctly. If you have a file that is not read in correctly,
    please open an issue on the pyxlma GitHub page.
    """
    if type(filenames) is str:
        filenames = [filenames]
    full_df = pd.DataFrame({})
    for filename in filenames:
        this_file = pd.read_csv(filename, parse_dates=['timestamp'])
        this_file['peakcurrent'] = this_file['peakcurrent']/1000
        this_file['type'] = this_file['type'].map({0:'CG',1:'IC'})
        this_file['semimajor'] = this_file['majoraxis']/2000
        this_file['semiminor'] = this_file['minoraxis']/2000
        this_file.drop(columns=['majoraxis','minoraxis'], inplace=True)
        rename = {'timestamp' : 'datetime', 'peakcurrent' : 'peak_current_kA', 'numbersensors' : 'num_stations', 'bearing' : 'ellipseangle'}
        this_file.rename(columns=rename, inplace=True)
        full_df = pd.concat([full_df, this_file])
    return full_df

    
class lmafile(object):
    def __init__(self,filename):
        """
        Pull the basic metadata from a '.dat.gz' LMA file

        startday : the date (datetime format)
        station_info_start : the line number (int) where the station information starts
        station_data_start : the line number (int) where the summarized station data starts
        station_data_end : the line number (int) end of the summarized station data
        maskorder : the order of stations in the station mask (str)
        names : column header names
        data_starts : the line number (int) where the VHF source data starts

        overview : summarized station data from file header (DataFrame, assumes fixed-width format)
        stations : station information from file header (DataFrame, assumes fixed-width format)

        """
        self.file = filename

        with open_gzip_or_dat(self.file) as f:
            for line_no, line in enumerate(f):
                if line.startswith(b'Analysis program:'):
                    analysis_program = line.decode().split(':')[1:]
                    self.analysis_program = ':'.join(analysis_program)[:-1]
                if line.startswith(b'Analysis program version:'):
                    analysis_program_version = line.decode().split(':')[1:]
                    self.analysis_program_version = ':'.join(analysis_program_version)[:-1]
                if line.startswith(b'File created:') | line.startswith(b'Analysis finished:'):
                    file_created = line.decode().split(':')[1:]
                    self.file_created = ':'.join(file_created)[:-1]
                if line.startswith(b'Location:'):
                    self.network_location = ':'.join(line.decode().split(':')[1:])[:-1].replace(' ','', 1)
                if line.startswith(b'Data start time:'):
                    timestring = line.decode().split()[-2:]
                    self.startday = dt.datetime.strptime(timestring[0],'%m/%d/%y')
                    # Full start time and second, likely unneeded
                    self.starttime = dt.datetime.strptime(timestring[0]+timestring[1],'%m/%d/%y%H:%M:%S')
                    # self.startsecond = (starttime-dt.datetime(starttime.year,starttime.month,starttime.day)).seconds
                if line.startswith(b'Number of seconds analyzed:'):
                    self.analyzed_sec = int(line.decode().split(':')[-1])
                # Find starting and ending rows for station information
                if line.startswith(b'Coordinate center'):
                    self.center_lat = float(line.decode().split()[-3])
                    self.center_lon = float(line.decode().split()[-2])
                    self.center_alt = float(line.decode().split()[-1])
                # Number of active stations
                if line.startswith(b'Number of active stations:'):
                    self.active_station_c_line = line_no
                    self.active_staion_c_count = line.decode().split()[-1]
                # Active stations
                if line.startswith(b'Active stations:'):
                    self.active_station_s_line = line_no
                    self.active_station_s = line.decode().split()[2:]
                if line.startswith(b'Minimum number of stations per solution:'):
                    self.minimum_stations = int(line.decode().split(':')[1])
                if line.startswith(b'Maximum reduced chi-squared:'):
                    self.maximum_chi2 = float(line.decode().split(':')[1])
                if line.startswith(b'Maximum chi-squared iterations:'):
                    self.maximum_chi2_iter = int(line.decode().split(':')[1])
                if line.startswith(b'Station information:'):
                    self.station_info_start = line_no
                if line.startswith(b'Station data:'):
                    self.station_data_start = line_no
                if line.startswith(b'Metric file:'):
                    self.station_data_end = line_no
                # Find mask list order
                if line.startswith(b'Station mask order:'):
                    self.maskorder = line.decode().split()[-1]
                # Pull data header
                if line.startswith(b'Data:'):
                    self.names = [x.strip(' ') for x in line.decode()[5:-1].split(",")]
                # Text format
                if line.startswith(b'Data format:'):
                    self.format = line.decode().split(' ')[2:]
                # Total number of events in file
                if line.startswith(b'Number of events:'):
                    self.events_line  = line_no
                    self.events_count = line.decode().split()[-1]
                # Find start line of the data
                if line.rstrip() == b"*** data ***":
                    break
        f.close()
        self.data_starts = line_no

        # Station overview information
        stations = pd.DataFrame(self.gen_sta_info(),
            columns=['ID','Name','Lat','Long','Alt','Delay Time'])
        overview = pd.DataFrame(self.gen_sta_data(),
            columns=['ID','Name','win(us)', 'data_ver', 'rms_error(ns)',
                     'sources','percent','<P/P_m>','active'])
        # Drop the station name column that has a redundant station letter code
        # as part of the name and join on station letter code.
        station_combo =  stations.set_index('ID').drop(columns=['Name']).join(
                             overview.set_index('ID'))
        self.stations = station_combo.reset_index(level=station_combo.index.names)

    def gen_sta_info(self):
        """ Parse the station info table from the header. Some files do not
        have fixed width columns, and station names may have spaces, so this
        function chops out the space-delimited columns to the left and right
        of the station names.
        """
        nstations = self.station_data_start-self.station_info_start-1
        with open_gzip_or_dat(self.file) as f:
            for i in range(self.station_info_start+1):
                line = next(f)
            for line in range(nstations):
                line = next(f)
                parts = line.decode("utf-8").split()
                name = ' '.join(parts[2:-6])
                sta_info, code = parts[0:2]
                yield (code, name) + tuple(parts[-6:-2])

    def gen_sta_data(self):
        """ Parse the station data table from the header. Some files do not
        have fixed width columns, and station names may have spaces, so this
        function chops out the space-delimited columns to the left and right
        of the station names.
        """
        nstations = self.station_data_start-self.station_info_start-1

        with open_gzip_or_dat(self.file) as f:
            for i in range(self.station_data_start+1):
                line = next(f)
            for line in range(nstations):
                line = next(f)
                parts = line.decode("utf-8").split()
                name = ' '.join(parts[2:-7])
                sta_info, code = parts[0:2]
                yield (code, name) + tuple(parts[-7:])

    def readfile(self):
        """
        Read data from '.dat.gz' file and return a Pandas Dataframe using the
        headers in the datafile.

        Datetime' holds the second of day into datetime format

        Station ID (letter identifier) columns each contain booleans (1/0)
        if the station contributed to the source

        'Station Count' column containes the total number of contributing
        stations for each source
        """
        # Read in data. Catch case where there is no data.
        try:
            if self.file.endswith('.gz'):
                comp = 'gzip'
            else:
                comp = None
            lmad = pd.read_csv(self.file,compression=comp,sep='\\s+',
                               header=None,skiprows=self.data_starts+1,on_bad_lines='skip')
            lmad.columns = self.names
        except pd.errors.EmptyDataError:
            lmad = pd.DataFrame(columns=self.names)

        # Convert seconds column to new datetime-formatted column
        lmad.insert(1,'Datetime',pd.to_timedelta(lmad['time (UT sec of day)'], unit='s')+self.startday)

        # Parse out which stations contributed into new columns for each station
        col_names = self.stations.Name.values
        self.mask_ints = mask_to_int(lmad["mask"])
        for index,items in enumerate(self.maskorder[::-1]):
            col_names[index] = items+'_'+self.stations.Name.values[index]
            lmad.insert(8,col_names[index],
                        (self.mask_ints>>index)%2)
        # Count the number of stations contributing and put in a new column
        lmad.insert(8,'Station Count',lmad[col_names].sum(axis=1).astype('uint8'))
        self.station_contrib_cols = col_names

        # Version for using only station symbols. Not as robust.
        # for index,items in enumerate(self.maskorder[::-1]):
        #     lmad.insert(8,items,(mask_to_int(lmad["mask"])>>index)%2)
        # # Count the number of stations contributing and put in a new column
        # lmad.insert(8,'Station Count',lmad[list(self.maskorder)].sum(axis=1))

        return lmad
