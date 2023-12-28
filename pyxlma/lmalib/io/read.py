import xarray as xr
import pandas as pd
import numpy as np
import gzip
import datetime as dt

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
        attr_vals =  all_attrs[k]
        set_of_values = set(attr_vals)
        if len(set_of_values) == 1:
            final_attrs[k] = tuple(set_of_values)[0]
        else:
            final_attrs[k] = '; '.join(attr_vals)
    # print(final_attrs)

    # Get just the pure-station variables
    lma_station_data = xr.concat(
        [d.drop_dims(['number_of_events']) for d in lma_data],
        dim='number_of_files'
    )
    # print(lma_station_data)

    # Get just the pure-event variables
    lma_event_data = xr.concat(
        [d.drop_dims(['number_of_stations']).drop(
            ['network_center_latitude',
             'network_center_longitude',
             'network_center_altitude',]
          ) for d in lma_data],
        dim='number_of_events'
    )
    # print(lma_event_data)

    # Get any varaibles with joint dimensions,
    # i.e., ('number_of_events', 'number_of_stations')
    event_contributing_stations = xr.concat(
        [d['event_contributing_stations'] for d in lma_data],
        dim='number_of_events'
    )
    # print(event_contributing_stations)

    # Find the mean of the station data, and then add back the event data
    ds = xr.merge([lma_station_data.mean(dim='number_of_files'),
                   lma_event_data],
                  compat='equals')
    # ... and then restore the contributing stations.
    # Note the coordinate variables are being used as labels to ensure data remain aligned.
    ds['event_contributing_stations'] = event_contributing_stations

    # Restore the global attributes
    ds.attrs.update(final_attrs)

    return ds

def dataset(filenames, sort_time=True):
    """ Create an xarray dataset of the type returned by
        pyxlma.lmalib.io.cf_netcdf.new_dataset for each filename in filenames
    """
    lma_data = []
    starttime = None
    next_event_id = 0
    for filename in filenames:
        lma_file = lmafile(filename)
        if starttime is None:
            starttime = lma_file.starttime
        else:
            starttime = min(lma_file.starttime, starttime)
        # Accounting for empty files
        try:
            ds = to_dataset(lma_file, event_id_start=next_event_id).set_index(
                {'number_of_stations':'station_code', 'number_of_events':'event_id'})
            lma_data.append(ds)
            next_event_id += ds.dims['number_of_events']
        except:
            raise
    ds = combine_datasets(lma_data)
    ds = ds.reset_index(('number_of_events', 'number_of_stations'))
    if 'number_of_events_' in ds.coords:
        # Older xarray versions appended a trailing underscore. reset_coords then dropped
        # converted the coordinate variables into regular variables while not touching
        # the original dimension name, allowing us to rename. In newer versions, the variable
        # names are never changed at the reset_index step, so the renaming step modifies
        # the dimension name, too.
        resetted = ('number_of_events_', 'number_of_stations_')
        ds = ds.reset_coords(resetted)
        ds = ds.rename({resetted[0]:'event_id',
                        resetted[1]:'station_code'})
    else:
        # The approach for newer xarray versions requires we explicitly rename only the variables.
        # The generic "rename" in the block above renames vars, coords, and dims.
        resetted = ('number_of_events', 'number_of_stations')
        ds = ds.reset_coords(resetted)
        ds = ds.rename_vars({resetted[0]:'event_id',
                        resetted[1]:'station_code'})
    if sort_time:
        ds = ds.sortby('event_time')
    return ds, starttime

def to_dataset(lma_file, event_id_start=0):
    """ lma_file: an instance of an lmafile object

    returns an xarray dataset of the type returned by
        pyxlma.lmalib.io.cf_netcdf.new_dataset
    """
    from pyxlma.lmalib.io.cf_netcdf import new_dataset
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

    return ds


def nldn(filenames):
    """
    Read Viasala NLDN file and return a pandas dataframe with appropriate column names
    """
    if type(filenames) is str:
        filenames = [filenames]
    full_df = pd.DataFrame({})
    for filename in filenames:
        this_file = pd.read_csv(filename, delim_whitespace=True, header=None, 
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
    Read Earth Networks Total Lightning Network file and return a pandas dataframe with appropriate column names
    """
    if type(filenames) is str:
        filenames = [filenames]
    full_df = pd.DataFrame({})
    for filename in filenames:
        this_file = pd.read_csv(filename, parse_dates=['timestamp'])
        this_file['peakcurrent'] = this_file['peakcurrent']/1000
        this_file['type'] = this_file['type'].map({0:'CG',1:'IC'})
        rename = {'timestamp' : 'datetime', 'peakcurrent' : 'peak_current_kA', 'sensors' : ''}
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
                    self.network_location = ':'.join(line.decode().split(':')[1:])[:-1]
                if line.startswith(b'Data start time:'):
                    timestring = line.decode().split()[-2:]
                    self.startday = dt.datetime.strptime(timestring[0],'%m/%d/%y')
                    # Full start time and second, likely unneeded
                    self.starttime = dt.datetime.strptime(timestring[0]+timestring[1],'%m/%d/%y%H:%M:%S')
                    # self.startsecond = (starttime-dt.datetime(starttime.year,starttime.month,starttime.day)).seconds
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
            lmad = pd.read_csv(self.file,compression=comp,delim_whitespace=True,
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
