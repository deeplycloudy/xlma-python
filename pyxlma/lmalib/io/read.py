import pandas as pd
import numpy as np
import gzip
import datetime as dt

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

class lmafile:
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
        
        with gzip.open(self.file) as f: 
            for line_no, line in enumerate(f):
                if line.startswith(b'Data start time:'):
                    timestring = line.decode().split()[-2:]
                    self.startday = dt.datetime.strptime(timestring[0],'%m/%d/%y')
                    # Full start time and second, likely unneeded
                    self.starttime = dt.datetime.strptime(timestring[0]+timestring[1],'%m/%d/%y%H:%M:%S')
                    # self.startsecond = (starttime-dt.datetime(starttime.year,starttime.month,starttime.day)).seconds
                # Find starting and ending rows for station information
                if line.startswith(b'Coordinate center'):
                    self.center_lat = line.decode().split()[-3]
                    self.center_lon = line.decode().split()[-2]
                    self.center_alt = line.decode().split()[-1]
                # Number of active stations
                if line.startswith(b'Number of active stations:'):
                    self.active_station_c_line = line_no
                    self.active_staion_c_count = line.decode().split()[-1]
                # Active stations
                if line.startswith(b'Active stations:'):
                    self.active_station_s_line = line_no
                    self.active_station_s = line.decode().split()[2:]
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
        self.overview = pd.read_fwf(self.file,compression='gzip',
                                    colspecs=[[10,11],[13,30],[30,35],[35,43],[43,48],
                                              [48,56],[56,61],[61,68],[68,73]],
                                    names=['ID', 'Name','win(us)', 'dec_win(us)', 
                                           'data_ver', 'rms_error(ns)', 
                                           'sources','<P/P_m>','active'],
                                    header=None,skiprows=self.station_data_start+1, 
                                    nrows=self.station_data_start-self.station_info_start-1)
        # Station Locations
        self.stations = pd.read_fwf(self.file,compression='gzip',
                                    colspecs=[[10,11],[13,32],[32,43],[44,56],[56,66],[66,70]],
                                    names=['ID', 'Name','Lat','Long','Alt','Delay Time'],
                                    header=None,skiprows=self.station_info_start+1, 
                                    nrows=self.station_data_start-self.station_info_start-1)
        
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
        # Read in data
        lmad = pd.read_csv(self.file,compression='gzip',delim_whitespace=True,
                            header=None,skiprows=self.data_starts+1,error_bad_lines=False)
        lmad.columns = self.names
        
        # Convert seconds column to new datetime-formatted column
        lmad.insert(1,'Datetime',pd.to_timedelta(lmad['time (UT sec of day)'], unit='s')+self.startday)
        
        # Parse out which stations contributed into new columns for each station
        col_names = self.stations.Name.values
        for index,items in enumerate(self.maskorder[::-1]):
            col_names[index] = items+'_'+self.stations.Name.values[index]
            lmad.insert(8,col_names[index],
                        (mask_to_int(lmad["mask"])>>index)%2)
        # Count the number of stations contributing and put in a new column
        lmad.insert(8,'Station Count',lmad[col_names].sum(axis=1))

        # Version for using only station symbols. Not as robust.
        # for index,items in enumerate(self.maskorder[::-1]):
        #     lmad.insert(8,items,(mask_to_int(lmad["mask"])>>index)%2)
        # # Count the number of stations contributing and put in a new column
        # lmad.insert(8,'Station Count',lmad[list(self.maskorder)].sum(axis=1))

        return lmad
