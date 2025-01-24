# from lmatools.io.LMA_h5_file import LMAh5File
# from pyxlma.lmalib.io.cf_netcdf import new_dataset
#
# # mapping from numpy name to the definition in pyxlma.lmalib.io.cf_netcdf.event_template.
# event_np_to_cf_variables = {
#     'time'
# }
#
#
# def events_np_to_xarray(events, base_date):
#     N = events.shape[0]
#     for v in events.columns:
#
#
#
# def to_dataset(filename):
#     lmah5 = LMAh5File(filename, min_points=1)
#     for events, flashes in lmah5.gen_events_flashes()
#         ev_xr = events_np_to_xarray(events, lmah5.base_date)


# optional
# flash_volume

# one or more
# flash_center_*
# flash_init_*


import datetime as dt
import numpy as np

class LMAdata(object):
    """Helper class to read LMA data using lmatools LMAdataFile.
    
    Warning
    -------
    This class is provided for backwards compatibility with lmatools.
    It is highly encouraged to use the functions in pyxlma.lmalib.io.read in new code.
    """
    def __init__(self, filename, mask_length, **kwargs):
        """Initialize LMAdata object.
        
        Parameters
        ----------
        filename : str
            Path to LMA data file.
        mask_length : int
            Length of the hexadecimal station mask in the LMA data file.
        **kwargs
            Filter parameters to use when reading LMA data. Valid keys are 'stn', 'chi2', and 'alt'.
        """
        from lmatools.io.LMAarrayFile import LMAdataFile
        self.lma = LMAdataFile(filename, mask_length=mask_length)
        self.get_date()
        self.limit_data(**kwargs)

    def get_date(self):
        """Get date from LMA data file."""
        for line in self.lma.header:
            if line[0:10] == 'Data start':
                datestr = line[17:25]
                break
        mo = int(datestr[0:2])
        dy = int(datestr[3:5])
        yr = int(datestr[6:8])
        if yr < 90:
            yr += 2000
        else:
            yr += 1900
        self.datetime = []
        for sod in self.lma.data['time']:
            self.datetime.append(
                dt.datetime(yr, mo, dy) + dt.timedelta(seconds=sod))
        self.datetime = np.array(self.datetime)
        self.year = yr
        self.month = mo
        self.day = dy

    def limit_data(self, **kwargs):
        """Limit LMA data based on filter parameters."""
        good1 = (self.lma.stations >= kwargs['stn']) & \
            (self.lma.chi2 <= kwargs['chi2']) & (self.lma.alt < kwargs['alt'])
        good2 = np.logical_and(
            self.lma.data['time'] >= 0,
            self.lma.data['time'] < 86400)
        good = np.logical_and(good1, good2)
        self.data = self.lma.data[good]
        self.datetime = self.datetime[good]
