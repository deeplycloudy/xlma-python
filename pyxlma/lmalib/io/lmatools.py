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
# def to_xarray(filename):
#     lmah5 = LMAh5File(filename, min_points=1)
#     for events, flashes in lmah5.gen_events_flashes()
#         ev_xr = events_np_to_xarray(events, lmah5.base_date)


# optional
# flash_volume

# one or more
# flash_center_*
# flash_init_*


from lmatools.lmaio.LMAarrayFile import LMAdataFile

class LMAdata(object):

    def __init__(self, filename, **kwargs):
        self.lma = LMAdataFile(filename, mask_length=kwargs['mask_length'])
        self.get_date()
        self.limit_data(**kwargs)

    def get_date(self):
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
        good1 = (self.lma.stations >= kwargs['stn']) & \
            (self.lma.chi2 <= kwargs['chi2']) & (self.lma.alt < kwargs['alt'])
        good2 = np.logical_and(
            self.lma.data['time'] >= 0,
            self.lma.data['time'] < 86400)
        good = np.logical_and(good1, good2)
        self.data = self.lma.data[good]
        self.datetime = self.datetime[good]
