import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime as dt
from matplotlib.ticker import Formatter, FormatStrFormatter, MaxNLocator

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

proj_cart = ccrs.PlateCarree(central_longitude=-95)


# # Add a note about plotting counties by default if metpy is available in docs, and how to add your own map data without relying on built-ins.
# reader = shpreader.Reader('UScounties/UScounties.shp')
# counties = list(reader.geometries())
# COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())
try:
    from metpy.plots import USCOUNTIES
    county_scales = ['20m', '5m', '500k']
    COUNTIES = USCOUNTIES.with_scale(county_scales[0])
except ImportError:
    COUNTIES = None


M2KM = 1000.0
COORD_TH = [0.1, 0.8, 0.83, 0.1]
COORD_PLAN = [0.1, 0.1, 0.65, 0.5]
COORD_LON = [0.1, 0.65, 0.65, 0.1]
COORD_LAT = [0.8, 0.1, 0.13, 0.5]
COORD_HIST = [0.8, 0.65, 0.13, 0.1]
xdiv = 0.01
ydiv = 0.01
zdiv = 0.1


class FractionalSecondFormatter(Formatter):
    def __init__(self, axis):
        self._axis = axis

    def __call__(self, x, pos=None):
        """ Formats seconds of the day to HHMM:SS.SSSSSS, with the fractional
            part varying in length based on the total interval.
            Maximum resolution is 1 microsecond, due to limitation in datetime.
        """

        tick_date = md.num2date(x)
        interval = [md.num2date(d) for d in self._axis.get_xlim()]
        delta_sec = (interval[1] - interval[0]).total_seconds()
        if (delta_sec < 30):
            fmt = '%S'
        else:
            fmt = '%H%M:%S'

        # for most plots, it seems like pos=1 is the first label, even though pos=0 is also requested.
        # some plots do in fact plot the label for both pos=0 and pos=1, so go with 1 for safety
        if pos == 1:
            fmt = '%H%M:%S'

        # This could be generated algorithmically - the pattern is obvious.
        frac_fmt = '%.6f'
        if delta_sec > 0.00005:
            frac_fmt = '%.5f'
        if delta_sec > 0.0005:
            frac_fmt = '%.4f'
        if delta_sec > 0.005:
            frac_fmt = '%.3f'
        if delta_sec > 0.05:
            frac_fmt = '%.2f'
        if delta_sec > 0.5:
            frac_fmt = '%.1f'
        if delta_sec > 5:
            frac_fmt = '%.0f'

        if pos is None:
            # Be verbose for the status readout
            fmt = '%H%M:%S'
            frac_fmt = '%.6f'

        # if pos is not None:
        #     print x, delta_sec, frac_fmt, frac_fmt % (tick_date.microsecond/1.0e6)

        time_str = tick_date.strftime(fmt)
        frac_str = frac_fmt % (tick_date.microsecond/1.0e6)
        return time_str + frac_str[1:]


class XlmaPlot(object):

    def __init__(self, data, stime, subplot_labels, bkgmap, readtype='xarray',inset=False,
        **kwargs):
        """
        data = an data structure matching the pyxlma formats:
            1. pyxlma.lmalib.io.lmatools.LMAdata.data for reading with lmatools
            2. pyxlma.lmalib.io.read.lma_file.readfile() for a pandas dataframe
              with same headers as LYLOUT files.
            3. an xarray dataset of the type returned by
              pyxlma.lmalib.io.read.dataset.
        stime = start time datetime object
        readtype = fileread options "lmatools", "xarray", or "pandas" dataframe
        inset_buffer = additional lat/lon padding on inset image showing location of image
        inset_size = plot size of inset image in left corner of plan view plot
        """
        self.data = data
        self.readtype = readtype
        self.stime = stime
        self.subplot_labels = subplot_labels
        self.bkgmap = bkgmap
        self.inset = inset
        self.majorFormatter = FormatStrFormatter('%.1f')
        self.data_exists = True
        self.setup_figure(**kwargs)
        self.time_height()
        self.lon_alt()
        self.histogram()
        self.plan_view()
        self.lat_alt()
        if self.inset==True:
            self.inset_view(**kwargs)

    def setup_figure(self, **kwargs):
        if self.readtype == 'lmatools':
            try:
                self.datetime = self.data.datetime
            except:
                self.data_exists=False
                self.datetime = [self.stime,self.stime+dt.timedelta(minutes=10)]
            self.dt_init = dt.datetime(
                self.data.year, self.data.month, self.data.day)
        if self.readtype == 'pandas':
            from pandas.plotting import register_matplotlib_converters
            register_matplotlib_converters()
            try:
                self.datetime = self.data.Datetime
            except AttributeError:
                self.data_exists=False
                self.datetime = [self.stime,self.stime+dt.timedelta(minutes=10)]
            self.dt_init = dt.datetime(
                self.stime.year, self.stime.month, self.stime.day)
        if self.readtype == 'xarray':
            from pandas.plotting import register_matplotlib_converters
            register_matplotlib_converters()
            try:
                self.datetime = self.data.event_time.to_dataframe().event_time
            except:
                self.data_exists=False
                self.datetime = [self.stime,self.stime+dt.timedelta(minutes=10)]
            self.dt_init = dt.datetime(
                self.stime.year, self.stime.month, self.stime.day)
        if 'density' in kwargs.keys():
            self.density = kwargs['density']
        else:
            self.density = False
        if self.data_exists==True:
            self._subset_data(**kwargs)
            self._setup_colors(**kwargs)
        else:
            self._define_bounds(**kwargs)
        self.fig = plt.figure(figsize=(8.5, 11))
        self.ax_th = self.fig.add_axes(COORD_TH)
        if self.bkgmap == False:
            self.ax_plan = self.fig.add_axes(COORD_PLAN)
        if self.bkgmap == True:
            self.ax_plan = self.fig.add_axes(COORD_PLAN,projection=ccrs.PlateCarree())
        self.ax_lon = self.fig.add_axes(COORD_LON)
        self.ax_lat = self.fig.add_axes(COORD_LAT)
        self.ax_hist = self.fig.add_axes(COORD_HIST)
        self.yticks = 5 * np.arange(6)
        if 'title' in kwargs.keys():
            self.title = kwargs['title']
        else:
            schi = 'chisqr = ' + str(kwargs['chi2'])
            if self.density:
                self.title = \
                    self.tlim[0].strftime('%Y%m%d Source Density, ') + schi
            else:
                self.title = \
                    self.tlim[0].strftime('%Y%m%d Sources by Time, ') + schi
        self.xbins = int((self.xlim[1] - self.xlim[0]) / xdiv)
        self.ybins = int((self.ylim[1] - self.ylim[0]) / ydiv)
        self.zbins = int((self.zlim[1] - self.zlim[0]) / zdiv)
        self.tbins = 300

    def time_height(self):
        if self.data_exists == True:
            if self.readtype == 'lmatools':
                t_data = self.data.data['time'][self.cond]
                z_data = self.data.data['alt'][self.cond]/M2KM
            if self.readtype == 'pandas':
                t_data = self.data['Datetime'][self.cond]
                z_data = self.data['alt(m)'][self.cond]/M2KM
            if self.readtype == 'xarray':
                # t_data = self.data['event_time'][self.cond].data
                t_data = self.data['event_time'][self.cond].data # - np.asarray(self.dt_init, dtype='datetime64[ns]')
                z_data = self.data['event_altitude'][self.cond].data/M2KM
            if np.size(t_data)==0:
                self.data_exists = False
            else:
                if self.density:
                    # Note that the need for the call to date2num is probably a bug.
                    # See https://github.com/matplotlib/matplotlib/issues/17319.
                    if self.readtype != 'lmatools':
                        self.ax_th.hist2d(md.date2num(t_data), z_data, bins=[self.tbins, self.zbins],
                            density=True, cmap=self.cmap, cmin=0.00001)
                    else:
                        self.ax_th.hist2d(t_data,z_data,bins=[self.tbins, self.zbins],
                            normed=True, cmap=self.cmap, cmin=0.00001)
                else:
                    self.ax_th.scatter(t_data, z_data, c=self.c,
                        vmin=self.vmin, vmax=self.vmax, cmap=self.cmap, s=self.s,
                        marker='o', edgecolors='none')

        self.ax_th.set_xlabel('Time (UTC)')
        self.ax_th.set_ylabel('Altitude (km)')
        self.ax_th.set_yticks(self.yticks)
        self.ax_th.set_ylim(self.zlim)
        self.ax_th.set_title(self.title)
        self.ax_th.minorticks_on()
        # Now fix the ticks and labels on the time axis
        # Ticks
        tstep = int(1e6*(self.tlim[1] - self.tlim[0]).total_seconds()/5)
        if tstep < 5000000:
            tfmt = '%H:%M:%S.%f'
        else:
            tfmt = '%H:%M:%S000'
        if self.readtype == 'lmatools':
        # if self.readtype != 'xarray':
            sod_start = (self.tlim[0] - self.dt_init).total_seconds()
            xticks = [sod_start + i*tstep*1e-6 for i in range(6)]
            self.ax_th.set_xlim(xticks[0], xticks[-1])
            self.ax_th.set_xticks(xticks)
            # Tick labels
            dt1 = self.tlim[0]
            dt2 = dt1 + dt.timedelta(microseconds=tstep)
            dt3 = dt2 + dt.timedelta(microseconds=tstep)
            dt4 = dt3 + dt.timedelta(microseconds=tstep)
            dt5 = dt4 + dt.timedelta(microseconds=tstep)
            dt6 = dt5 + dt.timedelta(microseconds=tstep)
            self.ax_th.set_xticklabels([
                dt1.strftime(tfmt)[:-3], dt2.strftime(tfmt)[:-3],
                dt3.strftime(tfmt)[:-3], dt4.strftime(tfmt)[:-3],
                dt5.strftime(tfmt)[:-3], dt6.strftime(tfmt)[:-3]])
        else:
            self.ax_th.set_xlim(self.tlim[0], self.tlim[1])
            # self.ax_th.xaxis.set_major_locator(MaxNLocator(6))
            self.ax_th.xaxis.set_major_formatter(FractionalSecondFormatter(self.ax_th))
        # Subplot letter
        if self.subplot_labels == True:
            plt.text(0.05, 0.8, '(a)', fontsize='x-large', weight='bold',
                     horizontalalignment='center', verticalalignment='center',
                     transform=self.ax_th.transAxes)

    def lon_alt(self):
        if self.data_exists == True:
            if self.readtype == 'lmatools':
                lon_data = self.data.data['lon'][self.cond]
                alt_data = self.data.data['alt'][self.cond]/M2KM
            if self.readtype == 'pandas':
                lon_data = self.data['lon'][self.cond]
                alt_data = self.data['alt(m)'][self.cond]/M2KM
            if self.readtype == 'xarray':
                lon_data = self.data['event_longitude'][self.cond].data
                alt_data = self.data['event_altitude'][self.cond].data/M2KM
            if self.density:
                self.ax_lon.hist2d(lon_data, alt_data,
                    bins=[self.xbins, self.zbins], density=True, cmap=self.cmap,
                    cmin=0.00001)
            else:
                self.ax_lon.scatter(lon_data, alt_data, c=self.c,
                    vmin=self.vmin, vmax=self.vmax, cmap=self.cmap, s=self.s,
                    marker='o', edgecolors='none')
        self.ax_lon.set_ylabel('Altitude (km MSL)')
        self.ax_lon.set_yticks(self.yticks)
        self.ax_lon.set_ylim(self.zlim)
        if hasattr(self, 'xlim'):
            self.ax_lon.set_xlim(self.xlim)
        self.ax_lon.minorticks_on()
        # self.ax_lon.xaxis.set_major_formatter(self.majorFormatter)
        if self.subplot_labels == True:
            plt.text(0.065, 0.80, '(b)', fontsize='x-large', weight='bold',
                     horizontalalignment='center', verticalalignment='center',
                     transform=self.ax_lon.transAxes)

    def histogram(self):
        if self.data_exists == True:
            if self.readtype == 'lmatools':
                alt_data = self.data.data['alt'][self.cond]/M2KM
            if self.readtype == 'pandas':
                alt_data = self.data['alt(m)'][self.cond]/M2KM
            if self.readtype == 'xarray':
                alt_data = self.data['event_altitude'][self.cond].data/M2KM
            self.ax_hist.hist(alt_data, orientation='horizontal',
                              density=True, bins=80, range=(0, 20))
            plt.text(0.25, 0.10, str(len(alt_data)) + ' src',
                 fontsize='small',
                 horizontalalignment='left', verticalalignment='center',
                 transform=self.ax_hist.transAxes)
        else:
            self.ax_hist.text(0.02,1,'No Sources',fontsize=12)
        self.ax_hist.set_xticks([0, 0.1, 0.2, 0.3])
        self.ax_hist.set_yticks(self.yticks)
        self.ax_hist.set_ylim(self.zlim)
        self.ax_hist.set_xlim(0, 0.3)
        self.ax_hist.set_xlabel('Freq')
        self.ax_hist.minorticks_on()
        if self.subplot_labels == True:
            plt.text(0.30, 0.80, '(c)', fontsize='x-large', weight='bold',
                     horizontalalignment='center', verticalalignment='center',
                     transform=self.ax_hist.transAxes)
        

    def plan_view(self):
        if self.data_exists == True:
            if self.readtype == 'lmatools':
                lon_data = self.data.data['lon'][self.cond]
                lat_data = self.data.data['lat'][self.cond]
            if self.readtype == 'pandas':
                lon_data = self.data['lon'][self.cond]
                lat_data = self.data['lat'][self.cond]
            if self.readtype == 'xarray':
                lon_data = self.data['event_longitude'][self.cond].data
                lat_data = self.data['event_latitude'][self.cond].data
            if self.density:
                self.ax_plan.hist2d(lon_data, lat_data,
                    bins=[self.xbins, self.ybins], density=True, cmap=self.cmap,
                    cmin=0.00001)
            else:
                self.ax_plan.scatter(lon_data, lat_data, c=self.c,
                    vmin=self.vmin, vmax=self.vmax, cmap=self.cmap, s=self.s,
                    marker='o', edgecolors='none')
        if self.bkgmap == True:
            if COUNTIES is not None:
                self.ax_plan.add_feature(COUNTIES, facecolor='none', edgecolor='gray')
            self.ax_plan.add_feature(cfeature.BORDERS)
            self.ax_plan.add_feature(cfeature.STATES.with_scale('10m'))
        self.ax_plan.set_xlabel('Longitude (degrees)')
        self.ax_plan.set_ylabel('Latitude (degrees)')
        if hasattr(self, 'xlim'):
            self.ax_plan.set_xlim(self.xlim)
        if hasattr(self, 'ylim'):
            self.ax_plan.set_ylim(self.ylim)
        self.ax_plan.minorticks_on()
        self.ax_plan.xaxis.set_major_formatter(self.majorFormatter)
        if self.bkgmap == True:
            self.ax_plan.set_xticks(self.ax_plan.get_xticks())
            self.ax_plan.set_yticks(self.ax_plan.get_yticks())
        if self.bkgmap == True:
            self.ax_plan.set_extent([self.xlim[0], self.xlim[1],
                                     self.ylim[0], self.ylim[1]])
        if self.subplot_labels == True:
            plt.text(0.065, 0.95, '(d)', fontsize='x-large', weight='bold',
                     horizontalalignment='center', verticalalignment='center',
                     transform=self.ax_plan.transAxes)

    def lat_alt(self):
        if self.data_exists == True:
            if self.readtype == 'lmatools':
                alt_data = self.data.data['alt'][self.cond]/M2KM
                lat_data = self.data.data['lat'][self.cond]
            if self.readtype == 'pandas':
                alt_data = self.data['alt(m)'][self.cond]/M2KM
                lat_data = self.data['lat'][self.cond]
            if self.readtype == 'xarray':
                alt_data = self.data['event_altitude'][self.cond].data/M2KM
                lat_data = self.data['event_latitude'][self.cond].data
            if self.density:
                self.ax_lat.hist2d(alt_data, lat_data,
                    bins=[self.zbins, self.ybins], density=True, cmap=self.cmap,
                    cmin=0.00001)
            else:
                self.ax_lat.scatter(alt_data, lat_data, c=self.c,
                    vmin=self.vmin, vmax=self.vmax, cmap=self.cmap, s=self.s,
                    marker='o', edgecolors='none')
        self.ax_lat.set_xlabel('Altitude (km MSL)')
        self.ax_lat.set_xticks(self.yticks)
        self.ax_lat.set_xlim(self.zlim)
        if hasattr(self, 'ylim'):
            self.ax_lat.set_ylim(self.ylim)
        self.ax_lat.minorticks_on()
        for xlabel_i in self.ax_lat.get_yticklabels():
            xlabel_i.set_fontsize(0.0)
            xlabel_i.set_visible(False)
        if self.subplot_labels == True:
            plt.text(0.30, 0.95, '(e)', fontsize='x-large', weight='bold',
                     horizontalalignment='center', verticalalignment='center',
                     transform=self.ax_lat.transAxes)

    def inset_view(self, **kwargs):
        if 'inset_buffer' in kwargs.keys():
            self.buffer = kwargs['inset_buffer']
        else:
            self.buffer = 0.5
        if 'inset_size' in kwargs.keys():
            self.inset_size = kwargs['inset_size']
        else:
            self.inset_size = 0.15
        self.inset = self.fig.add_axes([0.02, 0.01, 0.02+self.inset_size, 
                                                    0.01+self.inset_size],projection=ccrs.PlateCarree())
        if self.data_exists==True:
            if self.readtype == 'lmatools':
                lon_data = self.data.data['lon'][self.cond2]
                lat_data = self.data.data['lat'][self.cond2]
            if self.readtype == 'pandas':
                lon_data = self.data['lon'][self.cond2]
                lat_data = self.data['lat'][self.cond2]
            if self.readtype == 'xarray':
                lon_data = self.data['event_longitude'][self.cond2].data
                lat_data = self.data['event_latitude'][self.cond2].data
            self.inset.hist2d(lon_data, lat_data,
                        bins=[int((self.xlim[1]+self.buffer*2 - self.xlim[0]) / xdiv), 
                              int((self.ylim[1]+self.buffer*2 - self.ylim[0]) / ydiv)],
                        density=True, cmap=self.cmap,
                        cmin=0.00001)

        if self.bkgmap == True:
            if COUNTIES is not None:
                self.inset.add_feature(COUNTIES, facecolor='none', edgecolor='gray')
            self.inset.add_feature(cfeature.BORDERS)
            self.inset.add_feature(cfeature.STATES.with_scale('10m'))
            self.inset.set_extent([self.xlim[0]-self.buffer, self.xlim[1]+self.buffer, 
                                   self.ylim[0]-self.buffer, self.ylim[1]+self.buffer])
        self.inset.plot([self.xlim[0],self.xlim[0],self.xlim[1],self.xlim[1],self.xlim[0]],
                        [self.ylim[0],self.ylim[1],self.ylim[1],self.ylim[0],self.ylim[0]],
                        'k')


    def _define_bounds(self,**kwargs):
        if 'xlim' in kwargs.keys():
            self.xlim = kwargs['xlim']
        else:
            self.xlim = [-100,-96]

        if 'ylim' in kwargs.keys():
            self.ylim = kwargs['ylim']
        else:
            self.ylim = [33,37]

        if 'zlim' in kwargs.keys():
            self.zlim = kwargs['zlim']
        else:
            self.zlim = [0, 20]

        if 'tlim' in kwargs.keys():
            self.tlim = kwargs['tlim']
        else:
            self.tlim = (self.datetime[0], self.datetime[-1])

    def _subset_data(self, **kwargs):
        if self.readtype == 'lmatools':
            if 'chi2' in kwargs.keys():
                self.cond = self.data.data['chi2'] <= kwargs['chi2']
            else:
                self.cond = self.data.data['chi2'] <= 5
            if self.inset == True:
                self.cond2 = self.cond.copy()
                if 'zlim' in kwargs.keys():
                    self.zlim = kwargs['zlim']
                else:
                    self.zlim = (0, 20)
                tmpcond = np.logical_and(self.data.data['alt']/M2KM >= self.zlim[0],
                                         self.data.data['alt']/M2KM <= self.zlim[1])
                self.cond2 = np.logical_and(self.cond2, tmpcond)
            if 'xlim' in kwargs.keys():
                self.xlim = kwargs['xlim']
                tmpcond = np.logical_and(self.data.data['lon'] >= self.xlim[0],
                                         self.data.data['lon'] <= self.xlim[1])
                self.cond = np.logical_and(self.cond, tmpcond)
            if 'ylim' in kwargs.keys():
                self.ylim = kwargs['ylim']
                tmpcond = np.logical_and(self.data.data['lat'] >= self.ylim[0],
                                         self.data.data['lat'] <= self.ylim[1])
                self.cond = np.logical_and(self.cond, tmpcond)
            if 'zlim' in kwargs.keys():
                self.zlim = kwargs['zlim']
            else:
                self.zlim = (0, 20)
            tmpcond = np.logical_and(self.data.data['alt']/M2KM >= self.zlim[0],
                                     self.data.data['alt']/M2KM <= self.zlim[1])
            self.cond = np.logical_and(self.cond, tmpcond)

        if self.readtype == 'pandas':
            if 'chi2' in kwargs.keys():
                self.cond = self.data['reduced chi^2'] <= kwargs['chi2']
            else:
                self.cond = self.data.data['reduced chi^2'] <= 5
            if self.inset == True:
                self.cond2 = self.cond.copy()
                if 'zlim' in kwargs.keys():
                    self.zlim = kwargs['zlim']
                else:
                    self.zlim = (0, 20)
                tmpcond = np.logical_and(self.data['alt(m)']/M2KM >= self.zlim[0],
                                         self.data['alt(m)']/M2KM <= self.zlim[1])
                self.cond2 = np.logical_and(self.cond2, tmpcond)
            if 'xlim' in kwargs.keys():
                self.xlim = kwargs['xlim']
                tmpcond = np.logical_and(self.data['lon'] >= self.xlim[0],
                                         self.data['lon'] <= self.xlim[1])
                self.cond = np.logical_and(self.cond, tmpcond)
            if 'ylim' in kwargs.keys():
                self.ylim = kwargs['ylim']
                tmpcond = np.logical_and(self.data['lat'] >= self.ylim[0],
                                         self.data['lat'] <= self.ylim[1])
                self.cond = np.logical_and(self.cond, tmpcond)
            if 'zlim' in kwargs.keys():
                self.zlim = kwargs['zlim']
            else:
                self.zlim = (0, 20)
            tmpcond = np.logical_and(self.data['alt(m)']/M2KM >= self.zlim[0],
                                     self.data['alt(m)']/M2KM <= self.zlim[1])
            self.cond = np.logical_and(self.cond, tmpcond)


        if self.readtype == 'xarray':
            if 'chi2' in kwargs.keys():
                self.cond = self.data['event_chi2'] <= kwargs['chi2']
            else:
                self.cond = self.data['event_chi2'] <= 5.0
            if self.inset == True:
                self.cond2 = self.cond.copy()
                if 'zlim' in kwargs.keys():
                    self.zlim = kwargs['zlim']
                else:
                    self.zlim = (0, 20)
                tmpcond = np.logical_and(self.data['event_altitude']/M2KM >= self.zlim[0],
                                         self.data['event_altitude']/M2KM <= self.zlim[1])
                self.cond2 = np.logical_and(self.cond2, tmpcond)
            if 'xlim' in kwargs.keys():
                self.xlim = kwargs['xlim']
                tmpcond = ((self.data['event_longitude'] >= self.xlim[0]) &
                           (self.data['event_longitude'] <= self.xlim[1]))
                self.cond = np.logical_and(self.cond, tmpcond)
            if 'ylim' in kwargs.keys():
                self.ylim = kwargs['ylim']
                tmpcond = ((self.data['event_latitude'] >= self.ylim[0]) &
                           (self.data['event_latitude'] <= self.ylim[1]))
                self.cond = np.logical_and(self.cond, tmpcond)
            if 'zlim' in kwargs.keys():
                self.zlim = kwargs['zlim']
            else:
                self.zlim = (0, 20)
            tmpcond = ((self.data['event_altitude']/M2KM >= self.zlim[0]) &
                       (self.data['event_altitude']/M2KM <= self.zlim[1]))
            self.cond = np.logical_and(self.cond, tmpcond)

    def _setup_colors(self, **kwargs):
        self.cmap = kwargs['cmap']

        if 'tlim' in kwargs.keys():
            self.tlim = kwargs['tlim']
            tmpcond = np.logical_and(self.datetime >= self.tlim[0],
                                     self.datetime <= self.tlim[1])
            self.cond = np.logical_and(np.asarray(self.cond),
                                       np.asarray(tmpcond))
            self.vmax = (self.tlim[1] -
                         self.datetime[self.cond].min()).total_seconds()
            ldiff = self.datetime[self.cond] - self.datetime[self.cond].min()
            ldf = []
            for df in ldiff:
                ldf.append(df.total_seconds())
            self.c = np.array(ldf)
        else:
            self.tlim = (self.datetime[0], self.datetime[-1])
            if self.readtype == 'lmatools':
                self.c = self.data.data['time'][self.cond] - \
                    self.data.data['time'][self.cond].min()
                self.vmax = self.data.data['time'][self.cond].max() - \
                    self.data.data['time'][self.cond].min()
            if self.readtype == 'pandas':
                self.c = self.data['time (UT sec of day)'][self.cond] - \
                    self.data['time (UT sec of day)'][self.cond].min()
                self.vmax = self.data['time (UT sec of day)'][self.cond].max() - \
                    self.data['time (UT sec of day)'][self.cond].min()
            if self.readtype == 'xarray':
                self.c = self.data['event_time'][self.cond] - \
                    self.data['event_time'][self.cond].min()
                self.vmax = self.data['event_time'][self.cond].max() - \
                    self.data['event_time'][self.cond].min()
        self.vmin = 0
        if 's' in kwargs.keys():
            self.s = kwargs['s']
        else:
            self.s = 5

