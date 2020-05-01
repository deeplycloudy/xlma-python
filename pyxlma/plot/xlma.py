import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime as dt
from matplotlib.ticker import FormatStrFormatter

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

proj_cart = ccrs.PlateCarree(central_longitude=-95)


# Add a note about plotting counties by default if metpy is available in docs, and how to add your own map data without relying on built-ins.
# reader = shpreader.Reader('/Users/vannac/Documents/UScounties/UScounties.shp')
# reader = shpreader.Reader('/home/vanna/status_plots/UScounties/UScounties.shp')
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


class XlmaPlot(object):

    def __init__(self, data, stime, subplot_labels, bkgmap, readtype, **kwargs):
        """
        data = an data structure matching the lmatools or pyxlma formats:
            1. pyxlma.lmalib.io.lmatools.LMAdata.data for reading with lmatools
            2. pyxlma.lmalib.io.read.lma_file.readfile() for a pandas dataframe
              with same headers as LYLOUT files.
        stime = start time datetime object
        readtype = fileread options "lmatools" or "pandas" dataframe
        """
        self.data = data
        self.readtype = readtype
        if self.readtype == 'pandas':
            self.stime = stime
        self.subplot_labels = subplot_labels
        self.bkgmap = bkgmap
        self.majorFormatter = FormatStrFormatter('%.1f')
        self.setup_figure(**kwargs)
        self.time_height()
        self.lon_alt()
        self.histogram()
        self.plan_view()
        self.lat_alt()

    def setup_figure(self, **kwargs):
        if self.readtype == 'lmatools':
            self.datetime = self.data.datetime
            self.dt_init = dt.datetime(
                self.data.year, self.data.month, self.data.day)
        if self.readtype == 'pandas':
            self.datetime = self.data.Datetime
            self.dt_init = dt.datetime(
                self.stime.year, self.stime.month, self.stime.day)
        self._subset_data(**kwargs)
        self._setup_colors(**kwargs)
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
        if self.density:
            if self.readtype == 'lmatools':
                self.ax_th.hist2d(
                    self.data.data['time'][self.cond],
                    self.data.data['alt'][self.cond]/M2KM,
                    bins=[self.tbins, self.zbins],
                    density=True, cmap=self.cmap, cmin=0.00001)
            if self.readtype == 'pandas':
                self.ax_th.hist2d(
                    self.data['time (UT sec of day)'][self.cond],
                    self.data['alt(m)'][self.cond]/M2KM,
                    bins=[self.tbins, self.zbins],
                    density=True, cmap=self.cmap, cmin=0.00001)
        else:
            if self.readtype == 'lmatools':
                self.ax_th.scatter(
                    self.data.data['time'][self.cond],
                    self.data.data['alt'][self.cond]/M2KM, c=self.c,
                    vmin=self.vmin, vmax=self.vmax, cmap=self.cmap, s=self.s,
                    marker='o', edgecolors='none')
            if self.readtype == 'pandas':
                self.ax_th.scatter(
                    self.data['time (UT sec of day)'][self.cond],
                    self.data['alt(m)'][self.cond]/M2KM, c=self.c,
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
        if tstep < 5000000:
            tfmt = '%H:%M:%S.%f'
        else:
            tfmt = '%H:%M:%S000'
        self.ax_th.set_xticklabels([
            dt1.strftime(tfmt)[:-3], dt2.strftime(tfmt)[:-3],
            dt3.strftime(tfmt)[:-3], dt4.strftime(tfmt)[:-3],
            dt5.strftime(tfmt)[:-3], dt6.strftime(tfmt)[:-3]])
        # Subplot letter
        if self.subplot_labels == True:
            plt.text(0.05, 0.8, '(a)', fontsize='x-large', weight='bold',
                     horizontalalignment='center', verticalalignment='center',
                     transform=self.ax_th.transAxes)

    def lon_alt(self):
        if self.density:
            if self.readtype == 'lmatools':
                self.ax_lon.hist2d(
                    self.data.data['lon'][self.cond],
                    self.data.data['alt'][self.cond]/M2KM,
                    bins=[self.xbins, self.zbins], density=True, cmap=self.cmap,
                    cmin=0.00001)
            if self.readtype == 'pandas':
                self.ax_lon.hist2d(
                    self.data['lon'][self.cond],
                    self.data['alt(m)'][self.cond]/M2KM,
                    bins=[self.xbins, self.zbins], density=True, cmap=self.cmap,
                    cmin=0.00001)
        else:
            if self.readtype == 'lmatools':
                self.ax_lon.scatter(
                    self.data.data['lon'][self.cond],
                    self.data.data['alt'][self.cond]/M2KM, c=self.c,
                    vmin=self.vmin, vmax=self.vmax, cmap=self.cmap, s=self.s,
                    marker='o', edgecolors='none')
            if self.readtype == 'pandas':
                self.ax_lon.scatter(
                    self.data['lon'][self.cond],
                    self.data['alt(m)'][self.cond]/M2KM, c=self.c,
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
        if self.readtype == 'lmatools':
            self.ax_hist.hist(self.data.data['alt'][self.cond]/M2KM,
                              orientation='horizontal',
                              density=True, bins=80, range=(0, 20))

        if self.readtype == 'pandas':
            self.ax_hist.hist(self.data['alt(m)'][self.cond]/M2KM,
                              orientation='horizontal',
                              density=True, bins=80, range=(0, 20))
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
        plt.text(0.25, 0.10,
#                  str(len(self.data.data['alt'][self.cond])) + ' src',
                 str(len(self.data['alt(m)'][self.cond])) + ' src',
                 fontsize='small',
                 horizontalalignment='left', verticalalignment='center',
                 transform=self.ax_hist.transAxes)

    def plan_view(self):
        if self.density:
            if self.readtype == 'lmatools':
                self.ax_plan.hist2d(
                    self.data.data['lon'][self.cond],
                    self.data.data['lat'][self.cond],
                    bins=[self.xbins, self.ybins], density=True, cmap=self.cmap,
                    cmin=0.00001)
            if self.readtype == 'pandas':
                self.ax_plan.hist2d(
                    self.data['lon'][self.cond],
                    self.data['lat'][self.cond],
                    bins=[self.xbins, self.ybins], density=True, cmap=self.cmap,
                    cmin=0.00001)
        else:
            if self.readtype == 'lmatools':
                self.ax_plan.scatter(
                    self.data.data['lon'][self.cond],
                    self.data.data['lat'][self.cond], c=self.c,
                    vmin=self.vmin, vmax=self.vmax, cmap=self.cmap, s=self.s,
                    marker='o', edgecolors='none')
            if self.readtype == 'pandas':
                self.ax_plan.scatter(
                    self.data['lon'][self.cond],
                    self.data['lat'][self.cond], c=self.c,
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
        if self.density:
            if self.readtype == 'lmatools':
                self.ax_lat.hist2d(
                    self.data.data['alt'][self.cond]/M2KM,
                    self.data.data['lat'][self.cond],
                    bins=[self.zbins, self.ybins], density=True, cmap=self.cmap,
                    cmin=0.00001)
            if self.readtype == 'pandas':
                self.ax_lat.hist2d(
                    self.data['alt(m)'][self.cond]/M2KM,
                    self.data['lat'][self.cond],
                    bins=[self.zbins, self.ybins], density=True, cmap=self.cmap,
                    cmin=0.00001)
        else:
            if self.readtype == 'lmatools':
                self.ax_lat.scatter(
                    self.data.data['alt'][self.cond]/M2KM,
                    self.data.data['lat'][self.cond], c=self.c,
                    vmin=self.vmin, vmax=self.vmax, cmap=self.cmap, s=self.s,
                    marker='o', edgecolors='none')
            if self.readtype == 'pandas':
                self.ax_lat.scatter(
                    self.data['alt(m)'][self.cond]/M2KM,
                    self.data['lat'][self.cond], c=self.c,
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

    def _subset_data(self, **kwargs):
        if self.readtype == 'lmatools':
            if 'chi2' in kwargs.keys():
                self.cond = self.data.data['chi2'] <= kwargs['chi2']
            else:
                self.cond = self.data.data['chi2'] <= 5
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

    def _setup_colors(self, **kwargs):
        self.cmap = kwargs['cmap']
        if 'density' in kwargs.keys():
            self.density = kwargs['density']
        else:
            self.density = False
        if 'tlim' in kwargs.keys():
            self.tlim = kwargs['tlim']
            tmpcond = np.logical_and(self.datetime >= self.tlim[0],
                                     self.datetime <= self.tlim[1])
            self.cond = np.logical_and(self.cond, tmpcond)
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
        self.vmin = 0
        if 's' in kwargs.keys():
            self.s = kwargs['s']
        else:
            self.s = 5

