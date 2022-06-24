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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

GeoAxes._pcolormesh_patched = Axes.pcolormesh

# proj_cart = ccrs.PlateCarree(central_longitude=-95)

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

COORD_TH = [0.1, 0.8, 0.83, 0.1]
COORD_PLAN = [0.1, 0.1, 0.65, 0.5]
COORD_LON = [0.1, 0.65, 0.65, 0.1]
COORD_LAT = [0.8, 0.1, 0.13, 0.5]
COORD_HIST = [0.8, 0.65, 0.13, 0.1]

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


class BlankPlot(object):
    """
    Generate a matching plot setup with no data

    Requires:
    stime = starting time
    xlim, ylim = bounds of the domain
    title = title string
    tlim = list of start and end times

    Will include map information if bkgmap==True
    """
    def __init__(self, stime, bkgmap=True, **kwargs):
        self.zlim = kwargs['zlim']
        self.tlim = kwargs['tlim']
        self.ylim = kwargs['ylim']
        self.xlim = kwargs['xlim']
        self.bkgmap = bkgmap
        self.stime = stime
        self.majorFormatter = FormatStrFormatter('%.2f')
        self.dt_init = dt.datetime(
                self.stime.year, self.stime.month, self.stime.day)
        self.plot(**kwargs)


    def plot(self, **kwargs):
        self.fig = plt.figure(figsize=(8.5, 11))
        self.ax_th = self.fig.add_axes(COORD_TH)
        if self.bkgmap == True:
            self.ax_plan = self.fig.add_axes(COORD_PLAN,projection=ccrs.PlateCarree())
        else:
            self.ax_plan = self.fig.add_axes(COORD_PLAN)
        self.ax_lon = self.fig.add_axes(COORD_LON)
        self.ax_lat = self.fig.add_axes(COORD_LAT)
        self.ax_hist = self.fig.add_axes(COORD_HIST)
        self.yticks = 5 * np.arange(6)
        self.title = kwargs['title']

        # Time altitude panel
        self.ax_th.set_xlabel('Time (UTC)')
        self.ax_th.set_ylabel('Altitude (km)')
        self.ax_th.set_yticks(self.yticks)
        self.ax_th.set_ylim(self.zlim)
        self.ax_th.set_title(self.title)
        self.ax_th.minorticks_on()
        tstep = int(1e6*(self.tlim[1] - self.tlim[0]).total_seconds()/5)
        if tstep < 5000000:
            tfmt = '%H:%M:%S.%f'
        else:
            tfmt = '%H:%M:%S000'
        self.ax_th.set_xlim(self.tlim[0], self.tlim[1])
        self.ax_th.xaxis.set_major_formatter(FractionalSecondFormatter(self.ax_th))

        # Longitude-Altitue
        self.ax_lon.set_ylabel('Altitude (km MSL)')
        self.ax_lon.set_yticks(self.yticks)
        self.ax_lon.set_ylim(self.zlim)
        self.ax_lon.set_xlim(self.xlim)
        self.ax_lon.minorticks_on()
        for xlabel_i in self.ax_lon.get_xticklabels():
            xlabel_i.set_fontsize(0.0)
            xlabel_i.set_visible(False)

        # Height-VHF count
        self.ax_hist.set_xticks([0, 0.1, 0.2, 0.3])
        self.ax_hist.set_yticks(self.yticks)
        self.ax_hist.set_ylim(self.zlim)
        self.ax_hist.set_xlim(0, 0.3)
        self.ax_hist.set_xlabel('Freq')
        self.ax_hist.minorticks_on()

        # Altitude-Latitude
        self.ax_lat.set_xlabel('Altitude (km MSL)')
        self.ax_lat.set_xticks(self.yticks)
        self.ax_lat.set_xlim(self.zlim)
        self.ax_lat.set_ylim(self.ylim)
        self.ax_lat.minorticks_on()
        for xlabel_i in self.ax_lat.get_yticklabels():
            xlabel_i.set_fontsize(0.0)
            xlabel_i.set_visible(False)

        # Plan view
        if self.bkgmap==True:
            self.ax_plan.add_feature(COUNTIES, facecolor='none', edgecolor='gray')
            self.ax_plan.add_feature(cfeature.BORDERS)
            self.ax_plan.add_feature(cfeature.STATES.with_scale('10m'))
        self.ax_plan.set_xlabel('Longitude (degrees)')
        self.ax_plan.set_ylabel('Latitude (degrees)')
        self.ax_plan.set_xlim(self.xlim)
        self.ax_plan.set_ylim(self.ylim)


        # lon_formatter = LongitudeFormatter(number_format='.2f',
        #                                    degree_symbol='',
        #                                    dateline_direction_label=False)
        # lat_formatter = LatitudeFormatter(number_format='.2f',
        #                                   degree_symbol='')
        # self.ax_plan.xaxis.set_major_formatter(lon_formatter)
        # self.ax_plan.yaxis.set_major_formatter(lat_formatter)
        #
        self.ax_plan.minorticks_on()
        self.ax_plan.xaxis.set_major_formatter(self.majorFormatter)
        self.ax_plan.yaxis.set_major_formatter(self.majorFormatter)
        self.ax_plan.set_xticks(self.ax_lon.get_xticks())
        self.ax_plan.set_yticks(self.ax_lat.get_yticks())
        if self.bkgmap==True:
            self.ax_plan.set_extent([self.xlim[0], self.xlim[1],
                                     self.ylim[0], self.ylim[1]])


def subplot_labels(plot):
    """
    Place letters on each subplot panel.

    Returns a list of matplotlib text artists
    """
    a = plt.text(0.05, 0.8, '(a)', fontsize='x-large', weight='bold',
                     horizontalalignment='center', verticalalignment='center',
                     transform=plot.ax_th.transAxes)
    b = plt.text(0.065, 0.80, '(b)', fontsize='x-large', weight='bold',
                     horizontalalignment='center', verticalalignment='center',
                     transform=plot.ax_lon.transAxes)
    c = plt.text(0.30, 0.80, '(c)', fontsize='x-large', weight='bold',
                     horizontalalignment='center', verticalalignment='center',
                     transform=plot.ax_hist.transAxes)
    d = plt.text(0.065, 0.95, '(d)', fontsize='x-large', weight='bold',
                     horizontalalignment='center', verticalalignment='center',
                     transform=plot.ax_plan.transAxes)
    e = plt.text(0.30, 0.95, '(e)', fontsize='x-large', weight='bold',
                     horizontalalignment='center', verticalalignment='center',
                     transform=plot.ax_lat.transAxes)
    return [a,b,c,d,e]


def inset_view(plot, lon_data, lat_data, xlim, ylim, xdiv, ydiv,
               buffer=0.5, inset_size=0.15, plot_cmap = 'magma', bkgmap = True):
    """
    Overlay an inset panel of size 'inset_size' showing a plan-view histogram
    of sources at xdiv, ydiv intervals and outlining a box over xlim and ylim
    with buffer of 'buffer' lat/lon degrees in the image.

    Add background map features if 'bkgmap' == True
    """
    inset = plot.fig.add_axes([0.02, 0.01, 0.02+inset_size,
                              0.01+inset_size],projection=ccrs.PlateCarree())

    inset.hist2d(lon_data, lat_data,
                bins=[np.arange(xlim[0]-buffer, xlim[1]+buffer+xdiv, xdiv),
                      np.arange(ylim[0]-buffer, ylim[1]+buffer+ydiv, ydiv)],
                density=True, cmap=plot_cmap,
                cmin=0.00001)

    if bkgmap == True:
        if COUNTIES is not None:
            inset.add_feature(COUNTIES, facecolor='none', edgecolor='gray')
        inset.add_feature(cfeature.BORDERS)
        inset.add_feature(cfeature.STATES.with_scale('10m'))
        inset.set_extent([xlim[0]-buffer, xlim[1]+buffer,
                          ylim[0]-buffer, ylim[1]+buffer])
    inset.plot([xlim[0],xlim[0],xlim[1],xlim[1],xlim[0]],
               [ylim[0],ylim[1],ylim[1],ylim[0],ylim[0]],'k')
    return inset