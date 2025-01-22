import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime as dt
from matplotlib.ticker import Formatter, FormatStrFormatter, MaxNLocator
from matplotlib.dates import AutoDateLocator

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
    """Helper class to format fractional seconds in time labels on BlankPlot."""
    def __init__(self, axis):
        """Create a FractionalSecondFormatter object.
        
        Parameters
        ----------
        axis : matplotlib.axis.Axis
            The axis to format.
        """
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
    """Generate LMA plot setup with no data.

    Generates a matplotlib figure in the style of the base plot from XLMA (plan view, cross sections, alt histogram, alt-timeseries axes).

    Attributes
    ----------
    zlim : iterable of floats
        Altitude limits [min, max]
    tlim : iterable of datetimes
        Time limits [start, end]
    ylim : iterable of floats
        Latitude limits [min, max]
    xlim : iterable of floats
        Longitude limits [min, max]
    bkgmap : bool
        Whether ax_plan is a GeoAxes or standard matplotlib axis.
    ax_plan : matplotlib.axes.Axes or cartopy.mpl.geoaxes.GeoAxes
        Plan view axis
    ax_th : matplotlib.axes.Axes
        Time-altitude axis
    ax_lon : matplotlib.axes.Axes
        Longitude-altitude axis
    ax_lat : matplotlib.axes.Axes
        Latitude-altitude axis
    ax_hist : matplotlib.axes.Axes
        Altitude histogram axis
    fig : matplotlib.figure.Figure
        The figure object    
    """
    def __init__(self, stime, bkgmap=True, **kwargs):
        """Create a BlankPlot object.

        Parameters
        ----------
        stime : datetime
            Starting time for the plot
        bkgmap : bool, default=True
            Whether to include background map projection and add country/state borders.
            If False, the axis will be a standard matplotlib axis. If True, the axis will be a cartopy GeoAxes with the PlateCarree projection.
            If MetPy is installed and importable, the plot will include US county borders.
        zlim : iterable of floats
            Altitude limits [min, max]
        tlim : iterable of datetimes
            Time limits [start, end]
        ylim : iterable of floats
            Latitude limits [min, max]
        xlim : iterable of floats
            Longitude limits [min, max]
        title : str
            The title to use for the plot.
        """
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

    def set_ax_plan_labels(self):
        """Sets the labels on the planview axis to match the cross section axes.

        If the background map is enabled, ensure the plan view axis has the same extent as the cross section axes.
        """
        self.ax_plan.set_xticks(self.ax_lon.get_xticks())
        self.ax_plan.set_yticks(self.ax_lat.get_yticks())
        if self.bkgmap==True:
            self.ax_plan.set_extent([self.xlim[0], self.xlim[1],
                                     self.ylim[0], self.ylim[1]])

    def plot(self, **kwargs):
        """Draw the Blank Plot"""
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
        # Importing pandas results in it overriding Matplotlib default AutoDateLocator,
        # which in turn prevents more than one tick displaying for short time intervals.
        # See this blog entry for details.
        # https://notebook.community/azjps/matplotlib-tick-formatters/ notebooks/microsecond_precision At small time intervals,
        # So, restore Matplotlib's AutoDateLocator, which as of version >3.3 correctly
        # handles smaller time intervals. 3.2 did not!
        self.ax_th.xaxis.set_major_locator(AutoDateLocator())

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
            if COUNTIES != None:
                self.ax_plan.add_feature(COUNTIES, facecolor='none', edgecolor='gray')
            self.ax_plan.add_feature(cfeature.BORDERS)
            self.ax_plan.add_feature(cfeature.STATES.with_scale('10m'))
        self.ax_plan.set_xlabel('Longitude (degrees)')
        self.ax_plan.set_ylabel('Latitude (degrees)')


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
        self.set_ax_plan_labels()
        self.ax_plan.set_xlim(self.xlim)
        self.ax_plan.set_ylim(self.ylim)
        self.ax_plan.set_aspect('auto')

def subplot_labels(plot):
    """Place letters on each panel of a BlankPlot.

    Parameters
    ----------
    plot : BlankPlot
        The BlankPlot object to label.
    
    Returns
    -------
    list of matplotlib.text.Text
        Handles of the text objects for each label.
    
    Notes
    -----
    This function is useful for creating publication-quality figures with multiple panels which require labeling.
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
    """Overlay an inset panel showing a plan-view 2D histogram of sources.

    Parameters
    ----------
    plot : BlankPlot
        The BlankPlot object to add the inset to.
    lon_data : array_like
        longitudes of sources to be added to the histogram
    lat_data : array_like
        latitudes of sources to be added to the histogram
    xlim : iterable of floats
        x (or longitude) limits [min, max]
    ylim : iterable of floats
        y (or latitude) limits [min, max]
    xdiv : float
        x (or longitude) bin width for histogram
    ydiv : float
        y (or latitude) bin width for histogram
    buffer : float, default=0.5
        x/y buffer to be added/subtracted from xlim/ylim in the histogram bin edges
    inset_size : float, default=0.15
        Size of the inset panel as a fraction of the figure size
    plot_cmap : str, default='magma'
        Colormap to use for the histogram
    bkgmap : bool, default=True
        Whether to include background map projection and add country/state borders.

    Returns
    -------
    inset : cartopy.mpl.geoaxes.GeoAxes or matplotlib.axes.Axes
        The axis object containing the inset histogram.
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