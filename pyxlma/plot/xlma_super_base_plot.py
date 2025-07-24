import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime as dt
import pyart
import os
import glob
from pathlib import Path
import pandas as pd
import h5py
from matplotlib.ticker import Formatter, FormatStrFormatter, MaxNLocator
from matplotlib.dates import AutoDateLocator
import matplotlib.cm as cm
from scipy import spatial
import matplotlib.colors as mcolors
import wradlib
from geopy import distance

import pyxlma.plot.radar as lmarad

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

GeoAxes._pcolormesh_patched = Axes.pcolormesh

windows_os = True
file_type = 'h5'
file_suffix = 'flash.h5'
tbuffer = 1*60
max_chi = 1
min_stations = 6
max_dist = 2.5e3
m = -10

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

COORD_TH = [0.05, 0.8, 0.415, 0.1] #Time-Height
COORD_PLAN = [0.05, 0.1, 0.325, 0.5] #LMA Plan
COORD_LON = [0.05, 0.65, 0.325, 0.1] #Longitude-Height
COORD_LAT = [0.4, 0.1, 0.065, 0.5] #Latitude-Height
COORD_HIST = [0.4, 0.65, 0.065, 0.1] #Histogram-Height
COORD_RAD = [0.51, 0.535, 0.26, 0.405] #Radar Plan
COORD_VEL = [0.51, 0.1, 0.26, 0.385] #Leader Speed Plot
COORD_RCS = [0.82, 0.85, 0.17, 0.1] #Radar Cross Section
COORD_VCS = [0.82, 0.7, 0.17, 0.1] #Velocity Cross Section
COORD_SWCS = [0.82, 0.55, 0.17, 0.1] #Spectrum Width Cross Section
COORD_DRCS = [0.82, 0.4, 0.17, 0.1] #Differential Reflectivity Cross Section
COORD_SDPCS = [0.82, 0.25, 0.17, 0.1] #Specific Differential Phase Cross Section
COORD_CCCS = [0.82, 0.1, 0.17, 0.1] #Correlation Coefficient Cross Section

LMA_dir = '/Users/BenLa/.spyder-py3'

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


class SuperBlankPlot(object):
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
        self.radar_data = kwargs['radar_data']
        self.points = kwargs['points']
        self.plot(**kwargs)

    def set_ax_plan_labels(self):
        self.ax_plan.set_xticks(self.ax_lon.get_xticks())
        self.ax_plan.set_yticks(self.ax_lat.get_yticks())
        if self.bkgmap==True:
            self.ax_plan.set_extent([self.xlim[0], self.xlim[1],
                                     self.ylim[0], self.ylim[1]])
# 
    def plot(self, **kwargs):
        self.fig = plt.figure(figsize=(19, 11))
        self.ax_th = self.fig.add_axes(COORD_TH)
        if self.bkgmap == True:
            self.ax_plan = self.fig.add_axes(COORD_PLAN,projection=ccrs.PlateCarree())
        else:
            self.ax_plan = self.fig.add_axes(COORD_PLAN)
        self.ax_lon = self.fig.add_axes(COORD_LON)
        self.ax_lat = self.fig.add_axes(COORD_LAT)
        self.ax_hist = self.fig.add_axes(COORD_HIST)
        #if self.bkgmap == True:
        self.ax_rad = self.fig.add_axes(COORD_RAD, projection=ccrs.PlateCarree())
        radar_data = self.radar_data
        points = self.points
        
        closest_time = {abs(self.tlim[0].timestamp() - date.timestamp()) : date for date in radar_data[-1]}
        result = closest_time[min(closest_time.keys())]
        display = pyart.graph.GridMapDisplay(radar_data[radar_data[-1].index(result)])
        
        self.ax_vel = self.fig.add_axes(COORD_VEL)
        self.ax_rcs = self.fig.add_axes(COORD_RCS)#, self.fig.colorbar(cm.ScalarMappable(norm = mcolors.Normalize(-32, 65, 'False'), cmap='pyart_HomeyerRainbow'), label='Reflectivity (dBZ)', cax = plt.axes((0.96, 0.85, 0.007, 0.1)))
        self.ax_vcs = self.fig.add_axes(COORD_VCS)#, self.fig.colorbar(cm.ScalarMappable(norm = mcolors.Normalize(-40, 40, 'False'), cmap='pyart_NWSVel'), label='Velocity (m/s)', cax = plt.axes((0.96, 0.7, 0.007, 0.1)))
        self.ax_swcs = self.fig.add_axes(COORD_SWCS)#, self.fig.colorbar(cm.ScalarMappable(norm = mcolors.Normalize(0, 14.1, 'False'), cmap='pyart_NWS_SPW'), label='Spectrum Width (m/s)', cax = plt.axes((0.96, 0.55, 0.007, 0.1)))
        self.ax_drcs = self.fig.add_axes(COORD_DRCS)#, self.fig.colorbar(cm.ScalarMappable(norm = mcolors.Normalize(-2, 6, 'False'), cmap='pyart_RefDiff'), label='ZDR (DB)', cax = plt.axes((0.96, 0.4, 0.007, 0.1)))
        self.ax_sdpcs = self.fig.add_axes(COORD_SDPCS)#, self.fig.colorbar(cm.ScalarMappable(norm = mcolors.Normalize(-1, 4.1, 'False'), cmap='pyart_SCook18'), label='Specific Differential \n Phase (degrees)', cax = plt.axes((0.96, 0.25, 0.007, 0.1)))
        self.ax_cccs = self.fig.add_axes(COORD_CCCS)#, self.fig.colorbar(cm.ScalarMappable(norm = mcolors.Normalize(0.7, 1.03, 'False'), cmap='pyart_SCook18'), label='Correlation Coefficient', cax = plt.axes((0.96, 0.1, 0.007, 0.1)))
        self.yticks = 5 * np.arange(6)
        self.title = kwargs['title']


        # Time altitude panel [OK]
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

        # Longitude-Altitue [OK]
        self.ax_lon.set_ylabel('Altitude (km MSL)')
        self.ax_lon.set_yticks(self.yticks)
        self.ax_lon.set_ylim(self.zlim)
        self.ax_lon.set_xlim(self.xlim)
        self.ax_lon.minorticks_on()
        for xlabel_i in self.ax_lon.get_xticklabels():
            xlabel_i.set_fontsize(0.0)
            xlabel_i.set_visible(False)

        # Height-VHF count [OK]
        self.ax_hist.set_xticks([0, 0.1, 0.2, 0.3])
        self.ax_hist.set_yticks(self.yticks)
        self.ax_hist.set_ylim(self.zlim)
        self.ax_hist.set_xlim(0, 0.3)
        self.ax_hist.set_xlabel('Freq')
        self.ax_hist.minorticks_on()

        # Altitude-Latitude [OK]
        self.ax_lat.set_xlabel('Altitude (km MSL)')
        self.ax_lat.set_xticks(self.yticks)
        self.ax_lat.set_xlim(self.zlim)
        self.ax_lat.set_ylim(self.ylim)
        self.ax_lat.minorticks_on()
        for xlabel_i in self.ax_lat.get_yticklabels():
            xlabel_i.set_fontsize(0.0)
            xlabel_i.set_visible(False)

        # Plan view [OK]
        if self.bkgmap==True:
            self.ax_plan.add_feature(COUNTIES, facecolor='none', edgecolor='gray')
            self.ax_plan.add_feature(cfeature.BORDERS)
            self.ax_plan.add_feature(cfeature.STATES.with_scale('10m'))
        self.ax_plan.set_xlabel('Longitude (degrees)')
        self.ax_plan.set_ylabel('Latitude (degrees)')
        self.ax_plan.set_xlim(self.xlim)
        self.ax_plan.set_ylim(self.ylim)
        self.ax_plan.set_aspect('auto')

        # Radar plan view
        plt.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(vmin=-20, vmax=60), cmap = 'HomeyerRainbow'), ax = self.ax_rad)
        
        # Leader speed plot
        self.ax_vel.set_ylabel('Distance From Origin (m)')
        self.ax_vel.set_xlabel('Time From Origin (s)')
        self.ax_vel.set_title('Lightning Leader Speed')
        norm = mcolors.Normalize(vmin=0, vmax=6)
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap='cool'), ax=self.ax_vel, label='Altitude (km)', spacing='proportional')
        
        steps = ((distance.geodesic((points[1], points[0]), (points[3], points[2])).km)+((distance.geodesic((points[1], points[0]), (points[3], points[2])).km)/10))
        #print((points[1], points[0]), (points[3], points[2]))
        
        # Reflectivity Cross Section
        display.plot_cross_section("reflectivity", [points[1], points[0]], [points[3], points[2]], steps = steps, cmap = 'pyart_HomeyerRainbow', title = '', axislabels = ('', 'Height above \n Radar (km)'), colorbar_label = 'Reflectivity (dbZ)', vmin = -20, vmax = 70, ax = self.ax_rcs)
        
        # Velocity Cross Section
        display.plot_cross_section('velocity', [points[1], points[0]], [points[3], points[2]], steps = steps, cmap = 'pyart_NWSVel', title = '', axislabels = ('', 'Height above \n Radar (km)'), colorbar_label = 'Velocity (m/s)', vmin = -40, vmax = 40, ax = self.ax_vcs)
        
        # Spectrum Width Cross Section
        display.plot_cross_section("spectrum_width", [points[1], points[0]], [points[3], points[2]], steps = steps, cmap = 'pyart_NWS_SPW', title = '', axislabels = ('', 'Height above \n Radar (km)'), colorbar_label = 'Spectrum Width (m/s)', vmin = 0, vmax = 14.1, ax = self.ax_swcs)

        # Differential Reflectivity Cross Section
        display.plot_cross_section("differential_reflectivity", [points[1], points[0]], [points[3], points[2]], steps = steps, cmap = 'pyart_RefDiff', title = '', axislabels = ('', 'Height above \n Radar (km)'), colorbar_label = 'ZDR (DB)', vmin = -2, vmax = 6, ax = self.ax_drcs)

        # Specific Differential Phase Cross Section
        display.plot_cross_section('differential_phase', [points[1], points[0]], [points[3], points[2]], steps = steps, cmap = 'pyart_SCook18', title = '', axislabels = ('', 'Height above \n Radar (km)'), colorbar_label = 'Specific Differential \n Phase (degrees)', vmin = -1, vmax = 4.1, ax = self.ax_sdpcs)

        # Correlation Coefficient Cross Section
        display.plot_cross_section("cross_correlation_ratio", [points[1], points[0]], [points[3], points[2]], steps = steps, cmap = 'pyart_SCook18', title = '', axislabels = ('Distance Across Cross Section (km)', 'Height above \n Radar (km)'), colorbar_label = 'Correlation Coefficient', vmin = 0.7, vmax = 1.03, ax = self.ax_cccs)

        self.ax_plan.minorticks_on()
        self.ax_plan.xaxis.set_major_formatter(self.majorFormatter)
        self.ax_plan.yaxis.set_major_formatter(self.majorFormatter)
        self.set_ax_plan_labels()

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