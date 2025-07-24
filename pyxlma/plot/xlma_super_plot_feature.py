import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xradar as xd
import matplotlib.dates as md
from math import radians, cos, sin, asin, sqrt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import pyart
import datetime as dt
from datetime import datetime, timedelta
import pickle 
import os 
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyxlma.plot.leader_speed import get_time_distance, time_distance_plot
import pyxlma.plot.radar as lmarad
#from pyxlma.plot.xlma_super_base_plot import SuperBlankPlot
from geopy import distance
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from metpy.interpolate import cross_section

from scipy import spatial

try:
    from metpy.plots import USCOUNTIES
    county_scales = ['20m', '5m', '500k']
    COUNTIES = USCOUNTIES.with_scale(county_scales[0])
except ImportError:
    COUNTIES = None    


def subset(lon_data, lat_data, alt_data, time_data, chi_data, station_data,
           xlim, ylim, zlim, tlim, xchi, stationmin):
    """
    Generate a subset of x,y,z,t of sources based on maximum
    reduced chi squared and given x,y,z,t bounds

    Returns: longitude, latitude, altitude, time and boolean arrays
    """
    
    selection = ((alt_data>zlim[0])&(alt_data<zlim[1])&
                 (lon_data>xlim[0])&(lon_data<xlim[1])&
                 (lat_data>ylim[0])&(lat_data<ylim[1])&
                 (time_data>tlim[0])&(time_data<tlim[1])&
                 (chi_data<=xchi)&(station_data>=stationmin)
                 )
    
    alt_data = alt_data[selection]
    #print(alt_data)
    lon_data = lon_data[selection]
    #print(lon_data)
    lat_data = lat_data[selection]
    #print(lat_data)
    time_data = time_data[selection]
    #print(time_data)
    return lon_data, lat_data, alt_data, time_data, selection


def color_by_time(time_array, tlim):
    """
    Generates colormap values for plotting VHF sources by time in a
    given time window

    Returns: min, max values, array by time
    """
    vmax = (tlim[1] - time_array.min()).total_seconds()
    ldiff = time_array - time_array.min()
    ldf = []
    for df in ldiff:
        ldf.append(df.total_seconds())
    c = np.array(ldf)
    vmin = 0

    return vmin, vmax, c


def setup_hist(lon_data, lat_data, alt_data, time_data,
               xbins, ybins, zbins, tbins):
    """
    Create 2D VHF historgrams for combinations of x,y,z,t
    in specified intervals
    """
    alt_lon, _, _ = np.histogram2d(lon_data, alt_data, [xbins,zbins])
    alt_lat, _, _ = np.histogram2d(alt_data, lat_data, [zbins,ybins])
    alt_time, _, _ = np.histogram2d(md.date2num(time_data), alt_data, [tbins,zbins])
    lat_lon, _, _ = np.histogram2d(lon_data, lat_data, [xbins,ybins])
    return alt_lon, alt_lat, alt_time, lat_lon


def plot_super_points(bk_plot, lon_data, lat_data, alt_data, time_data,
                  plot_cmap=None, plot_s=40, plot_vmin=None, plot_vmax=None, plot_c=None, edge_color='face',
                  edge_width=0, add_to_histogram=True, marker='o', radar_data = None, points = None, **kwargs):
    """
    Plot scatter points on an existing bk_plot object given x,y,z,t for each
    and defined plotting colormaps and ranges
    """
    colour_map = LinearSegmentedColormap.from_list('mycmap', ['black', 'black', 'black'])
    start = dt.datetime(2022, 11, 16)
    flash_events = pd.DataFrame({'lon': lon_data, 'lat': lat_data, 'alt': alt_data, 'time': time_data})
    flash_events = flash_events.sort_values(by='time')
    
    #if radar_data != None: print(list(flash_events['time'])[0])
    mx, my, xx, xy, max_z = points[0], points[1], points[2], points[3], points[4]
    max_dist = 2.5e3
    
    if radar_data != None:
            closest_time = min(radar_data[-1], key=lambda sub: abs(sub-list(flash_events['time'])[0]))
            start = closest_time

    end = start+dt.timedelta(seconds=7.5*60)
    flash_event_time = pd.Series(flash_events.time)
    selection_event = (flash_event_time>=start-dt.timedelta(seconds=1*60))&(flash_event_time<end)

    lma_z_ktyx = flash_events.alt.values[selection_event]
    lma_x_ktyx, lma_y_ktyx = flash_events.lon.values[selection_event], flash_events.lat.values[selection_event]
    #print('Old LMA X', lma_x_ktyx)
    #print('Old LMA Y', lma_y_ktyx)

    # Look for LMA sources near the radar grid points
    new_angle = np.arctan2(xy-my,xx-mx) # Angle of cross section
    # Recenter on the left point
    new_lma_x = lma_x_ktyx-mx
    #print('New LMA X', new_lma_x)
    new_lma_y = lma_y_ktyx-my
    #print('New LMA Y', new_lma_y)
    # Rotate the coordinate frame
    new_lma_r = new_lma_x*np.cos(-new_angle) - new_lma_y*np.sin(-new_angle) # Cartesian x
    new_dists = new_lma_y*np.cos(-new_angle) + new_lma_x*np.sin(-new_angle) # Cartesian y
    #print('Radius', new_lma_r)
    #print("Distance", new_dists)
    cs_x = []
    cs_z = []
    NewC = []
    #radar_art_display = None
    #art_display_rad = None
    
    if np.sum((new_dists<max_dist)&(new_dists>-max_dist))>0:
        cs_x = new_lma_r[(new_dists<max_dist)&(new_dists>-max_dist)]
        cs_x = cs_x*100
        #print('x', cs_x)
        cs_z = lma_z_ktyx[(new_dists<max_dist)&(new_dists>-max_dist)]
        #print('z', cs_z)
        NewC = [plot_c[0]]*len(cs_z)
        #print('c', NewC)
    steps = ((distance.geodesic((points[1], points[0]), (points[3], points[2])).km)+((distance.geodesic((points[1], points[0]), (points[3], points[2])).km)/10))
    #print('Beginning')
    # before **kwargs was added to the function call, the following arguments
    # were specified as keywords separately. This allows backwards compatibility:
    if plot_cmap is None:
        plot_cmap = kwargs.pop('cmap', plot_cmap)
    if plot_s is None:
        plot_s = kwargs.pop('s', plot_s)
    if plot_vmin is None:
        plot_vmin = kwargs.pop('vmin', plot_vmin)
    if plot_vmax is None:
        plot_vmax = kwargs.pop('vmax', plot_vmax)
    if plot_c is None:
        plot_c = kwargs.pop('c', plot_c)
    if edge_color == 'face':
        edge_color = kwargs.pop('edgecolors', edge_color)
    if edge_width == 0:
        edge_width = kwargs.pop('linewidths', edge_width)
    
    art_plan = bk_plot.ax_plan.scatter(lon_data, lat_data,
                            c=plot_c, vmin=plot_vmin, vmax=plot_vmax, cmap=plot_cmap,
                            s=plot_s, marker=marker, linewidths=edge_width, edgecolors=edge_color, **kwargs)
    #print('plan', alt_data)
    art_th = bk_plot.ax_th.scatter(time_data, alt_data,
                          c=plot_c, vmin=plot_vmin, vmax=plot_vmax, cmap=plot_cmap,
                          s=plot_s, marker=marker, linewidths=edge_width, edgecolors=edge_color, **kwargs)
    #print('th', alt_data)
    art_lon = bk_plot.ax_lon.scatter(lon_data, alt_data,
                          c=plot_c, vmin=plot_vmin, vmax=plot_vmax, cmap=plot_cmap,
                          s=plot_s, marker=marker, linewidths=edge_width, edgecolors=edge_color, **kwargs)
    #print('lon', alt_data)
    art_lat = bk_plot.ax_lat.scatter(alt_data, lat_data,
                          c=plot_c, vmin=plot_vmin, vmax=plot_vmax, cmap=plot_cmap,
                          s=plot_s, marker=marker, linewidths=edge_width, edgecolors=edge_color, **kwargs)
    #print('lat', alt_data)
    cnt, bins, art_hist = bk_plot.ax_hist.hist(alt_data, orientation='horizontal',
                         density=True, bins=80, range=(0, 20), color='black')
    
    if radar_data != None:
        bk_plot.ax_rad.add_feature(COUNTIES, facecolor='none', edgecolor='gray')
        bk_plot.ax_rad.add_feature(cfeature.BORDERS)
        bk_plot.ax_rad.add_feature(cfeature.STATES.with_scale('10m'))
        bk_plot.ax_rad.set_title(f'{start} UTC \n Equivelent Reflectivity Factor', loc = 'center')
        grid = bk_plot.ax_rad.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--', color='gray')
        radar_display = pyart.graph.GridMapDisplay(radar_data[radar_data[-1].index(start)], debug=True)
        data = radar_display.grid.fields['reflectivity']['data'][1]
        radar = bk_plot.ax_rad.pcolormesh(radar_display.grid.point_longitude['data'][0], radar_display.grid.point_latitude['data'][0], data, cmap = 'HomeyerRainbow', vmin = -20, vmax = 60)
        grid.xlabels_top = False
        grid.ylabels_right = False
        #print(radar_display)
        bk_plot.ax_rad.plot([points[0], points[2]], [points[1], points[3]], color = '#6a8f9c', linewidth = 2)
        #print(radar_art_display)
    #else:
        #radar_art_display = bk_plot.ax_rad.scatter(lon_data[0], lat_data[0], alpha = 0, marker = '.')
    #if radar_art_display != None: radar_art_display.show()    
    art_rad = bk_plot.ax_rad.scatter(lon_data, lat_data, 
                            c=plot_c, vmin=plot_vmin, vmax=plot_vmax, cmap=colour_map,
                            s=plot_s, marker=marker, linewidths=edge_width, edgecolors=edge_color, **kwargs)
    #print(art_rad)
    if (len(flash_events['alt'])==0):
        art_vel = None
        print(" ")
    else:
        if(type(lat_data) == pd.core.series.Series):
            lat_data = lat_data.to_numpy()
        if(type(lon_data) == pd.core.series.Series):
            lon_data = lon_data.to_numpy()
        if(type(time_data) == pd.core.series.Series):
            time_data = time_data.to_numpy()
            
        first = np.nanargmin(time_data)
        
        distance_from_origin, time_from_origin = get_time_distance(lat_data, lon_data, time_data, 
                                                                   lat_data[first], lon_data[first], time_data[first])
        
        print('Num Points: ', len(distance_from_origin))
        
        art_vel = time_distance_plot(bk_plot.ax_vel, time_from_origin, distance_from_origin, 
                                     c=alt_data, vmin=0, vmax=6, cmap='cool',
                                     s=plot_s, marker='o', linewidths=edge_width, edgecolors=edge_color)

    if radar_data != None: 
            proj_params = radar_display.grid.get_projparams()
            radar_crs = ccrs.AzimuthalEquidistant(central_longitude=proj_params['lon_0'], central_latitude=proj_params['lat_0'])
            projection_info = radar_crs.to_cf()
            ds = radar_display.grid.to_xarray().metpy.assign_crs(projection_info)
            ds = cross_section(ds, [points[1], points[0]], [points[3], points[2]], steps, 'linear').set_coords(('lat', 'lon'))
            ds['z'] = ds['z']/1000
            ds['reflectivity'].plot(y='z', vmin=-20, vmax=60, ax=bk_plot.ax_rcs, cmap='HomeyerRainbow', add_colorbar=False)
            ds['velocity'].plot(y='z', vmin=-40, vmax=40, ax=bk_plot.ax_vcs, cmap='NWSVel', add_colorbar=False)
            ds['spectrum_width'].plot(y='z', vmin=0, vmax=14.1, ax=bk_plot.ax_swcs, cmap='NWS_SPW', add_colorbar=False)
            ds['differential_reflectivity'].plot(y='z', vmin=-2, vmax=6, ax=bk_plot.ax_drcs, cmap='RefDiff', add_colorbar=False)
            ds['differential_phase'].plot(y='z', vmin=-1, vmax=4.1, ax=bk_plot.ax_sdpcs, cmap='SCook18', add_colorbar=False)
            ds['cross_correlation_ratio'].plot(y='z', vmin=0.7, vmax=1.03, ax=bk_plot.ax_cccs, cmap='SCook18', add_colorbar=False)
            #radar_display.mappables.append(plot)
            bk_plot.ax_rcs.set_ylabel('')
            bk_plot.ax_rcs.set_xlabel('')
            bk_plot.ax_rcs.set_title('')
            bk_plot.ax_rcs.set_ylim(top=max_z)
            bk_plot.ax_vcs.set_ylabel('')
            bk_plot.ax_vcs.set_xlabel('')
            bk_plot.ax_vcs.set_title('')
            bk_plot.ax_vcs.set_ylim(top=max_z)
            bk_plot.ax_swcs.set_ylabel('')
            bk_plot.ax_swcs.set_xlabel('')
            bk_plot.ax_swcs.set_title('')
            bk_plot.ax_swcs.set_ylim(top=max_z)
            bk_plot.ax_drcs.set_ylabel('')
            bk_plot.ax_drcs.set_xlabel('')
            bk_plot.ax_drcs.set_title('')
            bk_plot.ax_drcs.set_ylim(top=max_z)
            bk_plot.ax_sdpcs.set_ylabel('')
            bk_plot.ax_sdpcs.set_xlabel('')
            bk_plot.ax_sdpcs.set_title('')
            bk_plot.ax_sdpcs.set_ylim(top=max_z)
            bk_plot.ax_cccs.set_ylabel('')
            bk_plot.ax_cccs.set_xlabel('')
            bk_plot.ax_cccs.set_title('')
            bk_plot.ax_cccs.set_ylim(top=max_z)
            
    art_rcs = bk_plot.ax_rcs.scatter(cs_x, cs_z,                
                            c=NewC, vmin=plot_vmin, vmax=plot_vmax, cmap = colour_map,
                            s=plot_s, marker=marker, linewidths=edge_width, edgecolors=edge_color, **kwargs)
    art_vcs = bk_plot.ax_vcs.scatter(cs_x, cs_z,                
                            c=NewC, vmin=plot_vmin, vmax=plot_vmax, cmap = colour_map,
                            s=plot_s, marker=marker, linewidths=edge_width, edgecolors=edge_color, **kwargs)
    art_swcs = bk_plot.ax_swcs.scatter(cs_x, cs_z,               
                            c=NewC, vmin=plot_vmin, vmax=plot_vmax, cmap = colour_map,
                            s=plot_s, marker=marker, linewidths=edge_width, edgecolors=edge_color, **kwargs)
    art_drcs = bk_plot.ax_drcs.scatter(cs_x, cs_z,                
                            c=NewC, vmin=plot_vmin, vmax=plot_vmax, cmap = colour_map,
                            s=plot_s, marker=marker, linewidths=edge_width, edgecolors=edge_color, **kwargs)
    art_sdpcs = bk_plot.ax_sdpcs.scatter(cs_x, cs_z,               
                            c=NewC, vmin=plot_vmin, vmax=plot_vmax, cmap = colour_map,
                            s=plot_s, marker=marker, linewidths=edge_width, edgecolors=edge_color, **kwargs)
    art_cccs = bk_plot.ax_cccs.scatter(cs_x, cs_z,               
                            c=NewC, vmin=plot_vmin, vmax=plot_vmax, cmap = colour_map,
                            s=plot_s, marker=marker, linewidths=edge_width, edgecolors=edge_color, **kwargs)

    art_txt = plt.text(0.25, 0.10, str(len(alt_data)) + ' src',
             fontsize='small', horizontalalignment='left',
             verticalalignment='center',transform=bk_plot.ax_hist.transAxes)
    #print(display_rad)
    art_out = [art_plan, art_th, art_lon, art_lat, art_rad, art_rcs, art_vcs, art_swcs, art_drcs, art_sdpcs, art_cccs]
    if art_vel is not None:
        art_out.extend(art_vel)
    if radar_data != None:
        for lists in bk_plot.ax_rad.collections:
                art_out.insert(-1, lists)
        for lists2 in bk_plot.ax_rcs.collections:
                art_out.insert(-1, lists2)
        for lists3 in bk_plot.ax_vcs.collections:
                art_out.insert(-1, lists3)
        for lists4 in bk_plot.ax_swcs.collections:
                art_out.insert(-1, lists4)
        for lists5 in bk_plot.ax_drcs.collections:
                art_out.insert(-1, lists5)
        for lists6 in bk_plot.ax_sdpcs.collections:
                art_out.insert(-1, lists6)
        for lists7 in bk_plot.ax_cccs.collections:
                art_out.insert(-1, lists7)
        #art_out.extend(radar_art_display)

    if add_to_histogram:
        cnt, bins, art_hist = bk_plot.ax_hist.hist(alt_data, orientation='horizontal',
                            density=True, bins=80, range=(0, 20), color='black')
        art_txt = plt.text(0.25, 0.10, str(len(alt_data)) + ' src',
                fontsize='small', horizontalalignment='left',
                verticalalignment='center',transform=bk_plot.ax_hist.transAxes)
        # art_hist is a tuple of patch objects. Make it a flat list of artists
        art_out.append(art_txt)
        art_out.append(art_hist)
    #print(art_out)
    return art_out

def plot_3d_grid(bk_plot, xedges, yedges, zedges, tedges,
                alt_lon, alt_lat, alt_time, lat_lon,
                alt_data, plot_cmap):
    """
    Plot gridded fields on an existing bk_plot given x,y,z,t grids and
    respective grid edges
    """
    alt_lon[alt_lon==0]=np.nan
    alt_lat[alt_lat==0]=np.nan
    lat_lon[lat_lon==0]=np.nan
    alt_time[alt_time==0]=np.nan
    bk_plot.ax_lon.pcolormesh( xedges, zedges,  alt_lon.T, cmap=plot_cmap, vmin=0)
    bk_plot.ax_lat.pcolormesh( zedges, yedges,  alt_lat.T, cmap=plot_cmap, vmin=0)
    bk_plot.ax_plan.pcolormesh(xedges, yedges,  lat_lon.T, cmap=plot_cmap, vmin=0)
    bk_plot.ax_th.pcolormesh(  tedges, zedges, alt_time.T, cmap=plot_cmap, vmin=0)
    bk_plot.ax_hist.hist(alt_data, orientation='horizontal',
                         density=True, bins=80, range=(0, 20))
    plt.text(0.25, 0.10, str(len(alt_data)) + ' src',
             fontsize='small', horizontalalignment='left',
             verticalalignment='center',transform=bk_plot.ax_hist.transAxes)
    
def plot_super_2d_network_points(bk_plot, netw_data, actual_height=None, fake_ic_height=18, fake_cg_height=1,
                        color_by='time', pos_color='blue', neg_color='red', **kwargs):
    """
    Plot points from a 2D lightning mapping neworks (ie, NLDN, ENTLN, etc)

    Parameters
    ----------
    bk_plot : `pyxlma.plot.xlma_base_plot.BlankPlot`
        A BlankPlot object to plot the data on
    netw_data : `pandas.DataFrame` or `xarray.Dataset`
        data object with columns/variables 'longitude', 'latitude', 'type' (CG/IC), and 'datetime'
    actual_height : `numpy.ndarray` or `pandas.Series` or `xarray.DataArray`
        the hieghts of the events to be plotted (default None, fake_ic_height and fake_cg_height used)
    fake_ic_height : float
        the altitude to plot IC points (default 18 km)
    fake_cg_height : float
        the altitude to plot CG points (default 1 km)
    color_by : ['time', 'polarity']
        Whether to color the points by time or polarity. Default 'time'. Ignored if **kwargs contains 'c'.
    pos_color : str
        color for positive points (default 'red') if color_by='polarity'
    neg_color : str
        color for negative points (default 'blue') if color_by='polarity'
    **kwargs
        additional keyword arguments to pass to plt.scatter

    Returns
    -------
    art_out2 : list
        nested lists of artists created by plot_points (first list CG, second list IC)

    """

    plot_c = kwargs.pop('c', None)
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    marker = kwargs.pop('marker', '^') 
    marker2 = kwargs.pop('marker', 'v') 
    if actual_height is not None:
        netw_data['height'] = actual_height

    if plot_c is not None:
        netw_data['plot_c'] = color_by_time(netw_data.datetime, bk_plot.tlim)[2]
    elif color_by == 'polarity':
        netw_data['plot_c'] = plot_c
    elif color_by == 'time':
        pass
    else:
        raise ValueError("color_by must be 'time' or 'polarity'")
    
    cgs = netw_data[netw_data['type']==0].copy()
    #print(netw_data['type']==0)
    #print(cgs)
    ics = netw_data[netw_data['type']==1].copy()
    #print(ics)

    if actual_height is None:
        cgs['height'] = np.full_like(cgs.longitude, fake_cg_height)
        ics['height'] = np.full_like(ics.longitude, fake_ic_height)
    art_out2 = []
    if color_by == 'polarity':
        cgpos = cgs[cgs.peak_current_kA>0]
        cgneg = cgs[cgs.peak_current_kA<0]
        icpos = ics[ics.peak_current_kA>0]
        icneg = ics[ics.peak_current_kA<0]
        #Negative CG plotting
        print("Negative CG")
        if not cgneg.empty:
                art_out2.extend(plot_super_points(bk_plot, cgneg.longitude, cgneg.latitude, cgneg.height*1000,
                                                  cgneg.datetime, plot_c=neg_color, marker=marker2, add_to_histogram=False, **kwargs))
        else: print('empty')
        #Positive CG plotting
        print('Positive CG')
        if not cgpos.empty:
                art_out2.extend(plot_super_points(bk_plot, cgpos.longitude, cgpos.latitude, cgpos.height*1000,
                                                  cgpos.datetime, plot_c=pos_color, marker=marker2, add_to_histogram=False, **kwargs))
        else: print('empty')
        #Negative IC plotting
        print('Negative IC')
        if not icneg.empty:
                art_out2.extend(plot_super_points(bk_plot, icneg.longitude, icneg.latitude, icneg.height*1000,
                                                  icneg.datetime, plot_c=neg_color, marker=marker, add_to_histogram=False, **kwargs))
        else: print('empty')
        #Positive IC plotting
        print('Positive IC')
        if not icpos.empty:
                art_out2.extend(plot_super_points(bk_plot, icpos.longitude, icpos.latitude, icpos.height*1000,
                                                  icpos.datetime, plot_c=pos_color, marker=marker, add_to_histogram=False, **kwargs))
        else: print('empty')
    else:
        art_out2.extend(plot_super_points(bk_plot, cgs.longitude, cgs.latitude, cgs.height*1000,
                    cgs.datetime, c=cgs.plot_c, vmin=vmin, vmax=vmax, marker=marker2, add_to_histogram=False, **kwargs))
        art_out2.extend(plot_super_points(bk_plot, ics.longitude, ics.latitude, ics.height*1000,
                    ics.datetime, c=ics.plot_c, vmin=vmin, vmax=vmax, marker=marker, add_to_histogram=False, **kwargs))
    return art_out2
