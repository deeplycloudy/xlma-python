import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as md


def subset(lon_data, lat_data, alt_data, time_data, chi_data,station_data,
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
    lon_data = lon_data[selection]
    lat_data = lat_data[selection]
    time_data = time_data[selection]
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


def plot_points(bk_plot, lon_data, lat_data, alt_data, time_data,
                  plot_cmap=None, plot_s=None, plot_vmin=None, plot_vmax=None, plot_c=None, edge_color='face',
                  edge_width=0, add_to_histogram=True, **kwargs):
    """
    Plot scatter points on an existing bk_plot object given x,y,z,t for each
    and defined plotting colormaps and ranges
    """

    # before **kwargs was added to the function call, the following arguments
    # were specified as keywords separately. This allows backwards compatibility:
    if plot_cmap is None:
        plot_cmap = kwargs.pop('cmap', kwargs.pop('plot_cmap', None))
    if plot_s is None:
        plot_s = kwargs.pop('s', kwargs.pop('plot_s', None))
    if plot_vmin is None:
        plot_vmin = kwargs.pop('vmin', kwargs.pop('plot_vmin', None))
    if plot_vmax is None:
        plot_vmax = kwargs.pop('vmax', kwargs.pop('plot_vmax', None))
    if plot_c is None:
        plot_c = kwargs.pop('c', kwargs.pop('plot_c', None))
    if edge_color == 'face':
        edge_color = kwargs.pop('edgecolors', kwargs.pop('edge_color', 'face'))
    if edge_width == 0:
        edge_width = kwargs.pop('linewidths', kwargs.pop('edge_width', 0))
    
    art_plan = bk_plot.ax_plan.scatter(lon_data, lat_data,
                            c=plot_c,vmin=plot_vmin, vmax=plot_vmax, cmap=plot_cmap,
                            s=plot_s,marker='o', linewidths=edge_width, edgecolors=edge_color, **kwargs)
    art_th = bk_plot.ax_th.scatter(time_data, alt_data,
                          c=plot_c,vmin=plot_vmin, vmax=plot_vmax, cmap=plot_cmap,
                          s=plot_s,marker='o', linewidths=edge_width, edgecolors=edge_color, **kwargs)
    art_lon = bk_plot.ax_lon.scatter(lon_data, alt_data,
                          c=plot_c,vmin=plot_vmin, vmax=plot_vmax, cmap=plot_cmap,
                          s=plot_s,marker='o',  linewidths=edge_width, edgecolors=edge_color, **kwargs)
    art_lat = bk_plot.ax_lat.scatter(alt_data, lat_data,
                          c=plot_c,vmin=plot_vmin, vmax=plot_vmax, cmap=plot_cmap,
                          s=plot_s,marker='o', linewidths=edge_width, edgecolors=edge_color, **kwargs)
    art_out = [art_plan, art_th, art_lon, art_lat, art_txt]

    if add_to_histogram:
        cnt, bins, art_hist = bk_plot.ax_hist.hist(alt_data, orientation='horizontal',
                            density=True, bins=80, range=(0, 20), color='black')
        art_txt = plt.text(0.25, 0.10, str(len(alt_data)) + ' src',
                fontsize='small', horizontalalignment='left',
                verticalalignment='center',transform=bk_plot.ax_hist.transAxes)
        # art_hist is a tuple of patch objects. Make it a flat list of artists
        art_out.extend(art_hist)
    return art_out

def plot_3d_grid(bk_plot, xedges, yedges, zedges, tedges,
                alt_lon, alt_lat, alt_time, lat_lon,
                alt_data, plot_cmap=None, **kwargs):
    """
    Plot gridded fields on an existing bk_plot given x,y,z,t grids and
    respective grid edges
    """
    plot_cmap = kwargs.pop('cmap', kwargs.pop('plot_cmap', None))
    plot_vmin = kwargs.pop('vmin', 0)

    alt_lon[alt_lon==0]=np.nan
    alt_lat[alt_lat==0]=np.nan
    lat_lon[lat_lon==0]=np.nan
    alt_time[alt_time==0]=np.nan
    bk_plot.ax_lon.pcolormesh( xedges, zedges,  alt_lon.T, cmap=plot_cmap, vmin=plot_vmin, **kwargs)
    bk_plot.ax_lat.pcolormesh( zedges, yedges,  alt_lat.T, cmap=plot_cmap, vmin=plot_vmin, **kwargs)
    bk_plot.ax_plan.pcolormesh(xedges, yedges,  lat_lon.T, cmap=plot_cmap, vmin=plot_vmin, **kwargs)
    bk_plot.ax_th.pcolormesh(  tedges, zedges, alt_time.T, cmap=plot_cmap, vmin=plot_vmin, **kwargs)
    bk_plot.ax_hist.hist(alt_data, orientation='horizontal',
                         density=True, bins=80, range=(0, 20))
    plt.text(0.25, 0.10, str(len(alt_data)) + ' src',
             fontsize='small', horizontalalignment='left',
             verticalalignment='center',transform=bk_plot.ax_hist.transAxes)