import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import matplotlib.dates as md
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from pyxlma.plot.xlma_base_plot import BlankPlot


def subset(lon_data, lat_data, alt_data, time_data, chi_data,station_data,
           xlim, ylim, zlim, tlim, xchi, stationmin):
    """Generate a subset of x,y,z,t of sources based on maximum reduced chi squared and given x,y,z,t bounds

    Parameters
    ----------
    lon_data : array_like
        longitudes of the sources
    lat_data : array_like
        latitudes of the sources
    alt_data : array_like
        altitudes of the sources
    time_data : array_like
        times of the sources
    chi_data : array_like
        reduced chi squared values of the sources
    station_data : array_like
        the number of stations receiving each source
    xlim : iterable
        [min, max] longitude bounds
    ylim : iterable
        [min, max] latitude bounds
    zlim : iterable
        [min, max] altitude bounds
    tlim : iterable
        [min, max] time bounds
    xchi : float
        maximum reduced chi squared value to be allowed
    stationmin : int
        minimum number of stations to be allowed
    
    Returns
    -------
    lon_data : numpy.ndarray
        longitudes of the sources
    lat_data : numpy.ndarray
        latitudes of the sources
    alt_data : numpy.ndarray
        altitudes of the sources
    time_data : numpy.ndarray
        times of the sources
    selection : numpy.ndarray
        boolean array of the sources that meet the criteria
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


def color_by_time(time_array, tlim=None):
    """Generates colormap values for plotting scatter points by time in a given time window

    Parameters
    ----------
    time_array : array_like
        (N_points,) array of the times of the sources.
        Array can be of datetime.datetime, numpy.datetime64, pandas.Timestamp, or float type
    tlim : iterable
        [min, max] time bounds
    
    Returns
    -------
    vmin : float
        vmin of color data
    vmax : float
        vmax of color data
    c : numpy.ndarray
        data representing the color for each point. Can be passed to the 'c' kwarg of matplotlib scatter, plot, etc.
    """
    if tlim is None:
        tlim = np.array([np.atleast_1d(time_array.min())[0],
                         np.atleast_1d(time_array.max())[0]])
    nsToS = 1e9
    time_array = np.array(time_array).astype('datetime64[ns]').astype(float)/nsToS
    tlim = np.atleast_1d(tlim).astype('datetime64[ns]').astype(float)/nsToS
    
    ldiff = time_array - tlim[0]

    vmax = tlim[1] - tlim[0]
    c = ldiff.astype(float)
    vmin = 0

    return vmin, vmax, c


def setup_hist(lon_data, lat_data, alt_data, time_data,
               xbins, ybins, zbins, tbins):
    """Create 2D VHF historgrams for combinations of x,y,z,t in specified intervals.

    Parameters
    ----------
    lon_data : array_like
        longitudes of the points to histogram
    lat_data : array_like
        latitudes of the points to histogram
    alt_data : array_like
        altitudes of the points to histogram
    time_data : array_like
        times of the points to histogram
    xbins : int
        number of bins for the longitude histogram
    ybins : int
        number of bins for the latitude histogram
    zbins : int
        number of bins for the altitude histogram
    tbins : int
        number of bins for the time histogram
    
    Returns
    -------
    alt_lon : numpy.ndarray
        2D histogram of altitude vs longitude
    alt_lat : numpy.ndarray
        2D histogram of altitude vs latitude
    alt_time : numpy.ndarray
        2D histogram of altitude vs time
    lat_lon : numpy.ndarray
        2D histogram of latitude vs longitude
    """
    alt_lon, _, _ = np.histogram2d(lon_data, alt_data, [xbins,zbins])
    alt_lat, _, _ = np.histogram2d(alt_data, lat_data, [zbins,ybins])
    alt_time, _, _ = np.histogram2d(md.date2num(time_data), alt_data, [tbins,zbins])
    lat_lon, _, _ = np.histogram2d(lon_data, lat_data, [xbins,ybins])
    return alt_lon, alt_lat, alt_time, lat_lon


def plot_points(bk_plot, lon_data, lat_data, alt_data, time_data,
                  plot_cmap=None, plot_s=None, plot_vmin=None, plot_vmax=None, plot_c=None, edge_color='face',
                  edge_width=0, add_to_histogram=True, marker='o', **kwargs):
    """Plot scatter points on an existing bk_plot object given x,y,z,t for each axis.
    
    Parameters
    ----------
    bk_plot : BlankPlot
        The BlankPlot object to plot the data on
    lon_data : array_like
        longitudes of the points.
    lat_data : array_like
        latitudes of the points.
    alt_data : array_like
        altitudes of the points.
    time_data : array_like
        times of the points.
    plot_cmap : str
        colormap to use for the points
    plot_s : float
        size of the points
    plot_vmin : float
        minimum value for the colormap
    plot_vmax : float
        maximum value for the colormap
    plot_c : array_like
        color data for the points. If None, color by time
    edge_color : str, default='face'
        color of the edge of the points. Use 'face' to match the face color
    edge_width : float, default=0
        width of the edge of the points
    add_to_histogram : bool, default=True
        whether to add the points to the time_altitude histogram
    marker : str, default='o'
        marker style for the points
    **kwargs
        additional keyword arguments to pass to plt.scatter

    Returns
    -------
    art_out : list
        list of artists created by the scatter plot, in order of planview, time-height, lon cross section, lat cross section, 
        and (if add_to_histogram is True) alt histogram
    
    Notes
    -----
    Before **kwargs was added to the function call, the plot_cmap, plot_s, plot_vmin, plot_vmax, plot_c, edge_color, edge_width, and marker arguments
    were specified directly as keywords. They are still included for backwards compatibility,
    however, using "cmap", "s", "vmin", "vmax", "c", "edgecolors", "linewidths", and "marker" in **kwargs is highly preferred.
    """

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
                            c=plot_c,vmin=plot_vmin, vmax=plot_vmax, cmap=plot_cmap,
                            s=plot_s, marker=marker, linewidths=edge_width, edgecolors=edge_color, **kwargs)
    art_th = bk_plot.ax_th.scatter(time_data, alt_data,
                          c=plot_c,vmin=plot_vmin, vmax=plot_vmax, cmap=plot_cmap,
                          s=plot_s, marker=marker, linewidths=edge_width, edgecolors=edge_color, **kwargs)
    art_lon = bk_plot.ax_lon.scatter(lon_data, alt_data,
                          c=plot_c,vmin=plot_vmin, vmax=plot_vmax, cmap=plot_cmap,
                          s=plot_s, marker=marker, linewidths=edge_width, edgecolors=edge_color, **kwargs)
    art_lat = bk_plot.ax_lat.scatter(alt_data, lat_data,
                          c=plot_c,vmin=plot_vmin, vmax=plot_vmax, cmap=plot_cmap,
                          s=plot_s, marker=marker, linewidths=edge_width, edgecolors=edge_color, **kwargs)
    art_out = [art_plan, art_th, art_lon, art_lat]

    if add_to_histogram:
        cnt, bins, art_hist = bk_plot.ax_hist.hist(alt_data, orientation='horizontal',
                            density=True, bins=80, range=(0, 20), color='black')
        art_txt = plt.text(0.25, 0.10, str(len(alt_data)) + ' src',
                fontsize='small', horizontalalignment='left',
                verticalalignment='center',transform=bk_plot.ax_hist.transAxes)
        # art_hist is a tuple of patch objects. Make it a flat list of artists
        art_out.append(art_txt)
        art_out.append(art_hist)
    return art_out


def plot_2d_network_points(bk_plot, netw_data, actual_height=None, fake_ic_height=18, fake_cg_height=1,
                        color_by='time', pos_color='red', neg_color='blue', **kwargs):
    """Plot points from a 2D lightning mapping neworks (ie, NLDN, ENTLN, etc)

    Parameters
    ----------
    bk_plot : BlankPlot
        A BlankPlot object to plot the data on
    netw_data : pandas.DataFrame or xarray.Dataset
        data object with columns/variables 'longitude', 'latitude', 'type' (CG/IC), and 'datetime'
    actual_height : array_like
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
    art_out : list
        nested lists of artists created by plot_points (first list CG, second list IC)

    """

    plot_c = kwargs.pop('c', None)
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    marker = kwargs.pop('marker', '^')
    if actual_height is not None:
        netw_data['height'] = actual_height

    if plot_c is not None:
        netw_data['plot_c'] = plot_c
    elif color_by == 'time':
        netw_data['plot_c'] = color_by_time(netw_data.datetime, bk_plot.tlim)[2]
    elif color_by == 'polarity':
        pass
    else:
        raise ValueError("color_by must be 'time' or 'polarity'")
    
    cgs = netw_data[netw_data['type']=='CG'].copy()
    ics = netw_data[netw_data['type']=='IC'].copy()

    if actual_height is None:
        cgs['height'] = np.full_like(cgs.longitude, fake_cg_height)
        ics['height'] = np.full_like(ics.longitude, fake_ic_height)
    art_out = []
    if color_by == 'polarity':
        cgpos = cgs[cgs.peak_current_kA>0]
        cgneg = cgs[cgs.peak_current_kA<0]
        icpos = ics[ics.peak_current_kA>0]
        icneg = ics[ics.peak_current_kA<0]
        art_out.append(plot_points(bk_plot, cgneg.longitude, cgneg.latitude, cgneg.height,
                    cgneg.datetime, c=neg_color, marker=marker, add_to_histogram=False, **kwargs))
        art_out.append(plot_points(bk_plot, cgpos.longitude, cgpos.latitude, cgpos.height,
                    cgpos.datetime, c=pos_color, marker=marker, add_to_histogram=False, **kwargs))
        art_out.append(plot_points(bk_plot, icneg.longitude, icneg.latitude, icneg.height,
                    icneg.datetime, c=neg_color, marker=marker, add_to_histogram=False, **kwargs))
        art_out.append(plot_points(bk_plot, icpos.longitude, icpos.latitude, icpos.height,
                    icpos.datetime, c=pos_color, marker=marker, add_to_histogram=False, **kwargs))
    else:
        art_out.append(plot_points(bk_plot, cgs.longitude, cgs.latitude, cgs.height,
                    cgs.datetime, c=cgs.plot_c, vmin=vmin, vmax=vmax, marker=marker, add_to_histogram=False, **kwargs))
        art_out.append(plot_points(bk_plot, ics.longitude, ics.latitude, ics.height,
                    ics.datetime, c=ics.plot_c, vmin=vmin, vmax=vmax, marker=marker, add_to_histogram=False, **kwargs))
    return art_out


def plot_glm_events(glm, bk_plot, fake_alt=[0, 1], should_parallax_correct=True, poly_kwargs={}, vlines_kwargs={}):
    """Plot event-level data from a glmtools dataset on a pyxlma.plot.xlma_base_plot.BlankPlot object.

    Events that occupy the same pixel have their energies summed and plotted on the planview axis, event locations
    are plotted on the lat/lon/time axes with an altitude specified as fake_alt.
    Requires glmtools to be installed.

    Parameters
    ----------
    glm : xarray.Dataset
        A glmtools dataset of GLM data to plot
    bk_plot : BlankPlot
        A BlankPlot object to plot the data on
    fake_alt : list
        the axes relative coordinates to plot the vertical lines for GLM events in the cross section, default [0, 1],
        the full height of the axes.
    should_parallax_correct : bool
        whether to correct the GLM event locations for parallax effect. See [Bruning et. al 2019, figure 5](https://doi.org/10.1029/2019JD030874).
    poly_kwargs : dict
        dictionary of additional keyword arguments to be passed to matplotlib Polygon
    vlines_kwargs : dict
        dictionary of additional keyword arguments to be passed to matplotlib vlines

    Returns
    -------
    art_out : list
        Handle to matplotlib polygon collection for the planview axis
    """
    from cartopy import crs as ccrs
    from glmtools.io.glm import get_lutevents
    from glmtools.io.ccd import load_pixel_corner_lookup, quads_from_corner_lookup
    from glmtools.io.lightning_ellipse import lightning_ellipse_rev, ltg_ellpse_rev
    from pyxlma.coords import GeostationaryFixedGridSystem, GeographicSystem

    unique_ds = get_lutevents(glm)
    evrad = unique_ds.lutevent_energy

    x_lut, y_lut, corner_lut = load_pixel_corner_lookup()
    x_lut = x_lut * 1.0e-6
    y_lut = y_lut * 1.0e-6
    corner_lut = corner_lut*1e-6

    event_polys = quads_from_corner_lookup(x_lut, y_lut, corner_lut,
        unique_ds.lutevent_x, unique_ds.lutevent_y)

    glm['lutevent_corner_x'] = xr.DataArray(event_polys[:,:,0], dims=['lutevent_id', 'number_of_corners'])
    glm['lutevent_corner_y'] = xr.DataArray(event_polys[:,:,1], dims=['lutevent_id', 'number_of_corners'])

    cx = glm.lutevent_corner_x
    cy = glm.lutevent_corner_y
    cz = np.zeros_like(cx)
    sat_ecef_height = glm.nominal_satellite_height.data.astype(np.float64)*1000
    if should_parallax_correct:
        ellps_rev_ver = ltg_ellpse_rev(glm.product_time.data.astype('datetime64[s]').item())
        ltg_ellps_re, ltg_ellps_rp = lightning_ellipse_rev[ellps_rev_ver]
        gfgs_ellipse = [ltg_ellps_re, ltg_ellps_rp]
    else:
        gfgs_ellipse = 'WGS84'
        ltg_ellps_re = None
        ltg_ellps_rp = None
    geofixcs = GeostationaryFixedGridSystem(subsat_lon=glm.lon_field_of_view.data.item(), ellipse=gfgs_ellipse,
                                            sweep_axis='x', sat_ecef_height=sat_ecef_height)
    grs80lla = GeographicSystem()
    ltg_lon, ltg_lat, ltg_alt = grs80lla.fromECEF(*geofixcs.toECEF(cx,cy,cz))
    poly_verts = []
    for polynum in range(ltg_lon.shape[0]):
        poly_lons = ltg_lon[polynum, :]
        poly_lats = ltg_lat[polynum, :]
        this_poly_verts = np.vstack([poly_lons, poly_lats]).T
        poly_verts.append(this_poly_verts)
    if hasattr(bk_plot.ax_plan, 'projection'):
        map_proj = bk_plot.ax_plan.projection
        if map_proj == ccrs.PlateCarree():
            transformed_pv = poly_verts
        else:
            transformed_pv = [map_proj.transform_points(ccrs.PlateCarree(),
                                                    this_poly_verts[:, 0],
                                                    this_poly_verts[:, 1])[:, 0:2]
                                                    for this_poly_verts in poly_verts]
    else:
        transformed_pv = poly_verts
    patches = [Polygon(pv, closed=True) for pv in transformed_pv]
    pc = PatchCollection(patches, **poly_kwargs)
    pc.set_array(evrad.data)
    bk_plot.ax_plan.add_collection(pc)
    th_handle = bk_plot.ax_th.vlines(glm.event_time_offset.data, fake_alt[0], fake_alt[1], transform=bk_plot.ax_th.get_xaxis_transform(), **vlines_kwargs)
    lon_handle = bk_plot.ax_lon.vlines(glm.event_lon, fake_alt[0], fake_alt[1], transform=bk_plot.ax_lon.get_xaxis_transform(), **vlines_kwargs)
    lat_handle = bk_plot.ax_lat.hlines(glm.event_lat, fake_alt[0], fake_alt[1], transform=bk_plot.ax_lat.get_yaxis_transform(), **vlines_kwargs)
    art_out = [pc, th_handle, lon_handle, lat_handle]
    return art_out


def plot_3d_grid(bk_plot, xedges, yedges, zedges, tedges,
                alt_lon, alt_lat, alt_time, lat_lon,
                alt_data, plot_cmap=None, **kwargs):
    """Plot gridded fields on an existing bk_plot given x,y,z,t grids and respective grid edges.

    Parameters
    ----------
    bk_plot : BlankPlot
        The BlankPlot object to plot the data on
    xedges : array_like
        The edges of the bins of the histogram along the x (or longitude) dimension
    yedges : array_like
        The edges of the bins of the histogram along the y (or latitude) dimension
    zedges : array_like
        The edges of the bins of the histogram along the z (or altitude) dimension
    tedges : array_like
        The edges of the bins of the histogram along the time dimension
    alt_lon : array_like
        2D histogram of altitude vs longitude
    alt_lat : array_like
        2D histogram of altitude vs latitude
    alt_time : array_like
        2D histogram of altitude vs time
    lat_lon : array_like
        2D histogram of latitude vs longitude
    alt_data : array_like
        1D histogram of source altitudes
    plot_cmap : str
        colormap to use for the points
    **kwargs
        additional keyword arguments to pass to pcolormesh

    Notes
    -----
    Before **kwargs was added to the function call, the plot_cmap was specified directly as a keyword. This is still included
    for backwards compatibility, however, using "cmap" in **kwargs is highly preferred.
    """

    plot_cmap = kwargs.pop('cmap', plot_cmap)
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