import numpy as np
import xarray as xr
from scipy.spatial import Delaunay, ConvexHull, QhullError
from scipy.special import factorial
from pyxlma.lmalib.traversal import OneToManyTraversal

def local_cartesian(lon, lat, alt, lonctr, latctr, altctr):
    """Converts lat, lon, altitude points to x, y, z distances from a center point.
    
    Parameters
    ----------
    lon : array_like
        Longitude of points in degrees.
    lat : array_like
        Latitude of points in degrees.
    alt : array_like
        Altitude of points in meters.
    lonctr : float
        Longitude of the center point in degrees.
    latctr : float
        Latitude of the center point in degrees.
    altctr : float
        Altitude of the center point in meters.
    
    Returns
    -------
    x : array_like
        x distance in meters from the center point.
    y : array_like
        y distance in meters from the center point.
    z : array_like
        z distance in meters from the center point.
    """
    Re = 6378.137e3           #Earth's radius in m
    latavg, lonavg, altavg = latctr, lonctr, altctr
    x = Re * (np.radians(lon) - np.radians(lonavg)) * np.cos(np.radians(latavg))
    y = Re * (np.radians(lat) - np.radians(latavg))
    z = alt - altavg
    return x,y,z

def hull_volume(xyz):
    """Calculate the volume of the convex hull of 3D (X,Y,Z) LMA data.
        
    Parameters
    ----------
    xyz : array_like
        A (N_points, 3) array of point locations in space.
    
    Returns
    -------
    volume : float
        The volume of the convex hull.
    vertices : array_like
        The vertices of the convex hull.
    simplex_volumes : array_like
        The volumes of the simplices that make up the convex hull.

    Raises
    ------
    QhullError
        If the convex hull cannot be computed. This is usually because too few points with little spacing are provided. See `perturb_vertex`.
    
    """
    if xyz.shape[1] != 3:
        raise ValueError("Input must be an array of shape (N_points, 3).")

    tri = Delaunay(xyz[:,0:3])
    vertices = tri.points[tri.simplices]

    # This is the volume formula in
    # https://github.com/scipy/scipy/blob/master/scipy/spatial/tests/test_qhull.py#L106
    # Except the formula needs to be divided by ndim! to get the volume, cf.,
    # http://en.wikipedia.org/wiki/Simplex#Geometric_properties
    # Credit Pauli Virtanen, Oct 14, 2012, scipy-user list

    q = vertices[:,:-1,:] - vertices[:,-1,None,:]
    simplex_volumes = (1.0 / factorial(q.shape[-1])) * np.fromiter(
           (np.linalg.det(q[k,:,:]) for k in range(tri.nsimplex)) , dtype=float)
    # print vertices.shape # number of simplices, points per simplex, coords
    # print q.shape

    # The simplex volumes have negative values since they are oriented
    # (think surface normal direction for a triangle
    volume=np.sum(np.abs(simplex_volumes))
    return volume, vertices, simplex_volumes


def perturb_vertex(x,y,z,machine_eps=1.0):
    """Add a random, small perturbation to an x, y, z point.
    
    With only a few points, QHull can error on a degenerate first simplex -
    all points are coplanar to machine precision. Add a random perturbation no
    greater than machine_eps to the first point in x, y, and z. Rerun QHull
    with the returned x,y,z arrays.

    Parameters
    ----------
    x : array_like
        x coordinates of points.
    y : array_like
        y coordinates of points.
    z : array_like
        z coordinates of points.
    machine_eps : float, default=1.0
        The maximum absolute value of the perturbation. (Perturbation can be positive or negative.)

    Returns
    -------
    x : array_like
        x coordinates of points with perturbation added to the first point.
    y : array_like
        y coordinates of points with perturbation added to the first point.
    z : array_like
        z coordinates of points with perturbation added to the first point.
    
    Notes
    -----
    This function is provided to work around the QHullError that can be raised by `pyxlma.lmalib.flash.properties.hull_volume` when the input points are coplanar.
    """
    perturb = 2*machine_eps*np.random.random(size=3)-machine_eps
    x[0] += perturb[0]
    y[0] += perturb[1]
    z[0] += perturb[2]
    return (x,y,z)

def event_hull_area(x,y,z):
    """Compute the 2D area of the convex hull of a set of x, y points.

    Parameters
    ----------
    x : array_like
        (N_points, 1) array of x coordinates of points.
    y : array_like
        (N_points, 1) array of y coordinates of points.
    z : array_like
        (N_points, 1) array of z coordinates of points [unused].

    Returns
    -------
    area : float
        The area of the convex hull of the points.

    Notes
    -----
    For more info on convex hull area and flash size, see [Bruning and MacGorman 2013](https://doi.org/10.1175/JAS-D-12-0289.1).
    """
    pointCount = x.shape[0]
    area = 0.0
    if pointCount > 3:
        try:
            # find the convex hull and calculate its area
            cvh = ConvexHull(np.vstack((x,y)).T)
            # NOT cvh.area - it is the perimeter in 2D.
            # cvh.area is the surface area in 3D.
            area = cvh.volume
        except IndexError:
            # tends to happen when a duplicate point causes the point count to
            # drop to 2, leading to a degenerate polygon with no area
            print('Setting area to 0 for flash with points %s, %s'
                            % (x, y))
        except KeyError:
            # hull indexing has problems here
            print('Setting area to 0 for flash with points %s, %s'
                            % (x, y))
        except QhullError:
            print("Perturbing one source to help area triangulation for "
                  "flash with {0} points".format(x.shape[0]))
            x,y,z = perturb_vertex(x,y,z)
            cvh = ConvexHull(np.vstack((x,y)).T)
            # NOT cvh.area - it is the perimeter in 2D.
            # cvh.area is the surface area in 3D.
            area = cvh.volume
    return area

def event_hull_volume(x,y,z):
    """Compute the 3D volume of the convex hull of a set of x, y points.

    Parameters
    ----------
    x : array_like
        (N_points, 1) array of x coordinates of points.
    y : array_like
        (N_points, 1) array of y coordinates of points.
    z : array_like
        (N_points, 1) array of z coordinates of points.

    Returns
    -------
    volume : float
        The volume of the convex hull of the points.
    """
    pointCount = x.shape[0]
    volume = 0.0
    if pointCount > 4:
        # Need four points to make at least one tetrahedron.
        try:
            volume, vertices, simplex_volumes = hull_volume(np.vstack(
                                                                (x,y,z)).T)
        except QhullError:
            print("Perturbing one source to help volume triangulation for "
                  "flash with {0} points".format(x.shape[0]))
            x,y,z = perturb_vertex(x,y,z)
            volume, vertices, simplex_volumes = hull_volume(np.vstack(
                                                                (x,y,z)).T)
    return volume


def rrbe(zinit):
    """Compute the runway breakeven threshold electric field.

    Uses a given initiation altitude for a lightning flash assuming a surface breakdown electric field
    threshold of 281 kV/m, following [Marshall et al. 2005](https://doi.org/10.1029/2004GL021802).

    Parameters
    ----------
    zinit : float
        The altitude of the flash initiation point in meters.

    Returns
    -------
    e_init : float
        The electric field at the initiation point in kV/m.
    """
    #Compute scaled air density with height.
    rho_air = 1.208 * np.exp(-(zinit/8.4))
    e_init  = 232.6*rho_air
    return(e_init)

def event_discharge_energy(z,area):
    """Estimate the electrical energy discharged by lightning flashes using a simple capacitor model.

    Parameters
    ----------
    z : array_like
        The altitude of the flash initiation points.
    area : array_like
        The area of the convex hull of the flash initiation points.

    Returns
    -------
    energy : array_like
        The energy discharged by the flash in Joules.

    Notes
    -----
    - Model assumes plates area defined by convex hull area, and separation
    between the 73 and 27th percentiles of each flash's vertical source
    distributions.
    - Only the initiation electric field (e_init) between the plates at the height
    of flash initiation to find the critical charge density (sigma_crit) on the plates,
    sigma_crit = epsilon * e_init -> epsilon = permitivity of air
    - Model considers the ground as a perfect conductor (image charging),

    """
    #Compute separation between charge plates (d) and critial charge density (sigma_crit):
    e          = 8.858*1e-12 #[C/Vm] permittivity of air
    if len(z) <=1: #if only one point exists for a series of altitudes, assume a separation of 0.
        d = 0
    else:          #use the percentiles or the two points to compute the separation (d)
        d          = np.abs(np.percentile(z,73) - np.percentile(z,27))
    zinit      = z.iloc[0]
    e_init     = rrbe(zinit*1e-3)*1e3
    sigma_crit = (e * e_init)
    eta_c      = 0.004 #Scale the energy to depict the fraction of energy neutralized by
                       #each flash in the capacitor model
    #Capacitor model:
    w = 4 * ((sigma_crit**2. * d * area)/(2* e)) #The quantity for appears when considering image charges (2*sigma)^2=4sigma^2
    return(w*eta_c)


def flash_stats(ds, area_func=None, volume_func=None):
    """Compute flash statistics from LMA data.

    Calculates the following variables for each flash in the dataset:

    - flash_time_start
    - flash_time_end
    - flash_duration
    - flash_init_latitude
    - flash_init_longitude
    - flash_init_altitude
    - flash_area
    - flash_volume
    - flash_energy
    - flash_center_latitude
    - flash_center_longitude
    - flash_center_altitude
    - flash_power
    - flash_event_count

    Parameters
    ----------
    ds : xarray.Dataset
        An LMA dataset that has flash clustering applied (i.e., has a `flash_id` and `event_parent_flash_id` variable).
    area_func : callable, optional
        A function that computes the area of the convex hull of a set of points. If None, `event_hull_area` is used.
    volume_func : callable, optional
        A function that computes the volume of the convex hull of a set of points. If None, `event_hull_volume` is used.
    
    Returns
    -------
    ds : xarray.Dataset
        LMA dataset with the computed flash statistics added as variables.
    """
    if area_func is None:
        area_func = lambda df: event_hull_area(df['event_x'].array,
                                               df['event_y'].array,
                                               df['event_z'].array)
    if volume_func is None:
        volume_func = lambda df: event_hull_area(df['event_x'].array,
                                               df['event_y'].array,
                                               df['event_z'].array)

    if not('event_x' in ds.variables):
        x,y,z = local_cartesian(ds.event_longitude,
                                ds.event_latitude,
                                ds.event_altitude,
                                ds.network_center_longitude,
                                ds.network_center_latitude,
                                0.0,
                                )
        ds['event_x'] = xr.DataArray(x, dims=['number_of_events'])
        ds['event_y'] = xr.DataArray(y, dims=['number_of_events'])
        ds['event_z'] = xr.DataArray(z, dims=['number_of_events'])

    # === new approach ====
    # Convert to pandas dataframe here before doing groupby
    event_vars_needed = ['event_x', 'event_y', 'event_z', 'event_id',
        'event_parent_flash_id', 'event_time',
        'event_longitude', 'event_latitude', 'event_altitude',
        'event_time', 'event_power']
    df = ds[event_vars_needed].to_dataframe().set_index('event_id')
    time_sorted_df = df.sort_values(by=['event_time'])

    fl_gb = df.groupby('event_parent_flash_id')

    # Summarize properties of each flash by operating on the groupby.
    # .loc with indexes or bools, .iloc with integer positions
    n_events = fl_gb.size() # index is the event_parent_flash_id
    # print(n_events)
    # event_parent_flash_id
    # 0         1
    # 1         1
    # 2        23
    # 3         1
    # 4         1
    #          ..
    # 41774     1
    # 41775     1
    # 41776     1
    # 41777     1
    # 41778     1
    # Length: 41779, dtype: int64

    # Index of first_ and last_event_df and are the event_id
    # first_event = fl_gb['event_time'].idxmin()
    # first_event_df = df.loc[first_event]
    # last_event = fl_gb['event_time'].idxmax()
    # last_event_df = df.loc[last_event]
    # # since the first/last df above has as index the original event ID, it can
    # # be used to index the original dataframe (before groupby) to get the
    # # corresponding flash ID. Use that flash ID as the index for assignment
    # # to the flash_id array.
    # This works, but we can just read the parent flash ID value directly since
    # .loc[first_event] already filters to one event per flash.
    # first_flidx = df.loc[first_event_df.index]['event_parent_flash_id'].values
    # last_flidx = df.loc[last_event_df.index]['event_parent_flash_id'].values

    # Per StackOverflow, idxmin/max are very slow. Here's an alternative:
    # https://stackoverflow.com/questions/55932560/
    first_event_df = time_sorted_df.drop_duplicates('event_parent_flash_id',
                                                    keep='first')
    last_event_df = time_sorted_df.drop_duplicates('event_parent_flash_id',
                                                   keep='last')
    first_flidx = first_event_df['event_parent_flash_id'].values
    last_flidx = last_event_df['event_parent_flash_id'].values
    # --first_event_df--
    #                event_x       event_y       event_z  event_parent_flash_id
    # event_id
    # 0         1.847077e+06 -5.560333e+06 -2.406689e+09                      0
    # 1         2.502132e+04  2.629595e+05 -7.198652e+05                      1
    # 2         1.862935e+04 -8.411014e+04 -9.948140e+03                      2
    # 3        -6.067912e+03 -1.257087e+05 -1.163374e+04                      3
    # 4         4.146723e+04 -3.249454e+04 -3.303795e+04                      4
    # ...                ...           ...           ...                    ...
    # 119285   -7.507991e+04 -5.834397e+04 -6.107464e+04                  41774
    # 119286   -6.284118e+04 -2.101047e+04 -6.247398e+04                  41775
    # 119287    3.876299e+04  6.409286e+04 -6.131979e+04                  41776
    # 119288    3.070174e+04 -3.972189e+04 -1.196834e+04                  41777
    # 119289    4.131542e+04 -3.370233e+04 -8.183331e+04                  41778

    mean_event_df = fl_gb[['event_latitude',
                           'event_longitude',
                           'event_altitude',]].mean()
    sum_event_df = fl_gb[['event_power']].sum()
    # Index of mean_event_df and sum_event_df are the event_parent_flash_id
    mean_flidx = mean_event_df.index
    sum_flidx = sum_event_df.index

    # Might be able to speed this up with 'aggregate' instead of 'apply'
    # could only run on those where point counts meet threshold, instead
    # testing inside the function
    # Index of event_area and event_volume are event_parent_flash_id
    event_area = fl_gb.apply(area_func, include_groups=False)
    event_volume = fl_gb.apply(volume_func, include_groups=False)

    #Compute flash discharge energy using parallel plate capacitor
    event_energy = fl_gb.apply(lambda df1: event_discharge_energy(df1['event_z'],
                                                                 event_area[df1.name]), include_groups=False)


    # set the index for the original dataset's flash dimension to the flash_id
    # and use the event_parent_flash_id from the aggregations above to assign
    # the data for that flash.
    ds=ds.set_index(number_of_flashes='flash_id')
    ds['flash_event_count'][n_events.index] = n_events.values
    ds['flash_center_longitude'][mean_flidx] = mean_event_df['event_longitude']
    ds['flash_center_latitude'][mean_flidx] = mean_event_df['event_latitude']
    ds['flash_center_altitude'][mean_flidx] = mean_event_df['event_altitude']
    ds['flash_init_longitude'][first_flidx] = first_event_df['event_longitude']
    ds['flash_init_latitude'][first_flidx] = first_event_df['event_latitude']
    ds['flash_init_altitude'][first_flidx] = first_event_df['event_altitude']
    ds['flash_time_start'][first_flidx] = first_event_df['event_time']
    ds['flash_time_end'][last_flidx] = last_event_df['event_time']
    ds['flash_area'][event_area.index] = event_area.values/1.0e6
    ds['flash_volume'][event_volume.index] = event_volume.values/1.0e9
    ds['flash_power'][sum_flidx] = sum_event_df['event_power']

    #Assign to varialbe in dataframe:
    ds['flash_energy'][event_energy.index] = event_energy.values * 1e-9 #Returns in GJ

    # recreate flash_id variable
    ds['flash_duration'] = ds['flash_time_end'] - ds['flash_time_start']
    ds.reset_index('number_of_flashes')
    ds['flash_id']=ds['number_of_flashes']

    return ds

def filter_flashes(ds, prune=True, **kwargs):
    """Filter flashes by their properties.
    
    Allows removing unwanted flashes from an LMA dataset based on their properties.
    After the flashes are removed by the criteria, the dataset can be pruned to remove any events that
    are not assoicated with any flashes (or events that were associated with now-removed flashes).

    Parameters
    ----------
    ds : xarray.Dataset
        An LMA dataset that has flash clustering applied (i.e., has a `flash_id` and `event_parent_flash_id` variable).
    prune : bool, default=True
        If True, remove events not associated with any flashes.
    **kwargs
        Variable names and ranges to filter by. The name of the keyword argument is used as the variable name,
        and ranges are given as a tuple of (min, max) values. Either end of the range can be None to skip it.

    Returns
    -------
    ds : xarray.Dataset
        LMA dataset with the flashes filtered by the given criteria.
    """
    # keep all points
    good = np.ones(ds.flash_id.shape, dtype=bool)
    # print("Starting flash count: ", good.sum())
    if 'flash_event_count' in ds.variables:
        if np.all(ds.flash_event_count == np.iinfo(np.uint32).max):
            raise ValueError('Before filtering a dataset by flash properties, call flash_stats on the dataset to compute flash properties.')
    for v, (vmin, vmax) in kwargs.items():
        if vmin is not None:
            good &= (ds[v] >= vmin).data
            # print("Flashes left after min for ", v, ": ", good.sum())
        if vmax is not None:
            good &= (ds[v] <= vmax).data
            # print("Flashes left after max for ", v, ": ", good.sum())

    flash_subset = ds[{'number_of_flashes':good}]
    if prune:
        new_flash_ids = list(set(flash_subset.flash_id.data))
        tree = OneToManyTraversal(ds, ('flash_id', 'event_id'),
                                      ('event_parent_flash_id',))
        return tree.reduce_to_entities('flash_id', new_flash_ids)
    else:
        return flash_subset
