import numpy as np
import xarray as xr
from scipy.spatial import Delaunay, ConvexHull
from scipy.special import factorial
from scipy.spatial.qhull import QhullError

def local_cartesian(lon, lat, alt, lonctr, latctr, altctr):
    Re = 6378.137e3           #Earth's radius in m
    latavg, lonavg, altavg = latctr, lonctr, altctr
    x = Re * (np.radians(lonavg) - np.radians(lon)) * np.cos(np.radians(latavg))
    y = Re * (np.radians(latavg) - np.radians(lat))
    z = altavg - alt
    return x,y,z

def hull_volume(xyz):
    """ Calculate the volume of the convex hull of 3D (X,Y,Z) LMA data.
        xyz is a (N_points, 3) array of point locations in space. """
    assert xyz.shape[1] == 3

    tri = Delaunay(xyz[:,0:3])
    vertices = tri.points[tri.vertices]

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

def event_hull_area(x,y,z):
    pointCount = x.shape[0]
    area = 0.0
    if pointCount > 2:
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
    return area

def event_hull_volume(x,y,z):
    pointCount = x.shape[0]
    volume = 0.0
    if pointCount > 3:
        # Need four points to make at least one tetrahedron.
        try:
            volume, vertices, simplex_volumes = hull_volume(np.vstack(
                                                                (x,y,z)).T)
        except QhullError:
            # this can happen with a degenerate first simplex - all points are
            # coplanar to machine precision. Try again, after adding a tiny amount
            # to the first point.
            print("Perturbing one source to help triangulation for flash with "
                  "{0} points".format(flash.pointCount))
            # we can tolerate perturbing by no more than 1 m
            machine_eps = 1.0 # np.finfo(x.dtype).eps
            perturb = 2*machine_eps*np.random.random(size=3)-machine_eps
            x[0] += perturb[0]
            y[0] += perturb[1]
            z[0] += perturb[2]
            volume, vertices, simplex_volumes = hull_volume(np.vstack(
                                                                (x,y,z)).T)
    return volume


def flash_stats(ds):
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
    event_area = fl_gb.apply(lambda df: event_hull_area(df['event_x'],
                                                        df['event_y'],
                                                        df['event_z']))
    event_volume = fl_gb.apply(lambda df: event_hull_volume(df['event_x'],
                                                            df['event_y'],
                                                            df['event_z']))

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
    ds['flash_area'][event_area.index] = event_area.values
    ds['flash_volume'][event_volume.index] = event_volume.values
    ds['flash_power'][sum_flidx] = sum_event_df['event_power']

    # recreate flash_id variable
    ds['flash_duration'] = ds['flash_time_start'] - ds['flash_time_end']
    ds.reset_index('number_of_flashes')
    ds['flash_id']=ds['number_of_flashes']

    return ds