import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from scipy.special import factorial
from scipy.spatial.qhull import QhullError

def local_cartesian(lon, lat, alt):
    Re = 6378.137e3           #Earth's radius in m
    latavg, lonavg, altavg = lat.mean(), lon.mean(), alt.mean()
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

def event_hull(ds):
    x,y,z = local_cartesian(ds.event_longitude,
                            ds.event_latitude,
                            ds.event_altitude)
    pointCount = ds.dims['number_of_events']
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
    return (x,y,z), area, volume


def flash_stat_iter(fl_gb):
    for fl_id, ds in fl_gb:
        first_event = np.argmin(ds['event_time'])
        last_event = np.argmax(ds['event_time'])
        init_lon = ds['event_longitude'][first_event]
        init_lat = ds['event_latitude'][first_event]
        init_alt = ds['event_altitude'][first_event]
        ctr_lon = ds['event_longitude'].mean()
        ctr_lat = ds['event_latitude'].mean()
        ctr_alt = ds['event_altitude'].mean()
        n_events = ds.dims['number_of_events']
        init_time = ds['event_time'][first_event]
        end_time = ds['event_time'][last_event]
        total_power = ds.event_power.sum()
        (x,y,z), area, volume = event_hull(ds)
#         mean_stations = ds.event_stations.mean()
#         mean_chi2 = ds.event_chi2.mean()

        yield(fl_id, n_events, ctr_lon, ctr_lat, ctr_alt,
              init_lon, init_lat, init_alt,
              init_time.data, end_time.data,
              area*1e-6, volume*1e-9,
              total_power, # mean_stations, mean_chi2
             )

def flash_stats(ds):
    fl_gb = ds.groupby('event_parent_flash_id')
    n_flashes = len(fl_gb.groups)
    fl_stat_dtype = [('fl_id', 'u8'),
                     ('flash_event_count', 'u8'),
                     ('flash_center_longitude', 'f8'),
                     ('flash_center_latitude', 'f8'),
                     ('flash_center_altitude', 'f8'),
                     ('flash_init_longitude', 'f8'),
                     ('flash_init_latitude', 'f8'),
                     ('flash_init_altitude', 'f8'),
                     ('flash_time_start', 'datetime64[ns]'),
                     ('flash_time_end', 'datetime64[ns]'),
                     ('flash_area','f8'),
                     ('flash_volume','f8'),
                     ('event_total_power','f8'),
#                      ('event_mean_stations', 'f4'),
#                      ('event_mean_chi2', 'f4'),
                    ]
    ev_iter = flash_stat_iter(fl_gb)
    agg = np.fromiter(ev_iter, dtype=fl_stat_dtype, count=n_flashes)
    ds=ds.set_index(number_of_flashes='flash_id')
    to_assign = ('flash_event_count',
                 'flash_center_longitude',
                 'flash_center_latitude',
                 'flash_center_altitude',
                 'flash_init_longitude',
                 'flash_init_latitude',
                 'flash_init_altitude',
                 'flash_time_start',
                 'flash_time_end',
                 'flash_area',
                 'flash_volume',
                )
    for var in to_assign:
        ds[var][agg['fl_id']] = agg[var]
    ds['flash_duration'] = ds['flash_time_start'] - ds['flash_time_end']
    ds.reset_index('number_of_flashes')
    ds['flash_id']=ds['number_of_flashes']
    return ds