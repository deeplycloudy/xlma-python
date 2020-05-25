import xarray as xr
import numpy as np
import datetime

from pyxlma.coords import GeographicSystem
import pyxlma.lmalib.io.cf_netcdf as cf_netcdf

def cluster_dbscan(X, Y, Z, T, min_points=1):
    """ Identify clusters in spatiotemporal data X, Y, Z, T.

    min_points is used as min_samples in the call to DBSCAN.

    Returns an unsigned 64 bit integer array of cluster labels.
    Noise points identified by DBSCAN are assigned the maximum value
    that can be represented by uint64, i.e., np.iinfo(np.uint64).max or
    18446744073709551615.
    """
    from sklearn.cluster import DBSCAN
    coords = np.vstack((X, Y, Z, T)).T
    db = DBSCAN(eps=1.0, min_samples=min_points, metric='euclidean')
    clusters = db.fit(coords)
    noise = (clusters.labels_ == -1)
    labels = clusters.labels_.astype('uint64')
    labels[noise] = np.iinfo(np.uint64).max
    return labels

def cluster_flashes(events, distance=3000.0, time=0.15):
    """

    events: xarray dataset conforming to the pyxlma CF NetCDF format

    distance: spatial separation in meters. Used for normalization of
    time: number
    """

    geoCS = GeographicSystem()
    X,Y,Z = geoCS.toECEF(events.event_longitude.data,
        events.event_latitude.data,
        events.event_altitude.data)
    Xc, Yc, Zc = geoCS.toECEF(events.network_center_longitude.data,
        events.network_center_latitude.data,
        events.network_center_altitude.data)
    X_norm = (X-Xc)/distance
    Y_norm = (Y-Yc)/distance
    Z_norm = (Z-Zc)/distance

    T_rel = (events.event_time.data - events.event_time.data.min())
    T_norm = (T_rel/np.timedelta64(1, 's')) / time
    labels = cluster_dbscan(X_norm, Y_norm, Z_norm, T_norm, min_points=1)
    n_labels = len(set(labels))

    flash_ds = cf_netcdf.new_dataset(flashes=n_labels)

    # Relabel any noise points with the fill value specified in the CF spec.
    noise = (labels == np.iinfo(np.uint64).max)
    labels[noise] = flash_ds.flash_id.attrs['_FillValue']
    flash_ds['flash_id'][:] = np.unique(labels)

    # ds = xr.merge([events, flash_ds], compat='override')
    ds = events.merge(flash_ds, compat='override')
    label_kwargs = cf_netcdf.__template_dataset['data_vars']['event_parent_flash_id']
    label_kwargs.pop('dtype') # given by the labels array
    ds['event_parent_flash_id'] = xr.DataArray(labels, **label_kwargs)

    # Add the CF parent/child metadata
    ds.flash_id.attrs.update({'child': 'event_id', 'cf_role': 'tree_id'})

    # Add other metadata
    ds.flash_distance_separation_threshold.data = distance
    ds.flash_time_separation_threshold.data = time
    ds.flash_duration_threshold.data = np.nan
    ds.flash_time_merge_threshold.data = np.nan
    ds.attrs['flash_algorithm_name'] = 'pyxlma DBSCAN'
    ds.attrs['flash_algorithm_version'] = '0.1'
    ds.attrs['title'] = ds.attrs['title'] + " L2 flashes."
    ds.attrs['production_date'] = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S +00:00")

    return ds