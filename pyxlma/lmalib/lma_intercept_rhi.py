from pyxlma.coords import RadarCoordinateSystem, TangentPlaneCartesianSystem, GeographicSystem
import numpy as np
import datetime as dt
import pandas as pd


def rcs_to_tps(radar_latitude, radar_longitude, radar_altitude, radar_azimuth):
    """Find the unit vector coordinates (east, north, up) of the plane of a radar RHI scan.
    
    Creates a azimuth, elevation, range and tangent plane cartesian system at the radar's latitude and longitude,
    and converts the RHI azimuth direction to the tangent plane coordinate system.

    Parameters
    ----------
    radar_latitude : float
        Latitude of the radar in degrees.
    radar_longitude : float
        Longitude of the radar in degrees.
    radar_altitude : float
        Altitude of the radar in meters.
    radar_azimuth : float
        Azimuth of the RHI scan in degrees.

    Returns
    ----------
    X : numpy.ndarray
        A 1x2 array representing the start and end points eastward component of the RHI scan.
    Y : numpy.ndarray
        A 1x2 array representing the start and end points northward component of the RHI scan.
    Z : numpy.ndarray
        A 1x2 array representing the start and end points upward component of the RHI scan.
    """

    # Coordinates Systems
    rcs = RadarCoordinateSystem(radar_latitude, radar_longitude, radar_altitude)
    tps = TangentPlaneCartesianSystem(radar_latitude, radar_longitude, radar_altitude)

    # - Elevations, azimuth, range
    r = np.array([0, 1])

    els = np.array([0])
    els = np.tensordot(els, np.ones_like(r), axes=0)

    azi = np.array([radar_azimuth])
    az = np.tensordot(azi, np.ones_like(r), axes=0)

    a, b, c = rcs.toECEF(r,az,els)
    abc = np.vstack((a,b,c))
    # ECEF to TPS
    n = tps.toLocal(abc)
    X = n[0,:]
    Y = n[1,:]
    Z = n[2,:]

    X = np.reshape(X,  (1, 2))
    Y = np.reshape(Y,  (1, 2))
    Z = np.reshape(Z,  (1, 2))
    
    return X, Y, Z


def geo_to_tps(event_longitude, event_latitude, event_altitude, tps_latitude, tps_longitude, tps_altitude):
    """
    Convert the latitude, longitude, and altitude of LMA VHF sources to x/y/z distances (in meters) from an arbitrary latitude, longitude, and altitude.

    Creates a tangent plane cartesian system at the latitude, longitude, and altitude provided, and converts the LMA VHF sources to the tangent plane coordinate system.

    Parameters
    ----------
    event_longitude : xarray.Dataset
        An LMA dataset containing latitude, longitude, and altitude of LMA VHF sources.
    tps_latitude : float
        Latitude of the tangent plane in degrees.
    tps_longitude : float
        Longitude of the tangent plane in degrees.
    tps_altitude : float
        Altitude of the tangent plane in meters.
    
    Returns
    ----------
    Xlma : numpy.ndarray
        A 1xN array representing the eastward distance (in meters) of the tangent plane center to the LMA VHF sources.
    Ylma : numpy.ndarray
        A 1xN array representing the northward distance (in meters) of the tangent plane center to the LMA VHF sources.
    Zlma : numpy.ndarray
        A 1xN array representing the upward distance (in meters) of the tangent plane center to the LMA VHF sources.
    """
    # GeographicSystem GEO - Lat, lon, alt
    geo = GeographicSystem()
    # Tangent Plane Cartesian System TPS - 
    tps = TangentPlaneCartesianSystem(tps_latitude, tps_longitude, tps_altitude)

    # GEO to TPS

    d, e, h = geo.toECEF(event_longitude, event_latitude, event_altitude)
    

    deh = np.vstack((d,e,h))
    m = tps.toLocal(deh)
    
    Xlma = m[0]
    Ylma = m[1]
    Zlma = m[2]
    
    return Xlma,Ylma,Zlma


def ortho_proj_lma(event_longitude, event_latitude, event_altitude, radar_latitude, radar_longitude, radar_altitude, radar_azimuth):
    """
    Convert the latitude, longitude, and altitude of LMA VHF sources to distance along, distance from, and height above the ground of a radar RHI scan.
    
    Creates tangent plane at the radar's location and converts the LMA VHF sources to the tangent plane coordinate system, then rotates the tangent plane in the direction of the scan azimuth.

    
    Parameters
    ----------
    lma_file : xarray.Dataset
        An LMA dataset containing latitude, longitude, and altitude of N number of LMA VHF sources.
    radar_latitude : float
        Latitude of the radar in degrees.
    radar_longitude : float
        Longitude of the radar in degrees.
    radar_altitude : float
        Altitude of the radar in meters.
    radar_azimuth : float
        Azimuth of the RHI scan in degrees.

    Returns
    ----------
    lma_file_loc : numpy.ndarray
        A Nx3 array representing the distance along, distance from, and height above the ground (in m) of the LMA VHF sources.

    """
    Xlma,Ylma,Zlma = geo_to_tps(event_longitude, event_latitude, event_altitude, radar_latitude, radar_longitude, radar_altitude)
    X, Y, Z = rcs_to_tps(radar_latitude, radar_longitude, radar_altitude, radar_azimuth)

    lon_ini1 = X[0,0]
    lat_ini1 = Y[0,0]
    lon_fin1 = X[0,-1]
    lat_fin1 = Y[0,-1]


    dlon1 = lon_fin1 - lon_ini1 # dx
    dlat1 = lat_fin1 - lat_ini1 # dy
    ds1 = np.array((dlon1,dlat1))
    norm_ds1 = np.linalg.norm(ds1)
    cross_ds1 = np.tensordot(ds1, ds1, (0,0))

    # LMA
    lma_file_n = np.column_stack((Xlma, Ylma))

    lma_file_loc_par = np.zeros(shape=(len(lma_file_n), 2))
    lma_file_loc_perp = np.zeros(shape=(len(lma_file_n), 2))
    lma_file_loc  = np.zeros(shape=(len(lma_file_n), 3))

    #
    # ##################################
    #
    #   (Xlma[i],Ylma[i]).ds1   .ds1
    #   ----------------------
    #        ds1 . ds1
    #
    # ##################################
    #
    lma_file_loc_tensor_x = np.tensordot(ds1,lma_file_n,(0,1))
    lma_file_loc_par = np.tensordot((lma_file_loc_tensor_x / cross_ds1 ),ds1,0)

    #
    # #######################################################################
    #
    #     (Xlma[i],Ylma[i])     _     (Xlma[i],Ylma[i]).ds1   .ds1
    #                                ----------------------
    #                                       ds1 . ds1
    #
    ##########################################################################
    #
    lma_file_loc_perp = lma_file_n - lma_file_loc_par

    #
    lma_file_loc[:,0] = np.sqrt(lma_file_loc_par[:,0]**2 + lma_file_loc_par[:,1]**2)
    if radar_azimuth <= 90 or radar_azimuth >= 270:
        lma_file_loc[:,0][lma_file_loc_par[:,1] < 0] = -lma_file_loc[:,0][lma_file_loc_par[:,1] < 0]
    elif radar_azimuth >= 180 and radar_azimuth < 270:
        lma_file_loc[:,0][lma_file_loc_par[:,0] > 0] = -lma_file_loc[:,0][lma_file_loc_par[:,0] > 0]
    else:
        lma_file_loc[:,0][lma_file_loc_par[:,0] < 0] = -lma_file_loc[:,0][lma_file_loc_par[:,0] < 0]
    lma_file_loc[:,1] = np.sqrt(lma_file_loc_perp[:,0]**2 + lma_file_loc_perp[:,1]**2)
    lma_file_loc[:,2] = Zlma

    return lma_file_loc


def find_points_near_rhi(event_longitude, event_latitude, event_altitude, event_time,
                         radar_latitude, radar_longitude, radar_altitude, radar_azimuth, radar_scan_time,
                         distance_threshold=1000, time_threshold=30):
    """
    Find the LMA VHF sources near a radar RHI scan.
    
    Creates tangent plane at the radar's location and converts the LMA VHF sources to the tangent plane coordinate system, then rotates the tangent plane in the direction of the scan azimuth.
    Filters RHI scan points based on distance and time thresholds.

    
    Parameters
    ----------
    event_longitude : array_like
        An array of the latitudes of events to be transformed.
    event_latitude : array_like
        An array of the latitudes of events to be transformed.
    event_altitude : array_like
        An array of the altitudes of events to be transformed.
    event_time : array_like
        An array of the times of events to be transformed.
    radar_latitude : float
        Latitude of the radar in degrees.
    radar_longitude : float
        Longitude of the radar in degrees.
    radar_altitude : float
        Altitude of the radar in meters.
    radar_azimuth : float
        Azimuth of the RHI scan in degrees.
    radar_scan_time : datetime.datetime or numpy.datetime64 or pandas.Timestamp
        Time of the RHI scan.
    distance_threshold : float
        Maximum distance from the radar to the LMA VHF sources in meters. Default is 1000.
    time_threshold : float
        Number of seconds before and after the RHI scan time to include LMA VHF sources. Default is 30. (total length: 1 minute)


    Returns
    ----------
    lma_range : numpy.ndarray
        A 1D array representing the distance along the tangent plane in the direction of the RHI scan.
    lma_dist : numpy.ndarray
        A 1D array representing the distance from the radar RHI scan plane to each filtered LMA point.
    lma_alt : numpy.ndarray
        A 1D array representing the height above the tangent plane centered at radar level of each filtered LMA point.
    point_mask : numpy.ndarray
        A 1D array of booleans representing the VHF points that were included in the return.
    """

    radar_azimuth = radar_azimuth % 360

    radar_scan_time = np.array([radar_scan_time]).astype('datetime64[s]').astype(dt.datetime)

    projected_lma = ortho_proj_lma(event_longitude, event_latitude, event_altitude,
                                   radar_latitude, radar_longitude, radar_altitude, radar_azimuth)
    lma_range = projected_lma[:,0]
    lma_dist = projected_lma[:,1]
    lma_alt = projected_lma[:,2]

    if isinstance(event_time, pd.Series):
        event_time = event_time.values
    lma_times = event_time.astype('datetime64[s]').astype(dt.datetime)
    point_mask = np.ones_like(lma_range, dtype=bool)
    if distance_threshold is not None:
        point_mask = np.logical_and(point_mask, lma_dist < distance_threshold)
    if time_threshold is not None:
        point_mask = np.logical_and(point_mask,
                                    np.abs(lma_times - radar_scan_time) < dt.timedelta(seconds=time_threshold))
    lma_range = lma_range[point_mask]
    lma_dist = lma_dist[point_mask]
    lma_alt = lma_alt[point_mask]
    return lma_range, lma_dist, lma_alt, point_mask


def find_lma_points_near_rhi(lma_file, radar_latitude, radar_longitude, radar_altitude, radar_azimuth, radar_scan_time, distance_threshold=1000, time_threshold=30):
    """
    Find the LMA VHF sources near a radar RHI scan.
    
    Creates tangent plane at the radar's location and converts the LMA VHF sources to the tangent plane coordinate system, then rotates the tangent plane in the direction of the scan azimuth.
    Filters RHI scan points based on distance and time thresholds.

    
    Parameters
    ----------
    lma_file : xarray.Dataset
        An LMA dataset containing latitude, longitude, and altitude, and event_id of N number of LMA VHF sources.
    radar_latitude : float
        Latitude of the radar in degrees.
    radar_longitude : float
        Longitude of the radar in degrees.
    radar_altitude : float
        Altitude of the radar in meters.
    radar_azimuth : float
        Azimuth of the RHI scan in degrees.
    radar_scan_time : datetime.datetime or numpy.datetime64 or pandas.Timestamp
        Time of the RHI scan.
    distance_threshold : float
        Maximum distance from the radar to the LMA VHF sources in meters. Default is 1000.
    time_threshold : float
        Number of seconds before and after the RHI scan time to include LMA VHF sources. Default is 30. (total length: 1 minute)


    Returns
    ----------
    lma_range : numpy.ndarray
        A 1D array representing the distance along the tangent plane in the direction of the RHI scan.
    lma_dist : numpy.ndarray
        A 1D array representing the distance from the radar RHI scan plane to each filtered LMA point.
    lma_alt : numpy.ndarray
        A 1D array representing the height above the tangent plane centered at radar level of each filtered LMA point.
    point_mask : numpy.ndarray
        A 1D array of booleans representing the VHF points that were included in the return.
    """
    return find_points_near_rhi(lma_file.event_longitude.data, lma_file.event_latitude.data, lma_file.event_altitude.data,
                                lma_file.event_time.data, radar_latitude, radar_longitude, radar_altitude, radar_azimuth,
                                radar_scan_time, distance_threshold, time_threshold)
