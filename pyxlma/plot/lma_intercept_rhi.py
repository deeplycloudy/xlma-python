from pyxlma.coords import RadarCoordinateSystem, TangentPlaneCartesianSystem, GeographicSystem
import numpy as np  


def rcs_to_tps(radar_latitude, radar_longitude, radar_altitude, radar_azimuth):
    """
    Given a rhi radar file read by pyart in range elevation azimuth coordinates,
    it returns the tangent plane coordinates X(pointing east),Y(pointing north),Z(local height) of it.    
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


def geo_to_tps(lma_file, radar_latitude, radar_longitude, radar_altitude):
    """
    Given a lma file read by lmatools in latitude, longitude, altitude coordinates,it returns the tangent plane coordinates Xlma(pointing east),Ylma(pointing north),Z(local height) of it.
    """
    # GeographicSystem GEO - Lat, lon, alt
    geo = GeographicSystem()
    # Tangent Plane Cartesian System TPS - 
    tps = TangentPlaneCartesianSystem(radar_latitude, radar_longitude, radar_altitude)

    # GEO to TPS

    d, e, h = geo.toECEF(lma_file.event_longitude.data, lma_file.event_latitude.data, lma_file.event_altitude.data)
    

    deh = np.vstack((d,e,h))
    m = tps.toLocal(deh)
    
    Xlma = m[0]
    Ylma = m[1]
    Zlma = m[2]
    
    return Xlma,Ylma,Zlma


def ortho_proj_lma(lma_file, radar_latitude, radar_longitude, radar_altitude, radar_azimuth):
    """
    Given a lma file read by lmatools and radar file read by pyart, it transforms both datasets to tangent plane coordinate system,
    and returns the lma sources coordinates rotated with x:pointing along the RHI scan, y:orthogonal to x, counterclockwise and z:local height.
    """
    Xlma,Ylma,Zlma = geo_to_tps(lma_file, radar_latitude, radar_longitude, radar_altitude)
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
    elif radar_azimuth > 180 and radar_azimuth < 270:
        lma_file_loc[:,0][lma_file_loc_par[:,0] > 0] = -lma_file_loc[:,0][lma_file_loc_par[:,0] > 0]
    else:
        lma_file_loc[:,0][lma_file_loc_par[:,0] < 0] = -lma_file_loc[:,0][lma_file_loc_par[:,0] < 0]
    lma_file_loc[:,1] = np.sqrt(lma_file_loc_perp[:,0]**2 + lma_file_loc_perp[:,1]**2)
    lma_file_loc[:,2] = Zlma


    return lma_file_loc


def find_points_near_rhi(lma_file, radar_latitude, radar_longitude, radar_altitude, radar_azimuth, radar_scan_time, distance_threshold=1000, time_threshold=60):

    radar_azimuth = radar_azimuth % 360

    radar_scan_time = np.array([radar_scan_time]).astype('datetime64[s]')

    projected_lma = ortho_proj_lma(lma_file, radar_latitude, radar_longitude, radar_altitude, radar_azimuth)
    lma_range = projected_lma[:,0]
    lma_dist = projected_lma[:,1]
    lma_alt = projected_lma[:,2]

    lma_times = lma_file.event_time.data.astype('datetime64[s]')
    points_i_want = (lma_dist < distance_threshold) & (np.abs(lma_times - radar_scan_time).astype(float) < time_threshold)
    lma_range = lma_range[points_i_want]
    lma_dist = lma_dist[points_i_want]
    lma_alt = lma_alt[points_i_want]
    lma_ids = lma_file.event_id.data[points_i_want]

    return lma_range, lma_dist, lma_alt, lma_ids
