from pyxlma.coords import *
import pytest
import numpy as np
from sklearn.neighbors import KDTree

test_lats = np.array([33.5, 1.0, 0.0, 0.0, 0.0, 10.0, -10.0, 33.606968])
test_lons = np.array([-101.5, -75.0, -85.0, -65.0, -75.0, -75.0, -75.0, -101.822625])
test_alts = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 984.0])

test_ecef_X = np.array([-1061448.75418035, 1650533.58831094, 555891.26758132,
                        2695517.17208404, 1650783.32787306, 1625868.32721344,
                        1625868.32721344, -1089633.44245767])
test_ecef_Y = np.array([-5217187.30723133, -6159875.21117539, -6353866.26310279,
                        -5780555.22988658, -6160807.25190988, -6067823.20357756,
                        -6067823.20357756, -5205511.43302535])
test_ecef_Z = np.array([3500334.28802236, 110568.77482457, 0, 0,
                        0, 1100248.54773536, -1100248.54773536, 3510766.26631805])

def test_geographic():
    geosys = GeographicSystem()
    ecef_coords = geosys.toECEF(test_lons, test_lats, test_alts)
    lons, lats, alts = geosys.fromECEF(*ecef_coords)

    assert np.allclose(ecef_coords[0], test_ecef_X)
    assert np.allclose(ecef_coords[1], test_ecef_Y)
    assert np.allclose(ecef_coords[2], test_ecef_Z)
    assert np.allclose(lons, test_lons)
    assert np.allclose(lats, test_lats)
    assert np.allclose(alts, test_alts)

def test_geographic_one_point():
    geosys = GeographicSystem()
    ecef_coords = geosys.toECEF(np.atleast_1d(test_lons[-1]), np.atleast_1d(test_lats[-1]), np.atleast_1d(test_alts[-1]))
    print(len(np.atleast_1d(test_lons[-1]).shape))
    lons, lats, alts = geosys.fromECEF(*ecef_coords)

    assert np.allclose(ecef_coords[0], test_ecef_X[-1])
    assert np.allclose(ecef_coords[1], test_ecef_Y[-1])
    assert np.allclose(ecef_coords[2], test_ecef_Z[-1])
    assert np.allclose(lons[0], test_lons[-1])
    assert np.allclose(lats[0], test_lats[-1])
    assert np.allclose(alts[0], test_alts[-1])

def test_geographic_custom_r_both():
    geosys = GeographicSystem(r_equator=6378.137, r_pole=6356.752)
    ecef_coords = geosys.toECEF(test_lons, test_lats, test_alts)
    lons, lats, alts = geosys.fromECEF(*ecef_coords)
    assert np.allclose(lons, test_lons)
    assert np.allclose(lats, test_lats)
    assert np.allclose(alts, test_alts)

def test_geographic_custom_r_eq():
    geosys = GeographicSystem(r_equator=6378.137)
    ecef_coords = geosys.toECEF(test_lons, test_lats, test_alts)
    lons, lats, alts = geosys.fromECEF(*ecef_coords)
    assert np.allclose(lons, test_lons)
    assert np.allclose(lats, test_lats)
    assert np.allclose(alts, test_alts)

def test_geographic_custom_r_pole():
    geosys = GeographicSystem(r_pole=6356.752)
    ecef_coords = geosys.toECEF(test_lons, test_lats, test_alts)
    lons, lats, alts = geosys.fromECEF(*ecef_coords)
    assert np.allclose(lons, test_lons)
    assert np.allclose(lats, test_lats)
    assert np.allclose(alts, test_alts)

def test_equidistant_cylindrical():
    eqsys = MapProjection()
    ecef_coords = eqsys.toECEF(0, 0, 0)
    x, y, z = eqsys.fromECEF(*ecef_coords)
    assert np.allclose(x, 0)
    assert np.allclose(y, 0)
    assert np.allclose(z, 0)

def test_equidistant_cylindrical_custom_point():
    eqsys = MapProjection(ctrLat=test_lats[-1], ctrLon=test_lons[-1])
    ecef_coords = eqsys.toECEF(0, 0, 0)
    x, y, z = eqsys.fromECEF(*ecef_coords)
    assert np.allclose(x, 0)
    assert np.allclose(y, 0)
    assert np.allclose(z, 0)


# def test_px_grid():
#     lon = np.arange(-105, -99.9, 0.5)
#     x_coord = np.arange(0, len(lon))
#     lat = np.arange(30, 35.1, 0.5)
#     y_coord = np.arange(0, len(lat))
#     lon, lat = np.meshgrid(lon, lat)
#     pxgrid = PixelGrid(lon, lat, KDTree, x_coord, y_coord)
#     ecef_coords = pxgrid.toECEF(np.array(7), np.array(7), np.array(0))
#     x, y, z = pxgrid.fromECEF(*ecef_coords)

#     assert np.allclose(ecef_coords[0], test_ecef_X[0])
#     assert np.allclose(ecef_coords[1], test_ecef_Y[0])
#     assert np.allclose(ecef_coords[2], test_ecef_Z[0])

#     assert np.allclose(x, 7)
#     assert np.allclose(y, 7)
#     assert np.allclose(z, 0)

def test_satellite_system():
    sat = GeostationaryFixedGridSystem(subsat_lon=75.2)
    ecef_coords = sat.toECEF(0.01, 0.01, 0.01)
    x, y, z = sat.fromECEF(*ecef_coords)

    assert np.allclose(x, 0.01)
    assert np.allclose(y, 0.01)
    assert np.allclose(z, 0.01)


def test_radar_system_height():
    ADRAD_rcs = RadarCoordinateSystem(30.6177, -96.3365, 114)
    tornado_ground_range, beam_height_agl = ADRAD_rcs.getGroundRangeHeight(17150, 1.4)
    assert np.allclose(tornado_ground_range, 17144.013390611748)
    assert np.allclose(beam_height_agl, 550.2784673999995)

def test_radar_system_elevation():
    ADRAD_rcs = RadarCoordinateSystem(30.6177, -96.3365, 114)
    tornado_slant_range, radar_elevation = ADRAD_rcs.getSlantRangeElevation(17144.013390611748, 550.2784673999995)
    print(tornado_slant_range, radar_elevation)
    assert np.allclose(tornado_slant_range, 17150)
    assert np.allclose(radar_elevation, 1.4)

def test_radar_system_lla():
    ADRAD_rcs = RadarCoordinateSystem(30.6177, -96.3365, 114)
    tornado_lon, tornado_lat, tornado_alt = ADRAD_rcs.toLonLatAlt(np.atleast_1d(17150), np.atleast_1d(228), np.atleast_1d(1.4))
    assert np.allclose(tornado_lat, 30.51415605367721)
    assert np.allclose(tornado_lon, -96.46923405085701)
    assert np.allclose(tornado_alt, 550.2784674)

def test_radar_system_ecef():
    ADRAD_rcs = RadarCoordinateSystem(30.6177, -96.3365, 114)
    tornado_x, tornado_y, tornado_z = ADRAD_rcs.toECEF(np.atleast_1d(17150), np.atleast_1d(228), np.atleast_1d(1.4))
    tornado_r, tornado_az, tornado_el = ADRAD_rcs.fromECEF(tornado_x, tornado_y, tornado_z)
    assert np.allclose(tornado_r, 17150)
    assert np.allclose(tornado_az, 228)
    assert np.allclose(tornado_el, 1.4)

def test_tpcs():
    tpcs = TangentPlaneCartesianSystem(ctrLat=test_lats[-1], ctrLon=test_lons[-1], ctrAlt=test_alts[-1])
    ecef_coords = tpcs.toECEF(100, 100, 100)
    x, y, z = tpcs.fromECEF(*ecef_coords)
    assert np.allclose(x, 100)
    assert np.allclose(y, 100)
    assert np.allclose(z, 100)

def test_tpcs_local():
    tpcs = TangentPlaneCartesianSystem(ctrLat=test_lats[-1], ctrLon=test_lons[-1], ctrAlt=test_alts[-1])
    ecef_coords = tpcs.toLocal(np.array([[100, 100, 100], [200, 200, 200], [300, 300, 300]]))
    local_coords = tpcs.fromLocal(ecef_coords)
    assert np.allclose(local_coords, np.array([[100, 100, 100], [200, 200, 200], [300, 300, 300]]))
