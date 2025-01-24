from __future__ import absolute_import
import pyproj as proj4
from numpy import *
from numpy.linalg import norm

# def radians(degrees):
    # return deg2rad(asarray(degrees))
    # return array(degrees) * pi / 180.0

# def degrees(radians):
    # return rad2deg(asarray(radians))
    # return array(radians) * 180.0 / pi


class CoordinateSystem(object):
    """Superclass representing a generic coordinate system. Subclasses represent specific coordinate systems.

    Each subclass coordinate system must be able to convert data to a common coordinate system,
    which is chosen to be Earth-Centered, Earth-Fixed (ECEF) cartesian.

    This is implemented by the fromECEF and toECEF methods in each coordinate system object.
    
    User code is responsible for taking data in its native coord system,
    transforming it using to/fromECEF using the a coord system appropriate to the data, and then
    transforming that data to the final coordinate system using another coord system.

    Subclasses should maintain an attribute ERSxyz that can be used in transformations to/from an ECEF cartesian system, e.g.
    
    ```py
    import pyproj as proj4
    self.ERSxyz = proj4.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    self.ERSlla = proj4.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    projectedData = proj4.Transformer.from_crs(self.ERSlla.crs, self.ERSxyz.crs).transform(lon, lat, alt)
    ```
    
    The ECEF system has its origin at the center of the earth, with the +Z toward the north pole, +X toward (lat=0, lon=0), and +Y right-handed orthogonal to +X, +Z

    Depends on [pyproj](https://github.com/pyproj4/pyproj) to handle the ugly details of
    various map projections, geodetic transforms, etc.

    *"You can think of a coordinate system as being something like character encodings,
    but messier, and without an obvious winner like UTF-8." - Django OSCON tutorial, 2007*
    http://toys.jacobian.org/presentations/2007/oscon/tutorial/

    Notes
    -----
    This class is not intended to be instantiated directly. Instead, use one of the subclasses documented on the "Transforms" page.
    """

    # WGS84xyz = proj4.Proj(proj='geocent',  ellps='WGS84', datum='WGS84')

    def coordinates(self):
        """Return a tuple of standarized coordinate names"""
        raise NotImplementedError()

    def fromECEF(self, x, y, z):
        """Take ECEF x, y, z values and return x, y, z in the coordinate system defined by the object subclass"""
        raise NotImplementedError()

    def toECEF(self, x, y, z):
        """Take x, y, z in the coordinate system defined by the object subclass and return ECEF x, y, z"""
        raise NotImplementedError()


class GeographicSystem(CoordinateSystem):
    """Coordinate system defined using latitude, longitude, and altitude.
    An ellipsoid is used to define the shape of the earth. Latitude and longitude represent the
    location of a point on the ellipsoid, and altitude is the height above the ellipsoid.

    Attributes
    ----------
    ERSlla : pyproj.Proj
        A Proj object representing the geographic coordinate system.
    ERSxyz : pyproj.Proj
        A Proj object representing the Earth-Centered, Earth-Fixed (ECEF) cartesian coordinate system.
    """
    def __init__(self, ellipse='WGS84', datum='WGS84',
                 r_equator=None, r_pole=None):
        """Initialize a GeographicSystem object.

        Parameters
        ----------

        ellipse : str, default='WGS84'
            Ellipse name recognized by pyproj.
            
            *Ignored if r_equator or r_pole are provided.*
        datum : str, default='WGS84'
            Datum name recognized by pyproj.
            
            *Ignored if r_equator or r_pole are provided.*
        r_equator : float, optional
            Semi-major axis of the ellipse in meters.
            
            *If only one of r_equator or r_pole is provided, the resulting ellipse is assumed to be spherical.*
        r_pole : float, optional
            Semi-minor axis of the ellipse in meters.
            
            *If only one of r_equator or r_pole is provided, the resulting ellipse is assumed to be spherical.*
        """
        if (r_equator is not None) | (r_pole is not None):
            if r_pole is None:
                r_pole=r_equator
            if r_equator is None:
                r_equator=r_pole
            self.ERSlla = proj4.Proj(proj='latlong', a=r_equator, b=r_pole)
            self.ERSxyz = proj4.Proj(proj='geocent', a=r_equator, b=r_pole)
        else:
            # lat lon alt in some earth reference system
            self.ERSlla = proj4.Proj(proj='latlong', ellps=ellipse, datum=datum)
            self.ERSxyz = proj4.Proj(proj='geocent', ellps=ellipse, datum=datum)
    def toECEF(self, lon, lat, alt):
        """Converts longitude, latitude, and altitude to Earth-Centered, Earth-Fixed (ECEF) X, Y, Z coordinates.

        Parameters
        ----------
        lon : float or array_like
            Longitude in decimal degrees East of the Prime Meridian.
        lat : float or array_like
            Latitude in decimal degrees North of the equator.
        alt: float or array_like
            Altitude in meters above the ellipsoid.

        Returns
        -------
        X : float or array_like
            ECEF X in meters from the center of the Earth.
        Y : float or array_like
            ECEF Y in meters from the center of the Earth.
        Z : float or array_like
            ECEF Z in meters from the center of the Earth.
        """
        lat = atleast_1d(lat) # proj doesn't like scalars
        lon = atleast_1d(lon)
        alt = atleast_1d(alt)
        if (lat.shape[0] == 0): return lon, lat, alt # proj doesn't like empties
        projectedData = array(proj4.Transformer.from_crs(self.ERSlla.crs, self.ERSxyz.crs).transform(lon, lat, alt))
        if len(projectedData.shape) == 1:
            return projectedData[0], projectedData[1], projectedData[2]
        else:
            return projectedData[0,:], projectedData[1,:], projectedData[2,:]

    def fromECEF(self, x, y, z):
        """Converts Earth-Centered, Earth-Fixed (ECEF) X, Y, Z to longitude, latitude, and altitude coordinates.

        Parameters
        ----------
        x : float or array_like
            ECEF X in meters from the center of the Earth.
        y : float or array_like
            ECEF Y in meters from the center of the Earth.
        z : float or array_like
            ECEF Z in meters from the center of the Earth.

        Returns
        -------
        lon : float or array_like
            Longitude in decimal degrees East of the Prime Meridian.
        lat : float or array_like
            Latitude in decimal degrees North of the equator.
        alt: float or array_like
            Altitude in meters above the ellipsoid.
        """
        x = atleast_1d(x) # proj doesn't like scalars
        y = atleast_1d(y)
        z = atleast_1d(z)
        if (x.shape[0] == 0): return x, y, z # proj doesn't like empties
        projectedData = array(proj4.Transformer.from_crs(self.ERSxyz.crs, self.ERSlla.crs).transform(x, y, z))
        if len(projectedData.shape) == 1:
            return projectedData[0], projectedData[1], projectedData[2]
        else:
            return projectedData[0,:], projectedData[1,:], projectedData[2,:]


class MapProjection(CoordinateSystem):
    """Coordinate system defined using meters x, y, z in a specified map projection.
    Wraps pyproj, and uses its projecion names. Converts location in any map projection to ECEF, and vice versa.

    Attributes
    ----------
    ERSxyz : pyproj.Proj
        A Proj object representing the Earth-Centered, Earth-Fixed (ECEF) cartesian coordinate system.
    projection : pyproj.Proj
        A Proj object representing the map projection.
    ctrLat : float
        Latitude of the center of the map projection in decimal degrees North of the equator, if required for the projection.
    ctrLon : float
        Longitude of the center of the map projection in decimal degrees East of the Prime Meridian, if required for the projection.
    ctrAlt : float
        Altitude of the center of the map projection in meters.
    geoCS : GeographicSystem
        GeographicSystem object used to convert the map projection's center to ECEF.
    cx : float
        X coordinate of the map projection's center in meters.
    cy : float
        Y coordinate of the map projection's center in meters.
    cz : float
        Z coordinate of the map projection's center in meters.
    """

    def __init__(self, projection='eqc', ctrLat=None, ctrLon=None, ellipse='WGS84', datum='WGS84', **kwargs):
        """Initialize a MapProjection object.

        Parameters
        ----------
        projection : str, default='eqc'
            Projection name recognized by pyproj. Defaults to 'eqc' (equidistant cylindrical).
        ctrLat : float, optional
            Latitude of the center of the map projection in decimal degrees North of the equator, if required for the projection.
        ctrLon : float, optional
            Longitude of the center of the map projection in decimal degrees East of the Prime Meridian, if required for the projection.
        ellipse : str, default='WGS84'
            Ellipse name recognized by pyproj.
        datum : str, default='WGS84'
            Datum name recognized by pyproj.
        **kwargs
            Additional keyword arguments passed to pyproj.Proj()
        """
        self.ERSxyz = proj4.Proj(proj='geocent', ellps=ellipse, datum=datum)
        self.projection = proj4.Proj(proj=projection, ellps=ellipse, datum=datum, **kwargs)
        self.ctrLat=ctrLat
        self.ctrLon=ctrLon
        self.ctrAlt=0.0
        self.geoCS = GeographicSystem()
        self.cx, self.cy, self.cz = 0, 0, 0
        self.cx, self.cy, self.cz = self.ctrPosition()

    def ctrPosition(self):
        """Get the map projection's center position as projected in the specified map projection.

        Returns
        -------
        cx : float
            X coordinate of the map projection's center in meters.
        cy : float
            Y coordinate of the map projection's center in meters.
        cz : float
            Z coordinate of the map projection's center in meters.
        """
        if (self.ctrLat != None) & (self.ctrLon != None):
            ex, ey, ez = self.geoCS.toECEF(self.ctrLon, self.ctrLat, self.ctrAlt)
            cx, cy, cz = self.fromECEF(ex, ey, ez)
        else:
            cx, cy, cz = 0, 0, 0
        return cx, cy, cz

    def toECEF(self, x, y, z):
        """Converts x, y, z meters in the map projection to Earth-Centered, Earth-Fixed (ECEF) X, Y, Z coordinates.

        Parameters
        ----------
        x : float or array_like
            x position in the map projection in meters.
        y : float or array_like
            y position in the map projection in meters.
        z: float or array_like
            z position in the map projection in meters.

        Returns
        -------
        X : float or array_like
            ECEF X in meters from the center of the Earth.
        Y : float or array_like
            ECEF Y in meters from the center of the Earth.
        Z : float or array_like
            ECEF Z in meters from the center of the Earth.
        """
        x += self.cx
        y += self.cy
        z += self.cz
        projectedData = array(proj4.Transformer.from_crs(self.projection.crs, self.ERSxyz.crs).transform(x, y, z))
        if len(projectedData.shape) == 1:
            px, py, pz = projectedData[0], projectedData[1], projectedData[2]
        else:
            px, py, pz = projectedData[0,:], projectedData[1,:], projectedData[2,:]
        return px, py, pz

    def fromECEF(self, x, y, z):
        """Converts x, y, z meters in the map projection to Earth-Centered, Earth-Fixed (ECEF) X, Y, Z coordinates.

        Parameters
        ----------
        x : float or array_like
            x position in the map projection in meters.
        y : float or array_like
            y position in the map projection in meters.
        z: float or array_like
            z position in the map projection in meters.

        Returns
        -------
        X : float or array_like
            ECEF X in meters from the center of the Earth.
        Y : float or array_like
            ECEF Y in meters from the center of the Earth.
        Z : float or array_like
            ECEF Z in meters from the center of the Earth.
        """
        projectedData = array(proj4.Transformer.from_crs(self.ERSxyz.crs, self.projection.crs).transform(x, y, z))
        if len(projectedData.shape) == 1:
            px, py, pz = projectedData[0], projectedData[1], projectedData[2]
        else:
            px, py, pz = projectedData[0,:], projectedData[1,:], projectedData[2,:]
        return px-self.cx, py-self.cy, pz-self.cz

class PixelGrid(CoordinateSystem):
    """Coordinate system defined using arbitrary pixel coordinates in a 2D pixel array.

    Attributes
    ----------
    geosys : GeographicSystem
        GeographicSystem object used to convert pixel coordinates to ECEF.
    lookup : object
        Lookup object used to find the nearest pixel to a given lat/lon. See `__init__` for more details.
    x : list or array_like
        1D integer array of pixel row IDs.
    y : list or array_like
        1D integer array of pixel column IDs.
    lons : array_like
        2D array of longitudes of pixel centers.
    lats : array_like
        2D array of latitudes of pixel centers.
    alts : array_like
        2D array of altitudes of pixel centers.
    """
    def __init__(self, lons, lats, lookup, x, y, alts=None, geosys=None):
        """Initialize a PixelGrid object.

        Parameters
        ----------
        lons : array_like
            2D array of longitudes of pixel centers.
        lats : array_like
            2D array of latitudes of pixel centers.
        lookup : object
            Object with instance method `query` that accepts a single argument, a (N,2) array of lats, lons and
            returns a tuple of distances to the nearest pixel centers and the pixel ID of the nearest pixel to each
            requested lat/lon. A [`sklearn.neighbors.KDTree`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html) is the intended use of this argument, but any class with a `query` method is accepted.
        
            Example of a valid lookup object:

                test_events = np.vstack([(-101.5, 33.5), (-102.8, 32.5), (-102.81,32.5)])
                distances, idx = lookup.query(test_events)
                loni, lati = lons[X[idx], Y[idx]], lats[X[idx], Y[idx]]
        
        x : list or array_like
            1D integer array of pixel row IDs
        y : list or array_like
            1D integer array of pixel column IDs
        alts : array_like, optional
            2D array of altitudes of pixel centers. If None, zeros are assumed.
        geosys : GeographicSystem, optional
            GeographicSystem object used to convert pixel coordinates to ECEF. If None, a GeographicSystem instance with default arguments is created.

        Notes
        -----
        When converting toECEF, which accepts pixel coordinates,
        the z pixel coordinate is ignored, as it has no meaning.
        When converting fromECEF, zeros in the shape of x are returned as the z
        coordinate.
        """
        if geosys is None:
            self.geosys = GeographicSystem()
        else:
            self.geosys = geosys
        self.lookup = lookup
        self.x = x
        self.y = y
        self.lons = lons
        self.lats = lats
        if alts is None:
            alts = zeros_like(lons)
        self.alts = alts

    def toECEF(self, x, y, z=None):
        """Converts x, y pixel IDs to Earth-Centered, Earth-Fixed (ECEF) X, Y, Z coordinates.

        Parameters
        ----------
        x : float or array_like
            row ID of the pixel in the pixel grid.
        y : float or array_like
            column ID of the pixel in the pixel grid.
        z : object, optional
            unused. If provided, it is ignored.

        Returns
        -------
        X : float or array_like
            ECEF X in meters from the center of the Earth.
        Y : float or array_like
            ECEF Y in meters from the center of the Earth.
        Z : float or array_like
            ECEF Z in meters from the center of the Earth.
        """
        x = x.astype('int64')
        y = y.astype('int64')
        lons = self.lons[x, y]
        lats = self.lats[x, y]
        alts = self.alts[x, y]
        return self.geosys.toECEF(lons, lats, alts)

    def fromECEF(self, x, y, z):
        """Converts Earth-Centered, Earth-Fixed (ECEF) X, Y, Z to x, y pixel ID coordinates.

        Parameters
        ----------
        x : float or array_like
            ECEF X in meters from the center of the Earth.
        y : float or array_like
            ECEF Y in meters from the center of the Earth.
        z : float or array_like
            ECEF Z in meters from the center of the Earth.

        Returns
        -------
        x : array_like
            row ID of the pixel in the pixel grid.
        y : array_like
            column ID of the pixel in the pixel grid.
        z : array_like
            Zeros array in the shape of x.
        """
        lons, lats, alts = self.geosys.fromECEF(x, y, z)
        locs = vstack((lons.flatten(), lats.flatten())).T
        if locs.shape[0] > 0:
            distances, idx = self.lookup.query(locs)
        else:
            idx = []
        x = squeeze(self.x[idx])
        y = squeeze(self.y[idx])
        return x, y, zeros_like(x)

class GeostationaryFixedGridSystem(CoordinateSystem):
    """Coordinate system defined using scan angles from the perspective of a geostationary satellite.

        The pixel locations are a 2D grid of scan angles (in radians) from the perspective of a 
        geostationary satellite above an arbitrary ellipsoid.

        Attributes
        ----------
        ECEFxyz : pyproj.Proj
            A Proj object representing the Earth-Centered, Earth-Fixed (ECEF) cartesian coordinate system.
        fixedgrid : pyproj.Proj
            A Proj object representing the geostationary fixed grid coordinate system.
        h : float
            Height of the satellite in meters above the specified ellipsoid.
        """
    def __init__(self, subsat_lon=0.0, subsat_lat=0.0, sweep_axis='y',
                 sat_ecef_height=35785831.0,
                 ellipse='WGS84'):
        """Initialize a GeostationaryFixedGridSystem object.

        Parameters
        ----------
        subsat_lon : float, default=0.0
            Longitude of the subsatellite point in degrees.
        subsat_lat : float, default=0.0
            Latitude of the subsatellite point in degrees.
        sweep_axis : str, default='y'
            Axis along which the satellite sweeps. 'x' or 'y'. Use 'x' for GOES
            and 'y' for EUMETSAT.
        sat_ecef_height : float, default=35785831.0
            Height of the satellite in meters above the specified ellipsoid. Defaults to the height of the GOES satellite.
        ellipse : str or iterable, default='WGS84'
            A string representing a known ellipse to pyproj, or iterable of [a, b] (semi-major
            and semi-minor axes) of the ellipse. Default is 'WGS84'.
        """
        if type(ellipse) == str:
            ellipse_args = {'ellps': ellipse}
        elif hasattr(ellipse, '__iter__') and len(ellipse) == 2:
            rf = semiaxes_to_invflattening(ellipse[0], ellipse[1])
            ellipse_args = {'a': ellipse[0], 'rf': rf}
        else:
            raise ValueError("Ellipse must be a string or iterable of [a, b].")
        self.ECEFxyz = proj4.Proj(proj='geocent', **ellipse_args)
        self.fixedgrid = proj4.Proj(proj='geos', lon_0=subsat_lon,
            lat_0=subsat_lat, h=sat_ecef_height, x_0=0.0, y_0=0.0,
            units='m', sweep=sweep_axis, **ellipse_args)
        self.h=sat_ecef_height

    def toECEF(self, x, y, z):
        """Converts x, y, z satellite scan angles to Earth-Centered, Earth-Fixed (ECEF) X, Y, Z coordinates.

        Parameters
        ----------
        x : float or array_like
            horizontal scan angle in radians from the perspective of the satellite.
        y : float or array_like
            vertical scan angle in radians from the perspective of the satellite.
        z : float or array_like
            altitude above the ellipsoid expressed as a fraction of the satellite's height above the ellipsoid.

        Returns
        -------
        X : float or array_like
            ECEF X in meters from the center of the Earth.
        Y : float or array_like
            ECEF Y in meters from the center of the Earth.
        Z : float or array_like
            ECEF Z in meters from the center of the Earth.
        """
        X, Y, Z = x*self.h, y*self.h, z*self.h
        return proj4.Transformer.from_crs(self.fixedgrid.crs, self.ECEFxyz.crs).transform(X, Y, Z)

    def fromECEF(self, x, y, z):
        """Converts Earth-Centered, Earth-Fixed (ECEF) X, Y, Z to longitude, latitude, and altitude coordinates.

        Parameters
        ----------
        x : float or array_like
            ECEF X in meters from the center of the Earth.
        y : float or array_like
            ECEF Y in meters from the center of the Earth.
        z : float or array_like
            ECEF Z in meters from the center of the Earth.

        Returns
        -------
        x : float or array_like
            horizontal scan angle in radians from the perspective of the satellite.
        y : float or array_like
            vertical scan angle in radians from the perspective of the satellite.
        z : float or array_like
            altitude above the ellipsoid expressed as a fraction of the satellite's height above the ellipsoid.
        """
        X, Y, Z = proj4.Transformer.from_crs(self.ECEFxyz.crs, self.fixedgrid.crs).transform(x, y, z)
        return X/self.h, Y/self.h, Z/self.h

# class AltitudePreservingMapProjection(MapProjection):
#     def toECEF(self, x, y, z):
#         px, py, pz = super(AltitudePreservingMapProjection, self).toECEF(x, y, z)
#         return px, py, z
#
#     def fromECEF(self, x, y, z):
#         px, py, pz = super(AltitudePreservingMapProjection, self).fromECEF(x, y, z)
#         return px, py, z

class RadarCoordinateSystem(CoordinateSystem):
    """Coordinate system defined using the range, azimuth, and elevation angles from a radar.
        
        Locations are defined using the latitude, longitude, and altitude of the radar and the azimuth and elevation angles of the radar beam.

        Attributes
        ----------
        ctrLat : float
            Latitude of the radar in decimal degrees North of the equator.
        ctrLon : float
            Longitude of the radar in decimal degrees East of the Prime Meridian.
        ctrAlt : float
            Altitude of the radar in meters above sea level.
        datum : str
            Datum name recognized by pyproj.
        ellps : str
            Ellipse name recognized by pyproj.
        lla : pyproj.Proj
            A Proj object representing the geographic coordinate system.
        xyz : pyproj.Proj
            A Proj object representing the Earth-Centered, Earth-Fixed (ECEF) cartesian coordinate system.
        Requator : float
            Equatorial radius of the earth in meters.
        Rpolar : float
            Polar radius of the earth in meters.
        flattening : float
            Ellipsoid flattening parameter.
        eccen : float
            First eccentricity squared.
        effectiveRadiusMultiplier : float
            Multiplier to scale the earth's radius to account for the beam bending due to atmospheric refraction.
    """

    def __init__(self, ctrLat, ctrLon, ctrAlt, datum='WGS84', ellps='WGS84', effectiveRadiusMultiplier=4/3):
        """Initialize a RadarCoordinateSystem object.

        Parameters
        ----------

        ctrLat : float
            Latitude of the radar in decimal degrees North of the equator.
        ctrLon : float
            Longitude of the radar in decimal degrees East of the Prime Meridian.
        ctrAlt : float
            Altitude of the radar in meters above sea level.
        datum : str, default='WGS84'
            Datum name recognized by pyproj.
        ellps : str, default='WGS84'
            Ellipse name recognized by pyproj.
        effectiveRadiusMultiplier : float, default=4/3
            Multiplier to scale the earth's radius to account for the beam bending due to atmospheric refraction.
        """
        self.ctrLat = float(ctrLat)
        self.ctrLon = float(ctrLon)
        self.ctrAlt = float(ctrAlt)
        self.datum=datum
        self.ellps=ellps

        self.lla = proj4.Proj(proj='latlong', ellps=self.ellps, datum=self.datum)
        self.xyz = proj4.Proj(proj='geocent', ellps=self.ellps, datum=self.datum)

        self.Requator, _, _ = proj4.Transformer.from_crs(self.lla.crs, self.xyz.crs).transform(0,0,0) # Equatorial radius  - WGS-84 value = 6378137.0
        _, _, self.Rpolar = proj4.Transformer.from_crs(self.lla.crs, self.xyz.crs).transform(0,90,0) # Polar radius  - WGS-84 value = 6356752.314
        self.flattening = (self.Requator-self.Rpolar)/self.Requator

        self.eccen = (2.0-self.flattening)*self.flattening   # First eccentricity squared - WGS-84 value = 0.00669437999013
        self.effectiveRadiusMultiplier = effectiveRadiusMultiplier

    def getGroundRangeHeight(self, r, elevationAngle):
        """Convert slant range (along the beam) and elevation angle into ground range and height.
        Ground range given in great circle distance and height above the surface of the ellipsoid.
        Follows [Doviak and Zrnic 1993, eq. 2.28.](https://doi.org/10.1016/C2009-0-22358-0)
        
        Parameters
        ----------
        r : float or array_like
            slant range in meters.
        elevationAngle : float or array_like
            elevation angle in degrees above the horizon.

        Returns
        -------
        s : float or array_like
            Ground range (great circle distance) in meters.
        z : float or array_like
            Height above the surface of the ellipsoid in meters.

        Warning
        -------
        By default, an imaginary earth radius of 4/3 the actual earth radius is assumed to correct for atmospheric refraction.
        This is a common assumption in radar meteorology, but may not be accurate in all cases.
        """

        #Double precison arithmetic is crucial to proper operation.
        lat = self.ctrLat * pi / 180.0
        elev = array(elevationAngle * pi / 180.0, dtype='float64')
        slantr = array(r, dtype='float64')

        #figure out earth's radius at radar's lat ... non-spherical earth model
        e2 = self.eccen           # First eccentricity squared - WGS-84 value = 0.00669437999013
        a = self.Requator         # Equatorial radius  - WGS-84 value = 6378137.0
        Rearth = a/sqrt(1-e2*(sin(lat))**2) # radius of curvature

        Rprime = self.effectiveRadiusMultiplier * Rearth

        # Eqns 2.28b,c in Doviak and Zrnic 1993
        # Radar altitude is tacked on at the end, which isn't part of their derivation. At 100 km, it's
        #   worth < 10 m range error total for a radar at 500 m MSL. For 250 m gate spacing (typical at S-band),
        #   this is not too important.
        h = sqrt(slantr**2.0 + Rprime**2.0 + 2*slantr*Rprime*sin(elev)) - Rprime
        s = Rprime * arcsin( (slantr*cos(elev)) / (Rprime + h) )

        h += self.ctrAlt

        return s, h

    def getSlantRangeElevation(self, groundRange, z):
        """Convert ground range (great circle distance) and height above the earth's surface to slant range (along the beam) and elevation angle.
        Follows [Doviak and Zrnic 1993, eq. 2.28.](https://doi.org/10.1016/C2009-0-22358-0)
        
        Parameters
        ----------
        groundRange : float or array_like
            Ground range (great circle distance) in meters.
        z : float or array_like
            Height above the surface of the ellipsoid in meters.
        
        Returns
        -------
        r : float or array_like
            Slant range in meters.
        el : float or array_like
            Elevation angle in degrees above the horizon.
        """

        lat = self.ctrLat * pi / 180.0

        #figure out earth's radius at radar's lat ... non-spherical earth model
        e2 = self.eccen           # First eccentricity squared - WGS-84 value = 0.00669437999013
        a = self.Requator         # Equatorial radius  - WGS-84 value = 6378137.0
        Rearth = a/sqrt(1-e2*(sin(lat))**2) # radius of curvature

        Rprime = self.effectiveRadiusMultiplier * Rearth

        h = array(z - self.ctrAlt, dtype='float64')
        s = array(groundRange, dtype='float64')

        # Use law of cosines (Side-Angle-Side triangle theorem) with
        # R', R'+h as sides and s/R' as the angle to get slant range
        r  = sqrt(Rprime**2.0 + (Rprime+h)**2.0 - 2*(Rprime+h)*Rprime*cos(s/Rprime))
        # Inverse of eq. 2.28c in Doviak and Zrnic 1993
        # Will return NaN for r=0, and only positive angles
        el = atleast_1d(arccos((Rprime+h) * sin(s/Rprime) / r))
        # Below gives all negative angles
        # el = arcsin((Rprime+h) * sin(s/Rprime) / r) - (pi/2.0)

        # If elevation angle is negative, the triangle will be acute
        acute = atleast_1d( (Rprime+h)*(Rprime+h) < (Rprime*Rprime + r*r) )
        el[acute] *= -1
        el *= 180.0 / pi

        return r, el

    def toLonLatAlt(self, r, az, el):
        """Convert slant range r, azimuth az, and elevation el to latitude, longitude, altitude coordiantes.
        
        Parameters
        ----------
        r : float or array_like
            Slant range in meters.
        az : float or array_like
            Azimuth angle in degrees clockwise from North.
        el : float or array_like
            Elevation angle in degrees above the horizon.
        
        Returns
        -------
        lon : float or array_like
            Longitude in decimal degrees East of the Prime Meridian.
        lat : float or array_like
            Latitude in decimal degrees North of the equator.
        z : float or array_like
            Altitude in meters above the surface of the ellipsoid.
        """
        geodetic = proj4.Geod(ellps=self.ellps)

        try:
            n = max((az.size, r.size))
        except AttributeError:
            n = max((len(az), len(r)))

        dist, z = self.getGroundRangeHeight(r,el)
        lon, lat, backAz = geodetic.fwd([self.ctrLon]*n, [self.ctrLat]*n, az, dist)
        return lon, lat, z

    def toECEF(self, r, az, el):
        """Converts range, azimuth, and elevation to Earth-Centered, Earth-Fixed (ECEF) X, Y, Z coordinates.

        Parameters
        ----------
        r : float or array_like
            Distance in meters along the radar beam from the target to the radar.
        az : float or array_like
            Azimuth angle of the target in degrees clockwise from North.
        el: float or array_like
            Elevation angle of the target in degrees above the horizon.

        Returns
        -------
        X : float or array_like
            ECEF X in meters from the center of the Earth.
        Y : float or array_like
            ECEF Y in meters from the center of the Earth.
        Z : float or array_like
            ECEF Z in meters from the center of the Earth.
        """
        geoSys = GeographicSystem()
        lon, lat, z = self.toLonLatAlt(r, az, el)
        return geoSys.toECEF(lon, lat, z.ravel())

    def fromECEF(self, x, y, z):
        """Converts Earth-Centered, Earth-Fixed (ECEF) X, Y, Z to longitude, latitude, and altitude coordinates.

        Parameters
        ----------
        x : float or array_like
            ECEF X in meters from the center of the Earth.
        y : float or array_like
            ECEF Y in meters from the center of the Earth.
        z : float or array_like
            ECEF Z in meters from the center of the Earth.

        Returns
        -------
        r : float or array_like
            Distance in meters along the radar beam from the target to the radar.
        az : float or array_like
            Azimuth angle of the target in degrees clockwise from North.
        el: float or array_like
            Elevation angle of the target in degrees above the horizon.
        """
        # x = np.atleast1d(x)
        geoSys = GeographicSystem()
        geodetic = proj4.Geod(ellps=self.ellps)

        try:
            n = x.size
        except AttributeError:
            n = len(x)

        lon, lat, z = geoSys.fromECEF(x, y, z)
        radarToGateAz, gateToRadarAz, dist = geodetic.inv([self.ctrLon]*n, [self.ctrLat]*n, lon, lat)
        az = array(radarToGateAz)   #radarToGateAz may be a list.
        # change negative azimuths to positive
        az[az < 0.0] += 360.0

        #have height, ground range, azimuth. need to get elev angle and slant range from ground range and height
        r, el = self.getSlantRangeElevation(dist, z)

        return r, az, el

class TangentPlaneCartesianSystem(CoordinateSystem):
    """Coordinate system defined by a meters relative to plane tangent to the earth at a specified location.

    Attributes
    ----------
    ctrLat : float
        Latitude of the center of the local tangent plane in decimal degrees North of the equator.
    ctrLon : float
        Longitude of the center of the local tangent plane in decimal degrees East of the Prime Meridian.
    ctrAlt : float
        Altitude of the center of the local tangent plane in meters above the ellipsoid.
    centerECEF : numpy.ndarray
        ECEF X, Y, Z coordinates of the center of the local tangent plane.
    TransformToLocal : numpy.ndarray
        Rotation matrix to convert from ECEF to local tangent plane coordinates.
    """

    def __init__(self, ctrLat=0.0, ctrLon=0.0, ctrAlt=0.0):
        """Initialize a TangentPlaneCartesianSystem object.
        
        Parameters
        ----------
        ctrLat : float, default=0.0
            Latitude of the center of the local tangent plane in decimal degrees North of the equator.
        ctrLon : float, default=0.0
            Longitude of the center of the local tangent plane in decimal degrees East of the Prime Meridian.
        ctrAlt : float, default=0.0
            Altitude of the center of the local tangent plane in meters above the ellipsoid.
        """
        self.ctrLat = float(ctrLat)
        self.ctrLon = float(ctrLon)
        self.ctrAlt = float(ctrAlt)

        ERSlla = proj4.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        ERSxyz = proj4.Proj(proj='geocent',  ellps='WGS84', datum='WGS84')
        self.centerECEF = array(proj4.Transformer.from_crs(ERSlla.crs, ERSxyz.crs).transform(ctrLon, ctrLat, ctrAlt))

        #location of point directly above local center
        aboveCenterECEF = array(proj4.Transformer.from_crs(ERSlla.crs, ERSxyz.crs).transform(ctrLon, ctrLat, self.ctrAlt+1e3))

        #normal vector to earth's surface at the center is the local z direction
        n = aboveCenterECEF - self.centerECEF
        n = n / norm(n)
        localz = n[:,None] #make a column vector

        # n (dot) x = d defines a plane for normal vector n and position vector x on the plane
        d = dot(n, aboveCenterECEF)

        #north = array((northx, northy, northz))

        #http://www.euclideanspace.com/maths/geometry/elements/plane/index.htm
        #matrix to project point onto a plane defined by the normal vector n.
        P = identity(3,float) - transpose(vstack((n,n,n))) * vstack((n,n,n))

        # Point just to the north of the center on earth's surface, projected onto the tangent plane
        # This calculation seems like it should only be done with latitude/north since the local x
        #   direction curves away along a non-straight line when projected onto the plane
        northCenterECEF = array(proj4.Transformer.from_crs(ERSlla.crs, ERSxyz.crs).transform(self.ctrLon, self.ctrLat+1.01, self.ctrAlt))
        localy = dot(P, northCenterECEF[:,None] )
        localy = localy / norm(localy)


        #local x is y (cross) z to get an orthogonal system
        localx = transpose(cross(localy.transpose(), localz.transpose()))
        localx = localx / norm(localx)


        ECEFx = array((1.0, 0.0, 0.0))[:,None]
        ECEFy = array((0.0, 1.0, 0.0))[:,None]
        ECEFz = array((0.0, 0.0, 1.0))[:,None]

        #
        # Calculate the transformation matrix TM to go from
        #   the earth-centered earth-fixed (ECEF) system to the local tangent plane system
        # http://www.spenvis.oma.be/spenvis/help/background/coortran/coortran.html, http://mathworld.wolfram.com/DirectionCosine.html
        # (X1, X2, X3) are the direction cosines of the X-direction of the b-system, expressed in function of X, Y and Z of the a-system
        # b system = local tangent plane system     a system = ECEF system
        # [vb_x]   [[x1, x2, x3]  [va_x
        # [vb_y] =  [y1, y2, y3]   va_y
        # [vb_z]    [z1, z2, z3]]  va_z]
        # va = transpose(M) vb
        x1 = dot(localx.transpose(), ECEFx) # / abs(localx) ... don't need since normalized
        x2 = dot(localx.transpose(), ECEFy)
        x3 = dot(localx.transpose(), ECEFz)
        y1 = dot(localy.transpose(), ECEFx) # / abs(localx) ... don't need since normalized
        y2 = dot(localy.transpose(), ECEFy)
        y3 = dot(localy.transpose(), ECEFz)
        z1 = dot(localz.transpose(), ECEFx) # / abs(localx) ... don't need since normalized
        z2 = dot(localz.transpose(), ECEFy)
        z3 = dot(localz.transpose(), ECEFz)
        self.TransformToLocal = array([[x1, x2, x3],
                                       [y1, y2, y3],
                                       [z1, z2, z3]]).squeeze()

    def fromECEF(self, x, y, z):
        """Converts x, y, z meters in the local tangent plane to Earth-Centered, Earth-Fixed (ECEF) X, Y, Z coordinates.

        Parameters
        ----------
        x : float or array_like
            x position in meters East of the tangent plane center.
        y : float or array_like
            y position in meters North of the tangent plane center.
        z: float or array_like
            z position in meters above the tangent plane center.

        Returns
        -------
        X : float or array_like
            ECEF X in meters from the center of the Earth.
        Y : float or array_like
            ECEF Y in meters from the center of the Earth.
        Z : float or array_like
            ECEF Z in meters from the center of the Earth.

        Warning
        -------

        The z coordinate input is **NOT** the altitude above the ellipsoid. It is the z position in the local tangent plane.
        Due to the curvature of the Earth, the TPCS z position and altitude difference increases with distance from the center of the TPCS.
        """
        data = vstack((x, y, z))
        tpXYZ = self.toLocal(data)
        if len(tpXYZ.shape) == 1:
            tpX, tpY, tpZ = tpXYZ[0], tpXYZ[1], tpXYZ[2]
        else:
            tpX, tpY, tpZ = tpXYZ[0,:], tpXYZ[1,:], tpXYZ[2,:]
        return tpX, tpY, tpZ

    def toECEF(self, x, y, z):
        """Converts  Earth-Centered, Earth-Fixed (ECEF) X, Y, Z to x, y, z meters in the local tangent plane.

        Parameters
        ----------
        x : float or array_like
            ECEF X in meters from the center of the Earth.
        y : float or array_like
            ECEF Y in meters from the center of the Earth.
        z : float or array_like
            ECEF Z in meters from the center of the Earth.

        Returns
        -------
        x : float or array_like
            x position in meters East of the tangent plane center.
        y : float or array_like
            y position in meters North of the tangent plane center.
        z: float or array_like
            z position in meters above the tangent plane center.

        Warnings
        --------
        The x and z output coordinates are not the great circle distance (the distance along the surface of the Earth). Distances between points should be compared in their ECEF coordinates.
        
        Similarly, the z coordinate output is **NOT** the altitude above the ellipsoid. It is the z position in the local tangent plane.
        Due to the curvature of the Earth, the TPCS z position and altitude difference increases with distance from the center of the TPCS.
        
        If you want to find the altitude of a point above the ellipsoid, use the `GeographicSystem` class.
        """
        data = vstack((x, y, z))
        ecXYZ = self.fromLocal(data)
        if len(ecXYZ.shape) == 1:
            ecX, ecY, ecZ = ecXYZ[0], ecXYZ[1], ecXYZ[2]
        else:
            ecX, ecY, ecZ = ecXYZ[0,:], ecXYZ[1,:], ecXYZ[2,:]
        return ecX, ecY, ecZ

    def toLocal(self, data):
        """Transforms 3xN array of ECEF X, Y, Z coordinates to the local tangent plane cartesian system.

        Parameters
        ----------
        data : array_like
            (3, N) array of data (position vectors) in the ECEF system, representing X, Y, Z meters from the center of the Earth.

        Returns
        -------
        local_data : array_like
            (3, N) array of data (position vectors) in the local tangent plane cartesian system, representing x, y, z meters from the center of the tangent plane.
        """
        local_data = array( [ dot(self.TransformToLocal, (v-self.centerECEF)[:,None])
                        for v in data[0:3,:].transpose()]
                    ).squeeze().transpose()
        return local_data

    def fromLocal(self, data):
        """Transforms 3xN array of ECEF X, Y, Z coordinates to the local tangent plane cartesian system.

        Parameters
        ----------
        data : array_like
            (3, N) array of data (position vectors) in the ECEF system, representing X, Y, Z meters from the center of the Earth.

        Returns
        -------
        ecef_data : array_like
            (3, N) array of data (position vectors) in the local tangent plane cartesian system, representing x, y, z meters from the center of the tangent plane.
        """
        #Transform from local to ECEF uses transpose of the TransformToLocal matrix
        ecef_data = array( [ (dot(self.TransformToLocal.transpose(), v) + self.centerECEF)
                        for v in data[0:3,:].transpose()]
                    ).squeeze().transpose()
        return ecef_data


def semiaxes_to_invflattening(semimajor, semiminor):
    """ Calculate the inverse flattening from the semi-major and semi-minor axes of an ellipse"
    
    Parameters
    ----------
    semimajor : float
        Semi-major axis of the ellipse
    semiminor : float
        Semi-minor axis of the ellipse
    
    Returns
    -------
    rf : float
        Inverse flattening of the ellipse
    """
    rf = semimajor/(semimajor-semiminor)
    return rf


def centers_to_edges(x):
    """ Create an array of length N+1 edge locations from an array of lenght N grid center locations.
    
    Parameters
    ----------
    x : array_like
        (N,) array of locations of the centers 
    
    Returns
    -------
    xedge : numpy.ndarray
        (N+1,) array of locations of the edges

    Notes
    -----
    In the interior, the edge positions set to the midpoints
    of the values in x. For the outermost edges, half the 
    closest dx is assumed to apply.
    """
    xedge=zeros(x.shape[0]+1)
    xedge[1:-1] = (x[:-1] + x[1:])/2.0
    xedge[0] = x[0] - (x[1] - x[0])/2.0
    xedge[-1] = x[-1] + (x[-1] - x[-2])/2.0
    return xedge 


def centers_to_edges_2d(x):
    """Create a (N+1)x(M+1) array of edge locations from a
    NxM array of grid center locations.
    
    
    Parameters
    ----------
    x : array_like
        (N,M) array locations of the centers.
    
    Returns
    -------
    xedge : array_like
        (N+1,M+1) array of locations of the edges.
    
    Notes
    -----
    In the interior, the edge positions set to the midpoints
    of the values in x. For the outermost edges, half the 
    closest dx is assumed to apply. This matters for polar
    meshes, where one edge of the grid becomes a point at the
    polar coordinate origin; dx/2 is a half-hearted way of
    trying to prevent negative ranges.
    """
    xedge = zeros((x.shape[0]+1,x.shape[1]+1))
    # interior is a simple average of four adjacent centers
    xedge[1:-1,1:-1] = (x[:-1,:-1] + x[:-1,1:] + x[1:,:-1] + x[1:,1:])/4.0
    
    #         /\
    #        /\/\
    #       / /\ \
    #      /\/  \/\
    #     / /\  /\ \
    #    /\/  \/  \/\
    #   / /\  /\  /\ \
    #  /\/  \/  \/  \/\
    #4 \/\  /\  /\  /\/ 4
    # 3 \ \/  \/  \/ / 3 
    #    \/\  /\  /\/
    #   2 \ \/  \/ / 2  
    #      \/\  /\/
    #     1 \ \/ / 1
    #        \/\/
    #       0 \/ 0 = center ID of 0th dimension
    #
    
    # calculate the deltas along each edge, excluding corners
    xedge[1:-1,0] = xedge[1:-1, 1] - (xedge[1:-1, 2] - xedge[1:-1, 1])/2.0
    xedge[1:-1,-1]= xedge[1:-1,-2] - (xedge[1:-1,-3] - xedge[1:-1,-2])/2.0
    xedge[0,1:-1] = xedge[1,1:-1]  - (xedge[2,1:-1]  - xedge[1,1:-1])/2.0 
    xedge[-1,1:-1]= xedge[-2,1:-1] - (xedge[-3,1:-1] - xedge[-2,1:-1])/2.0
    
    # now do the corners
    xedge[0,0]  = xedge[1, 1] - (xedge[2, 2] - xedge[1, 1])/2.0
    xedge[0,-1] = xedge[1,-2] - (xedge[2,-3] - xedge[1,-2])/2.0
    xedge[-1,0] = xedge[-2,1] - (xedge[-3,2] - xedge[-2,1])/2.0 
    xedge[-1,-1]= xedge[-2,-2]- (xedge[-3,-3]- xedge[-2,-2])/2.0
    
    return xedge
