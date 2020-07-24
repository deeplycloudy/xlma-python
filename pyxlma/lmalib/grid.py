import numpy as np
import xarray as xr

def discretize(x, x0, dx, int_type='uint64', bounds_check=True):
    """ Calculate a unique location ID given some
        discretization interval and allowed range.

        Values less than x0 raise an exception of bounds_check=True,
        othewise the assigned index may wrap according when integer
        casting is performed.

        Arguments:
        x: coordinates, float array
        x0: minimum x value
        dx: discretization interval

        Keyword arguments:
        int_type: numpy dtype of x_id. 64 bit unsigned int by default, since 32
            bit is limited to a 65536 pixel square grid.

        Returns:
        x_id = unique pixel ID

        assert (np.array([0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3], dtype='uint64')
            == discretize(np.asarray([-2.0, -1.5, -1.0, -0.1, 0.0, 0.1, 0.4,
                          0.5, 0.6, 0.9, 1.0, 1.1]), -2.00, 1.0)).all()
    """
    if bounds_check:
        if (x<x0).any():
            raise ValueError("Some values are less than minimum")
    x_discr = ((x - x0) / dx).astype(int_type)
    return x_discr

def unique_ids(d, maxes, name='pixel_id'):
    """ Convert multidimensional integer pixel indices into a unique pixel_id.
    The maximum ID value is the product of all values in maxes.

    d is a list of 1D pixel ID arrays of the same length, d = [x, y, z, …]
    maxes is a list or array of the maximum value allowed along each coordinate
    in d.
    """
    cumprod = np.cumprod(maxes)
    # Example for 4D data
    #     d[0] + d[1]*maxes[0] + d[2]*maxes[0]*maxes[1] + d[3]*maxes[0]*maxes[1]*maxes[2]
    #     d[0] + d[1]*cumprod[0] + d[2]*cumprod[1] + d[3]*cumprod[2]
    #     discr = d[0]+ d[1:]*cumprod[:-1]
    discr = xr.DataArray(d[0].copy(), name=name)
    for di, cpi in zip(d[1:], cumprod[:-1]):
        discr += di*cpi
    return discr

def create_regular_grid(var_ranges, center_names=None):
    """
    var_ranges is a mapping from a variable name to (x0, x1, dx),
    the minimum, maximum, and spacing along that variable.
    Used as arguments to np.arange(x0, x1, dx), which should be thought of
    as the edges of the grid cells.

    This function will also calculate the grid cell centers (as is
    expected in the CF conventions) if a mapping {var_name:center_name} is
    provided in center_names.

    Dimensions are created with the same name as var_name
    (and, if provided, center_name) for later use in creating the actual grids.
    It is customary to create the grids with the grid center dimensions.

    Returns an xarray dataset representing a regular grid

    """
    dsg = {}

    for var_name, var_range in var_ranges.items():
        x0, x1, dx = var_range
        var_edge = np.arange(x0, x1, dx)
        dsg[var_name] = xr.DataArray(var_edge, name=var_name, dims=[var_name])
        if center_names is not None:
            # Can't use regular average with datetime variables
            # var_ctr = (var_edge[:-1] + var_edge[1:])/2.0
            var_ctr = var_edge[:-1] + dx/2.0
            ctr_name = center_names[var_name]
            dsg[ctr_name] = xr.DataArray(var_ctr, name=ctr_name,
                                         dims=[ctr_name])

    return xr.Dataset(dsg)

def groupby_regular_bins(dsg, ds, var_to_grid_map, pixel_id_var='pixel_id',
        append_indices=True):
    """
    dsg is a regular grid created by a call to create_regular_grid ds is the
    dataset to be grouped by its pixel location within dsg var_to_grid_map is a
    dict mapping from the variable in ds to the grid edge variable in dsg.

    The variables in var_to_grid_map must all be along the same dimension, and
    there should only be one variable for each corresponding dimension in the
    grid. If variables along along another dimension need to be grided to the
    same grid, call this function a second time with a different pixel_id_var.
    Do the same if multiple sets of variables along the same variable dimension
    need to be gridded.

    For example, consider counting the number of points within each 4-D grid
    cell. If I have N point_{x,y,z,t}s that have been clustered into M objects
    for which M cluster_center_{x,y,z,t}s and M cluster_first_point_{x,y,z,t}s
    have been identified, this function should be called three times, with a
    different pixel_id_var each time.

    The unique pixel index for each element along is stored in a new variable
    pixel_id.

    If append_indices is True, add the indices on the grid for each variable in
    var_ranges to var_name + '_parent_' + pixel_id_var.

    The regular grid spacing is calculated from the grid edge variable's (first
    - zeroth) elements. This spacing is used to calculate the grid index for
    each data variable. While it is not strictly necessary, since we could use
    search_sorted with the edges provided by dsg, the regularity of the grid
    allows us to extend to very large datasets because the regular grid index
    can be calculated without constructing the binary search tree used by
    search_sorted.

    """

    # Make sure all data are in range so we can skip the bounds check on the
    # call to discretize
    in_range, var_dim = None, None
    grid_edge_maxes = {}
    grid_edge_ranges = {}
    for var_name, grid_edge_var in var_to_grid_map.items():
        # Find the dimension of the source data
        if var_dim is None:
            var_dim = ds[var_name].dims[0]
        else:
            assert var_dim == ds[var_name].dims[0]
        # Get the grid spec along this dimension
        grid_edge_name = var_to_grid_map[var_name]
        grid_edge_var = dsg[grid_edge_name]
        x0, x1 = grid_edge_var.min(), grid_edge_var.max()
        dx = (x1-x0)/(grid_edge_var.shape[0]-1)
        ximax = discretize(np.asarray(x1), x0, dx)
        grid_edge_ranges[grid_edge_name] = (x0, x1, dx, ximax)
        # Mask out points along the source data dimension that
        # aren't on the grid.
        print(var_name)
        in_range_check = (ds[var_name]>=x0) & (ds[var_name]<x1)
        if in_range is None:
            in_range = in_range_check
        else:
            in_range &= in_range_check
    ds = ds[{var_dim:in_range}]
    # After selecting along this dimension, need to prune the tree of entities that are not parents and children

    # Get the index for each data variabile on the regular grid
    all_id = []
    maxes = []
    for var_name, grid_edge_var in var_to_grid_map.items():
        x0, x1, dx, ximax = grid_edge_ranges[grid_edge_var]
        xid = discretize(ds[var_name], x0, dx, bounds_check=False)
        all_id.append(xid)
        maxes.append(ximax)
        if append_indices is not False:
            # add to original dataset
            idx_name = var_name + '_parent_' + pixel_id_var
            ds[idx_name] = xid

    uids = unique_ids(all_id, maxes, name=pixel_id_var)
    ds[pixel_id_var] = uids

    if len(uids) > 0:
        group = ds.groupby(pixel_id_var)
    else:
        group = None

    return ds, group

def events_to_grid(ev_gb, dsg, grid_spatial_coords=['grid_time',
                        'grid_altitude', 'grid_latitude', 'grid_longitude'],
                   event_spatial_vars = ('event_altitude', 'event_latitude',
                        'event_longitude',),
                   to_grid = ('event_count', 'event_total_power',
                             'flash_extent_density', 'average_flash_area',
                             'stdev_flash_area', 'minimum_flash_area'),
                   min_points_per_flash=3):
    """ dataset after reduction and assignment of grid indexes, and groupby
    unique grid box id

    grid_spatial_coords should be in t, z, y, x order, and can be overridden
    for grids in other projection coordinates. If gridding to one of those
    coordinates is not needed, set it to None.
    """

    # Filter out pixel coordinates that aren't needed for this grid.
    all_px_vars = 'event_t_px', 'event_z_px', 'event_y_px', 'event_x_px'
    px_vars = tuple(pv for pv, g in
            zip(all_px_vars, grid_spatial_coords) if g is not None)
    all_pixel_source_vars = ('event_time',) + event_spatial_vars
    px_source_vars = tuple(pv for pv, g in
            zip(all_pixel_source_vars, grid_spatial_coords) if g is not None)

    n_lutevents = len(ev_gb.groups)

    # Create the dtype matching the sequence of data values that
    # wil be received from the iterator.
    # Unique grid box id
    event_px_dtype  = [('event_px', 'u8'),]
    # Pixel ID along each dimension of the grid
    event_px_dtype += [(px_name, 'u8') for px_name in px_vars]
    # Representative values at that grid box
    event_px_dtype += [('event_count', 'u8'),
                       ('flash_extent_density', 'u8'),
                       ('average_flash_area', 'f8'),
                       ('stdev_flash_area', 'f8'),
                       ('minimum_flash_area', 'f8'),
                       ('event_total_power','f8'),
                       ('event_mean_stations', 'f4'),
                       ('event_mean_chi2', 'f4'),
                      ]

    ev_iter = events_grid_iter(ev_gb, min_points_per_flash,px_source_vars, )
    agg = np.fromiter(ev_iter, dtype=event_px_dtype, count=n_lutevents, )
    coord_vars = [dsg[c] for c in grid_spatial_coords if c is not None]

    sel = tuple(agg[pv] for pv in px_vars)
    for var in to_grid:
        dsg[var] = xr.DataArray(coords=coord_vars)
        # need to index on the raw numpy array (.data) so we can use direct integer indexing
        # somehow we could probably use xarray's built-in indexing...
        dsg[var].data[sel] = agg[var]
    return dsg

def events_grid_iter(ev_gb, min_points_per_flash,
                     source_vars,
                     pixel_id_suffix='event_pixel_id',
                    ):
    """ spatial_vars can be overridden for gridding to projection coordinates
    with some other name
    """
    px_id_names = [sv + '_parent_' + pixel_id_suffix for sv in source_vars]
    for event_pixel_id, ds in ev_gb:
        # Given a groupby over globally unique pixel IDs, the IDs along each
        # dimension should also be identical for each point, so that we can
        # select the first point only.
#         for sv in px_id_names:
#             unq_ids = np.unique(ds[sv])
#             if len(unq_ids) != 1:
#                 print(unq_ids)
#                 print(ds)
        px_coord_ids = tuple(ds[sv][0] for sv in px_id_names)
        n_events = ds.dims['number_of_events']
        total_power = ds.event_power.sum()
        mean_stations = ds.event_stations.mean()
        mean_chi2 = ds.event_chi2.mean()

        all_flash_ids = np.unique(ds.event_parent_flash_id)
        big_flash_mask = (ds.flash_event_count >= min_points_per_flash)
        big_flash_ids = ds.flash_id[big_flash_mask]
        filtered_flash_ids = list(set(all_flash_ids.data) & set(big_flash_ids.data))
        flash_count = len(filtered_flash_ids)

        if flash_count > 0:
            ds_fl = ds.set_index({'number_of_flashes':'flash_id'})[{'number_of_flashes':filtered_flash_ids}]
            flash_mean_area = ds_fl.flash_area.mean()
            flash_std_area = ds_fl.flash_area.std()
            flash_min_area = ds_fl.flash_area.min()
        else:
            flash_mean_area = np.nan
            flash_std_area = np.nan
            flash_min_area = np.nan

        out = (event_pixel_id,)
        out += px_coord_ids
        out += (n_events, flash_count, flash_mean_area, flash_std_area, flash_min_area,
                total_power, mean_stations, mean_chi2)
        yield out

