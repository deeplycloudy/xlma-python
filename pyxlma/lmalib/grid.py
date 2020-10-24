import itertools
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

def assign_regular_bins(dsg, ds, var_to_grid_map, pixel_id_var='pixel_id',
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
        in_range_check = (ds[var_name]>=x0) & (ds[var_name]<x1)
        if in_range is None:
            in_range = in_range_check
        else:
            in_range &= in_range_check
    ds = ds[{var_dim:in_range}]
    # After selecting along this dimension, need to prune the tree of entities that are not parents and children

    # Get the index for each data variable on the regular grid
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

    # if len(uids) > 0:
    #     group = ds.groupby(pixel_id_var)
    # else:
    #     group = None

    return ds #, group

def events_to_grid(ds, dsg, grid_spatial_coords=['grid_time',
                        'grid_altitude', 'grid_latitude', 'grid_longitude'],
                   event_spatial_vars = ('event_altitude', 'event_latitude',
                        'event_longitude',),
                   pixel_id_var='event_pixel_id',
                   min_points_per_flash=3):
    """ dataset after reduction and assignment of grid indexes

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
    pixel_id_suffix='event_pixel_id'
    px_id_names = [sv + '_parent_' + pixel_id_suffix for sv in px_source_vars]

    grid_coord_vars = [dsg[c] for c in grid_spatial_coords if c is not None]

    # These variables will be used to calculate values on the grid.
    event_vars_needed = ['event_power', 'event_chi2', 'event_stations',
        'event_id', 'event_parent_flash_id', 'event_time',]
    event_vars_needed += list(event_spatial_vars) + list(px_id_names)
    event_vars_needed += [pixel_id_var]


    # Code that generates ev_gb (flash_sort.py) should filter out the small
    # flashes instead of doing so within the gb. Write a new utility function
    # to filter the whole dataset
    # should prune out small flashes, and associated events first. Also, for some reason there are large chi2 events still in the dataset that are associated with flashes - e.g., flash 2 in the test dataset.
    # all_flash_ids = np.unique(ds.event_parent_flash_id)
    # big_flash_mask = (ds.flash_event_count >= min_points_per_flash)
    # big_flash_ids = ds.flash_id[big_flash_mask]
    # filtered_flash_ids = list(set(all_flash_ids.data) & set(big_flash_ids.data))
    # flash_count = len(filtered_flash_ids)
    # print("filtered_flash_ids")
    # print(filtered_flash_ids)
    # print('---')

    # === Convert to Pandas since xarray's groupby is slow. ====
    # Open issue in xarray as of Oct2020.

    # events must be time sorted because we want to later drop duplicates
    # and keep the first event in each pixel in each flash.
    ev_df = ds[event_vars_needed].to_dataframe().sort_values(by=['event_time'])

    flash_vars_needed = ['flash_id', 'flash_area',]
    fl_df = ds[flash_vars_needed].to_dataframe()


    # ===== Summarize data at each grid box: approach =====
    # Steps are:
    # 1. Replicate flash data to each event in the flash -> ev_df[fl_var] = ...
    # 2. At each pixel, keep one event for each flash - we choose the first
    #    event. These are the flash extent density-type quantities.
    # 3. Group by event grid box (event_pixel_id) -> ev_gb
    # 4. Summarize properties of all events in each pixel, regardless of flash.
    #    These are the event density-type quantities


    # Step 1.
    # Replicate the flash_vars_needed to each event based on
    # that event's event_parent_flash_id
    fl_df_idx = fl_df.set_index('flash_id')
    ev_df_fl_idx = ev_df.set_index('event_parent_flash_id')
    flash_vars_needed.remove('flash_id')
    for fl_var in flash_vars_needed:
        ev_df_fl_idx[fl_var] = fl_df_idx[fl_var]
    ev_df = ev_df_fl_idx.reset_index().set_index('event_id')

    # Step 2.
    first_event_df = ev_df.drop_duplicates(
                        ['event_parent_flash_id'] + list(px_id_names),
                        keep='first').groupby('event_pixel_id')
    n_flashes = first_event_df.size()
    areas_this_pixel = first_event_df['flash_area']
    fl_mean_area = areas_this_pixel.mean()
    fl_std_area = areas_this_pixel.std()
    fl_min_area = areas_this_pixel.min()

    # Get the grid box for each pixel using the first event in each grid box.
    # Given a groupby over globally unique pixel IDs, the IDs along each
    # dimension should also be identical for each point, so that we can
    # select the first point only.
    #         for sv in px_id_names:
    #             unq_ids = np.unique(ds[sv])
    #             if len(unq_ids) != 1:
    #                 print(unq_ids)
    #                 print(ds)
    fl_px_coord_ids = tuple(first_event_df[sv].first() for sv in px_id_names)
    # # Assign flash variables to the grid
    # sel = tuple(agg[pv] for pv in px_vars)
    # for var in to_grid:
    #     dsg[var] = xr.DataArray(coords=grid_coord_vars)
    #     # somehow we could probably use xarray's built-in indexing...
    #     dsg[var].data[sel] = agg[var]

    sel = fl_px_coord_ids
    for var, var_data in [('flash_extent_density', n_flashes),
                          ('average_flash_area', fl_mean_area),
                          ('stdev_flash_area', fl_std_area),
                          ('minimum_flash_area', fl_min_area),
                         ]:
        dsg[var] = xr.DataArray(coords=grid_coord_vars)
        # need to index on the raw numpy array (.data)
        # so we can use direct integer indexing
        dsg[var].data[sel] = var_data

    # Step 3.
    ev_gb = ev_df.groupby('event_pixel_id')
    # number of target grid cells to be filled - this is actually pretty slow!
    # n_occupied_pixels = len(ev_gb.groups)

    # Get the grid box for each pixel using the first event in each grid box.
    # Given a groupby over globally unique pixel IDs, the IDs along each
    # dimension should also be identical for each point, so that we can
    # select the first point only.
    #         for sv in px_id_names:
    #             unq_ids = np.unique(ds[sv])
    #             if len(unq_ids) != 1:
    #                 print(unq_ids)
    #                 print(ds)
    # first_points = ev_gb
    ev_px_coord_ids = tuple(ev_gb[sv].first() for sv in px_id_names)

    # Step 4.
    n_events = ev_gb.size()
    sum_ev_df = ev_gb[['event_power']].sum()
    # mean_ev_df = ev_gb[['event_chi2', 'event_stations']].mean()

    # to assign to final dataset
    ev_total_power = sum_ev_df['event_power']
    # ev_mean_stations = mean_ev_df['event_stations']
    # ev_mean_chi2 = mean_ev_df['event_chi2']

    # Handled by n_flashes above. Should be the same!
    # unq_fl_this_px = ev_gb['event_parent_flash_id'].unique()
    # flash_count = unq_fl_this_px.count()

    sel = ev_px_coord_ids
    for var, var_data in [('event_count', n_events),
                          ('event_total_power', ev_total_power),
                         ]:
        dsg[var] = xr.DataArray(coords=grid_coord_vars)
        # need to index on the raw numpy array (.data)
        # so we can use direct integer indexing
        dsg[var].data[sel] = var_data

    return dsg
