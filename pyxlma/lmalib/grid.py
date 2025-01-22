import itertools
import numpy as np
import xarray as xr

def discretize(x, x0, dx, int_type='uint64', bounds_check=True):
    """Calculate a unique location ID given some discretization interval and allowed range.

    Values less than x0 raise an exception of bounds_check=True,
    othewise the assigned index may wrap according when integer
    casting is performed.

    Parameters
    ----------
    x : array_like
        coordinates of the points
    x0 : float
        minimum x value
    dx : float
        discretization interval

    int_type : str, default='uint64'
        numpy dtype of x_id. 64 bit unsigned int by default, since 32 bit is limited to a 65536 pixel square grid.

    Returns
    -------
    x_id : array_like
        unique pixel ID
    """
    # assert (np.array([0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3], dtype='uint64')
    #         == discretize(np.asarray([-2.0, -1.5, -1.0, -0.1, 0.0, 0.1, 0.4,
    #                       0.5, 0.6, 0.9, 1.0, 1.1]), -2.00, 1.0)).all()
    if bounds_check:
        if (x<x0).any():
            raise ValueError("Some values are less than minimum")
    x_discr = ((x - x0) / dx).astype(int_type)
    return x_discr

def unique_ids(d, maxes, name='pixel_id'):
    """Convert multidimensional integer pixel indices into a unique pixel_id.
    
    Parameters
    ----------
    d : array_like
        (N,) array of pixel IDs
    maxes : array_like
        (N,) array of maximum pixel values
    name : str, default='pixel_id'
        name of the output DataArray
    
    Returns
    -------
    discr : xarray.DataArray
        unique pixel ID
    
    Notes
    -----
    The maximum ID value is the product of all values in maxes.
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
    """Create a regular grid from a set of variable ranges.

    Parameters
    ----------
    var_ranges : dict
        A dictionary where the keys represent variable edge names and the values are a 3-element tuple of (start, stop, step).
        As in np.arange(start, stop, step), the stop value is not included in the range.
    center_names : dict, optional
        A dictionary where the keys represent variable edge names and the values are the corresponding center names.
        If unspecified, the center values are not calculated. The CF conventions expect the grid to be defined by the center values.

    Returns
    -------
    dsg : xarray.Dataset
        A dataset with coordinates representing a regular grid and no data variables.
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
    """Assign pixel IDs and variable parent pixel IDs to a dataset based on a regular grid.
    
    Parameters
    ----------
    dsg : xarray.Dataset
        a regular grid in the format of a dataset created by a call to create_regular_grid
    ds : xarray.Dataset
        dataset to be grouped by its pixel location within dsg
    var_to_grid_map : dict
        mapping of the variable name (keys) in ds to the grid edge variable (values) in dsg.
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
    pixel_id_var : str
        The name of the pixel ID variable to be created in ds.
        If append_indices is True, add the indices on the grid for each variable in
        var_ranges to var_name + '_parent_' + pixel_id_var.

    Returns
    -------
    ds : xarray.Dataset
        the original dataset with pixel IDs assigned
    
    Notes
    -----
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
        if ds.sizes[var_dim] < 1:
            # No data on the dim, set in_range to nothing and stop checking other vars
            in_range = []
            have_data = False
            break
        else:
            have_data = True
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
        if have_data:
            x0, x1, dx, ximax = grid_edge_ranges[grid_edge_var]
            xid = discretize(ds[var_name], x0, dx, bounds_check=False)
            maxes.append(ximax)
        else:
            xid = xr.DataArray(np.array([], dtype='uint64'))
        all_id.append(xid)
        if append_indices is not False:
            # add to original dataset
            idx_name = var_name + '_parent_' + pixel_id_var
            ds[idx_name] = xid

    if have_data:
        uids = unique_ids(all_id, maxes, name=pixel_id_var)
    else:
        uids = xr.DataArray(np.array([], dtype=all_id[0].dtype))
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
    """Assign LMA events to a binned gridded dataset.
    
    Assignes gridded flash products to an LMA dataset using a gridded and binned dataset
    (see create_regular_grid and assign_regular_bins).

    Parameters
    ----------
    ds : xarray.Dataset
        LMA dataset with event data
    dsg : xarray.Dataset
        Dataset with grid edge variables and pixel IDs assigned
    grid_spatial_coords : iterable[str], default=('grid_time', 'grid_altitude', 'grid_latitude', 'grid_longitude')
        Names of the grid edge variables in dsg in t, z, y, x order.
        If gridding to one of those coordinates is not needed, set it to None.
    event_spatial_vars : iterable[str], default=('event_altitude', 'event_latitude', 'event_longitude')
        Names of the event spatial variables in ds in t, z, y, x order.
    pixel_id_var : str, default='event_pixel_id'
        Name of the pixel ID variable to be created in ds.
    min_points_per_flash : int, default=3
        Minimum number of points required to form a flash.

    Returns
    -------
    dsg : xarray.Dataset
        Dataset with gridded flash products
    """

    # Filter out pixel coordinates that aren't needed for this grid.
    all_px_vars = 'event_t_px', 'event_z_px', 'event_y_px', 'event_x_px'
    px_vars = tuple(pv for pv, g in
            zip(all_px_vars, grid_spatial_coords) if g is not None)
    all_pixel_source_vars = ('event_time',) + event_spatial_vars
    px_source_vars = tuple(pv for pv, g in
            zip(all_pixel_source_vars, grid_spatial_coords) if g is not None)
    pixel_id_suffix=pixel_id_var
    px_id_names = [sv + '_parent_' + pixel_id_suffix for sv in px_source_vars]

    grid_coord_vars = [dsg[c] for c in grid_spatial_coords if c is not None]

    # These variables will be used to calculate values on the grid.
    event_vars_needed = ['event_power', 'event_chi2', 'event_stations',
        'event_id', 'event_parent_flash_id', 'event_time',]
    event_vars_needed += list(event_spatial_vars) + list(px_id_names)
    event_vars_needed += [pixel_id_var]

    # === Convert to Pandas since xarray's groupby is slow. ====
    # Open issue in xarray as of Oct2020.

    # events must be time sorted because we want to later drop duplicates
    # and keep the first event in each pixel in each flash.
    these_events = ds[event_vars_needed]
    if these_events[event_vars_needed[0]].shape[0] < 1:
        have_data = False
        n_flashes = None
        fl_mean_area = None
        fl_std_area = None
        fl_min_area = None
        fl_mean_energy = None
    else:
        have_data = True
        ev_df = these_events.to_dataframe().sort_values(by=['event_time'])

        flash_vars_needed = ['flash_id', 'flash_area', 'flash_energy']
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
                            keep='first').groupby(pixel_id_var)
        n_flashes = first_event_df.size()
        areas_this_pixel = first_event_df['flash_area']
        energy_this_pixel= first_event_df['flash_energy']
        fl_mean_area = areas_this_pixel.mean()
        fl_std_area = areas_this_pixel.std()
        fl_min_area = areas_this_pixel.min()
        fl_mean_energy=energy_this_pixel.mean()

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

    for var, var_data in [('flash_extent_density', n_flashes),
                          ('average_flash_area', fl_mean_area),
                          ('stdev_flash_area', fl_std_area),
                          ('minimum_flash_area', fl_min_area),
                          ('average_flash_energy',fl_mean_energy),
                         ]:
        dsg[var] = xr.DataArray(np.nan, coords=grid_coord_vars)
        # need to index on the raw numpy array (.data)
        # so we can use direct integer indexing
        if have_data:
            sel = fl_px_coord_ids
            dsg[var].data[sel] = var_data

    # Step 3.
    if have_data:
        ev_gb = ev_df.groupby(pixel_id_var)
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
    else:
        n_events = None
        ev_total_power = None

    for var, var_data in [('event_count', n_events),
                          ('event_total_power', ev_total_power),
                         ]:
        dsg[var] = xr.DataArray(coords=grid_coord_vars)
        # need to index on the raw numpy array (.data)
        # so we can use direct integer indexing
        if have_data:
            sel = ev_px_coord_ids
            dsg[var].data[sel] = var_data

    return dsg
