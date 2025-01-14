from pyxlma.xarray_util import concat_1d_dims

def combine_glm_l2(filenames):
    """Combine multiple GLM L2 files into a single dataset.

    Reads and concatenates multiple GLM L2 files into a single dataset.

    Parameters
    ----------
    filenames : list of str
        List of filenames to read.
    
    Returns
    -------
    combo : xarray.Dataset
        Combined GLM dataset.

    Notes
    -----
    This function requires the `glmtools` package to be installed.
    """
    from glmtools.io.glm import GLMDataset
    scalar_dim = 'segment'
    datasets=[]
    for fn in filenames:
        glm = GLMDataset(fn)
        ds = glm.dataset
        datasets.append(ds)
    combo = concat_1d_dims(datasets, stack_scalars=scalar_dim)

    # most scalar vars are simple metadata, assumed constant across files,
    # so we can just choose one of them. however, for some of them we want
    # to sum instead of select one.
    sum_scalars = ['event_count', 'group_count', 'flash_count']
    avg_scalars = ['percent_navigated_L1b_events', 'percent_uncorrectable_L0_errors']
    all_but_scalars = set(combo.dims)
    all_but_scalars.remove(scalar_dim)
    scalar_dataset = combo.drop_dims(all_but_scalars)
    scalar_vars = list(scalar_dataset.variables.keys())
    for var in scalar_vars:
        if var in sum_scalars:
            calc = combo[var].sum()
        elif var in avg_scalars:
            calc = combo[var].mean()
        else:
            calc = combo[var][0]
        combo[var] = calc

    # dimensions with length 2, that turn into dimensions with size 2*n_files
    # these bounds are assumed sequential / in order so just take the first and last.
    combo = combo[{'number_of_wavelength_bounds':[0, -1],
                   'number_of_time_bounds':[0, -1],
                   'number_of_field_of_view_bounds':[0, -1]}]

    # for some reason, this, in addition to other scalars like event_count
    # are being replicated along the number_of_wavelength_bounds dimension.
    # That doesn't make sense. This is a rough fix for product_time.
#     combo['product_time'] = xr.DataArray(combo.product_time[0], dims=None)
    return combo
