from collections import defaultdict
import xarray as xr
import numpy as np

def get_1d_dims(d):
    """
    Find all dimensions in a dataset that are purely 1-dimensional,
    i.e., those dimensions that are not part of a 2D or higher-D
    variable.
    
    arguments
        d: xarray Dataset
    returns
        dims1d: a list of dimension names
    """
    # Assume all dims coorespond to 1D vars
    dims1d = list(d.dims.keys())
    for varname, var in d.variables.items():
        if len(var.dims) > 1:
            for vardim in var.dims:
                if vardim in dims1d:
                    dims1d.remove(str(vardim))
    return dims1d
    
def gen_1d_datasets(d):
    """
    Generate a sequence of datasets having only those variables
    along each dimension that is only used for 1-dimensional variables.
    
    arguments
        d: xarray Dataset
    returns
        generator function yielding a sequence of single-dimension datasets
    """
    dims1d = get_1d_dims(d)
#     print(dims1d)
    for dim in dims1d:
        all_dims = list(d.dims.keys())
        all_dims.remove(dim)
        yield d.drop_dims(all_dims)

def get_1d_datasets(d):
    """
    Generate a list of datasets having only those variables
    along each dimension that is only used for 1-dimensional variables.
    
    arguments
        d: xarray Dataset
    returns
        a list of single-dimension datasets
    """
    return [d1 for d1 in gen_1d_datasets(d, *args, **kwargs)]

def get_scalar_vars(d):
    scalars = []
    for varname, var in d.variables.items():
        if len(var.dims) == 0:
            scalars.append(varname)
    return scalars

def concat_1d_dims(datasets, stack_scalars=None):
    """
    For each xarray Dataset in datasets, concatenate (preserving the order of datasets)
    all variables along dimensions that are only used for one-dimensional variables.
    
    arguments
        d: iterable of xarray Datasets
        stack_scalars: create a new dimension named with this value
            that aggregates all scalar variables and coordinates
    returns
        a new xarray Dataset with only the single-dimension variables
    """
    # dictionary mapping dimension names to a list of all
    # datasets having only that dimension
    all_1d_datasets = defaultdict(list)
    
    for d in datasets:
        scalars = get_scalar_vars(d)
        for d_1d_initial in gen_1d_datasets(d):
            # Get rid of scalars
            d_1d = d_1d_initial.drop(scalars)
            dims = tuple(d_1d.dims.keys())
            all_1d_datasets[dims[0]].append(d_1d)
        if stack_scalars:
            # restore scalars along new dimension stack_scalars
            scalar_dataset = xr.Dataset()
            for scalar_var in scalars:
                # promote from scalar to an array with a dimension, and remove
                # the coordinate info so that it's just a regular variable.
                as_1d = d[scalar_var].expand_dims(stack_scalars).reset_coords(drop=True)
                scalar_dataset[scalar_var] = as_1d # xr.DataArray(as_1d, dims=[stack_scalars])
            all_1d_datasets[stack_scalars].append(scalar_dataset)
    
    unified = xr.Dataset()
    for dim in all_1d_datasets:
        combined = xr.concat(all_1d_datasets[dim], dim, coords='minimal', data_vars='minimal')
        unified.update(combined)
    return unified

# datasets=[]
# for i, size in enumerate((4, 6)):
#     a = xr.DataArray(10*i + np.arange(size), dims='x')
#     b = xr.DataArray(10*i + np.arange(size/2), dims='y')
#     c = xr.DataArray(20*i + np.arange(size*3), dims='t')
#     d = xr.DataArray(11*i + np.arange(size*3), dims='t')
#     T = xr.DataArray(10*i + np.arange(size)**2, dims='x')
#     D = xr.DataArray(10*i + np.arange(size/2)**2, dims='y')
#     z = xr.DataArray(10*i + np.arange(size*4)**2, dims='z')
#     u = xr.DataArray(10*i + np.arange(size*5)**2, dims='u')
#     v = xr.DataArray(12*i + np.arange(size*5)**2, dims='u')
#     P = xr.DataArray(10*i + np.ones((size,int(size/2))), dims=['x', 'y'])
#     Q = xr.DataArray(20*i + np.ones((size,int(size/2))), dims=['x', 'y'])
#     d = xr.Dataset({'x':a,'y':b, 't':c, 'd':d, 'u':u, 'v':v, 'z':z, 'T':T, 'D':D, 'P':P, 'Q':Q})
#     datasets.append(d)
# #     datasets.append(d[{'x':slice(None, None), 'y':slice(0,0)}])
# for d in datasets: print(d,'\n')
# # xr.combine_by_coords(datasets,  coords='all')
# # xr.combine_nested(datasets, coords='all', data_vars='all')

# # print(get_1d_dims(d))
# assert(get_1d_dims(d)==['t', 'u', 'z'])
# # for d1 in get_1d_datasets(d):
# #     print(d1,'\n')
    
# combined = concat_1d_dims(datasets)
# print(combined)