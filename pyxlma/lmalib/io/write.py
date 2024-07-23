import numpy as np

def dataset(dataset, path):
    "Write an LMA dataset to a netCDF file"

    if 'flash_event_count' in dataset.data_vars:
        if np.all(dataset.flash_event_count == np.iinfo(np.uint32).max):
            dataset = dataset.drop_vars(['flash_init_latitude', 'flash_init_longitude', 'flash_init_altitude', 'flash_area', 'flash_volume', 'flash_energy', 'flash_center_latitude', 'flash_center_longitude', 'flash_center_altitude', 'flash_power', 'flash_event_count', 'flash_duration_threshold', 'flash_time_start', 'flash_time_end', 'flash_duration'])
    
    for var in dataset.data_vars:
        if np.all(dataset[var].data == np.nan):
            dataset = dataset.drop_vars(var)
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in dataset.data_vars}
    dataset = dataset.chunk('auto')
    dataset.to_netcdf(path, encoding=encoding)