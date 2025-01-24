## Installation

To install the latest "bleeding edge" commit of xlma-python, you can pip install the git repo directly:
```sh
pip install git+https://github.com/deeplycloudy/xlma-python`
```
Note that this does **NOT** automatically update in a conda environment when running `conda update --all`, you must fully reinstall the library to obtain new changes.

For a development install see [contributing](../contributing/).

## Dependencies
Required:

- xarray (I/O requires the netcdf4 backend)
- pandas
- numpy

Flash clustering:

- scikit-learn
- scipy
- pyproj

Plotting:

- matplotlib
- cartopy
- metpy (optionally, for US county lines)

GLM Plotting:

- glmtools (https://github.com/deeplycloudy/glmtools)

Interactive:

- jupyterlab (or, notebook)
- ipywidgets
- ipympl

Building:

- setuptools
- pytest-cov
- pytest-mpl
- lmatools (https://github.com/deeplycloudy/lmatools)
- ...and all of the above