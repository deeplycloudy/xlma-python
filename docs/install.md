## Installation
Clone this repostiory install with pip.

```sh
git clone https://github.com/deeplycloudy/xlma-python.git
cd xlma-python
pip install -e .
```

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