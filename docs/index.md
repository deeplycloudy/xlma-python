<img src="xlma_logo.svg" alt="logo" width="200"/>

# xlma-python
---
*A future, Python-based version of XLMA?*

## Motivation

xlma-python is used for analysis and visualization of [Lightning Mapping Array](https://doi.org/10.1029/1999GL010856) (LMA) data.  

XLMA is a venerable IDL GUI program that diplays VHF Lightning Mapping Array data. It was originally written by New Mexico Tech in the late 1990s. This repository represents progress on community-contributed ideas and code to facilitate reimplementing workflows and features from XLMA in Python.

## Features

### Data model and subsetting

xlma-python includes file IO support for the ASCII .dat format used by XLMA and `lma_analysis`. Datasets are read in to the widely-used [xarray.Dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html#xarray.Dataset) object. VHF sources and network information are added as multidimensional data, with metadata for each variable. Data can be easily saved to and read from the NetCDF format, allowing for conversion from ASCII to `xarray.Dataset` to a NetCDF file.

### Analysis

xlma-python uses a DBSCAN clustering algorithm from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) to assign groups of VHF sources recorded by the LMA into discrete lightning flashes, as well as tools to compute properties of the flashes. 

Subsetting of all varaibles sharing a dimension and fast indexing are supported. VHF Sources and clustered flashes can be filtered by their properties, and individual flashes/events can be interrogated easily.

### Display

Publication quality plots in the style of XLMA (with a plan view, latitude-altitude and longitude-altitude cross sections, altitude-histogram, and altitude-timeseries plots) are supported. These plots are generated with high level functions, using the popular [matplotlib](https://matplotlib.org/stable/index.html) plotting library.

### Interactive

There are several ongoing, parallel efforts to create an interactive analysis/display tool in python, using different GUI tools, each with their own pros/cons:

- This repository contains an "interactive" module which utilizes matplotlib/jupyter/ipywidgets for interactivity. This was easy to implement with the already existing matplotlib plotting tools, and allows for "good enough" performance in jupyter notebooks, which are widely used in geosciences.

- An older example using [Glue](https://github.com/glue-viz/glue) is also included in the examples directory.

- Other options (for external projects):
    - Vispy: OpenGL, so can display plots quickly, but will require more low-level coding. See the SatPy/PyTroll/SIFT efforts.
    - ipyvolume: originally written for volume rendering, but has grown to display points, etc. Browser/notebook based.
    - HoloViz Suite (holoviews/geoviews plotting libraries with bokeh backend, panel for dashboard layout): browser-based; Javascript and WebGL front end with Python backend.
        - HoloViz's Datashader might be useful as a method of data reduction prior to visualization even if we don't use HoloViews/Bokeh.
    - Yt: written by the astronomy community in Python ... is it fast enough?
    - lonboard: python binding for deck.gl, very fast vector rendering of many points, but limited interactive tools and ONLY allows rendering geospatial data (ie, not cross sections or histograms)
        - pydeck: lower-level deck.gl python binding which removes the geospatial data requirement but requires more work on the developer's side. 

## Prior art

- [`lmatools`](https://github.com/deeplycloudy/lmatools/)
    - Includes readers for LMA and NLDN data (using older methods from 2010)
    - Flash sorting and gridding
    - Has code for all necessary coordinate transforms.
- [`brawl4d`](https://github.com/deeplycloudy/brawl4d/) A working version of the basic GUI functionality of xlma.
    - Based on matplotlib; plots can be drag-repositioned. Slow for large numbers of data points.
    - Includes charge analyis that auto-saves to disk
    - At one point, could display radar data underneath LMA data
    - Built around a data pipeline, with a pool of data at the start, subsetting, projection, and finally display.
