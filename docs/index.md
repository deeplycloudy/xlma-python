<img src="xlma_logo_big_big_ltg.svg" alt="logo" width="200"/>

# xlma-python
---
*A future, Python-based version of XLMA?*

## Motivation

xlma-python is used for analysis and visualization of [Lightning Mapping Array](https://doi.org/10.1029/1999GL010856) (LMA) data.  

XLMA is a venerable IDL GUI program that diplays VHF Lightning Mapping Array data. It was originally written by New Mexico Tech in the late 1990s. This repository represents progress on community-contributed ideas and code to facilitate reimplementing workflows and features from XLMA in Python.

## Features

### Data model and subsetting

xlma-python includes file IO support for the ASCII .dat format used by XLMA and `lma_analysis`. Datasets are read in to the widely-used `xarray` Dataset object. VHF sources and network information are added as multidimensional data, with metadata for each variable. Data can be easily saved to and read from the NetCDF format, allowing for conversion from ASCII to `Dataset` to a NetCDF file.

### Analysis

xlma-python uses a DBSCAN clustering algorithm from scikit-learn to assign groups of VHF sources recorded by the LMA into discrete lightning flashes, as well as tools to compute properties of the flashes. 

Subsetting of all varaibles sharing a dimension and fast indexing are supported. VHF Sources and clustered flashes can be filtered by their properties, and individual flashes/events can be interrogated easily.

### Display

Keeping the core data structure and selection operations separate from dislpay is good programming practice. It is doubly important in Python, where there is not one obvious solution for high performance *and* publication-quality graphics as in IDL.

#### Plotting library

There are many options, so we want a design that:
1. Permits a GUI to provide the bounds of the current view (or a polygon lasso) to the data model, changing the subset
2. Allows many GUIs to read from the data model so that we maintain loose coupling as the GUI and plotting landscape evolves.

- Matplotlib: publication quality figures, mature, and can plot everything, including weather symbols. Perhaps too slow for interactive, colormapped scatterplots of O(1e6) points, as is common for LMA data.
- Vispy: OpenGL, so can display plots quickly, but will require more low-level coding. See the SatPy/PyTroll/SIFT efforts.
- ipyvolume: originally written for volume rendering, but has grown to display points, etc. Browser/notebook based.
- Bokeh/holoviews/geoviews: browser-based; Javascript and WebGL front end with Python backend. Works well with Binder.
- Datashader might be useful as a method of data reduction prior to visualization even if we don't use Bokeh.
- Yt - written by the astronomy community in Python … is it fast enough?

#### GUI

There is no obvious choice here, either.

- Jupyter notebooks
- JupyterLab with custom plugin
- PyQT
- [Glue](https://github.com/glue-viz/glue/wiki/SciPy-2019-Tutorial-on-Multi-dimensional-Linked-Data-Exploration-with-Glue) - seems about 60% there out of the box?! 

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
