import pytest
import xarray as xr
from pyxlma.plot.xlma_base_plot import *
from pyxlma.plot.xlma_plot_feature import *
from pyxlma.lmalib.grid import *
import datetime as dt
import pandas as pd
import matplotlib.dates as md

@pytest.mark.mpl_image_compare
def test_blank_plot():
    start_time = dt.datetime(2023, 12, 24, 0, 57, 0, 0)
    end_time = start_time + dt.timedelta(seconds=60)
    bk_plot = BlankPlot(start_time, bkgmap=True, xlim=[-103.5, -99.5], ylim=[31.5, 35.5], zlim=[0, 20], tlim=[start_time, end_time], title='XLMA Test Plot')
    return bk_plot.fig

@pytest.mark.mpl_image_compare
def test_blank_plot_labeled():
    start_time = dt.datetime(2023, 12, 24, 0, 57, 0, 0)
    end_time = start_time + dt.timedelta(seconds=60)
    bk_plot = BlankPlot(start_time, bkgmap=True, xlim=[-103.5, -99.5], ylim=[31.5, 35.5], zlim=[0, 20], tlim=[start_time, end_time], title='XLMA Test Plot')
    subplot_labels(bk_plot)
    return bk_plot.fig

@pytest.mark.mpl_image_compare
def test_plot_feature_plot_points_positional():
    start_time = dt.datetime(2023, 12, 24, 0, 57, 0, 0)
    end_time = start_time + dt.timedelta(seconds=60)
    bk_plot = BlankPlot(start_time, bkgmap=True, xlim=[-103.5, -99.5], ylim=[31.5, 35.5], zlim=[0, 20], tlim=[start_time, end_time], title='XLMA Test Plot')
    dataset = xr.open_dataset('tests/truth/lma_netcdf/lma.nc')
    times = pd.Series(dataset.event_time.data.flatten())
    vmin, vmax, colors = color_by_time(pd.to_datetime(times), (start_time, end_time))
    plot_points(bk_plot, dataset.event_longitude.data, dataset.event_latitude.data, dataset.event_altitude.data/1000,
                dataset.event_time.data, 'rainbow', 5, vmin, vmax, colors, 'k', 0.1, True)
    return bk_plot.fig

@pytest.mark.mpl_image_compare
def test_plot_feature_plot_points_old_kw():
    start_time = dt.datetime(2023, 12, 24, 0, 57, 0, 0)
    end_time = start_time + dt.timedelta(seconds=60)
    bk_plot = BlankPlot(start_time, bkgmap=True, xlim=[-103.5, -99.5], ylim=[31.5, 35.5], zlim=[0, 20], tlim=[start_time, end_time], title='XLMA Test Plot')
    dataset = xr.open_dataset('tests/truth/lma_netcdf/lma.nc')
    times = pd.Series(dataset.event_time.data.flatten())
    vmin, vmax, colors = color_by_time(pd.to_datetime(times), (start_time, end_time))
    plot_points(bk_plot, dataset.event_longitude.data, dataset.event_latitude.data, dataset.event_altitude.data/1000,
                dataset.event_time.data, plot_cmap='rainbow', plot_s=5, plot_vmin=vmin, plot_vmax=vmax,
                plot_c=colors, edge_color='k', edge_width=0.1, add_to_histogram=True)
    return bk_plot.fig

@pytest.mark.mpl_image_compare
def test_plot_feature_plot_points_new_kw():
    start_time = dt.datetime(2023, 12, 24, 0, 57, 0, 0)
    end_time = start_time + dt.timedelta(seconds=60)
    bk_plot = BlankPlot(start_time, bkgmap=True, xlim=[-103.5, -99.5], ylim=[31.5, 35.5], zlim=[0, 20], tlim=[start_time, end_time], title='XLMA Test Plot')
    dataset = xr.open_dataset('tests/truth/lma_netcdf/lma.nc')
    times = pd.Series(dataset.event_time.data.flatten())
    vmin, vmax, colors = color_by_time(pd.to_datetime(times), (start_time, end_time))
    plot_points(bk_plot, dataset.event_longitude.data, dataset.event_latitude.data, dataset.event_altitude.data/1000,
                dataset.event_time.data, cmap='rainbow', s=5, vmin=vmin, vmax=vmax,
                c=colors, edgecolors='k', linewidths=0.1, add_to_histogram=True)
    return bk_plot.fig


@pytest.mark.mpl_image_compare
def test_plot_feature_plot_points_new_kw_no_bkmap():
    start_time = dt.datetime(2023, 12, 24, 0, 57, 0, 0)
    end_time = start_time + dt.timedelta(seconds=60)
    bk_plot = BlankPlot(start_time, bkgmap=False, xlim=[-103.5, -99.5], ylim=[31.5, 35.5], zlim=[0, 20], tlim=[start_time, end_time], title='XLMA Test Plot')
    dataset = xr.open_dataset('tests/truth/lma_netcdf/lma.nc')
    times = pd.Series(dataset.event_time.data.flatten())
    vmin, vmax, colors = color_by_time(pd.to_datetime(times), (start_time, end_time))
    plot_points(bk_plot, dataset.event_longitude.data, dataset.event_latitude.data, dataset.event_altitude.data/1000,
                dataset.event_time.data, cmap='rainbow', s=5, vmin=vmin, vmax=vmax,
                c=colors, edgecolors='k', linewidths=0.1, add_to_histogram=True)
    return bk_plot.fig


@pytest.mark.mpl_image_compare
def test_plot_feature_plot_3d_grid():
    dataset = xr.open_dataset('tests/truth/lma_netcdf/lma.nc')
    x_edges = np.linspace(-103.5, -99.5, 100)
    y_edges = np.linspace(31.5, 35.5, 100)
    z_edges = np.linspace(0, 20, 100)
    t_edges = md.date2num(pd.date_range(start='2023-12-24T00:57:00', end='2023-12-24T00:58:00', periods=7).values)
    histograms = setup_hist(dataset.event_longitude.data, dataset.event_latitude.data, dataset.event_altitude.data/1000,
                            dataset.event_time.data, x_edges, y_edges, z_edges,  t_edges)
    start_time = dt.datetime(2023, 12, 24, 0, 57, 0, 0)
    end_time = start_time + dt.timedelta(seconds=60)
    bk_plot = BlankPlot(start_time, bkgmap=True, xlim=[-103.5, -99.5], ylim=[31.5, 35.5], zlim=[0, 20], tlim=[start_time, end_time], title='XLMA Test Plot')

    plot_3d_grid(bk_plot, x_edges, y_edges, z_edges, t_edges, *histograms, dataset.event_altitude.data/1000, cmap='plasma')
    return bk_plot.fig

@pytest.mark.mpl_image_compare
def test_plot_feature_plot_3d_grid_old_kw():
    dataset = xr.open_dataset('tests/truth/lma_netcdf/lma.nc')
    x_edges = np.linspace(-103.5, -99.5, 100)
    y_edges = np.linspace(31.5, 35.5, 100)
    z_edges = np.linspace(0, 20, 100)
    t_edges = md.date2num(pd.date_range(start='2023-12-24T00:57:00', end='2023-12-24T00:58:00', periods=7).values)
    histograms = setup_hist(dataset.event_longitude.data, dataset.event_latitude.data, dataset.event_altitude.data/1000,
                            dataset.event_time.data, x_edges, y_edges, z_edges,  t_edges)
    start_time = dt.datetime(2023, 12, 24, 0, 57, 0, 0)
    end_time = start_time + dt.timedelta(seconds=60)
    bk_plot = BlankPlot(start_time, bkgmap=True, xlim=[-103.5, -99.5], ylim=[31.5, 35.5], zlim=[0, 20], tlim=[start_time, end_time], title='XLMA Test Plot')

    plot_3d_grid(bk_plot, x_edges, y_edges, z_edges, t_edges, *histograms, dataset.event_altitude.data/1000, plot_cmap='plasma')
    return bk_plot.fig


@pytest.mark.mpl_image_compare
def test_plot_feature_inset_view():
    start_time = dt.datetime(2023, 12, 24, 0, 57, 0, 0)
    end_time = start_time + dt.timedelta(seconds=60)
    bk_plot = BlankPlot(start_time, bkgmap=True, xlim=[-103.5, -99.5], ylim=[31.5, 35.5], zlim=[0, 20], tlim=[start_time, end_time], title='XLMA Test Plot')
    dataset = xr.open_dataset('tests/truth/lma_netcdf/lma.nc')
    times = pd.Series(dataset.event_time.data.flatten())
    vmin, vmax, colors = color_by_time(pd.to_datetime(times), (start_time, end_time))
    plot_points(bk_plot, dataset.event_longitude.data, dataset.event_latitude.data, dataset.event_altitude.data/1000,
                dataset.event_time.data, cmap='rainbow', s=5, vmin=vmin, vmax=vmax,
                c=colors, edgecolors='k', linewidths=0.1, add_to_histogram=True)
    inset_view(bk_plot, dataset.event_longitude.data, dataset.event_latitude.data,
               [-102.75, -102.25], [32, 32.5], .01, .01)
    return bk_plot.fig