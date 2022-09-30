import numpy as np
import datetime
import xarray as xr
import pandas as pd

from matplotlib.dates import num2date

from pyxlma.plot.xlma_plot_feature import color_by_time, plot_points, setup_hist, plot_3d_grid, subset
from pyxlma.plot.xlma_base_plot import subplot_labels, inset_view, BlankPlot

from ipywidgets import Output
output = Output()


class AccumulatorAxesManager(object):
    def __init__(self, a, ax):
        self.accumulator = a
        self.ax = ax
        f = ax.figure
        f.canvas.mpl_connect('draw_event', a.draw_event)
        f.canvas.mpl_connect('button_release_event', a.mouse_up_event)
        f.canvas.mpl_connect('button_press_event', a.mouse_down_event)
        ax.callbacks.connect('xlim_changed', a.axis_limit_changed)
        ax.callbacks.connect('ylim_changed', a.axis_limit_changed)


class Accumulator(object):
    """ Provides for event callbacks for matplotlib drag/release events and
        axis limit changes by accumulating a series of event occurrences.
        Produces a single call to func after a user interacts with the plot.
        Also stores the axes that got the event, and passes them to func.
        Sample usage:
        from pylab import figure, show
        def simple(axes):
            print "update ", axes
        a = Accumulator(simple)
        f=figure()
        ax=f.add_subplot(111)
        plt=ax.plot(range(10))
        f.canvas.mpl_connect('draw_event', a.draw_event)
        f.canvas.mpl_connect('button_release_event', a.mouse_up_event)
        f.canvas.mpl_connect('button_press_event', a.mouse_down_event)
        ax.callbacks.connect('xlim_changed', a.axis_limit_changed)
        ax.callbacks.connect('ylim_changed', a.axis_limit_changed)
        show()
        """

    def __init__(self, func):
        self.func=func
        self.reset()
        self.mouse_up = True
        # print('did init')

    def reset(self):
        """ Reset flags after the update function is called.
            Mouse is tracked separately.
            """
        # print('reset')
        self.limits_changed = 0
        self.got_draw = False
        self.axes = None

    def axis_limit_changed(self, ax):
        # print('ax limits')
        self.limits_changed += 1
        self.axes = ax
        self.check_status()

    def draw_event(self, event):
        # print('draw event')
        if self.limits_changed > 0:
            # only care about draw events if one of the axis limits has changed
            self.got_draw=True
        self.check_status()

    def mouse_up_event(self, event):
        # print('mouse up')
        self.mouse_up = True
        self.check_status()

    def mouse_down_event(self, event):
        # print('mouse down')
        self.mouse_up = False

    def both_limits_changed(self):
        """ Both x and y limits changed and the mouse is up (not dragging)
            This condition takes care of the limits being reset outside of a
            dragging context, such as the view-reset (home) button on the
            Matplotlib standard toolbar.
            """
        # print("both_lim_chg")
        return (self.limits_changed >= 2) & self.mouse_up

    def interaction_complete(self):
        """ x, y, or both limits changed, and the mouse is up (not dragging).
            Also checks if matplotlib has done its final redraw of the screen,
            which comes after the call to *both* set_xlim and set_ylim
            have been triggered. The check for the draw event is the crucial
            step in not producing two calls to self.func.

            New problem: with zoom, after a reset, get a draw event. on next axes change, the draw
            event  combines with an axis change to trigger interaction_complete. Then get another
            reset and ax limit change, and draw, and this passes again.
            Fixed this by adding a check on self.limits_changed > 0  in draw_event.
        """
        # print("interaction_complete")
        return (self.limits_changed>0) & self.got_draw & self.mouse_up

    def check_status(self):
        if self.both_limits_changed() | self.interaction_complete():
            # print('both limits:', self.both_limits_changed(), ', interaction:', self.interaction_complete())
            self.func(self.axes)
            self.reset()


def event_space_time_limits(ds):
    xlim = ds.event_longitude.min().values, ds.event_longitude.max().values
    ylim = ds.event_latitude.min().values, ds.event_latitude.max().values
    zlim = ds.event_altitude.min().values/1000, ds.event_altitude.max().values/1000
    tlim = (pd.to_datetime(ds.event_time.min().values.astype('M8[us]')).to_pydatetime(),
            pd.to_datetime(ds.event_time.max().values.astype('M8[us]')).to_pydatetime())
    return xlim, ylim, zlim, tlim



class InteractiveLMAPlot(object):

    def __init__(self, ds, chimax=1.0, stationmin=6,
            plot_cmap='plasma', point_size=5, clon=-101.5, clat=33.5,
            xlim=None, ylim=None, zlim=None, tlim=None):
        """ lma_plot is as returned by pyxlma's BlankPlot

            xlim, ylim, zlim, and tlim are initial limits to use with the dataset;
            otherwise, the limits are taken from the limits of the data in ds.
        """
        xlim_ds, ylim_ds, zlim_ds, tlim_ds = event_space_time_limits(ds)

        if xlim is None: xlim = xlim_ds
        if ylim is None: ylim = ylim_ds
        if zlim is None: zlim = zlim_ds
        if tlim is None: tlim = tlim_ds
        # tlim = pd.to_datetime('2022-06-04T22:15'), pd.to_datetime('2022-06-04T22:20')

        self.stationmin = stationmin
        self.chimax = chimax
        self.plot_cmap = plot_cmap
        self.point_size = point_size
        self.clon = clon
        self.clat = clat
        self.widget_output = output


        # list of all artists in the plot that change when the view subset changes
        self.data_artists = []

        self._accumulator = Accumulator(self.limits_changed)
        self._managers = {}
        self.ds = ds

        self.bounds = {
            'x': xlim,
            'y': ylim,
            'z': zlim,
            't': tlim,
        }

        self.lma_plot = None
        self.inset_ax = None
        self.make_plot()
        self.make_plot_interactive()

    @output.capture()
    def make_plot_interactive(self):
            # Make the just-created plot interactive
        self._managers['xy'] = AccumulatorAxesManager(self._accumulator, self.lma_plot.ax_plan)
        self._managers['tz'] = AccumulatorAxesManager(self._accumulator, self.lma_plot.ax_th)
        self._managers['xz'] = AccumulatorAxesManager(self._accumulator, self.lma_plot.ax_lon)
        self._managers['zy'] = AccumulatorAxesManager(self._accumulator, self.lma_plot.ax_lat)


    def _axes_spacetime_limits(self):
        """ Get the current spacetime limits of the axes """
        xlim = self._managers['xy'].ax.get_xlim()
        ylim = self._managers['xy'].ax.get_ylim()
        zlim = self._managers['tz'].ax.get_ylim()
        t0, t1 = list(map(num2date, self.lma_plot.ax_th.get_xlim()))
        tlim = (t0.replace(tzinfo=None), t1.replace(tzinfo=None))
        return xlim, ylim, tlim, zlim

    @output.capture()
    def limits_changed(self, axes):
        """ updates self.bounds from the limits of all axes in the plot making sure to
            include changes from axes
        """

        # When we start out, we get the old axis limits, naively from the xy and tz
        # plots, and then make sure to update the limits for the axis that changed.

        xlim, ylim, tlim, zlim = self._axes_spacetime_limits()

        tag_changed = None
        for tag, mgr in self._managers.items():
            if axes == mgr.ax:
                tag_changed = tag

        if tag_changed == 'xy':
            xlim = self._managers['xy'].ax.get_xlim()
            ylim = self._managers['xy'].ax.get_ylim()
        if tag_changed == 'tz':
            t0, t1 = list(map(num2date, self.lma_plot.ax_th.get_xlim()))
            tlim = (t0.replace(tzinfo=None), t1.replace(tzinfo=None))
            zlim = self._managers['tz'].ax.get_ylim()
        if tag_changed == 'xz':
            xlim = self._managers['xz'].ax.get_xlim()
            zlim = self._managers['xz'].ax.get_ylim()
        if tag_changed == 'zy':
            zlim = self._managers['zy'].ax.get_xlim()
            ylim = self._managers['zy'].ax.get_ylim()

        # When a zoom happens, the plan view plot changes the region covered
        # by the axes (think of the frame) to ensure that the zoomed region
        # maintains a 1:1 aspect ratio in the map projection coordinates.
        # So, we also need to change the axis limits of the other axes by the
        # same amount.

        # Make it so that we can do time math and get total_seconds.
        # t0 = pd.to_datetime(tlim[0].astype('M8[us]')).to_pydatetime()
        # t1 = pd.to_datetime(tlim[1].astype('M8[us]')).to_pydatetime()
        # tlim = (t0, t1)

        self.bounds['x'] = xlim
        self.bounds['y'] = ylim
        self.bounds['z'] = zlim
        self.bounds['t'] = tlim

        self.make_plot()
        self.make_plot_interactive()


    @output.capture()
    def make_plot(self):
        # pull out relevant plot params from object attributes
        ds = self.ds
        plot_s = self.point_size
        plot_cmap = self.plot_cmap
        xlim, ylim = self.bounds['x'], self.bounds['y']
        zlim, tlim = self.bounds['z'], self.bounds['t']
        xchi = self.chimax
        stationmin = self.stationmin

        # Regularize time limits to the right datetime type.
        tlim_sub = pd.to_datetime(tlim[0]), pd.to_datetime(tlim[1])

        alt_data = ds.event_altitude.values/1000.0
        lon_data = ds.event_longitude.values
        lat_data = ds.event_latitude.values
        time_data = pd.Series(ds.event_time) # because time comparisons
        chi_data = ds.event_chi2.values
        station_data = ds.event_stations.values.astype(int)

        tstring = 'LMA {}-{}'.format(tlim_sub[0].strftime('%H%M'),
                                     tlim_sub[1].strftime('%H%M UTC %d %B %Y '))

        lon_set, lat_set, alt_set, time_set, selection = subset(
                   lon_data, lat_data, alt_data, time_data, chi_data, station_data,
                   xlim, ylim, zlim, tlim_sub, xchi, stationmin)

        # if self.lma_plot is not None:
        #     fig = self.lma_plot.fig
        # else:
        #     fig = None
        if self.lma_plot is None:
            self.lma_plot = BlankPlot(pd.to_datetime(tlim_sub[0]), bkgmap=True, #fig=fig,
                      xlim=xlim, ylim=ylim, zlim=zlim, tlim=tlim, title=tstring)
            # Add some subplot labels
            label_art = subplot_labels(self.lma_plot)
        else:
            # clear all the old plots of data
            for a in self.data_artists:
                # print("removing", a)
                a.remove()
            # reset the list of data artists.
            self.data_artists = []

        # Add a view of where the subset is
        # Not plotting this because removing the inset_ax created by this funcion does not
        # clean up the callbacks created by GeoAxes._boundary, causing "None" to be
        # passed as the axes to _trigger_patch_reclip
        # if self.inset_ax is not None:
        #     print(self.inset_ax.callbacks)
        #     self.inset_ax.remove()
        # xdiv = ydiv = 0.1
        # self.inset_ax = inset_view(self.lma_plot, lon_data, lat_data, xlim, ylim, xdiv, ydiv,
        #           buffer=0.5, inset_size=0.15, plot_cmap = 'plasma', bkgmap = True)

        # Add a range ring
        ring_art = self.lma_plot.ax_plan.tissot(rad_km=40.0, lons=self.clon, lats=self.clat, n_samples=80,
                          facecolor='none',edgecolor='k')
        self.data_artists.append(ring_art)
        # Add the station locations
        stn_art = self.lma_plot.ax_plan.plot(ds['station_longitude'],
                                       ds['station_latitude'], 'wD', mec='k', ms=5)
        self.data_artists.extend(stn_art)

        if len(lon_set)==0:
            no_src_art = self.lma_plot.ax_hist.text(0.02,1,'No Sources',fontsize=12)
            self.data_artists.append(no_src_art)
        else:
            plot_vmin, plot_vmax, plot_c = color_by_time(time_set, tlim_sub)
            src_art = plot_points(self.lma_plot, lon_set, lat_set, alt_set, time_set,
                              plot_cmap, plot_s, plot_vmin, plot_vmax, plot_c)
            self.data_artists.extend(src_art)

        # Cartopy's enforcement of a 1:1: aspect ratio results in extent of the
        # axes on the plot changing, causing the vertical projection panes
        # to no longer match. Turn off that feature.
        self.lma_plot.ax_plan.set_aspect('auto')

        # Zoom all the axes to the same limits. Don't emit a callback event,
        # so that we don't have an infinite loop.
        self.lma_plot.ax_lon.axis(xlim+zlim, emit=False)
        self.lma_plot.ax_lat.axis(zlim+ylim, emit=False)
        self.lma_plot.ax_th.axis(tlim+zlim, emit=False)
        # print(self.lma_plot.ax_plan.get_position())
        # print(self.lma_plot.ax_lat.get_position())
        # print(self.lma_plot.ax_lon.get_position())

        # Refresh the ticks from the non-Cartopy plot that knows how to
        # generate ticks. If this goes after the ax_plan.axis() below,
        # it causes the proportion of the Cartopy axes frame
        # to change within the figure.
        self.lma_plot.ax_plan.set_xticks(self.lma_plot.ax_lon.get_xticks())
        self.lma_plot.ax_plan.set_yticks(self.lma_plot.ax_lat.get_yticks())

        self.lma_plot.ax_plan.axis(xlim+ylim, emit=False)




        # Refresh the ticks from the non-Cartopy plot that knows how to
        # generate ticks.
        # self.lma_plot.ax_plan.set_xticks(self.lma_plot.ax_lon.get_xticks())
        # self.lma_plot.ax_plan.set_yticks(self.lma_plot.ax_lat.get_yticks())
        # This line does runs the above two lines, and then a set_extent, and
        # promptly crashes the iPy kernel.
        # self.lma_plot.set_ax_plan_labels()



