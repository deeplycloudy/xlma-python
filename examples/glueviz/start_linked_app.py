# script to launch glue programmatically
# From https://github.com/glue-viz/glue/wiki/SciPy-2019-Tutorial-on-Multi-dimensional-Linked-Data-Exploration-with-Glue#scripts
from glue.core import DataCollection
from glue.core.link_helpers import LinkSame
from glue.core.data_factories import load_data
from glue.app.qt.application import GlueApplication
from glue.viewers.scatter.qt.data_viewer import ScatterViewer
# from glue.viewers.image.qt.data_viewer import ImageViewer
from glue.external.echo import keep_in_sync

import config # from config import read_lma_data

lma_filename = '/data/GOES16oklmaMCS/LMA_171022/OK_LMA_2002_2005_171022_051000_0600.dat.gz'

lma = load_data(lma_filename)
dc = DataCollection([lma])

# dc.add_link(LinkSame(image.id['Right Ascension'], catalog.id['RAJ2000']))
# dc.add_link(LinkSame(image.id['Declination'], catalog.id['DEJ2000']))

app = GlueApplication(dc)

# image_viewer = app.new_data_viewer(ImageViewer)
# image_viewer.add_data(image)
# image_viewer.add_data(catalog)
# image_viewer.viewer_size = (500, 500)
# image_viewer.state.layers[0].percentile = 99

view_names = ('xy', 'xz', 'zy', 'tz')
sizes = ((500, 500), (500, 250), (250, 500), (500, 250))
posns = ((0, 500), (0, 250), (500, 500), (0, 0))
thevars = (('lon', 'lat'), ('lon', 'alt(m)'), ('alt(m)', 'lat'), ('time (UT sec of day)', 'alt(m)'))

views = {}
for view, size, posn, axvars in zip(view_names, sizes, posns, thevars):
    new_viewer = app.new_data_viewer(ScatterViewer)
    new_viewer.add_data(lma)
    new_viewer.viewer_size = size
    new_viewer.position = posn
    new_viewer.state.x_att = lma.id[axvars[0]]
    new_viewer.state.y_att = lma.id[axvars[1]]
    views[view] = new_viewer

# Create six-panel viewer manually, then this links limits
# viewer1 = application.viewers[0][0]
# viewer2 = application.viewers[0][1]
# Do this for all pairs, limits
# keep_in_sync(viewer1.state, 'x_min', viewer2.state, 'x_min')

def link_x(view1, view2):
    sync_x0 = keep_in_sync(view1.state, 'x_min', view2.state, 'x_min')
    sync_x1 = keep_in_sync(view1.state, 'x_max', view2.state, 'x_max')
    return sync_x0, sync_x1
def link_y(view1, view2):
    sync_y0 = keep_in_sync(view1.state, 'y_min', view2.state, 'y_min')
    sync_y1 = keep_in_sync(view1.state, 'y_max', view2.state, 'y_max')
    return sync_y0, sync_y1
def link_x_to_y(view1, view2):
    sync_xy0 = keep_in_sync(view1.state, 'x_min', view2.state, 'y_min')
    sync_xy1 = keep_in_sync(view1.state, 'x_max', view2.state, 'y_max')
    return sync_xy0, sync_xy1
def link_y_to_x(view1, view2):
    ync_yx0 = keep_in_sync(view1.state, 'y_min', view2.state, 'x_min')
    ync_yx1 = keep_in_sync(view1.state, 'y_max', view2.state, 'x_max')
    return sync_yx0, sync_yx1

# To keep sync active, need to hold a reference to the link sync objects.
# Probably should move this into a _pyxlma_state dict so we only add one thing
# to the app.
app._x_sync = link_x(views['xy'], views['xz'])
app._y_sync = link_y(views['xy'], views['zy'])
app._zt_sync = link_y(views['xz'], views['tz'])
app._zy_sync = link_x_to_y(views['zy'], views['tz'])

# Because of sync this updates all axes. Doesn't work with interactive zoom!?
views['tz'].state.y_min = 0.0
views['tz'].state.y_max = 20000.0


# programmatically create a subset

# def subset_from_axes_limits(subset):
    
# selection = (data.id['time'] > 6) & (data.id['time'] < 18)

# all_points = (lma.id['time'] > 0)
# dc.new_subset_group('Current selection', all_points)

# Show IPython console
# Show how to access the data collection
# data_collection or dc
# Show how to access a single dataset
# data = dc[0]
# Show information about data
# print(data)
# Show how to access a subset
# data.subsets[0].to_mask()
# Define new subset programmatically containing all daytime flights
# selection = (data.id['time'] > 6) & (data.id['time'] < 18)
# dc.new_subset_group('Daytime flights', selection)
# Define new subset programmatically containing the plane with the most points
# from collections import Counter
# count = Counter(data['callsign'])
# count.most_common(5)
# selection = data.id['callsign'] == 'N56418'
# data_collection.new_subset_group('N56418', selection)
# Access viewer state and modify it, e.g. changing what is shown on axes, or toggling options on/off
# viewer = application.viewers[0][0]
# viewer.state
# viewer.state.x_att = data.id['heading']


app.start()




# Can also sync x_att to keep the variable plotted on the axes in sync if
# desired, though I don't think we every want to let the user change that.

# viewer1.state
# Out[6]:
# <ScatterViewerState
#   aspect: auto
#   dpi: 71.30820648960312
#   layers: <CallbackList with 1 elements>
#   show_axes: True
#   x_att: lon
#   x_axislabel: lon
#   x_axislabel_size: 10
#   x_axislabel_weight: normal
#   x_log: False
#   x_max: -94.28925283925686
#   x_min: -98.07825293748292
#   x_ticklabel_size: 8
#   y_att: lat
#   y_axislabel: lat
#   y_axislabel_size: 10
#   y_axislabel_weight: normal
#   y_log: False
#   y_max: 37.31313050338403
#   y_min: 31.426154379432987
#   y_ticklabel_size: 8
# >

# Add a callback that triggers on xmin, xmax, ymin, ymax to update a selection 
# viewer1.state.add_callback

# Will want to hide the original dataset and only update the selection

# Load data, create selection object, hide orig data, create 

# histogram of chi and stations and graphically select those!

# Animation would need to update selection on basis of a clock. Examples?

# glue-geospatial has coord transforms
# https://github.com/glue-viz/glue-geospatial/tree/master/glue_geospatial