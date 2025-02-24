import numpy as np
from numpy.testing import assert_equal
import xarray as xr

from pyxlma.lmalib.traversal import OneToManyTraversal

def get_four_level_data():
    storm_id = [0,1,2] 
    flash_parent_storm_id = [0,0,0,2,2,2,2,2]
    flash_id =              [1,2,3,4,5,6,7,8]
    stroke_parent_flash_id = [1,1,2,3, 4, 4, 4, 6, 8, 8, 8]
    stroke_id =              [4,6,7,9,13,14,15,19,20,23,46]
    trig_parent_stroke_id = [4,4,4,4,6, 7, 7, 9,13,13,14,14,15,19,20,20,23,46]
    trig_id =               [1,3,5,8,9,10,12,16,18,19,20,22,23,25,26,30,31,32]
    trig_parent_storm_id =  [0,0,0,0,0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    trig_parent_flash_id =  [1,1,1,1,1, 2, 2, 3, 4, 4, 4, 4, 4, 6, 8, 8, 8, 8]

    storm_child_flash_count = [3,0,5]
    flash_child_stroke_count = [2,1,1,3,0,1,0,3]
    stroke_child_trig_count = [4,1,2,1,2,2,1,1,2,1,1]
    
    storm_child_stroke_count = [4,0,7]
    storm_child_trig_count = [8,0,10]
    
    storm_dim = 'nstorms'
    flash_dim = 'nflashes'
    stroke_dim = 'nstrokes'
    trig_dim = 'ntrigs'
    
    d = xr.Dataset({
        'storm_id': (storm_dim, storm_id),
        'flash_id': (flash_dim, flash_id),
        'flash_parent_storm_id': (flash_dim, flash_parent_storm_id),
        'stroke_id': (stroke_dim, stroke_id),
        'stroke_parent_flash_id': (stroke_dim, stroke_parent_flash_id),
        'trig_id': (trig_dim, trig_id),
        'trig_parent_stroke_id': (trig_dim, trig_parent_stroke_id),
        'trig_parent_flash_id': (trig_dim, trig_parent_flash_id),
        'trig_parent_storm_id': (trig_dim, trig_parent_storm_id),
        'storm_child_flash_count': (storm_dim, storm_child_flash_count),
        'storm_child_stroke_count': (storm_dim, storm_child_stroke_count),
        'storm_child_trig_count': (storm_dim, storm_child_trig_count),
        'flash_child_stroke_count': (flash_dim, flash_child_stroke_count),
        'stroke_child_trig_count': (stroke_dim, stroke_child_trig_count),
        })
    d = d.set_coords(['stroke_id', 'flash_id', 'storm_id', 'trig_id', 
                      'stroke_parent_flash_id', 'trig_parent_stroke_id', 
                      'flash_parent_storm_id'])
    assert len(flash_id) == len(flash_parent_storm_id)
    assert len(stroke_id) == len(stroke_parent_flash_id)
    assert len(trig_id) == len(trig_parent_stroke_id)
    assert sum(storm_child_flash_count) == len(flash_id)
    assert sum(storm_child_stroke_count) == len(stroke_id)
    assert sum(storm_child_trig_count) == len(trig_id)
    assert sum(flash_child_stroke_count) == len(stroke_id)
    assert sum(stroke_child_trig_count) == len(trig_id)
    return d

def get_four_level_data_traversal():
    d = get_four_level_data()

    entity_vars = ('storm_id', 'flash_id', 'stroke_id', 'trig_id')
    parent_vars = ('flash_parent_storm_id',
                   'stroke_parent_flash_id',
                   'trig_parent_stroke_id')
    traversal = OneToManyTraversal(d, entity_vars, parent_vars)
    return d, traversal

def get_four_level_empty_data_traversal():
    d = get_four_level_data()
    empty_all = {dim:slice(0,0) for dim in d.dims}
    d = d[empty_all]

    entity_vars = ('storm_id', 'flash_id', 'stroke_id', 'trig_id')
    parent_vars = ('flash_parent_storm_id',
                   'stroke_parent_flash_id',
                   'trig_parent_stroke_id')
    traversal = OneToManyTraversal(d, entity_vars, parent_vars)
    return d, traversal

def test_empty_traversal_init():
    d, traversal = get_four_level_empty_data_traversal()

def test_count_children():
    d, traversal = get_four_level_data_traversal()

    # validation data
    storm_child_count = d['storm_child_flash_count'].data
    flash_child_count = d['flash_child_stroke_count'].data
    stroke_child_count = d['stroke_child_trig_count'].data
    storm_child_stroke_count = d['storm_child_stroke_count'].data
    storm_child_trig_count = d['storm_child_trig_count'].data

    n_storms = traversal.count_children('storm_id')[0]
    assert_equal(storm_child_count, n_storms)
    n_flashes = traversal.count_children('flash_id')[0]
    assert_equal(flash_child_count, n_flashes)
    n_strokes = traversal.count_children('stroke_id')[0]
    assert_equal(stroke_child_count, n_strokes)

    all_counts = traversal.count_children('storm_id', 'trig_id')
    assert_equal(storm_child_count, all_counts[0])
    assert_equal(flash_child_count, all_counts[1])
    assert_equal(stroke_child_count, all_counts[2])

    grouper = d.groupby('trig_parent_storm_id').groups
    count = [len(grouper[eid]) if (eid in grouper) else 0
             for eid in d['storm_id'].data]
    assert_equal(storm_child_trig_count, count)

def test_replicate_parent_ids():
    d, traversal = get_four_level_data_traversal()
    trig_parent_storm_ids = traversal.replicate_parent_ids('storm_id',
                                                    'trig_parent_stroke_id')
    trig_parent_flash_ids = traversal.replicate_parent_ids('flash_id',
                                                    'trig_parent_stroke_id')
    trig_parent_stroke_ids = traversal.replicate_parent_ids('stroke_id',
                                                    'trig_parent_stroke_id')

    assert_equal(d['trig_parent_storm_id'].data, trig_parent_storm_ids)
    assert_equal(d['trig_parent_flash_id'].data, trig_parent_flash_ids)
    assert_equal(d['trig_parent_stroke_id'].data, trig_parent_stroke_ids)


def test_prune_from_middle():
    d, traversal = get_four_level_data_traversal()

    reduced_stroke_id = [13,15,23]
    d = traversal.reduce_to_entities('stroke_id', reduced_stroke_id)
    reduced_storm_id = [2,]
    reduced_flash_id = [4,8]
    reduced_trig_id = [18,19,23,31]
    assert_equal(d['storm_id'].data, reduced_storm_id)
    assert_equal(d['flash_id'].data, reduced_flash_id)
    assert_equal(d['stroke_id'].data, reduced_stroke_id)
    assert_equal(d['trig_id'].data, reduced_trig_id)

def test_prune_from_bottom():
    d, traversal = get_four_level_data_traversal()

    trig_idx = slice(7,10)
    reduced_storm_id = np.unique(d['trig_parent_storm_id'][trig_idx].data)
    reduced_flash_id = np.unique(d['trig_parent_flash_id'][trig_idx].data)
    reduced_stroke_id = np.unique(d['trig_parent_stroke_id'][trig_idx].data)
    reduced_trig_id = d['trig_id'][trig_idx].data
    d = traversal.reduce_to_entities('trig_id', reduced_trig_id)
    assert_equal(d['trig_id'].data, reduced_trig_id)
    assert_equal(d['stroke_id'].data, reduced_stroke_id)
    assert_equal(d['flash_id'].data, reduced_flash_id)
    assert_equal(d['storm_id'].data, reduced_storm_id)

def test_prune_from_top():
    d, traversal = get_four_level_data_traversal()
    reduced_storm_id = [1,]
    d = traversal.reduce_to_entities('storm_id', reduced_storm_id)
    reduced_stroke_id = np.asarray([])
    reduced_flash_id = np.asarray([])
    reduced_trig_id = np.asarray([])
    assert_equal(d['storm_id'], reduced_storm_id)
    assert_equal(d['flash_id'], reduced_flash_id)
    assert_equal(d['stroke_id'], reduced_stroke_id)
    assert_equal(d['trig_id'], reduced_trig_id)

    reduced_storm_id = [2,]
    d = traversal.reduce_to_entities('storm_id', reduced_storm_id)
    reduced_flash_id = [4,5,6,7,8]
    reduced_stroke_id = [13,14,15,19,20,23,46]
    reduced_trig_id = [18,19,20,22,23,25,26,30,31,32]
    assert_equal(d['storm_id'].data, reduced_storm_id)
    assert_equal(d['flash_id'].data, reduced_flash_id)
    assert_equal(d['stroke_id'].data, reduced_stroke_id)
    assert_equal(d['trig_id'].data, reduced_trig_id)