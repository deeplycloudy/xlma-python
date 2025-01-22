import collections, itertools
import numpy as np

# """ Code from glmtools, where there are also unit tests for this class.
#     TODO: port to work as an xarray accessor? And move unit tests here. Adapt
#         to automatically use cf-tree metadata.
# """

class OneToManyTraversal(object):
    """A class to allow traversal of a dataset where data variables have a one-to-many relationships.
        This object creates groupbys stored by dictionaries that make it
        convenient to look up the groupby given either the p

        Dimensions
        child_dim       grandchild_dim          greatgrandchild_dim

        entity_groups: a groupby for each of these variables, indexed by same
        child_id        grandchild_id           greatgrandchild_id

        parent_groups: a groupby for each of these variables, indexed by same
        None            parent_id_of_grandchild parent_id_of_greatgrandchild

        or, given parent id variable, get the groupby for the parent entity
        *along the same dimension*
        entity_groups_by_parent
        (e.g., parent_id_of_grandchild -> grandchild_id )

        or, given entity id variable, get the groupby for the parent
        *along the same dimension*
        parent_groups_by_entity
        (e.g., grandchild_id -> parent_id_of_grandchild)

        One will note that the hierarchy encoded by this group is a sort
        of virtual dimension that links set membership instead of coordinates.
        A future NetCDF spec could formalize this relationship structure
        to encourage the same library-level functionality as this class.
    """
    def __init__(self, dataset, entity_id_vars, parent_id_vars):
        """Initialize a OneToManyTraversal object.

        Parameters
        ----------
        dataset : xarray.Dataset
            The dataset to be traversed.
        entity_id_vars : iterable of str
            The names of the N variables to be traversed, in order from grandparent -> parent -> child -> grandchild -> ...
            Variables must be unique along the dimension of the variable.
        parent_id_vars : iterable of str
            The names of the N-1 variables that link the entities in entity_id_vars, in order from (grandparent_id_of_parent) -> (parent_id_of_child) -> (child_id_of_grandchild) -> ...

        """
        n_entities = len(entity_id_vars)
        n_parent_keys = len(parent_id_vars)
        if (n_parent_keys != (n_entities-1)):
            err_msg = "{0} entities may only have {1} parent-child keys"
            raise ValueError(err_msg.format(n_entities, n_entities-1))

        self.entity_id_vars = tuple(entity_id_vars)
        # There is no parent for the first entity.
        self.parent_id_vars = (None,) + tuple(parent_id_vars)

        # These groupby objects index the *dimension* crresponding to the
        # Group the ID of each dimension that describes the coordinates.
        self.entity_groups = collections.OrderedDict()
        # Group the parent-child keys.
        self.parent_groups = collections.OrderedDict()
        # Make it easy to look up the parent variable given the child
        #
        self.parent_groups_by_entity = collections.OrderedDict()
        self.entity_groups_by_parent = collections.OrderedDict()
        self.child_to_parent = collections.OrderedDict()
        self.parent_to_child = collections.OrderedDict()
        for (entity_var, parent_var) in self._descend():
            if dataset.sizes[dataset[entity_var].dims[0]] == 0:
                # No data, so groupby will fail in xarray > 0.13
                entity_grouper = None
            else:
                entity_grouper = dataset.groupby(entity_var)
            self.entity_groups[entity_var] = entity_grouper
            if parent_var is None:
                parent_grouper = None
            else:
                if dataset.sizes[dataset[parent_var].dims[0]] == 0:
                    # No data, so groupby will fail in xarray > 0.13
                    parent_grouper = None
                else:
                    parent_grouper = dataset.groupby(parent_var)
            self.parent_groups[parent_var] = parent_grouper
            self.child_to_parent[entity_var] = parent_var
            self.parent_to_child[parent_var] = entity_var
            self.parent_groups_by_entity[entity_var] = parent_grouper
            self.entity_groups_by_parent[parent_var] = entity_grouper
        self.dataset = dataset

    def _descend(self):
        """ Iterate from parent to children"""
        yield from zip(self.entity_id_vars, self.parent_id_vars)

    def _ascend(self):
        """ Iterate from children to parents"""
        yield from zip(self.entity_id_vars[::-1], self.parent_id_vars[::-1])

    def count_children(self, entity_id_var, child_entity_id_var=None):
        """Count the children of entity_id_var.

        Optionally, accumulate counts of children down to and including
        the level of child_entity_id_var. These are the counts from parent
        to immediate child.

        If replicate_parent_ids has been used to create 'bottom_parent_top_id',
        where the top and bottom are separated by a few levels, it is possible
        to get an aggregate count (matching the top parent dimension) of the
        children many generations below:
        ```py
            grouper = dataset.groupby('bottom_parent_top_id').groups
            count = [len(grouper[eid]) if (eid in grouper) else 0
                   for eid in d['top_id'].data]
            assert_equal(storm_child_trig_count, count)
        ```
        
        Parameters
        ----------
        entity_id_var : str
            The name of the variable to count children of.
        child_entity_id_var : str, optional
            The name of the lowest variable in the hierarchy to count children of.
            If None, only the immediate children are counted.
        
        Returns
        -------
        all_counts : tuple of numpy.ndarray
            A tuple of arrays, where each array is the count of children at the
            corresponding level in the hierarchy.
        """
        count_next = False
        all_counts = []
        for e_var, p_var in self._descend():
            if count_next == True:
                grouper = self.parent_groups[p_var].groups
                count = [len(grouper[eid]) if (eid in grouper) else 0
                         for eid in last_entity_ids]
                last_entity_ids = self.dataset[e_var].data
                all_counts.append(np.asarray(count))
            if (child_entity_id_var == None) | (e_var == child_entity_id_var):
                count_next = False
            if e_var == entity_id_var:
                count_next = True
                last_entity_ids = self.dataset[e_var].data
        return all_counts

    def replicate_parent_ids(self, entity_id_var, parent_id_var):
        """Replicate the IDs of the ancestors at the level of a child entity.

        If given a mapping of child->parent, this function can find the grandparents, great-grandparents, etc.

        Parameters
        ----------
        entity_id_var : str
            The name of the ancestor entity to find. Must be a variable originally specificied to `entity_id_vars` the class initialization.
        parent_id_var : str
            The name of initial the child->parent relationship to start ascending through the heirarchy with.
            Must be a variable originally specified to the `parent_id_vars` and must be lower than the value specified to `entity_id_var`

        Returns
        -------
        last_replicated_p_ids : numpy.ndarray
            The replicated IDs of the ancestors (at the level of `entity_id_var`) replicated to the level of the child in the child->parent
            relationship specified to `parent_id_var` 

        Notes
        -----
        This function is not strictly needed for queries up one level
        (where parent_id_var can be used directly), but is needed to ascend
        the hierarchy by two or more levels.

        Once the parent entity_ids have been replicated, they can be used
        with xarray's indexing functions to replicate any other variable
        along that parent entity's dimension.
        """
        # TODO: args should really be:
        # replicate_to_dim
        # replicate_var -> replicate_from_dim

        # Work from bottom up.
        # First, wait until the parent_id_var is reached
        # then use the groupby corresponding to the next level up
        # in the next time through the loop.
        collect = False
        dataset = self.dataset
        for e_var, p_var in self._ascend():
            if e_var == entity_id_var:
                # stop collecting if we reach the level of entity_id_var
                # We never do anything at the level of the final entity,
                # because the parent_ids collected from the level
                # below are sufficent to represent the final level
                collect = False
            if collect == True:
                # get the parent_id_var one level down
                grouper = self.entity_groups[e_var]
                e_idx = [grouper.groups[eid] for eid in last_replicated_p_ids]
                e_idx_flat = np.asarray(e_idx).flatten()

                # Need to index the whole dataset because the group_by indexes
                # indexes the whole dataset
                if len(e_idx_flat) == 0:
                    # xarray doesn't accept an empty array as a valid index
                    e_idx_flat = slice(0, 0)
                last_replicated_e_ids = dataset[e_var][e_idx_flat]
                last_replicated_p_ids = last_replicated_e_ids[p_var].data
            if p_var == parent_id_var:
                # We've reached the parent_id_var level, collect ids
                # from group at next level
                collect = True
                last_replicated_p_ids = dataset[p_var].data
        return last_replicated_p_ids
            # last_e_var, last_p_var = e_var, p_var

    def reduce_to_entities(self, entity_id_var, entity_ids):
        """Reduce the dataset to the ancestors and descendents of the given entity_ids that are within the given entity_id_var.

        Finds all ancestors and descendents of the specified IDs on the specified variable, and returns a filtered dataset containing only the requested
        entity IDs and their ancestors and children.

        Parameters
        ----------
        entity_id_var : str
            The variable to find the ancestors and descendants of.

        entity_ids : array_like
            The IDs to filter the dataset by.
        
        Returns
        -------
        dataset : xarray.Dataset
            original dataset filtered to the requested entity_ids along the entity_id_var and all ancestors and descendants.
        """
        entity_ids = np.asarray(entity_ids)

        dataset = self.dataset

        # The list of indices returned by the group dictionary correspond to
        # the indices automatically generated by xarray for each dimension,
        # and don't necessarily correspond to the values in the entity_id
        # variables.

        # First we descend through the dataset in order to prune out children
        # that whose parents are not in entity_ids
        prune = False
        for e_var, p_var in self._descend():
            if prune == True:
                p_group = self.parent_groups[p_var].groups
                e_iter = (np.atleast_1d(p_group[eid]) for eid in last_entity_ids
                          if eid in p_group)
                e_idx = list(itertools.chain.from_iterable(e_iter))
                if len(e_idx) == 0:
                    # xarray doesn't accept an empty array as a valid index
                    e_idx = slice(0, 0)
                indexer[dataset[e_var].dims[0]]=e_idx
                dataset = dataset[indexer]
                last_entity_ids = dataset[e_var].data

            # reset the indexer
            indexer = {}
            if e_var == entity_id_var:
                # start pruning once we reach the level of entity_id_var
                # At that level, we index on the entity itself, not the parent
                # index like all subsequent dimensions. This generates
                # the need to index along two dimensions at once the first time
                # through.
                prune = True
                e_group = self.entity_groups[e_var].groups
                e_iter = (np.atleast_1d(e_group[eid]) for eid in entity_ids
                          if eid in e_group)
                e_idx = list(itertools.chain.from_iterable(e_iter))
                last_entity_ids = entity_ids # == dataset[e_var].data
                if len(e_idx) == 0:
                    # xarray doesn't accept an empty array as a valid index
                    e_idx = slice(0, 0)
                # Interesting: what if we have a multidimensional
                # hierarchical cluster id? Can't index by dims[0] alone.
                indexer[dataset[e_var].dims[0]] = e_idx
        # Reset pruning now that the bottom has been reached
        prune = False

        # if prune first at the bottom of the hierarchy, the prune block
        # above doesn't run, so we need to use the indexer that was created.
        if len(indexer.keys()) > 0:
            dataset = dataset[indexer]

        # Now we ascend through the dataset in order to prune out parents
        # above the level of entity_id_var who do not contain at least one
        # child at the level of entity_id_var
        for e_var, p_var in self._ascend():
            if (prune == True):
                e_group = self.entity_groups[e_var].groups
                e_iter = (np.atleast_1d(e_group[eid]) for eid in last_entity_ids
                          if eid in e_group)
                e_idx = list(itertools.chain.from_iterable(e_iter))
                if len(e_idx) == 0:
                    # xarray doesn't accept an empty array as a valid index
                    e_idx = slice(0, 0)
                indexer[dataset[e_var].dims[0]]=e_idx
                dataset = dataset[indexer]
            if (p_var is not None):
                last_entity_ids = np.unique(dataset[p_var].data)
            else:
                # we've reached the top
                prune = False

            indexer = {}
            if (e_var == entity_id_var) & (p_var is not None):
                # start pruning once we reach the level of entity_id_var
                # but don't do anything until we ascend to the next level.
                prune = True
                last_entity_ids = np.unique(dataset[p_var].data)

        return dataset
