# Copyright 2011-2016 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

cimport numpy as np
from cpython cimport array
from .defs cimport gint

cdef struct Edge:
    gint tail, head

cdef class Graph:
    cdef int allow_negative_nodes
    cdef Edge *edges
    cdef gint capacity, size, _num_nodes
    cdef gint num_pp_edges, num_pn_edges, num_np_edges

    cpdef reserve(self, gint capacity)
    cpdef gint add_edge(self, gint tail, gint head) except -1
    cdef _add_edges_ndarray_int64(self, np.ndarray[np.int64_t, ndim=2] edges)
    cdef _add_edges_ndarray_int32(self, np.ndarray[np.int32_t, ndim=2] edges)

cdef class gintArraySlice:
    cdef gint *data
    cdef gint size

cdef class CGraph:
    cdef readonly bint twoway, edge_nr_translation
    cdef readonly gint num_nodes, num_edges, num_px_edges, num_xp_edges
    cdef array.array _heads_idxs
    cdef gint *heads_idxs
    cdef array.array _heads
    cdef gint *heads
    cdef array.array _tails_idxs
    cdef gint *tails_idxs
    cdef array.array _tails
    cdef gint *tails
    cdef array.array _edge_ids
    cdef gint *edge_ids
    cdef array.array _edge_ids_by_edge_nr
    cdef gint *edge_ids_by_edge_nr
    cdef gint edge_nr_end

    cpdef gintArraySlice out_neighbors(self, gint node)


cdef class CGraph_malloc(CGraph):
    pass

cdef class EdgeIterator:
    cdef CGraph graph
    cdef gint edge_id, tail
