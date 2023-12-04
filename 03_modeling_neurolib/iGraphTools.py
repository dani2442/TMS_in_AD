"""
From Gustavo Patow's WholeBrain 
=============================
MISCELLANOUS HELPER FUNCTIONS
=============================

This module contains miscellaneous helper WholeBrain I have defined for better
or more personalised use of the Python iGraph interface.

I/O AND DATA CONVERSIONS
========================
Array2iGraph
    Converts a 2D numpy array into an iGraph graph object.
"""
from __future__ import division, print_function

__author__ = "Gorka Zamora-Lopez"
__email__ = "Gorka.zamora@ymail.com"
__copyright__ = "Copyright 2015"
__license__ = "GPL"
__update__ = "22/11/2015"

import numpy as np
import igraph as ig

def Array2iGraph(adjmatrix, weighted=False):
    """Converts a 2D numpy array into an iGraph graph object.

    The function automatically detects whether the input network is directed,
    or is undirected but contains asymmetric weights.

    Parameters
    ----------
    adjmatrix : ndarray of rank-2
        The adjacency matrix of the network. Weighted links are ignored.
    weighted : boolean (optional)
        Specifies whether the network is weighted or not.

    Returns
    -------
    iggraph : graph object recognised by iGraph.
        The graph representation of the adjacency matrix with corresponding
        un/directed and/or un/weighted properties.
    """
    # 0) Security check
    assert len(np.shape(adjmatrix)) == 2, \
        'Input array not an adjacency matrix. Array dimension has to be 2.'

    # 1) Find out whether the network is directed or has asymmetric weights
    diff = np.abs(adjmatrix - adjmatrix.T)
    if diff.max() > 10**-6: directed = True
    else: directed = False
    del diff

    # 2) Declare the igraph graph object
    N = len(adjmatrix)
    iggraph = ig.Graph()

    # 2.1) Add the nodes
    iggraph.add_vertices(N)

    # 2.2) Add the links if the network is DIRECTED
    if directed:
        # Make the graph directed
        iggraph.to_directed()

        # Create a list with the links.
        idx = np.where(adjmatrix)
        links = []
        for l in range(len(idx[0])):
            links.append( (idx[0][l],idx[1][l]) )

        # Add the links in the graph.
        iggraph.add_edges(links)

        # If the network is directed, include the weights.
        if weighted:
            values = adjmatrix[idx]
            iggraph.es[:]['weight'] = values

    # 2.2) Add the links if the network is UNDIRECTED
    else:
        # Make a copy of the adjacency matrix to be modified.
        newadjmatrix = adjmatrix.copy()
        idx = np.triu_indices(N,k=1)
        newadjmatrix[idx] = 0

        # Create a list with the links.
        idx = np.where(newadjmatrix)
        links = []
        for l in range(len(idx[0])):
            links.append( (idx[0][l],idx[1][l]) )

        # Add the links in the graph.
        iggraph.add_edges(links)

        # If the network is directed, include the weights.
        if weighted:
            values = newadjmatrix[idx]
            iggraph.es[:]['weight'] = values

        # Clean trash
        del newadjmatrix

    return iggraph