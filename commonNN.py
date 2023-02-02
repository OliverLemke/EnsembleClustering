#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:31:25 2020

@author: Oliver
"""

import pathlib
import os
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import matplotlib as mpl
import Color_Mix


path = pathlib.Path().absolute()

def commonNN(Data,R,N,M,Ensemble=False,neighborlist=None,number_neighbors=None):
    """
    Common-Nearest-Neighbor Clustering

    Parameters
    ----------
    Data : array
        Data x*y with x being the data points and y the features/dimensions.
    R : float
        Distance cut-off.
    N : int
        Neighbor cut-off.
    M : int
        Minimal number of data points within a cluster.
    Ensemble : bool, optional
        If used for EnsembleClustering. If True, neighborlist and number_neighbors have to be provided. The default is False.
    neighborlist : list of array, optional
        Neighborlist containing neighbors for every data point. The default is None.
    number_neighbors : array, optional
        Number of neighbors for every data point. The default is None.

    Returns
    -------
    Cluster_list_filter : list of array
        Extracted clusters for the given input parameters.
    """
    
    if Ensemble:
        pass
    else:
        neighborlist, number_neighbors = get_neighborlist(Data, R)
    number_neighbors = noise_filter(number_neighbors, M)
    Cluster_list = clustering(neighborlist,number_neighbors, N)
    Cluster_list_filter = [cluster for cluster in Cluster_list if len(cluster)>=M]
    return Cluster_list_filter

def get_neighborlist(Data,R):
    """
    Get neighborlist for every data point.

    Parameters
    ----------
    Data : array
        Data x*y with x being the data points and y the features/dimensions.
    R : float
        Distance cut-off.

    Returns
    -------
    neighborlist : list of array, optional
        Neighborlist containing neighbors for every data point.
    number_neighbors : array, optional
        Number of neighbors for every data point.

    """
    Tree = spatial.cKDTree(Data)
    neighborlist = Tree.query_ball_tree(Tree,R)
    number_neighbors = np.asarray([len(item) for item in neighborlist])
    return neighborlist, number_neighbors

def noise_filter(number_neighbors,M):
    """
    Apply a Noise filter filtering out all data points with less than M neighbors.

    Parameters
    ----------
    number_neighbors : array, optional
        Number of neighbors for every data point.
    M : int
        Minimal number of data points within a cluster.

    Returns
    -------
    number_neighbors : array, optional
        Number of neighbors for every data point with at least M neighbors.

    """
    number_neighbors[number_neighbors<M]=0
    return number_neighbors

def cluster_assignment(neighborlist,number_neighbors, N):
    """
    Assign data points for a single cluster (Cluster initialisaton and expansion for data point with most neighbors)

    Parameters
    ----------
    neighborlist : list of array, optional
        Neighborlist containing neighbors for every data point.
    number_neighbors : array, optional
        Number of neighbors for every data point.
    N : int
        Neighbor cut-off.

    Returns
    -------
    current_cluster : list of ind
        Assigned data points for one cluster.

    """
    current_cluster = [np.where(number_neighbors==np.max(number_neighbors))[0][0]]
    [current_cluster.append(neighbor) for current in current_cluster 
                      for neighbor in neighborlist[current] 
                      if neighbor not in current_cluster
                      if len(np.intersect1d(neighborlist[current],neighborlist[neighbor])) >= N]    
    return current_cluster

def clustering(neighborlist,number_neighbors, N):
    """
    Clustering of the data points

    Parameters
    ----------
    neighborlist : list of array, optional
        Neighborlist containing neighbors for every data point.
    number_neighbors : array, optional
        Number of neighbors for every data point.
    N : int
        Neighbor cut-off.

    Returns
    -------
    Cluster_list : list of arrays
        Extracted clusters. All indices within a list element belong to the same cluster.

    """
    Cluster_list=[]
    while np.sum(number_neighbors!=0):
        Cluster = cluster_assignment(neighborlist,number_neighbors, N)
        number_neighbors[Cluster]=0
        Cluster_list.append(Cluster)
    return Cluster_list

def get_distance_matrix(Data):
    """
    Distance matrix for Clustering

    Parameters
    ----------
    Data : array
        Data x*y with x being the data points and y the features/dimensions.

    Returns
    -------
    dist : array
        x*x Distance matrix.

    """
    
    dist = spatial.distance_matrix(Data,Data)
    return dist

def plot_2d(Data, labels, Dim = (0,1), axislabels = None, output="Clusters.png"):
    """
    Plot 2D representation of cluster assignment

    Parameters
    ----------
    Data : array
        Data x*y with x being the data points and y the features/dimensions.
    labels : array
        Array containing labels for every data point. 0 corresponds to Noise.
    Dim : tuple, optional
        Dimensions to be plotted. The default is (0,1).
    axislabels : list of str, optional
        axislabels. The default is None.
    output : str, optional
        Name of the output file. The default is "Clusters.png".
    """
    
    fig,ax = plt.subplots()
    if 0 in labels:
        ax.scatter(Data[:,Dim[0]], Data[:,Dim[1]], c = labels, cmap = mpl.colors.ListedColormap(Color_Mix.colors(len(np.unique(labels)))))
    else:
        ax.scatter(Data[:,Dim[0]], Data[:,Dim[1]], c = labels, cmap = mpl.colors.ListedColormap(Color_Mix.colors_noNoise(len(np.unique(labels)))))
    if axislabels:
        ax.set_xlabel(axislabels[0])
        ax.set_ylabel(axislabels[1])
    else:
        ax.set_xlabel("Dim "+str(Dim[0]))
        ax.set_ylabel("Dim "+str(Dim[1]))
    plt.tight_layout()
    plt.savefig(os.path.join(path,output))
    plt.show()
    return

def plot_hist(Data,nbins=50,output="Hist.png"):
    """
    Plot distance distribution

    Parameters
    ----------
    Data : array
        Data x*y with x being the data points and y the features/dimensions.
    nbins : int, optional
        Number of bins. The default is 50.
    output : str, optional
        Name of the output file. The default is "Hist.png".
    """
    
    hist = np.histogram(get_distance_matrix(Data),bins=nbins)
    fig,ax = plt.subplots()
    ax.plot((hist[1][:-1]+hist[1][1:])/2,hist[0])
    ax.fill_between((hist[1][:-1]+hist[1][1:])/2,np.zeros(len(hist[0])),hist[0],alpha=0.5)
    ax.set_xlim(0,(hist[1][-1]+hist[1][-2])/2)
    ax.set_ylim(0,np.max(hist[0]*1.05))
    ax.set_xlabel("Distance")
    ax.set_ylabel("Counts")
    plt.tight_layout()
    plt.savefig(os.path.join(path,output))
    plt.show()
    return

def get_labels(Cluster_list, num_datapoints):
    """
    Obtain an array with labels for every data point. 0 corresponds to Noise

    Parameters
    ----------
    Cluster_list : list of array
        Extracted clusters. All indices within a list element belong to the same cluster.
    num_datapoints : int
        Number of data points.

    Returns
    -------
    labels : array
        Array containing labels for every data point. 0 corresponds to Noise.
    """
    
    labels = np.zeros(num_datapoints)
    for label,cluster in enumerate(Cluster_list):
        labels[cluster]=label+1
    return labels

def get_outliers(Data, Cluster_list):
    """
    Get a list of outliers

    Parameters
    ----------
    Data : array
        Data x*y with x being the data points and y the features/dimensions.
    Cluster_list : list of array
        Extracted clusters. All indices within a list element belong to the same cluster.

    Returns
    -------
    Outliers : list of int
        List of outliers.
    """
    
    assigned_datapoints = [el for cluster in Cluster_list for el in cluster]
    all_datapoints = np.arange(len(Data))
    Outliers = [item for item in all_datapoints if item not in assigned_datapoints]
    return Outliers