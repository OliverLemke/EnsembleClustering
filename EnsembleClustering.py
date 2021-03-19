#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:56:05 2020

@author: Oliver
"""

import pathlib
import os
import numpy as np
from scipy import spatial
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import seaborn as sns
import Color_Mix
from sklearn.cluster import KMeans
import CNN
#import hdbscan

# General function that can use a label vector needed for transferability!!!
## Construct Coclustering Matrix

path = pathlib.Path().absolute()

def Ensemble_kMeans(Data,ks):
    """
    Ensemble Clustering using kMeans.

    Parameters
    ----------
    Data : array
        Data x*y with x being the data points and y the features/dimensions.
    ks : list of int
        Number of clusters k to be assigned.

    Returns
    -------
    coclustering : array
        Array containing the x*x coclustering matrix.
    """

    coclustering = np.zeros((len(Data),len(Data)))
    norm = 0
    for k in ks:
        model = KMeans(k,init="k-means++",n_init=10)
        labels = model.fit(Data).labels_
        Cluster_list = [np.where(labels==label)[0] for label in np.unique(labels)]            
        for cluster in Cluster_list:
            x, y = np.meshgrid(cluster,cluster)
            coclustering[x,y]+=1
        norm +=1
    return coclustering/norm   

# def Ensemble_hDBSCAN(Data,Min_cluster_size,Min_samples):
#     coclustering = np.zeros((len(Data),len(Data)))
#     norm = 0
#     for min_cluster_size in Min_cluster_size:
#         for min_samples in Min_samples:
#             model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,min_samples=min_samples)
#             labels = model.fit(Data).labels_+1
#             Cluster_list = [np.where(labels==label)[0] for label in np.unique(labels)[1:]]            
#             for cluster in Cluster_list:
#                 x, y = np.meshgrid(cluster,cluster)
#                 coclustering[x,y]+=1
#             norm +=1
#     return coclustering/norm   

def Ensemble_CNN(Data, Rs, Ns, M, full_list=False):
    """
    Ensemble Clustering using Common-Nearest-Neighbor algorithm.

    Parameters
    ----------
    Data : array
        Data x*y with x being the data points and y the features/dimensions.
    Rs : list of float
        Distance cut-offs R to be tested.
    Ns : list of int
        Neighbor cut-offs N to be tested.
    M : int
        Minimal number of data points within a cluster.
    full_list : bool, optional
        if True returns clustering results for every parameter set tested. The default is False.

    Returns
    -------
    coclustering : array
        Array containing the x*x coclustering matrix.
    """

    Tree = spatial.cKDTree(Data)
    coclustering = np.zeros((len(Data),len(Data)))
    norm = 0
    if full_list:
        Cluster_list_full={}
        for R in Rs:
            neighborlist, number_neighbors = get_neighborlist_tree(Tree, R)
            for N in Ns:
                nc = np.copy(number_neighbors)
                Cluster_list = CNN.CNN(Data, R, N, M, Ensemble=True, neighborlist=neighborlist, number_neighbors=nc)
                for cluster in Cluster_list:
                    x, y = np.meshgrid(cluster,cluster)
                    coclustering[x,y]+=1
                if Cluster_list:
                    norm +=1
                Cluster_list_full.update({str(R)+"_"+str(N):Cluster_list})
        return coclustering/norm, Cluster_list_full        
    else:
        for R in Rs:
            neighborlist, number_neighbors = get_neighborlist_tree(Tree, R)
            for N in Ns:
                nc = np.copy(number_neighbors)
                Cluster_list = CNN.CNN(Data, R, N, M, Ensemble=True, neighborlist=neighborlist, number_neighbors=nc)
                for cluster in Cluster_list:
                    x, y = np.meshgrid(cluster,cluster)
                    coclustering[x,y]+=1
                if Cluster_list:
                    norm +=1
        return coclustering/norm
    
#def Ensemble_CNN_auto(Data,Rs,Ns,M, Noise_level = 0.7, Cluster_difference = 0.5, full_list=False, output="Clusters.png"):
#    
#    if full_list:
#        Coclustering, Cluster_list_full = Ensemble_CNN(Data, Rs, Ns, M, full_list=full_list)
#    else:
#        Coclustering = Ensemble_CNN(Data, Rs, Ns, M)
#    Map, indices_dens = Cluster_Cooccurence(Coclustering, Noise_level, output=output[:-4]+"_Coclustering.png")
#    Cluster_list = get_clusters_from_map(Map, indices_dens, Cluster_difference, output=output[:-4]+"_Map.png")
#    labels = get_labels(Cluster_list, len(Data))
#    Ensemble_dict={}
#    Ensemble_dict.update({"Cooccurence":Coclustering})
#    Ensemble_dict.update({"Map":Map})
#    Ensemble_dict.update({"Clusters":Cluster_list})
#    Ensemble_dict.update({"Labels":labels})
#    Ensemble_dict.update({"indices_dens":indices_dens})
#    if full_list:
#        Ensemble_dict.update({"Clusters stepwise":Cluster_list_full})
#    return Ensemble_dict

def get_coclustering_from_labels(labels):
    """
    Get the Coclustering matrix for one clustering step.

    Parameters
    ----------
    labels : arr
        labels for every data point.

    Returns
    -------
    Cluster_list : list of arr
        list containg datapoints assigned per cluster as element.
    coclustering : arr
        Quadratic coclustering matrix for one clustering approach.
    """
    
    Cluster_list = [np.where(labels==label)[0] for label in np.unique(labels)]            
    coclustering = np.zeros((len(labels),len(labels)))
    for cluster in Cluster_list:
        x, y = np.meshgrid(cluster,cluster)
        coclustering[x,y]+=1
    return Cluster_list, coclustering

def get_coclustering_from_cluster_list(Cluster_list, num_Datapoints=None):
    """
    Get the Coclustering matrix for one clustering step.

    Parameters
    ----------
    Cluster_list : list of arr
        list containg datapoints assigned per cluster as element.
    num_Datapoints : int, optional
        Number of datapoints (needed if noise is removed during the clustering)

    Returns
    -------
    coclustering : arr
        Quadratic coclustering matrix for one clustering approach.
    """
    
    if not num_Datapoints:
        num_Datapoints = np.max([el for cluster in Cluster_list for el in cluster])+1
    coclustering = np.zeros((num_Datapoints,num_Datapoints))
    for cluster in Cluster_list:
        x, y = np.meshgrid(cluster,cluster)
        coclustering[x,y]+=1
    return coclustering

def get_neighborlist_tree(Tree,R):
    """
    Get the neighborlist using a tree object (REF)

    Parameters
    ----------
    Tree : obj
        Neighbor tree extracted from spatial.cKDTree() implemented in Scipy.
    R : float
        Distance cut-off.

    Returns
    -------
    neighborlist : list of array, optional
        Neighborlist containing neighbors for every data point.
    number_neighbors : array, optional
        Number of neighbors for every data point.
    """
    
    neighborlist = Tree.query_ball_tree(Tree,R)
    number_neighbors = np.asarray([len(item) for item in neighborlist])
    return neighborlist, number_neighbors

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

def Cluster_Cooccurence(Coclustering, Noise_level=0.7, cmap=cm.magma_r, output="Clustering_Coclustering.png"):
    """
    Clustering of the Coclustering matrix using hierarchical Ward clustering.

    Parameters
    ----------
    Coclustering : array
        Coclustering matrix for all data points.
    Noise_level : float, optional
        Minimal percentage a data point have to be assigned to a cluster to be not considered as noise. The default is 0.7.
    cmap : str, optional
        Colormap. The default is cm.magma_r.
    output : str, optional
        Name of the output file. The default is "Clustering_Coclustering.png".

    Returns
    -------
    Map : obj
        Map object returned by seaborn.clustermap.
    indices_dens : list of int
        Indices of data points not apointed to Noise.

    """
    Input_data = Coclustering
    indices_dens = np.where(np.diag(Input_data)>Noise_level)[0]
    x_indices,y_indices = np.meshgrid(indices_dens,indices_dens)
    Input_data_filtered = Input_data[x_indices,y_indices]
    Map = sns.clustermap(Input_data_filtered, method="ward",cmap=cmap)
    Map.savefig(os.path.join(path,output))
    return Map, indices_dens

# def get_clusters_from_map_diff(Map, indices_dens, Cluster_difference=0.5, cmap=cm.magma_r, suffix=""):
#     indices = np.asarray(Map.dendrogram_row.reordered_ind)
#     diagonal_indices = np.arange(1,len(Map.data2d)-1)
#     splits_right = np.where(np.asarray(Map.data2d)[diagonal_indices,diagonal_indices]-np.asarray(Map.data2d)[diagonal_indices,diagonal_indices+1]>Cluster_difference)[0]+1
#     splits_left = np.where(np.asarray(Map.data2d)[diagonal_indices,diagonal_indices]-np.asarray(Map.data2d)[diagonal_indices,diagonal_indices-1]>Cluster_difference)[0]
#     splits = np.unique(np.concatenate((splits_right,splits_left)))
#     splits = np.append(-1,splits)
#     splits = np.append(splits,len(Map.data2d)-1)
    
#     Clusters = [indices[splits[ind-1]+1:splits[ind]+1] for ind in range(1,len(splits))]
#     Cluster_list = [np.asarray([indices_dens[item] for item in cluster]) for cluster in Clusters]
#     plot_Cluster_assignment(Map, Cluster_list, cmap=cmap, suffix=suffix)
#     return Cluster_list

def get_clusters_from_map(Map, indices_dens, cmap=cm.magma_r, output="Coclustering_clustered.png", m=4):
    """
    Get clusters from Map using a cut-off m.

    Parameters
    ----------
    Map : obj
        Map object returned by seaborn.clustermap.
    indices_dens : list of int
        Indices of data points not apointed to Noise.
    cmap : str, optional
        Colormap. The default is cm.magma_r.
    output : str, optional
        Name of the ouput file. The default is "Coclustering_clustered.png".
    m : float, optional
        Parameter for cluster detection. The smaller the more clusters will be extracted. The default is 4.

    Returns
    -------
    Cluster_list : list of array
        Extracted clusters. All indices within a list element belong to the same cluster.

    """
    z = np.mean(Map.dendrogram_col.linkage[:,2])*m
    Assignment = fcluster(Map.dendrogram_col.linkage, z, criterion="distance")
    
    Assignment_reordered = Assignment[Map.dendrogram_row.reordered_ind]
    keys=[]
    for ind in Assignment_reordered: 
        if ind not in keys:
            keys.append(ind)
            
    Assignment_new = np.zeros_like(Assignment_reordered)
    for i,key in enumerate(keys):
        Assignment_new[np.where(Assignment==key)]=i+1
    
    Cluster_list = [indices_dens[np.where(Assignment_new==el)] for el in np.unique(Assignment_new)]
    plot_Cluster_assignment(Map, Cluster_list, cmap=cmap, output=output)
    return Cluster_list


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

def plot_Cluster_assignment(Map, Cluster_list, cmap=cm.magma_r, output="Coclustering_clustered.png"):
    """
    Plot the cocluster cluster assignment.

    Parameters
    ----------
    Map : obj
        Map object returned by seaborn.clustermap.
    Cluster_list : list of array
        Extracted clusters. All indices within a list element belong to the same cluster.
    cmap : str, optional
        Colormap. The default is cm.magma_r.
    output : str, optional
        Name of the output file. The default is "Coclustering_clustered.png".
    """

    fig, ax = plt.subplots()
    g = ax.matshow(Map.data2d, cmap = cmap)
    Shift=0
    for cluster in Cluster_list:
        Shift += len(cluster)
        ax.plot([-.5,len(Map.data2d)-.5],[Shift-0.5,Shift-0.5],c='k',linestyle=':')
        ax.plot([Shift-0.5,Shift-0.5],[-.5,len(Map.data2d)-.5],c='k',linestyle=':')
    ax.set_xlim(-.5,len(Map.data2d)-.5)
    ax.set_ylim(len(Map.data2d)-.5,-.5)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    cbar = fig.colorbar(g)
    cbar.ax.set_ylabel("Coclustering")
    plt.tight_layout()
    plt.savefig(os.path.join(path,output))
    plt.show()
    return

def plot_Coclustering(Coclustering, cmap=cm.magma_r, output="Coclustering.png"):
    """
    Plot Coclustering matrix

    Parameters
    ----------
    Coclustering : array
        Coclustering matrix.
    cmap : str optional
        Colormap. The default is cm.magma_r.
    output : str, optional
        Name of output file. The default is "Coclustering.png".
    """

    fig, ax = plt.subplots()
    g = ax.matshow(Coclustering, cmap = cm.magma_r)
    ax.set_xlim(-.5,len(Coclustering)-.5)
    ax.set_ylim(len(Coclustering)-.5,-.5)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    cbar = fig.colorbar(g)
    cbar.ax.set_ylabel("Coclustering")
    plt.tight_layout()
    plt.savefig(os.path.join(path,output))
    plt.show()
    return

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

#%%
#### Long term:
### Try to get automatic Thresholds by analyzing how many data sets have 1/no cluster, only dependency should be scanned range, only this parameter needed
### PCA instead of hierarchical -> assignment in PCA space
### Auto parameter definition # N with respect to parameter size, # R with respect to Distance-cutoff (percentage)
### Function to analyze outliers with respect to closest cluster
### Join EnsembleClustering algorithms
### Best prediction for m??? look for dynamic cut-off

#### Short term:
### Add prints for console
### Jupyter tutorial