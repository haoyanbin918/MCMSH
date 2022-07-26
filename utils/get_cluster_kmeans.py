import random
import h5py
import numpy as np 
from collections import defaultdict
from args import *

#This file is to generate the cluster set by using K-means.

def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    NB. points can have more dimensions than 2
    
    Returns a new point which is the center of all the points.
    """
    dimensions = len(points[0])
    new_centers = np.mean(points,1)

    return new_centers


def update_centers(data_set, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes 
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers where `k` is the number of unique assignments.
    """
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)
    for points in new_means.values():
        centers.append(np.mean(points,0))

    return centers


def assign_points(data_points, centers):
    """
    Given a data set and a list of points betweeen other points,
    assign each point to an index that corresponds to the index
    of the center point on it's proximity to that point. 
    Return a an array of indexes of centers that correspond to
    an index in the data set; that is, if there are N points
    in `data_set` the list we return will have N elements. Also
    If there are Y points in `centers` there will be Y unique
    possible values within the returned list.
    """
    assignments = []
    have = []
    for point in data_points:
        shortest = 9999999 # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        if shortest_index not in have:
            have.append(shortest_index)
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    """
    dis = np.sum((a-b)**2)
    return dis


def generate_k(data_set, k):
    """
    Given `data_set`, which is an array of arrays,
    find the minimum and maximum for each coordinate, a range.
    Generate `k` random points between the ranges.
    Return an array of the random points within the ranges.
    """
    centers = []
    num_sample = len(data_set)-1
    random_index = [random.randint(0,num_sample) for _ in range(k)]
    

    for _k in range(k):

        centers.append(data_set[random_index[_k]])

    return centers,random_index


def k_means(dataset, k):
    k_points,random_index = generate_k(dataset, k)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    itera = 0
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
        print(itera)
        itera = itera + 1
        if itera>15:
            break
    return new_centers


with h5py.File(train_feat_path,'r') as h5_file: 
    video_feats = h5_file['feats'][:] 
video_feats=np.mean(video_feats,1)

clusters = k_means(video_feats, 2000)
h5 = h5py.File(home_root+'/data/cluster.h5', 'w')
h5.create_dataset('feats',data = clusters)
h5.close()
print("get cluster done")