import msprime  # >= 0.5
import numpy as np
from math import sqrt
from scipy.stats import norm
import scipy
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import seaborn as sns
import tskit
import pandas as pd
import math
import multiprocessing
import matplotlib.pyplot as plt
from IPython.display import SVG

#Parameters to tune
DIFF_CONST = 0.01
DRAG = 1 #probably dont mess with this, it causes numerical instability
#You can change both these functions to define any force field on the tumor you want
def fx(x_coord): 
    return -x_coord
def fy(y_coord):
    return -y_coord


def brownian(start_pt, time, delta):
    x0 = start_pt
    r = norm.rvs(size=(2,), scale=delta*sqrt(time))
    return np.array(x0 + r)

    
def returnOUcoords(start_pt, time,drag, diff_const):
    sigma = time  # Standard deviation.
    mu = np.array(0)  # Mean.
    theta = drag
    division_factor = 10**(orderOfMagnitude(time) +2)
    D = diff_const #diffusion constant
    dt = time/division_factor  # Time step.
    T = time  # Total time.
    n = int(T / dt)  # Number of time steps.
    t = np.linspace(0., T, n)  # Vector of times.
    sigma_bis = np.sqrt(2 * D)
    sqrtdt = np.sqrt(dt)
    v = np.zeros((n,2))
    x = np.zeros((n,2))
    x[0] = start_pt
    v[0] = np.array([0,0])
    for i in range(n - 1):
        v[i + 1,0] = v[i,0] + theta*dt*(-(v[i,0] - mu)) + sigma_bis * sqrtdt * np.random.randn()
        v[i + 1,1] = v[i,1] + theta*dt*(-(v[i,1] - mu)) + sigma_bis * sqrtdt * np.random.randn()
        x[i+1, 0] = x[i, 0] + v[i, 0]*dt
        x[i+1, 1] = x[i, 1] + v[i,1]*dt
    return x[n-1]
def returnForceFieldcoords(start_pt, time,drag, diff_const):
    sigma = time  # Standard deviation.
    mu = np.array(0)  # Mean.
    theta = drag #drag coefficient
    D = diff_const #diffusion constant
    dt = 1 # Time step.
    T = time  # Total time.
    n = int(T / dt)  # Number of time steps.
    t = np.linspace(0., T, n)  # Vector of times.
    sigma_bis = np.sqrt(2 * D)
    sqrtdt = np.sqrt(dt)
    v = np.zeros((n,2))
    x = np.zeros((n,2))
    x[0] = start_pt
    v[0] = np.array([0,0])
    for i in range(n - 1):
        v[i + 1,0] = v[i,0] + fx(x[i,0]) + theta*dt*(-(v[i,0])) + sigma_bis * sqrtdt * np.random.randn()
        v[i + 1,1] = v[i,1] + fy(x[i,1]) + theta*dt*(-(v[i,1])) + sigma_bis * sqrtdt * np.random.randn()
        x[i+1, 0] = x[i, 0] + v[i,0]*dt
        x[i+1, 1] = x[i, 1] + v[i,1]*dt
    return x[n-1]
def getCoordsFromPhylogeny(tree, no_samples): 
    list_of_paths = []
    for u in range(no_samples):
        path = []
        while u != tskit.NULL:
            path.insert(0, u)
            u = tree.parent(u)
        list_of_paths.append(path)
    list_of_times = []
    for l in list_of_paths: 
        times = []
        for j in l[1:]: 
            times.append(tree.branch_length(j))
        list_of_times.append(times)
    largest_node = list_of_paths[0][0]
    coords = np.zeros((largest_node+1, 2))
    for path, time in zip(list_of_paths, list_of_times): 
        for i in range(len(path) -1): 
            if(np.array_equal(coords[path[i+1],:], np.zeros((2,)))):
                coords[path[i+1], :] = brownian(coords[path[i], :], time[i], 1)
    return coords[:no_samples, :]
#Assume tumor is constrained to a circle of area A
def getCoordsFromPhylogenyFixedArea(tree, no_samples, constrain_area): 
    Area_tumor = math.inf
    scaling_factor = 1
    while(Area_tumor > constrain_area): 
        list_of_paths = []
        for u in range(no_samples):
            path = []
            while u != tskit.NULL:
                path.insert(0, u)
                u = tree.parent(u)
            list_of_paths.append(path)
        list_of_times = []
        for l in list_of_paths: 
            times = []
            for j in l[1:]: 
                times.append(tree.branch_length(j)/scaling_factor)
            list_of_times.append(times)
        largest_node = list_of_paths[0][0]
        coords = np.zeros((largest_node+1, 2))
        for path, time in zip(list_of_paths, list_of_times): 
            for i in range(len(path) -1): 
                if(np.array_equal(coords[path[i+1],:], np.zeros((2,)))):
                    coords[path[i+1], :] = brownian(coords[path[i], :], time[i], 1)
        final_samples = coords[:no_samples, :]
        hull = ConvexHull(final_samples)
        Area_tumor = hull.volume
        scaling_factor +=1 
    return final_samples
def getCoordsOUProcess(tree, no_samples): 
    list_of_paths = []
    for u in range(no_samples):
        path = []
        while u != tskit.NULL:
            path.insert(0, u)
            u = tree.parent(u)
        list_of_paths.append(path)
    list_of_times = []
    for l in list_of_paths: 
        times = []
        for j in l[1:]: 
            times.append(tree.branch_length(j))
        list_of_times.append(times)
    largest_node = list_of_paths[0][0]
    coords = np.zeros((largest_node+1, 2))
    
    for path, time in zip(list_of_paths, list_of_times): 
        for i in range(len(path) -1): 
            if(np.array_equal(coords[path[i+1],:], np.zeros((2,)))):
                coords[path[i+1], :] = returnOUcoords(coords[path[i], :], time[i],DRAG, DIFF_CONST)
    return coords[:no_samples, :]
def getCoordsForceFieldProcess(tree, no_samples): 
    list_of_paths = []
    for u in range(no_samples):
        path = []
        while u != tskit.NULL:
            path.insert(0, u)
            u = tree.parent(u)
        list_of_paths.append(path)
    list_of_times = []
    for l in list_of_paths: 
        times = []
        for j in l[1:]: 
            times.append(tree.branch_length(j))
        list_of_times.append(times)
    largest_node = list_of_paths[0][0]
    coords = np.zeros((largest_node+1, 2))
    
    for path, time in zip(list_of_paths, list_of_times): 
        for i in range(len(path) -1): 
            if(np.array_equal(coords[path[i+1],:], np.zeros((2,)))):
                coords[path[i+1], :] = returnForceFieldcoords(coords[path[i], :], time[i],DRAG, DIFF_CONST)
    return coords[:no_samples, :]