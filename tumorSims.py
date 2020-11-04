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
from utils import *

#Parameters to tune
no_samples = 100
DIFF_CONST = 0.01
DRAG = 1 #probably dont mess with this, it causes numerical instability
#You can change both these functions to define any force field on the tumor you want
def fx(x_coord): 
    return -x_coord
def fy(y_coord):
    return -y_coord

#GENERATE PHYLOGENETIC RELATIONSHIP
tree_sequence = msprime.simulate(
     sample_size=no_samples, Ne=1e4, length=5e3, recombination_rate=2e-8,
     mutation_rate=2e-8, random_seed=10)
tree = tree_sequence.first()

def main(): 
	i = getCoordsForceFieldProcess(tree, no_samples)
	hull = ConvexHull(i)
	print(hull.volume)
	plt.scatter(i[:,0], i[:,1])
	tum = range(no_samples)
	for x, y, z in zip(i[:,0], i[:,1], tum):
		plt.annotate(z,(x,y), textcoords="offset points", xytext=(0,10), ha='center')
	for simplex in hull.simplices:
		plt.plot(i[simplex, 0], i[simplex, 1], 'k-')
	plt.show()
if __name__ == "__main__": 
	main()