# TumorSpatialSimulations

This script generates spatial coordinates of tumor cells based on evolutionary histories. You need to install msprime to generate a phylogenetic tree for the tumor cells. Change the diffusion constant and image directory in the parameters file to your needs. There are four main methods: regular brownian motion, constrained brownian motion, ornstein-uhlenbeck motion, and brownian motion within a potential.

You can adjust the number of samples, the diffusion rate of the frictional coefficient to reduce the tumor size/test different simulations.You can also adjust the potential/force field. Some changes may cause numerical instability so be sure to tune the parameters appropriately.
