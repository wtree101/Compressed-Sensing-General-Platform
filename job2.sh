#!/bin/bash

#################### The following is compulsory ####################
#SBATCH -J matlab_test
###Use queue (partition) q1
#SBATCH -p q1
### Use 1 nodes and 8 cores
#SBATCH -N 1 -n 8
#SBATCH --mail-user=twubi@connect.ust.hk
#SBATCH --mail-type=end
#####################################################################

matlab  < Phasediagram.m
