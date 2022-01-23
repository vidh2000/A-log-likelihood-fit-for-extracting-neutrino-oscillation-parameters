import numpy as np 
from functions import *
import numpy as np
import matplotlib.pyplot as plt
import os 
import random as rand
from copy import *
from functions import *
from minimisers import *
from differentiation import *

# Loading data files
current_path = os.path.dirname(__file__)
resources = os.path.join(current_path, "data")
events = np.loadtxt("data/events.txt")
events_simul = np.loadtxt("data/unoscillated_flux.txt")
E_bins = np.arange(0.025,200*0.05+0.025,0.05)


##############################################
##############################################
orig_theta_orig = np.pi/4
orig_dm2_orig = 1e-3
L=295
##############################################
##############################################

params = [events,events_simul,E_bins,L]


# 1D
fact=np.array([np.pi/4,1])
vec = np.array([0.9224302261560001,2.4])#2.172835587202903]
print("1D Minimum:", vec*fact)
sig = [0.01,0.1]
pos=[0]
eps=1e-15
pm_ers = pm_error_finder(nll,params,vec,sig,pos,eps)
pm_ers = pm_ers[0]
print("1D +- errors in theta:",pm_ers*(np.pi/4))
print("1D NLL:",nll(params,vec))
print("\n")

# 2d 
vec = np.array([0.9999999978307968,2.1728347070840557])
print("2D Minimum:", vec*fact)
sig = [0.01,0.1]
pos=[0,1]
eps=1e-15
pm_ers = pm_error_finder(nll,params,vec,sig,pos,eps)
print("pmers",pm_ers)
print("2D +- errors:",pm_ers[0]*(np.pi/4), pm_ers[1])
print("2D NLL:",nll(params,vec))
print("\n")

#3d
fact=np.array([np.pi/4,1,1])
vec=np.array([0.9498679648714824, 2.174719676542579, 0.987174041433529])
print("3D Minimum1:", vec*fact)
sig = [0.05,0.05,0.02]
pos=[0,1,2]
eps=1e-15
pm_ers = pm_error_finder(nll_cs,params,vec,sig,pos,eps)
print("pmers",pm_ers)
print("3D +- errors:",pm_ers[0]*(np.pi/4), pm_ers[1],pm_ers[2])
print("3D NLL:",nll_cs(params,vec))
print("\n")

fact=np.array([np.pi/4,1,1])
vec=np.array([1.0501320351744492, 2.174719676563728, 0.9871740415989932])
print("3D Minimum2:", vec*fact)
sig = [0.05,0.05,0.02]
pos=[0,1,2]
eps=1e-15
pm_ers = pm_error_finder(nll_cs,params,vec,sig,pos,eps)
print("pmers",pm_ers)
print("3D +- errors:",pm_ers[0]*(np.pi/4), pm_ers[1],pm_ers[2])
print("3D NLL:",nll_cs(params,vec))
print("\n")
