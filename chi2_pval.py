import numpy as np
import matplotlib.pyplot as plt
import os 
import random as rand
from copy import *
from functions import *
from minimisers import *

# Plotting parameters

plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {'text.usetex' : True,
          'font.size' : 16,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          'axes.labelsize':16,
          'legend.fontsize': 14,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16,
          'figure.figsize': [7.5,7.5/1.2],
                     }
plt.rcParams.update(params)

# Loading data files
current_path = os.path.dirname(__file__)
resources = os.path.join(current_path, "data")
events = np.loadtxt("data/events.txt")
events_simul = np.loadtxt("data/unoscillated_flux.txt")
E_bins = np.arange(0.025,200*0.05+0.025,0.05)

##############################################
##############################################
theta = 1
dm2 = 2.4
L=295
##############################################
##############################################


osc_rate_predict1 = lambda_i(events_simul,theta,dm2,L,E=E_bins,arr=True)
#1D

#2D
osc_rate_predict2 = lambda_i(events_simul,theta,2.172835587202903,L,E=E_bins,arr=True)
#plt.show()

# 3D
# 1.053244542298085, 2.122983095864706, 1.0571209432185358
params = [events,events_simul,E_bins,L]
osc_rate_predict3 = lambda_i_new(events_simul,0.9498679648714824, 2.174719676542579,L,E=E_bins,alpha=0.987174041433529,arr=True)
vec = [1,2.1690530534811217,0.9936495523131138]
#print("Newton min NLL =",nll_cs(params,vec))
#osc_rate_predict3 = lambda_i_new(events_simul,1.053244542298085,2.122983095864706,L,E=E_bins,alpha=1.0571209432185358,arr=True)
#vec = [1.053244542298085,2.122983095864706,1.0571209432185358]
#print("MC min NLL =",nll_cs(params,vec))

rchi1D = reduced_chi2(events,osc_rate_predict1,2,0)
rchi2D = reduced_chi2(events,osc_rate_predict2,2,0)
rchi3D = reduced_chi2(events,osc_rate_predict3,3,0)

print("REDUCED CHI^2")
print("Reduced Chi^2 value for 1D (pi/4, 2.4e-3):", rchi1D)
print("Reduced Chi^2 value for 2D (pi/4, 2.1728e-3):", rchi2D)
print("Reduced Chi^2 value for 3D (pi/4, 2.1690530534811217e-3,0.9936495523131138): chi^2_red =",rchi3D,"\n")

rchi1Dc = reduced_chi2(events,osc_rate_predict1,2,1)
rchi2Dc = reduced_chi2(events,osc_rate_predict2,2,1)
rchi3Dc = reduced_chi2(events,osc_rate_predict3,3,1)

print("REDUCED CHI^2 - William's correction")
print("Reduced Chi^2 value for 1D (pi/4, 2.4e-3):", rchi1Dc)
print("Reduced Chi^2 value for 2D (pi/4, 2.1728e-3):", rchi2Dc)
print("Reduced Chi^2 value for 3D (pi/4, 2.1690530534811217e-3,0.9936495523131138): chi^2_red =",rchi3Dc,"\n")


dof1d,dof2d,dof3d = 198,198,197
#print(rchi1D*dof1d,rchi2D*dof2d,rchi3D*dof3d)
eps=1e-6
p1D = p_value(rchi1D,dof1d,eps,0)#p_value(rchi1D,dof1d,eps)
p2D = p_value(rchi2D,dof2d,eps,0)
p3D = p_value(rchi3D,dof3d,eps,0) #CHI2 from chi^2=dof or?
print("\nP-VALUE")
print("Part 3 p-val=",p1D)
print("Part 4 p-val=",p2D)
print("Part 5 p-val=",p3D)
#print(chi2_CDF(rchi3D,dof3d,eps))

print("P-VALUE William's correction")
p1D = p_value(rchi1D,dof1d,eps,1)
p2D = p_value(rchi2D,dof2d,eps,1)
p3D = p_value(rchi3D,dof3d,eps,1)
print("Part 3 p-val=",p1D)
print("Part 4 p-val=",p2D)
print("Part 5 p-val=",p3D)







