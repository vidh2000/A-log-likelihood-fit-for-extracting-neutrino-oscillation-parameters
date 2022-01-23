import numpy as np
import matplotlib.pyplot as plt
import os 
import random as rand
from copy import *
from functions import *
from minimisers import *
from differentiation import *
# Plotting parameters

plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {'text.usetex' : True,
          'font.size' : 26,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          'axes.labelsize':20,
          'legend.fontsize': 15,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
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
orig_theta_orig = np.pi/4
orig_dm2_orig = 1e-3
L=295
##############################################
##############################################



####################################################################################
####################################################################################
"""
4.) 2-DIMENSIONAL MINIMISATION
"""
####################################################################################
####################################################################################
params = [events,events_simul,E_bins,L]
ht =  0.0005608878686509
hm =  0.0005608878686509
order = 4
dvec = [ht,hm]

"""
4.1) The univariate method
"""

# dm_arr = [1e-5,3.5e-3]
# theta_arr = [0.1,0.78]
# vec = [0.75,2.1e-3]
dm_arr = [2.0,2.3]
theta_arr = [0.1,2]
vec = [0.99,2.174]

results, vec_u = univariate_method(nll,[theta_arr,dm_arr],params,[0.9,2.15],dvec,4,1e-10)
theta,dm2 = results[0],results[1]
#theta = np.array(theta)#*orig_theta_orig
#dm2 = np.array(dm2)#*orig_dm2_orig
t0,dm2_0 = theta,dm2#[0]
#tunc,munc = theta[1],dm2[1]

"""
4.2) Simultaneous minimisation
"""

############ plot a contour plot of NLL(theta,dm2) ################

theta_arr = np.linspace(0,2,50)
dm2_arr = np.linspace(0,10,50)
NLL = NLL_mesh(theta_arr,dm2_arr,params)

title="3_NLL_2Dmesh"
plt.figure(title)
h = plt.contourf(theta_arr, dm2_arr*1e-3, NLL,alpha=1.0,cmap="nipy_spectral",levels=50)
plt.xlabel(r"$\theta_{23} \: [\pi/4]$")
plt.ylabel(r"$\Delta m^2_{23}$ [eV$^2$]")
clb = plt.colorbar()
clb.set_label(r'$NLL(\theta_{23},\Delta m^2_{23})$')
#plt.savefig("plots/4/"+title+".pdf",
           #dpi=1200, 
           #bbox_inches="tight")

################################################


results,vec_arr = newton_minimiser(nll,params,vec,dvec,order=order,eps=1e-10,N=1e3)
theta, dm2 = results[0], results[1]
theta = np.array(theta)
dm2 = np.array(dm2)
results_qn1, vec_arr_qn1 = quasi_newton_minimiser(nll,params,vec,dvec,#alph=1e-2
                   method="BFGS",order=order,alph=5e-2,alpha_factor=1e-2,eps=1e-10,N=1e3)
results_qn2, vec_arr_qn2 = quasi_newton_minimiser(nll,params,vec,dvec,
                   method="DFP",order=order,alph=5e-2,alpha_factor=1e-2,eps=1e-10,N=1e3)

results_g, vec_g = gradient_method(nll,params,vec,dvec,order=order,alph=1e-3,eps=1e-10,N=1e3)
theta_g, dm2_g = results_g[0],results_g[1]


print("\n 4.1) Univariate minimisation. Parameters that minimise NLL:")
print(f"theta = {t0} +- {0},\ndm2 = {dm2_0} +- {0}")

print("\n4.2) Newton method. Parameters that minimise NLL:")
print(f"Theta = {theta[0]} +- {theta[1]}")
print(f"dm2 = {dm2[0]} +- {dm2[1]}")

print("\n4.3) BFGS Quasi-Newton method. Parameters that minimise NLL:")
print(f"Theta = {results_qn1[0]}")
print(f"dm2 = {results_qn1[1]}")
print("\n4.3) DFP Quasi-Newton method. Parameters that minimise NLL:")
print(f"Theta = {results_qn2[0]}")
print(f"dm2 = {results_qn2[1]}")

print("\n4.4) Gradient descent method. Parameters that minimise NLL:")
print(f"Theta = {theta_g[0]} +- {theta_g[1]}")
print(f"dm2 = {dm2_g[0]} +- {dm2_g[1]}")

#print("\nDifference between univariate and simultaneous minimisation:")
#print(f"For theta = {abs(t0-theta[0])}")
#print(f"For dm2 = {abs(dm2_0-dm2[0])} \n")

# Univariate
xu = np.array([vec[0] for vec in np.array(vec_u)])*orig_theta_orig
yu = np.array([vec[1] for vec in np.array(vec_u)])
#Newton
xn = np.array([vec[0] for vec in np.array(vec_arr)])*orig_theta_orig
yn = np.array([vec[1] for vec in np.array(vec_arr)])
#Quasi newton
# BFGS
x_qn1 = np.array([vec[0] for vec in vec_arr_qn1])*orig_theta_orig
y_qn1 = np.array([vec[1] for vec in vec_arr_qn1])
# DFP
x_qn2 = np.array([vec[0] for vec in vec_arr_qn2])*orig_theta_orig
y_qn2 = np.array([vec[1] for vec in vec_arr_qn2])
# Gradient descent
x_g = np.array([vec[0] for vec in vec_g])*orig_theta_orig
y_g = np.array([vec[1] for vec in vec_g])

xmin = 1 #0.7853981555130009
ymin = 2.172835587202903#0.002172834606545901


plt.figure("Vec trajectory")
# Methods
plt.plot(xu,yu,marker=".",label="Univariate")
plt.plot(xn,yn,marker=".",label="Newton")
plt.plot(x_qn1,y_qn1,marker=".",label="Quasi-Newton BFGS")
plt.plot(x_qn2,y_qn2,marker=".",label="Quasi-Newton DFP")
plt.plot(x_g,y_g,marker=".",label="Gradient")
#Start, min
plt.scatter([vec[0]*orig_theta_orig], [vec[1]],label="Start",color="green",marker="s")
plt.scatter([xmin*orig_theta_orig],[ymin],label="Global minimum",marker="s",color="red")


plt.xlabel(r"$\theta_{23}$")
plt.ylabel(rf"$\Delta m^2 \: [10^{-3}]$")
plt.legend()
plt.grid()

# Monte Carlo
print("\n4.5) Monte Carlo Minimisation")

T_arr = np.linspace(1e-3,5,10)*1.0
global_minimum = MC_minimiser(nll,params,T_arr,[[min(theta_arr),max(theta_arr)],[min(dm_arr),max(dm_arr)]],
                        N_random=10,N_steps=10, sig=0.01,zoom=2,eps=1e-10)

plt.show()