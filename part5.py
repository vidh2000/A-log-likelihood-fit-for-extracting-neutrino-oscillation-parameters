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
          'font.size' : 16,
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

# CONSTANTS
theta = np.pi/4
dm2 = 2.4e-3
L=295



####################################################################################
####################################################################################
"""
5.) Neutrino interaction cross-section
"""
####################################################################################
####################################################################################


title="5_oscillated_event_rate_prediction_propto_E"
plt.figure(title)
theta_min = 1#0.785
dm2_min = 2.17#e-3
alpha_arr = [0.6,0.8,1,1.2,1.4]
for alpha in alpha_arr:
  osc_rate_predict = lambda_i_new(events_simul,theta_min,dm2_min,L,E=E_bins,alpha=alpha,arr=True)
  plt.plot(E_bins,np.array(osc_rate_predict),label=fr"$Prediction, \alpha={alpha}$",linewidth=2.0)

plt.bar(E_bins, np.array(events),width=0.05,label="Data",color="red")
plt.legend()
plt.xlabel("$Energy$ [GeV]")
plt.ylabel("$Flux$ [a.u]")
plt.grid()
#plt.savefig("plots/5/"+title+".pdf")

alpha_arr = np.linspace(0,10,2000)
title="NLL(alpha)"
plt.figure(title)
NLL = [nll_cs([events,events_simul,E_bins,L],[1,2.17,alpha]) for alpha in alpha_arr]
plt.plot(alpha_arr,NLL,label= r"$ \theta_{23} = \frac{\pi}{4}, \: \Delta m_{23}^2 = 2.17 \times 10^{-3}$ eV$^2$")
plt.legend()
#plt.grid()
plt.xlabel(r"$\alpha$")
plt.ylabel(r"NLL($\theta_{23},\Delta m_{23}^2, \alpha$)")

# plt.savefig("plots/5/"+title+".pdf",
           # dpi=1200, 
           # bbox_inches="tight")


ht = 4.806217384e-6
hm = 4.806217384e-6
ha = 4.806217384e-6


params = [events,events_simul,E_bins,L]
vec = [1.03,2.173,1]
dvec = [ht,hm,ha]


theta_arr = [0.5,1.5]
dm_arr = [2.0,2.4]
alpha_arr = [0.97,0.99]

# results, vec_u = univariate_method(nll,[theta_arr,dm_arr,alpha_arr],params,vec,dvec,2,1e-15)
# theta,dm2,alpha = results[0],results[1],results[2]
# print(vec_u)

# # print("\n5.1) Univariate parameters that minimise NLL:")
# # print(f"Theta = {theta[0]} +- {theta[1]}")
# # print(f"dm2 = {dm2[0]} +- {dm2[1]}")
# # print(f"alpha={alpha[0]} +- {alpha[1]}\n")

# results, vec_g = gradient_method(nll,params,vec,dvec,order=2,alph=1e-5,eps=1e-10,N=1e3)
# theta,dm2,alpha = results[0],results[1],results[2]

# print("\n5.2) Gradient descent parameters that minimise NLL:")
# print(f"Theta = {theta[0]} +- {theta[1]}")
# print(f"dm2 = {dm2[0]} +- {dm2[1]}")
# print(f"alpha={alpha[0]} +- {alpha[1]}\n")


# results, vec_arr_qn1 = quasi_newton_minimiser(nll,params,vec,dvec,alpha_factor=1e-2,
#                    method="BFGS",order=2,alph=1e-2,eps=1e-10,N=1e3)
# theta,dm2,alpha = results[0],results[1],results[2]

# print("\n5.4) BFGS Quasi-Newton parameters that minimise NLL:")
# print(f"Theta = {theta[0]} +- {theta[1]}")
# print(f"dm2 = {dm2[0]} +- {dm2[1]}")
# print(f"alpha={alpha[0]} +- {alpha[1]}\n")

# results, vec_arr_qn2 = quasi_newton_minimiser(nll,params,vec,dvec,alpha_factor=1e-2,
#                    method="DFP",order=2,alph=1e-2,eps=1e-10,N=1e3)
# theta,dm2,alpha = results[0],results[1],results[2]

# print("\n5.5) DFP Quasi-Newton parameters that minimise NLL:")
# print(f"Theta = {theta[0]} +- {theta[1]}")
# print(f"dm2 = {dm2[0]} +- {dm2[1]}")
# print(f"alpha={alpha[0]} +- {alpha[1]}\n")
# vec = [1.05,2.173,0.98]

results, vec_arr = newton_minimiser(nll_cs,params,vec,dvec,order=2,eps=1e-10,N=1e3) #singular matrix? goes to -infty?
theta,dm2,alpha = results[0],results[1],results[2]
print(results)

print("\n5.6) Newton parameters that minimise NLL:")
print(f"Theta = {theta[0]} +- {theta[1]}")
print(f"dm2 = {dm2[0]} +- {dm2[1]}")
print(f"alpha={alpha[0]} +- {alpha[1]}\n")

vec = [0.992,2.15,1]
results, vec_arr = newton_minimiser(nll_cs,params,vec,dvec,order=2,eps=1e-10,N=1e3) #singular matrix? goes to -infty?
theta,dm2,alpha = results[0],results[1],results[2]


print("\n5.6) Newton parameters that minimise NLL:")
print(f"Theta = {theta[0]} +- {theta[1]}")
print(f"dm2 = {dm2[0]} +- {dm2[1]}")
print(f"alpha={alpha[0]} +- {alpha[1]}\n")

T_arr = np.linspace(1e-3,5,10)*1.0

results = MC_minimiser(nll_cs,params,T_arr,[[0.9,1.1],[2.14,2.2], [0.9,1.1]], 10,100, 0.01, 2, 1e-10)
theta,dm2,alpha = results[0],results[1],results[2]

print("\n5.7) Monte Carlo parameters that minimise NLL:")
print(f"Theta = {theta[0]} +- {theta[1]}")
print(f"dm2 = {dm2[0]} +- {dm2[1]}")
print(f"alpha={alpha[0]} +- {alpha[1]}\n")

osc_rate_predict = lambda_i_new(events_simul,theta[0],dm2[0],L,E=E_bins,alpha=alpha[0],arr=True)

title="5_event_prediction_with_optimized_parameters_vs_data"
fig = plt.figure(title)
ax1 = fig.add_subplot(111)
ax1.bar(E_bins,np.array(osc_rate_predict),width=0.05,label=r"Expected average rate $\lambda^{new}(E)$",color="red",alpha=0.7)
ax1.bar([1], [0.001],width=0.05,label=r"Simulated experimental data from T2K",color="blue",alpha=0.7)

ax2 = ax1.twinx()
ax2.bar(E_bins, np.array(events),width=0.05,label=r"Simulated experimental data from T2K",color="blue",alpha=0.7)

ax1.legend()
plt.xlabel(r"$E$ [GeV]")
plt.ylabel(r"$N$")

ax1.set_xlabel(r"$E$ [GeV]")
ax1.set_ylabel(r"$N$")
ax2.set_ylabel(r"$N$")
ax1.yaxis.set_ticks([0,5,10,15,20])
ax2.yaxis.set_ticks([0,5,10,15,20])
clr = 'tab:red'
ax2.yaxis.label.set_color(clr)
ax2.spines["right"].set_edgecolor(clr)
ax2.tick_params(axis="y",color=clr,labelcolor=clr)
clr = 'tab:blue'
ax1.yaxis.label.set_color(clr)
ax1.spines["right"].set_edgecolor(clr)
ax1.tick_params(axis="y",color=clr,labelcolor=clr)

ax1.set_ylim(0,22)
ax2.set_ylim(0,22)
ax1.set_xlim(0,10)
plt.tight_layout()
# plt.savefig("plots/5/"+title+".pdf",
#            dpi=1200, 
#            bbox_inches="tight")

plt.show()
