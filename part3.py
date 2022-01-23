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
theta = 1#np.pi/4
dm2 = 2.4#e-3
L=295
##############################################
##############################################


"""
3.1.) The Data
"""

title = "1_observed_flux_E_distribution"
plt.figure("Events energy histogram")
plt.bar(E_bins,events,width=0.05)
plt.ylabel(r"$N$")
plt.xlabel("$Energy$ [GeV]")

#plt.savefig("plots/3/"+title+".pdf")


title = "1_unoscillated_flux_E_distribution"

plt.figure(title)
plt.bar(E_bins,events_simul,width=0.05)
plt.ylabel("$N$")
plt.xlabel("$Energy$ [GeV]")

plt.savefig("plots/3/"+title+".pdf")


title="1_unoscillated_and_oscillated_data_and_lambda(E)"

osc_rate_predict = lambda_i(events_simul,theta,dm2,L,E=E_bins,arr=True)
plt.figure(title)
plt.bar(E_bins,events_simul,width=0.05,label=r"Unoscillated $\nu_\mu$ data",edgecolor='red', color='None')
plt.bar(E_bins,events,width=0.05,label=r"Simulated $\nu_\mu$ data from T2K",edgecolor='blue', color='None')
plt.plot(E_bins,osc_rate_predict,label=r"$\lambda(\theta_{23}, \Delta m_{23}^2; \: E)$",color="black",linewidth=3.0)

plt.ylabel("$N$")
plt.xlabel("$E$ [GeV]")
plt.legend()
# plt.savefig("plots/3/"+title+".pdf",
#            dpi=1200, 
#            bbox_inches="tight")

"""
3.2.) Fit function
"""


title = "2_P_neutrino(E)"
plt.figure(title)
plt.plot(E_bins, P(1,2.4,L,E_bins))
plt.xlabel("$Energy$ [GeV]")
plt.ylabel(r"$P(\nu_{\mu} \rightarrow \nu_{\mu})$")
#plt.xscale("log")
#plt.grid()
#plt.savefig("plots/3/"+title+".pdf")


osc_rate_predict = lambda_i(events_simul,theta,dm2,L,E=E_bins,arr=True)

title="2_oscillated_event_rate_prediction"
plt.figure(title)
plt.bar(E_bins, events,width=0.05,label=r"Events",edgecolor='blue', color='None')
plt.plot(E_bins,osc_rate_predict,label=r"Simulated event rate $\lambda(E)$",color="orange",linewidth=3.0)
plt.legend()
plt.xlabel("$Energy$ [GeV]")
plt.ylabel(r"$N$")
#plt.grid()
#plt.savefig("plots/3/"+title+".pdf")


osc_rate_predict = lambda_i(events_simul,theta,2.172835587202903,L,E=E_bins,arr=True)

"""
3.3) Likelihood function
"""

dm2_arr = np.linspace(0,10,1000)
NLL = [nll([events,events_simul,E_bins,L],[0.7853981555130009/(np.pi/4),dm2]) for dm2 in dm2_arr]
title = "3_NLL(dm2)"
plt.figure(title)
plt.plot(dm2_arr*1e-3,NLL,label=r"$\theta_{23}=\pi/4$")
plt.legend()
plt.xlabel(r"$\Delta m^2_{23} \: [eV^2]$")
plt.ylabel("$NLL$")
#plt.savefig("plots/3/"+title+".pdf")

"""
3.4) Minimise
"""

theta_arr = np.arange(0,2.0,1e-2)
dm2 = 2.4#0.002172834606545901 *1e3
NLL = [nll([events,events_simul,E_bins,L],[theta,dm2]) for theta in theta_arr]

theta_min = 0.2
theta_max = 1.5
vec = [0.4,2.4]
params = [events,events_simul,E_bins,L]
theta_minimized, sig = parabolic1D_minimiser(theta_min,theta_max,nll,params,vec,
                        optimize_index=0,eps=1e-15,unc=True)
print(theta_minimized)

print("NLL at fixed dm^2=2.4e-3 is minimised at theta_23 with error from the curvature of the parabolic estimate =\n",
		theta_minimized*(np.pi/4),"+-",sig*(np.pi/4))


vec = [0.9224302261560001,2.4]
sig = [0.01,0.1]
pos=[0]
eps=1e-15
pm_ers = pm_error_finder(nll,params,vec,sig,pos,eps)
pm_ers = pm_ers[0]
print("1D +- errors in theta:",pm_ers*(np.pi/4))
print("\n")

vec = [1.000000007709211, 2.1728346997962342]
print("NLL:",nll(params,vec))




vec_min = [theta_minimized,2.4]
NLLmin = nll([events,events_simul,E_bins,L],vec_min)
vec= [vec_min[0]+pm_ers[0],2.4]
NLLp = nll([events,events_simul,E_bins,L],vec)-NLLmin
vec= [vec_min[0]+pm_ers[1],2.4]
NLLm = nll([events,events_simul,E_bins,L],vec)-NLLmin

title = "3_NLL(theta_23)"
fig = plt.figure(title)
ax1 = fig.add_subplot(111)
lns1 = ax1.plot(theta_arr,NLL, label=r"NLL($\theta_{23}$), $\Delta m^2_{23}=2.4\times 10^{-3}$ eV$^2$",color="blue")
ax1.set_xlabel(r"$\theta_{23} \: [\frac{\pi}{4}]$")
ax1.set_ylabel(r"NLL")

ax1.legend(loc="upper center")
# plt.savefig("plots/3/"+title+".pdf",
#            dpi=1200, 
#            bbox_inches="tight")


vec = [0.78,dm2]

dm2_min, sig_m = parabolic1D_minimiser(1,3,nll,params,vec,
                        optimize_index=1,eps=1e-15,unc=True)
"""
3.5) Find accuracy of fit result
"""
print("NLL at fixed theta_23=0.78 is minimised at dm2_23 with error from the curvature of the parabolic estimate =\n",
        dm2_min, "+-", sig_m)

plt.show()

theta_arr = np.arange(theta_min,theta_max,1e-5)


# err_plus,err_min = theta_pm_finder(events, events_simul,E_bins,L,theta_arr,theta,dm2,5e-3)
# print("NLL is minimised at: theta_23 +- [dtheta^+,dtheta^-] == ", theta_minimized,"+- [",err_plus,",",err_min,"]")
