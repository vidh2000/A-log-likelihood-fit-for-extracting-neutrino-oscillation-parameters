from minimisers import *
from functions import *
from differentiation import *
import numpy as np
import math
import matplotlib 


def points(vec_arr):
	x = [vec[0] for vec in vec_arr]
	y = [vec[1] for vec in vec_arr]
	return x,y

plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
#Options
params = {'text.usetex' : True,
          'font.size' : 16,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          'axes.labelsize':20, 
          'legend.fontsize': 15, #16
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'figure.figsize': [7.5,11/1.2],   #[7.5,11/1.2] is normal size for 1 plot. Changed cause of subplots
                     }
plt.rcParams.update(params)


n = 2
def f1(params,vec):
	if len(vec)==1:
		x = vec[0]
	if len(vec)==2:
		x = vec[0]
		y = vec[1]
	if len(vec)==3:
		x = vec[0]
		y = vec[1]
		z = vec[2]
	return #2*x**3+6*x*y**2-3*y**3-150*x  #x**n+y**n #+ x**2+y**2

def f2(params,vec):
	if len(vec)==1:
		x = vec[0]
	if len(vec)==2:
		x = vec[0]
		y = vec[1]
	if len(vec)==3:
		x = vec[0]
		y = vec[1]
		z = vec[2]
	return (x**4-x**2)*x*(x-1)# + y**2

def f_globalminima(params,vec):
	if len(vec)==1:
		x = vec[0]
	if len(vec)==2:
		x = vec[0]
		y = vec[1]
	if len(vec)==3:
		x = vec[0]
		y = vec[1]
		z = vec[2]
	return x**4+y**4-x*(x**2-4)-(y-4)**2

def inverted_sinc(params,vec):
	if len(vec)==1:
		x = vec[0]
	if len(vec)==2:
		x = vec[0]
		y = vec[1]
	if len(vec)==3:
		x = vec[0]
		y = vec[1]
		z = vec[2]
	return -np.sin(x)/x

def ackley_f(params,vec):
	if len(vec)==2:
		x = vec[0]
		y = vec[1]
	return -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2))) - np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))) + np.exp(1)+20

def testing_function(params,vec):
	if len(vec)==2:
		x = vec[0]
		y = vec[1]
	return x**3-12*x+y**3+3*y**2-9*y

def test3d_1(params,vec):
	if len(vec)==3:
		x = vec[0]
		y = vec[1]
		z = vec[2]
	return -np.sin(x)/x * np.sin(y)/y * np.sin(z)/z


def test3d_2(params,vec):
	if len(vec)==3:
		x = vec[0]
		y = vec[1]
		z = vec[2]
	return x**2+y**2+z**2


def test3d_3(params,vec):
	func = 0
	for x in vec:
		func += (x**4-16*x**2+5*x)/2	
	return func


params=0
vec = [-3,10]
h = 0.0005608878686509
dvec = [h for i in range(len(vec))]
eps = 1e-10
order=4


#!!!!!!!!!! vec is a starting point from which functions diverge towards minimum!!!!!!!!!!!!!!!#

vec = [1]
x = np.linspace(-7,7,500)
y = inverted_sinc(0,[x])
minimum, data = parabolic1D_minimiser(-5,2,inverted_sinc,0,vec,0,1e-10,unc=False)
title="parabolic_minimiser_testing"
fig = plt.figure(title)
ax1 = fig.add_subplot(111)
lns1 = ax1.plot(x,y,label=r"- sinc($x$)",color="blue")
lns5 = ax1.bar([-1.5],[0.01],width=1e-9,color="red",alpha=1,
				label=r"Parabolic method distribution of points")
#lns2 = ax1.plot([minimum],[inverted_sinc(0,[minimum])], marker="o",color="black",label=r"$x_{min}=7.0\times10^{-9}$")
ax2 = ax1.twinx()
lns3 = ax2.hist(data,bins=40, color="red",alpha=1,label="Distribution of x from the parabolic method",histtype="bar",
				density=False,stacked=False)
lns4 = ax2.plot([minimum],[inverted_sinc(0,[minimum])], marker=".",color="black",
				label=r"Parabolic method x-distribution")
#ax3.plot([minimum],[inverted_sinc(0,[minimum])], marker="o",color="black",label=r"$x_{min}=7.0\times10^{-9}$")
#ax3.yaxis.set_visible(False)
ax2.yaxis.set_ticks([0,5,10,15,20])
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$f(x)$")
ax2.set_ylabel(r"N")
clr = 'tab:red'
ax2.yaxis.label.set_color(clr)
ax2.spines["right"].set_edgecolor(clr)
ax2.tick_params(axis="y",color=clr,labelcolor=clr)
clr = 'tab:blue'
ax1.yaxis.label.set_color(clr)
ax1.spines["right"].set_edgecolor(clr)
ax1.tick_params(axis="y",color=clr,labelcolor=clr)

ax1.set_ylim(-1.5,0.6)
ax2.set_ylim(0,20)
ax1.set_xlim(-4,4)
#lns3 = matplotlib.lines.Line2D([0 for i in range(len(lns3[0]))], lns3[0])
#added these three lines

ax1.legend(loc="upper center")

print("1D sinc(x) minimum approximated with parabolic method:",minimum)
# plt.savefig("plots/minimisation/"+title+".pdf",
#            dpi=1200, 
#            bbox_inches="tight")




# # Minima for f1
# x0, y0 = 0, 0
# title = "3_NLL(theta_23)"
# fig = plt.figure(title)
# ax1 = fig.add_subplot(111)
# lns1 = ax1.plot(theta_arr,NLL, label=r"NLL($\theta_{23}$), $\Delta m^2_{23}=2.4\times 10^{-3}$ eV$^2$/c$^4$",color="blue")
# ax1.set_xlabel(r"$\theta_{23} \: [\frac{\pi}{4}]$")
# ax1.set_ylabel(r"NLL")

# ax2 = ax1.twinx()
# lns2 = ax2.plot(x_arr,[nll([events,events_simul,E_bins,L],[theta,dm2]) for theta in x_arr],
#                     label="Parabolic minimiser",color='red')
# ax2.set_ylabel(r"N")
# # added these three lines
# lns = lns1+lns2
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc=0)
# fig.tight_layout()
# plt.show()


# vec = [2.5,1.0]
# vec = [-7.0,1.0]
# vec = [2.5,2.0] 
# vec = [-7.0,-11.0]

##################################################################################
##                         2D
##################################################################################

x_arr = np.linspace(-4,5,100)
y_arr = np.linspace(-4,4,100)
x_arr = np.linspace(-4,4,500)
y_arr = np.linspace(-4,4,500)

func_val = f_mesh(x_arr,y_arr,ackley_f,0)

title="Ackley_f_minimisers_testing_2D"
#title="minimisers_testing_2D"
#vec_arr = [[-2,-2]]#[2.5,1.0],[-2.0,1.0],[2.5,2.0],[7.0,-5.0]]
vec=[-1,3]
vec=[-2,1.3]
plt.figure(title)
fig = plt.figure(title)
ax1 = fig.add_subplot(2,1,1)
h1 = ax1.contourf(x_arr, y_arr, func_val,alpha=1.0,cmap="nipy_spectral",levels=50)
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$y$")
clb1 = plt.colorbar(h1)
clb1.set_label(r'$f(x,y)$')

ax2 = fig.add_subplot(2,1,2)
h2 = ax2.contourf(x_arr, y_arr, func_val,alpha=1.0,cmap="nipy_spectral",levels=50)
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$y$")
clb2 = plt.colorbar(h2)
clb2.set_label(r'$f(x,y)$')

#print("Evaluating at:",vec)			-1,5   -2,5
mu, vec_u = univariate_method(ackley_f,[[-2.3,-0.9],[0.7,1.4]],params,vec,dvec,order,eps=eps) # converges well when vec = [-10,7], [x_ar = [-8,9] y_ar = [-9,7]]
#print("Univariate",mu)
mn, vec_n = newton_minimiser(ackley_f,params,vec,dvec,order,eps=eps,N=1e3)
#print("Newton",mn)
#mqn1, vec_qn1= quasi_newton_minimiser(ackley_f,params,vec,dvec,order,method="DFP",alph=1e-1,alpha_factor=1e-1,eps=eps,N=1e3)
#mqn2, vec_qn2 = quasi_newton_minimiser(ackley_f,params,vec,dvec,order,method="BFGS",alph=1e-1,alpha_factor=1e-1,eps=eps,N=1e3) 
mqn1, vec_qn1= quasi_newton_minimiser(ackley_f,params,vec,dvec,order,method="DFP",alph=2e-2,alpha_factor=1e-2,eps=eps,N=1e3)
mqn2, vec_qn2 = quasi_newton_minimiser(ackley_f,params,vec,dvec,order,method="BFGS",alph=5e-2,alpha_factor=1e-2,eps=eps,N=1e3) 
mg, vec_g = gradient_method(ackley_f,params,vec,dvec,order,alph=1.5e-2,eps=eps,N=1e3)

#print("DFP",mqn1)
#print("BFGS",mqn2)
#print("Gradient",mg)

xu,yu = points(vec_u)
xn,yn = points(vec_n)
xqn1,yqn1 = points(vec_qn1)
xqn2,yqn2 = points(vec_qn2)
xg,yg = points(vec_g)


ax1.plot(xu,yu,label="Univariate",marker=".",color="white")
ax1.plot(xn,yn,label="Newton",marker=".",color="blue")
ax1.plot(xqn1,yqn1,label="DFP",marker=".",color="violet")
ax1.plot(xqn2,yqn2,label="BFGS",marker=".",color="yellow")
ax1.plot(xg,yg, label="Gradient",marker=".",color="red")
ax1.text(-3.2,2.2,r"$x_0=(-2,1.3)$",usetex=True,
		bbox=dict(boxstyle="round",pad=0.3,#rounding_size=0.1,
				  fc="white",ec="black"))
ax1.text(1,-1,r"$x_{min}=(0,0)$",usetex=True,
		bbox=dict(boxstyle="round",pad=0.3,#rounding_size=0.1,
				  fc="white"))
ax1.scatter([vec[0]],[vec[1]],marker="x",color="black",s=70,zorder=9)
#plt.scatter([0],[0],marker="x",color="red",s=70,zorder=9)
ax1.legend(loc="lower left")

ax2.plot(xu,yu,label="Univariate",marker=".",color="white")
ax2.plot(xn,yn,label="Newton",marker=".",color="blue")
ax2.plot(xqn1,yqn1,label="DFP",marker=".",color="violet")
ax2.plot(xqn2,yqn2,label="BFGS",marker=".",color="yellow")
ax2.plot(xg,yg, label="Gradient",marker=".",color="red")

ax2.scatter([vec[0]],[vec[1]],marker="x",color="black",s=70,zorder=9)
#plt.scatter([0],[0],marker="x",color="red",s=70,zorder=9)
ax2.legend(loc="lower left")

#plt.legend()
#plt.grid()
#plt.xlim(-3.6,4.4)
#plt.ylim(-4,4)
#plt.xlim(2-0.7e-8,2+0.3e-8)
#plt.ylim(1-1.1e-9,1+5.8e-9)
ax1.set_xlim(-4,4)
ax1.set_ylim(-4,4)
ax2.set_xlim(-2.5,-1.5)
ax2.set_ylim(0.5,1.5)
fig.tight_layout()
# plt.savefig("plots/minimisation/"+title+".pdf",
#           dpi=1200, 
#           bbox_inches="tight")




x_arr = np.linspace(-4,5,100)
y_arr = np.linspace(-4,4,100)
x_arr = np.linspace(-3.6,4.4,500)
y_arr = np.linspace(-2,4,500)
#x_arr = np.linspace(-4,4,500)
#y_arr = np.linspace(-4,4,500)

func_val = f_mesh(x_arr,y_arr,testing_function,0)

#title="Ackley_f_minimisers_testing_2D"
title="minimisers_testing_2D"
#vec_arr = [[-2,-2]]#[2.5,1.0],[-2.0,1.0],[2.5,2.0],[7.0,-5.0]]
vec=[-1,3]
#vec=[-2,1.3]
plt.figure(title)
fig = plt.figure(title)
ax1 = fig.add_subplot(2,1,1)
h1 = ax1.contourf(x_arr, y_arr, func_val,alpha=1.0,cmap="nipy_spectral",levels=50)
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$y$")
clb1 = plt.colorbar(h1)
clb1.set_label(r'$f(x,y)$')

ax2 = fig.add_subplot(2,1,2)
h2 = ax2.contourf(x_arr, y_arr, func_val,alpha=1.0,cmap="nipy_spectral",levels=50)
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$y$")
clb2 = plt.colorbar(h2)
clb2.set_label(r'$f(x,y)$')

#print("Evaluating at:",vec)			-1,5   -2,5
mu, vec_u = univariate_method(testing_function,[[-1,2.5],[-1,3.2]],params,vec,dvec,order,eps=eps) # converges well when vec = [-10,7], [x_ar = [-8,9] y_ar = [-9,7]]
print("vec_u:",vec_u)
#print("Univariate",np.array(mu))#-np.array([2,1]))
mn, vec_n = newton_minimiser(testing_function,params,vec,dvec,order,eps=eps,N=1e3)
#print("Newton",mn)
mqn1, vec_qn1= quasi_newton_minimiser(testing_function,params,vec,dvec,order,method="DFP",alph=1e-1,alpha_factor=1e-1,eps=eps,N=1e3)
mqn2, vec_qn2 = quasi_newton_minimiser(testing_function,params,vec,dvec,order,method="BFGS",alph=1e-1,alpha_factor=1e-1,eps=eps,N=1e3) 
#mqn1, vec_qn1= quasi_newton_minimiser(testing_function,params,vec,dvec,order,method="DFP",alph=2e-2,alpha_factor=1e-2,eps=eps,N=1e3)
#mqn2, vec_qn2 = quasi_newton_minimiser(testing_function,params,vec,dvec,order,method="BFGS",alph=5e-2,alpha_factor=1e-2,eps=eps,N=1e3) 
mg, vec_g = gradient_method(testing_function,params,vec,dvec,order,alph=1.5e-2,eps=eps,N=1e3)

#print("DFP",mqn1)
#print("BFGS",mqn2)
#print("Gradient",mg)

xu,yu = points(vec_u)
xn,yn = points(vec_n)
xqn1,yqn1 = points(vec_qn1)
xqn2,yqn2 = points(vec_qn2)
xg,yg = points(vec_g)


ax1.plot(xu,yu,label="Univariate",marker=".",color="white")
ax1.plot(xn,yn,label="Newton",marker=".",color="limegreen")
ax1.plot(xqn1,yqn1,label="DFP",marker=".",color="violet")
ax1.plot(xqn2,yqn2,label="BFGS",marker=".",color="yellow")
ax1.plot(xg,yg, label="Gradient",marker=".",color="red")
ax1.text(-3.2,2.2,r"$x_0=(-1,3)$",usetex=True,
		bbox=dict(boxstyle="round",pad=0.3,#rounding_size=0.1,
				  fc="white",ec="black"))
ax1.text(1,-1,r"$x_{min}=(2,1)$",usetex=True,
		bbox=dict(boxstyle="round",pad=0.3,#rounding_size=0.1,
				  fc="white"))
ax1.scatter([vec[0]],[vec[1]],marker="x",color="black",s=70,zorder=9)
#plt.scatter([0],[0],marker="x",color="red",s=70,zorder=9)
ax1.legend(loc="lower left")

ax2.plot(xu,yu,label="Univariate",marker=".",color="white")
ax2.plot(xn,yn,label="Newton",marker=".",color="limegreen")
ax2.plot(xqn1,yqn1,label="DFP",marker=".",color="violet")
ax2.plot(xqn2,yqn2,label="BFGS",marker=".",color="yellow")
ax2.plot(xg,yg, label="Gradient",marker=".",color="red")

ax2.scatter([vec[0]],[vec[1]],marker="x",color="red",s=70,zorder=9)
ax2.scatter([2],[1],marker="x",color="red",s=70,zorder=9)
ax2.legend(loc="lower left")

#plt.legend()
#plt.grid()
#plt.xlim(-3.6,4.4)
#plt.ylim(-4,4)
#plt.xlim(2-0.7e-8,2+0.3e-8)
#plt.ylim(1-1.1e-9,1+5.8e-9)
ax1.set_xlim(-3.6,4.4)
ax1.set_ylim(-2,4)
ax2.set_xlim(2-0.7e-8,2+0.3e-8)
ax2.set_ylim(1-1.1e-9,1+5.8e-9)
fig.tight_layout()
plt.savefig("plots/minimisation/"+title+".pdf",
          dpi=1200, 
          bbox_inches="tight")







# print(vec_u[-3:])
# print(vec_n[-3:])
# print(vec_qn1[-3:])
# print(vec_qn2[-3:])
# print(vec_g[-3:])
# T_arr = np.linspace(1e-3,5,10)*1.0
# global_minimum = MC_minimiser(ackley_f,params,T_arr,[[-4,4], [-4,4]], 10,100, 0.01,2, 1e-15)
# print(global_minimum)

####################################################################################################
##########################         3D        ####################
####################################################################################################


N=3
vec = [-2.5 for i in range(N)]
#h = 0.0005608878686509
dvec = [h for i in range(len(vec))]


T_arr = np.linspace(1e-3,5,10)*1.0
"""
NLL(min)= 7.105427357601002e-15                                                                                                                                                                                                                                                                                                                                                              
TOTAL ITERATIONS: 510000                                                                                                                                                                                                                                                                                                                                                                     
MC minimum: [-6.028104942123491e-17, 5.06252279333629e-16]   
"""
eps=1e-10


mu, vec_u = univariate_method(test3d_3,[[-3,-1] for i in range(N)],params,vec,dvec,order,eps=eps) # converges well when vec = [-10,7], [x_ar = [-8,9] y_ar = [-9,7]]
mn, vec_n = newton_minimiser(test3d_3,params,vec,dvec,order,eps=eps,N=1e3)
mu = np.array(mu)
mn = np.array(mu)
mqn1, vec_qn1= quasi_newton_minimiser(test3d_3,params,vec,dvec,order,method="DFP",alph=1e-2,alpha_factor=1e-2,eps=eps,N=1e5)
mqn2, vec_qn2 = quasi_newton_minimiser(test3d_3,params,vec,dvec,order,method="BFGS",alph=1e-2,alpha_factor=1e-2,eps=eps,N=1e5) #works for alph=1e-6
mg, vec_g = gradient_method(test3d_3,params,vec,dvec,order,alph=1.5e-3,eps=eps,N=1e4)
global_minimum = MC_minimiser(test3d_3,params,T_arr,[[-3,5] for i in range(N)], 10,100, 0.01, 2, 1e-10)


print(f"MINIMIZING {N}-D Styblinski–Tang function with global minimum at x_i=−2.903534")
print("\nUnivariate",mu)
print("Newton",mn)
print("DFP",mqn1)
print("BFGS",mqn2)
print("Gradient", mg)

print("MC", global_minimum)
print("Some methods also show an uncertainty next to the result obtained by curvature at the minimum.")
plt.show()