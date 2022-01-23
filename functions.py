import numpy as np
import matplotlib.pyplot as plt
import os 
import random as rand
from copy import *
from minimisers import *
from differentiation import *
import math 
from scipy.special import gamma
from scipy.special import gammainc

def P(theta,dm2,L,E):
	"""
	The probability that the muon neutrino will be observed
	as a muon neutrino and will not have oscillated into a tau neutrino."
	"""
	theta = theta*np.pi/4
	dm2 = dm2*1e-3
	return 1- (np.sin(2*theta))**2 * (np.sin(1.267*dm2*L/E))**2 



def lambda_i(data,theta,dm2,L,E,arr):
	"""
	Finds lambda_i = P(E)*N(E) where E is energy 
	and N(E) is number of events in the energy bin with E.
	
	theta = value
	dm2 = value
	1. arr=True
	Finds an array of lambdas where E == E_arr
	2. arr=False
	Finds lambda_i for specific energy bin with energy E
	"""
	if arr==True:
		l_arr = []
		for i in range(len(data)):
			lambd = P(theta,dm2,L,E[i])*data[i]
			l_arr.append(lambd)
		return l_arr

	if arr==False:
		lambd = P(theta,dm2,L,E)*data
		return lambd

def lambda_i_new(data,theta,dm2,L,E,alpha,arr=False):
    """
    Finds lambda_i = P(E)*N(E) where E is energy 
    and N(E) is number of events in the energy bin with E.

    1. arr=True
    Finds an array of lambda_is for E_arr
    2. arr=False
    Finds lambda_i for specific energy bin
    """
    if arr==True:
        l_arr = []
        for i in range(len(data)):
            lambd = P(theta,dm2,L,E[i])*data[i]*alpha*E[i]
            l_arr.append(lambd)
        return l_arr

    if arr==False:
        lambd = P(theta,dm2,L,E)*data*alpha*E
        return lambd




def nll(params=[],vec=[]):
	"""
	Negative Log Likelihood
	For many bins, data = {m_i} denotes the observed number of neutrino events in
	bin i.

	For minimising both
	params=[osc_data,unosc_data,E_bins,L]
	osc_data,unosc_data = arrays corresponding to E_bins
	of oscillated and unoscillated datasets
	E_bins = array of energies
	
	vec = [theta,dm2]
	theta = VALUE not an array
	dm2 = VALUE not an array
	"""
	osc_data = params[0]
	unosc_data = params[1]
	E_bins = params[2]
	L = params[3]

	if not len(params)==4:
		raise ValueError("Need len(params)=4 i.e params=[osc_data,unosc_data,E_bins,L]")

	theta=vec[0]
	dm2=vec[1]
	# To optimise w.r.t theta and/or dm2
	# if len(params)==4:
	# 	if not len(vec)==2:
	# 		raise ValueError("NLL needs two inputs as vec i.e [theta,dm2]")
	# 	theta=vec[0]
	# 	dm2=vec[1]

	NLL = 0
	N=len(osc_data)
	for i in range(N):
		
		lbd = lambda_i(unosc_data[i],theta,dm2,L,E_bins[i],arr=False)
		m_fac = math.factorial(osc_data[i])
		NLL += lbd - osc_data[i]*np.log(lbd) + np.log(float(m_fac))

		# if osc_data[i]==0:
		# 	NLL += lbd	
		# else:
		# 	NLL += lbd-osc_data[i]+osc_data[i]*np.log(osc_data[i]/lbd)
	return NLL

def nll_cs(params,vec):
	"""
	Negative Log Likelihood
	For many bins, data = {m_i} denotes the observed number of neutrino events in
	bin i.

	For minimising both
	params=[osc_data,unosc_data,E_bins,L]
	osc_data,unosc_data = arrays corresponding to E_bins
	of oscillated and unoscillated datasets
	E_bins = array of energies

	vec = [theta,dm2,alpha]
	theta = VALUE not an array
	dm2 = VALUE not an array
	alpha= VALUE
	"""
	osc_data = params[0]
	unosc_data = params[1]
	E_bins = params[2]
	L = params[3]

	theta = vec[0]
	dm2 = vec[1]
	alpha=vec[2]

	NLL = 0
	N=len(osc_data)
	for i in range(N):
		
		lbd = lambda_i_new(unosc_data[i],theta,dm2,L,E_bins[i],alpha,arr=False)
		m_fac = math.factorial(osc_data[i])
		NLL += lbd - osc_data[i]*np.log(lbd) + np.log(float(m_fac))

		# if osc_data[i]==0:
		# 	NLL += lbd	
		# else:
		# 	NLL += lbd-osc_data[i]+osc_data[i]*np.log(osc_data[i]/lbd)
	return NLL



def theta_pm_finder(osc_data,unosc_data,E_bins,L,theta_arr,theta,dm2,eps):
	"""

	Finds errors in + and - direction
	i.e directional errors; distances from theta which minimizes NLL 
	to theta+- where (NLL-NLL_min)=0.5
	"""
	params = [osc_data,unosc_data,E_bins,L]
	vec = [theta,dm2]
	theta_minimiz = parabolic1D_minimiser(theta_arr[0],theta_arr[-1],nll,params,vec,0,1e-15)
	vec = [theta_minimiz,dm2]
	NLL_min = nll(params,vec)
	#print("NLL_min",NLL_min,"at",theta_minimiz)
	NLL = [nll(params,[t,dm2]) for t in theta_arr]
	# Find approximate index of the minimum point
	min_index = []
	for Nll in NLL:
		if abs(Nll-NLL_min)<eps:
			index = np.where(NLL == Nll)[0][0]
			min_index.append(index)
	min_index = min_index[0]

	# Find indices - and corresponding thetas for theta+-
	NLL_pm_indices = []
	for Nll in NLL:
		if abs(abs(Nll-NLL_min)-0.5)<eps:
			index = np.where(NLL == Nll)[0][0]
			NLL_pm_indices.append(index)
	indices = []
	top = []
	bot = []
	# get only 1 value of theta+- since you can get more very close (neighbouring) values
	for ind in NLL_pm_indices:
		if ind<min_index:
			bot.append(ind)
		if ind>min_index:
			top.append(ind)		

	indices.append(min(bot))
	indices.append(min(top))
	indices.sort()
	theta_pm = []
	for ind in indices:
		theta_pm.append(theta_arr[ind])
	
	theta_minus = min(theta_pm)
	theta_plus = max(theta_pm)

	err_plus = theta_plus-theta_minimiz
	err_min = theta_minimiz-theta_minus

	return err_plus, err_min 


def NLL_mesh(theta_arr,dm2_arr,params):
    mesh = []
    for m in dm2_arr:
        #print(f"{m/max(dm2_arr)}/1")
        row = np.array([])
        for t in theta_arr:
            vec = [t,m]
            NLL = nll(params,vec)
            row = np.append(row,NLL)
        mesh.append(row)
    return mesh

def f_mesh(x_arr,y_arr,f,params):
    mesh = []
    for m in y_arr:
        #print(f"{m/max(y_arr)}/1")
        row = np.array([])
        for t in x_arr:
            vec = [t,m]
            v = f(params,vec)
            row = np.append(row,v)
        mesh.append(row)
    return mesh

def outer_product(x,y):
	"""
	Finds outer product of vectors x,y
	"""
	M = np.zeros((len(x),len(y)))
	for i in range(len(x)):
		for j in range(len(y)):
			M[i][j] = x[i]*y[j]
	return M 


def reduced_chi2(osc_data,exp_data,N_param,correct):
	"""
	Calculates reduced chi^2 value 
	for datasets of expected mean values of
	unoscillated dataset.
	exp_data = {lambda_i}
	"""
	dof = len(osc_data)-N_param
	#print(dof)
	exp_data = np.array(exp_data)
	# 1sigma uncertainty on oscillated data (histogram bins)
	# is sqrt(N) where N is height of the bin
	sigs = np.sqrt(exp_data)
	rchi2 = 0
	for i in range(len(osc_data)):
		rchi2 += (osc_data[i]-exp_data[i])**2/sigs[i]**2
	rchi2 = rchi2/dof
	q = 1+(200**2-1)/(6*dof*464)
	if correct:
		return rchi2/q
	else:
		return rchi2
def ext_simpsons_int(f,a,b,eps):
	"""
	Extended simpsons rule to integrate
	any function f from a to b with 
	convergence criteria eps
	"""
	x = np.array([a,b])
	y = np.array([f(a),f(b)])
	h = b-a
	S1 = 1e-6
	S2 = 1e6
	I1 = h*1/2*np.sum(y)
	I2 = I1*1e9
	j = 0
	while abs((S2-S1)/S1) > eps:
		if j!=0:
			I1 = I2
		if j>1:
			S1 = S2

		x_new = np.array([(x[i]+x[i+1])/2 for i in range(len(x)-1)])
		y_new = f(x_new)
		
		I2 = 1/2*I1 + np.sum(h/2*y_new)
		if j==1:
			S1 = 4/3*I2 - 1/3*I1
		if j>1:
			S2 = 4/3*I2 - 1/3*I1

		h = h/2
		x = np.append(x,x_new)
		x = np.sort(x)

		j +=1
	return S2



def chi2_CDF(x,dof,eps,correct):
	"""
	CDF of chi^2 where
	dof = number of degress of freedom
	x = reduced chi2 value
	"""

	if correct:
		q = 1+(200**2-1)/(6*dof*464) #where 464 == total sample size
		x =np.array(x)*dof/q
	else:
		x = np.array(x)*dof
	
	#print("x",x)
	s = dof/2

	f = lambda t: np.exp((s-1)*np.log(t)-t)

	lower_inc_gamma = ext_simpsons_int(f,0,x/2,eps)
	gamma = ext_simpsons_int(f,0,1e4,eps) 

	F = lower_inc_gamma/gamma

	return F


def p_value(x,dof,eps,correct):

	return 1-chi2_CDF(x,dof,eps,correct)


def pm_error_finder(f,params,vec,sig,pos,eps):
	"""
	Bisection method to find +- errors.
	Finds points where NLL changes by 0.5
	from the value at minimum.

	sig, vec must be vectors.
	pos= array of indices you want results for
	Returns an array of uncertainties for
	each parameter dimension
	uncs = [+err,-err]
	"""
	vec = np.array(vec)
	n = len(vec)
	param_uncs = []
	func = lambda vec_i: f(params,vec_i)-f(params,vec)-0.5
	
	# find errors for each component - dimension
	for i in range(n):
		uncs = []

		#+ point
		xl = vec[i]
		xr = vec[i]+3*sig[i]
		x = (xl+xr)/2
		err = xr-xl
		#print("l,r",xl,xr)
		#print("x",x)
		err_old = deepcopy(err)*1e6
		while abs(err-err_old)>eps:
			err_old = deepcopy(err)
			# Find on which side estimation lies
			vec_i = deepcopy(vec)
			vec_i[i] = x
			y = func(vec_i)
			#print("y",y)
			if y>0:
				xr = deepcopy(x)
				#print("right")
			else:
				xl = deepcopy(x)
				#print("left")
			#print("l,r",xl,xr)
			# Update 
			x = (xl+xr)/2
			err = xr-xl 
		#print("+err=",x)
		# input +error
		uncs.append(deepcopy(x))
		#- point 
		xl = vec[i]-3*sig[i]
		xr = vec[i]
		x = (xl+xr)/2
		err = xr-xl
		err_old = deepcopy(err)*1e6
		while abs(err-err_old)>eps:
			err_old = deepcopy(err)
			# Find on which side estimation lies
			vec_i = deepcopy(vec)
			vec_i[i] = x 
			y = func(vec_i)
			if y>0:
				xl = deepcopy(x)
			else:
				xr = deepcopy(x)
			# Update 
			x = (xl+xr)/2
			err = xr-xl 
		#print("-err=",x)
		# Input -error
		uncs.append(deepcopy(x))
		param_uncs.append(uncs)


	res_all = [] 
	for i in range(n):
		component = vec[i]
		unc_arr = np.array(param_uncs[i])
		errs = unc_arr-component
		# print("vec",vec)
		# print("comp",component)
		# print("parmunc",unc_arr)
		# print("errrs",errs)
		res_all.append(unc_arr-component)
	#print("resal",res_all)
	results = [res_all[i] for i in pos]
	return results



