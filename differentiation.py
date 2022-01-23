import numpy as np 
import matplotlib.pyplot as plt
import os 
import random as rand
from copy import *
from minimisers import *
from functions import *
import itertools


def der1(f,params,vec,dvec,pos,order):
	"""
	First derivative of a function f(params,vec)
	for variable vec[pos] where pos is index position 
	of the variable w.r.t to which we differentiate.
	Order = O(h^order)... accuracy of the finite difference method
	"""

	if order==2:
		pts = [-1.0,1.0]
		coef = [-1/2,1/2]

	elif order==4:
		pts = [-2.0,-1.0,1.0,2.0]
		coef = [1/12,-2/3,2/3,-1/12]
	elif order==6:
		pts = [-3.0,-2.0,-1.0,1.0,2.0,3.0]
		coef = [-1/60,3/20,-3/4,3/4,-3/20,1/60]
	elif order==8:
		pts = [-4.0,-3.0,-2.0,-1.0,1.0,2.0,3.0,4.0]
		coef = [1/280,-4/105,1/5,-4/5,4/5,-1/5,4/105,-1/280]
	else:
		raise ValueError(f"Can't find derivative with O(h^{order})")
	der = 0
	vec_orig = deepcopy(np.array(vec))

	for i in range(len(coef)):
		# Reset vec to original values
		vec = vec_orig*1.0
		vec[pos] = vec[pos]+pts[i]*dvec[pos]
		der += coef[i]*f(params,vec)/dvec[pos]
	return der

def der2(f,params,vec,dvec,pos,order):
	"""
	Calculate second order derivative with
	accuracy O(h^order)
	For input description - refer to der1
	"""
	if order==2:
		pts = [-1.0,0.0,1.0]
		coef = [1.0,-2.0,1.0]
	elif order==4:
		pts = [-2.0,-1.0,0.0,1.0,2.0]
		coef = [-1/12,4/3,-5/2,4/3,-1/12]
	elif order==6:
		pts = [-3.0,-2.0,-1.0,0.0,1,2.0,3.0]
		coef = [1/90,-3/20,3/2,-49/18,3/2,-3/20,1/90] 
	elif order==8:
		pts = [-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0]
		coef = [-1/560,8/315,-1/5,8/5,-205/72,8/5,-1/5,8/315,-1/560]
	else:
		raise ValueError(f"Can't find derivative with O(h^{order})")

	der = 0
	vec_orig = deepcopy(np.array(vec))

	for i in range(len(coef)):
		# Reset vec to original values
		vec = vec_orig*1.0
		# Find term value
		vec[pos] = vec[pos]+pts[i]*dvec[pos]
		der += coef[i]*f(params,vec)/(dvec[pos]*dvec[pos])
		#print(f"{coef[i]}f({np.array(vec)-np.array(vec_orig)})")
	return der

def der2mix(f,params,vec,dvec,pos,order):
	"""
	Finds 2nd order mixed derivative w.r.t
	parameters at pos[0],pos[1].
	
	Inputs:
	pos = [int,int]
	order = int
	vec,dvec,params = list/array/tuple 
	"""

	if order==2:
		pts = [-1.0,1.0]
		coef = [-1/2,1/2]

	elif order==4:
		pts = [-2.0,-1.0,1.0,2.0]
		coef = [1/12,-2/3,2/3,-1/12]
	elif order==6:
		pts = [-3.0,-2.0,-1.0,1.0,2.0,3.0]
		coef = [-1/60,3/20,-3/4,3/4,-3/20,1/60]
	elif order==8:
		pts = [-4.0,-3.0,-2.0,-1.0,1.0,2.0,3.0,4.0]
		coef = [1/280,-4/105,1/5,-4/5,4/5,-1/5,4/105,-1/280]
	else:
		raise ValueError(f"Can't find derivative with O(h^{order})")

	der=0
	vec_orig = deepcopy(np.array(vec))
	
	for i in range(len(coef)):
		vec = vec_orig*1.0
		vec[pos[0]] = vec[pos[0]]+pts[i]*dvec[pos[0]]
		vec_orig1 = deepcopy(vec)
		for j in range(len(coef)):
			vec = vec_orig1*1.0
			vec[pos[1]] = vec[pos[1]]+pts[j]*dvec[pos[1]]
			der += coef[i]*coef[j]*f(params,vec)/(dvec[pos[0]]*dvec[pos[1]])
			#print(f"{coef[i]*coef[j]}f({np.array(vec)-np.array(vec_orig)})")
	return der

def nabla_f(f,params,vec,dvec,order):
	"""
	Finds gradient of function f at vector
	"vec" i.e vec=[theta_val,dm2_val,...] is a position in parameter space
	dvec = [h_theta,h_dm2,...]
	params=[a,b,c...]
	f = f(params,vec)
	""" 
	if len(vec)!=len(dvec):
		raise ValueError(f"vec and dvec don't match. len(vec)={len(vec)} != len(dvec)={len(dvec)}")

	# Find derivatives
	del_f = np.array([],np.float64)
	for i in range(len(vec)):
		#print(vec)
		der = der1(f,params,vec,dvec,i,order)
		del_f = np.append(del_f,der)

	return np.array(del_f,np.float64)


def hessian(f,params,vec,dvec,order):
	"""
	Finds n-dimensional symmatric Hessian matrix where
	n = len(vec) = len(dvec)
	O(h^order) error on derivatives
	"""

	n = len(vec)
	if len(vec)!=len(dvec):
		raise ValueError("vec and dvec array lengths don't match")

	# Derivatives - matrix entries

	# Second order derivatives - diagonal
	#print("vevec",vec)
	#print("dvec",dvec)
	#print("params",f,params)
	#print(der2(f,params,vec,dvec,0,order))
	#print(der2(f,params,vec,dvec,1,order))
	der2_arr = [der2(f,params,vec,dvec,i,order) for i in range(n)]
	#print("der2_arr",der2_arr)
	# Mixed derivatives - off-diagonal
	indices = [i for i in range(n)]
	combinations = list(itertools.combinations(indices, 2))
	der2mix_arr = [der2mix(f,params,vec,dvec,pos,order) for pos in combinations]

	H = np.zeros((n,n),np.float64)

	sum_ind = [i+1 for i in range(n)]
	for i in range(n):
		for j in range(n):
			if i==j:
				H[i][j] = der2_arr[i]
				if type(der2_arr[i])==int: 
					raise ValueError("int")
			else:
				for s in sum_ind:
					if (i+j)==s:
						H[i][j] = der2mix_arr[s-1]
						if type(der2mix_arr[s-1])==int: 
							raise ValueError("int")
			#print(H)

	return np.array(H,np.float64)

