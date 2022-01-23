import numpy as np
import matplotlib.pyplot as plt
import os 
import random as rand
from copy import *
from functions import *
from differentiation import *
from inspect import signature


def parabolic1D_minimiser(x_min,x_max,f,params,vec,optimize_index,eps,unc=False):
	"""
	Minimizes a function with some fixed parameters
	via x-axis (value x).

	Function f must have input parameters formated as:
	f(params=[],vec=[])

	if params={} then f must have input parameters as:
	f(vec=[])
	
	vec = [value] and we're trying to find the optimal value
	type(va)
	optimize_index  = index (position) of the vec component which we try to optimise 

	For 3 values x0,x1,x2 take maximum, minimum and average
    of some input array with limits x_min,x_max
	"""

	# Take care of functions which don't have any extra "params", but only vec = []
	sig = signature(f)
	N_max = 1e4
	x_arr = [vec[optimize_index]]
	if len(sig.parameters)==1:
		print("Optimising function with 0 extra parameters")

		x = [x_min,(x_min+x_max)/2,x_max]
		y = []
		y = np.array([])
		for element in x:
			vec[optimize_index] = element
			y = np.append(y, f(vec)*1)

		x_3_old = 1e3

		i=0
		while abs(x_3_old-min(x)) > eps:
			x_3_old = deepcopy(min(x))
			print(x)

			x3 = 1/2 * ((x[2]*x[2]-x[1]*x[1])*y[0] + (x[0]*x[0]-x[2]*x[2])*y[1] + (x[1]*x[1]-x[0]*x[0])*y[2])/\
						((x[2]-x[1])*y[0] 		   + (x[0]-x[2])*y[1] 			+ (x[1]-x[0])*y[2])

			x_arr.append(x3)
			x = np.append(x,x3)
			y = np.array([])
			for element in x:
				vec[optimize_index] = element
				y = np.append(y, f(vec)*1)

			#index_maxval = int(np.where(y==max(y))[0][0])
			index_maxval = np.argmax(y)

			x = x[x != x[index_maxval]]
			y = y[y != y[index_maxval]]
		
			i +=1

			if i>N_max:
				raise ValueError("Not converging")

		x_min_y = x[np.where(y == min(y))[0]][0]
		
		denom0 = (x[0]-x[1])*(x[0]-x[2])
		denom1 = (x[1]-x[0])*(x[1]-x[2])
		denom2 = (x[2]-x[1])*(x[2]-x[0]) 

		a = y[0]/denom0 + y[1]/denom1 + y[2]/denom2
		sigma = np.sqrt(1/abs(2*a))
		if unc:
			return x_min_y,sigma,x_arr
		else:
			return x_min_y, x_arr

	# If the function has parameters
	else:
		x = [x_min,(x_min+x_max)/2,x_max]
		y = []
		y = np.array([])
		for element in x:
			vec[optimize_index] = element
			y = np.append(y, f(params,vec)*1)

		y_3_old = 1e3
		x_min_y = 1e9
		x3 = 1e3
		i=0
		while abs(y_3_old-max(y)) > eps:
			y_3_old = deepcopy(max(y))

			prev_min = deepcopy(x3)
			x3 = 1/2 * ((x[2]*x[2]-x[1]*x[1])*y[0] + (x[0]*x[0]-x[2]*x[2])*y[1] + (x[1]*x[1]-x[0]*x[0])*y[2])/\
						((x[2]-x[1])*y[0] 		   + (x[0]-x[2])*y[1] 			+ (x[1]-x[0])*y[2])
			
			if abs(x3-prev_min)<1e-14:
				x3 = deepcopy(x_min_y)
				for comp in x:
					x_arr.append(comp)
				x = np.append(x,x3)
				y = np.array([])
				for element in x:
					vec[optimize_index] = element
					y = np.append(y, f(params,vec)*1)

				index_maxval = np.argmax(y)
				index_min = np.argmin(y)
				x_min_y = x[index_min]

				x = np.delete(x,index_maxval)
				y = np.delete(y,index_maxval)
				i +=1
				# End the loop
				y_3_old=max(y)+eps/10


			else:

				for comp in x:
					x_arr.append(comp)
				x = np.append(x,x3)
				y = np.array([])
				for element in x:
					vec[optimize_index] = element
					y = np.append(y, f(params,vec)*1)

				index_maxval = np.argmax(y)
				index_min = np.argmin(y)
				x_min_y = x[index_min]

				x = np.delete(x,index_maxval)
				y = np.delete(y,index_maxval)
				i +=1
			#print(i)
			if i>N_max:
				raise ValueError("Not converging")

		#x_min_y = x[np.where(y == min(y))[0]][0]
		
		denom0 = (x[0]-x[1])*(x[0]-x[2])
		denom1 = (x[1]-x[0])*(x[1]-x[2])
		denom2 = (x[2]-x[1])*(x[2]-x[0]) 

		a = y[0]/denom0 + y[1]/denom1 + y[2]/denom2
		sigma = np.sqrt(1/abs(2*a))

		if unc:
			return x_min_y,sigma#,x_arr
		else:
			return x_min_y, x_arr

def univariate_method(f,arrays,params,vec,dvec,order,eps):
	"""
	Optimizes a function f in n-dimensional parameter space.
	For 3 values x0,x1,x2 take maximum, minimum and average
	of some input array in arrays
	n = len(arrays) == number of dimensions we're optimizing in.

	Inputs:
	arrays = [[theta_min,theta_max],[dm2_min,dm2_max],...]
	arrays in arrays must not have the same limits
	vec = [theta_val,dm2_val,...]
	"""
	x = deepcopy(vec)
	n = len(arrays)
	xs = [[x[0],(x[0]+x[-1])/2, x[-1]] for x in arrays]

	# Start with given input values of vec = [v0,v1..]
	# i.e vec = [theta_vaule, dm2_value]
	# Begin with the first vector component
	y = np.array([])
	for el in xs[0]:
		x[0] = el
		y = np.append(y,f(params,x))

	y_max = max(y)
	y_3_old_main = 1e9

	vec_unc = [0 for i in range(n)]
	x = deepcopy(vec)
	x_arr = [deepcopy(x)]
	j = 0

	while abs(y_3_old_main-y_max)>eps:
		y_3_old_main = deepcopy(y_max)
		for m in range(n):
			# Minimise function w.r.t m-th input parameter fixing other parameters
			x[m],vec_unc[m] = parabolic1D_minimiser(min(xs[m]),max(xs[m]),f,params,x,m,eps,True)

			y_max = f(params,x)
			x_arr.append(deepcopy(x))
			j +=1

		if j>1e3:
			print(x)
			raise ValueError("Not converging, main loop")

	index_min = np.argmin([f(params,x_vec) for x_vec in x_arr])
	x = x_arr[index_min]
	print(f"Univariate i={j}")
	print("Min",np.array(x))
	print("NLL(min)=",f(params,x),"\n")
	return [[x[i],vec_unc[i]] for i in range(len(x))], x_arr


def newton_minimiser(f,params,vec,dvec,order,eps,N):
	"""
	Simultaneously minimizes function f by varying parameters in vec
	using Newton's method.
	Returns: vec[theta_minimised,dm2_minimised,...]
	"""
	if len(vec)!=len(dvec):
		raise ValueError(f"vec and dvec don't match. len(vec)={len(vec)} != len(dvec)={len(dvec)}")
	n = len(vec)
	x = np.array(vec)*1.0
	x_arr = [vec]
	x_old = np.array([1e3,1e3])
	i=0
	while abs(np.linalg.norm(x)-np.linalg.norm(x_old))>eps:
		x_old = deepcopy(x)
		H = hessian(f,params,x,dvec,order)
		if np.linalg.det(H)==0:
			print("check x",x)
			print("H",H)
		H_inv = np.linalg.inv(H)
		
		del_f = nabla_f(f,params,x,dvec,order)
		delta = np.matmul(H_inv,del_f)
		
		for p in range(len(delta)):
			if np.sign(delta[p]) != np.sign(del_f[p]):
				delta[p] *= -1
		x = x-delta
		x_arr.append(x)

		if i>N:
			print("Newton Method isn't converging")
			unc_arr = [np.sqrt(abs(1/H[i,i])) for i in range(n)]
			results = [[x[i],unc_arr[i]] for i in range(n)]
			return results, x_arr
		#print(i)
		i +=1
	unc_arr = [np.sqrt(abs(1/H[i,i])) for i in range(n)]
	results = [[x[i],unc_arr[i]] for i in range(n)]

	print(f"Newton i={i}")
	print("Min",x)
	print("NLL(min)=",f(params,x),"\n")
	return results, x_arr

	
def quasi_newton_minimiser(f,params,vec,dvec,order,alph,alpha_factor,method,eps,N):
	"""
	Minimizes function f in n-dimensional parameter space
	using Quasi-Newton method.
	See Newton method for parameters.
	alph<<1
	alpha_factor states how much alph is reduced in the first step.
	i.e alph = alph*alpha_factor
	
	Method of minimisation: "DFP", "BFGS", BLAS?
	"""
	if len(vec)!=len(dvec):
		raise ValueError(f"vec and dvec don't match. len(vec)={len(vec)} != len(dvec)={len(dvec)}")

	n = len(vec)

	x = np.array(vec,float)*1.0
	x_arr = [vec]
	x_old = np.array([1e3,1e3],float)
	G = np.identity(n)
	alph_old = deepcopy(alph)

	if method=="DFP":
		i=0
		while abs(np.linalg.norm(x)-np.linalg.norm(x_old))>eps:

			if i==0:
				alph*=alpha_factor
			else:
				alph = alph_old
			x_old = deepcopy(x)

			# Find new x (new vec)
			del_f = nabla_f(f,params,x,dvec,order)

			step = alph*np.matmul(G,del_f)
			for p in range(len(step)):
				if np.sign(step[p]) != np.sign(del_f[p]):
					step[p] *= -1

			x = x-step
			x_arr.append(x)

			# Update parameters - update matrix G
			delta = x-x_old
			gamma = nabla_f(f,params,x,dvec,order)-nabla_f(f,params,x_old,dvec,order)
			out_delt = np.outer(delta,delta)
			out_gamma = np.outer(gamma,gamma)

			G = G + out_delt/(np.dot(gamma,delta)) - \
					np.matmul(np.matmul(G,out_gamma), G)/(np.matmul(np.matmul(gamma,G),gamma))
			

			if i>N:
				print(f"Quasi-Newton {method} isn't converging")

				return x, x_arr
			i +=1

		print(f"Quasi Newton {method}: i={i}")
		print("Min",np.array(x))
		print("NLL(min)=",f(params,x),"\n")
		return x, x_arr

	if method=="BFGS":
		i=0
		while abs(np.linalg.norm(x)-np.linalg.norm(x_old))>eps:

			if i==0:
				alph*=alpha_factor
			else:
				alph = alph_old	
	
			x_old = deepcopy(x)

			# Find new x (new vec)
			del_f = nabla_f(f,params,x,dvec,order)
			
			step = alph*np.matmul(G,del_f)
			for p in range(len(step)):
				if np.sign(step[p]) != np.sign(del_f[p]):
					step[p] *= -1

			x = x-step
			x_arr.append(x)

			# Update parameters - update matrix G
			delta = x-x_old
			gamma = nabla_f(f,params,x,dvec,order)-nabla_f(f,params,x_old,dvec,order)
			I = np.identity(n)
			
			G = np.matmul(np.matmul((I-np.outer(delta,gamma)/np.dot(gamma,delta)),G),
				(I-np.outer(gamma,delta)/np.dot(gamma,delta))) + \
				np.outer(delta,delta)/np.dot(gamma,delta) 
				
	

			if i>N:
				print(f"Quasi-Newton {method} isn't converging")
		
				return x, x_arr
			i +=1

		print(f"Quasi Newton {method}: i={i}")
		print("Min",np.array(x))
		print("NLL(min)=",f(params,x),"\n")
		return x, x_arr

	
def gradient_method(f,params,vec,dvec,order,alph,eps,N):
	"""
	Simultaneously minimizes function f by varying parameters in vec
	using gradient descent method.
	Returns: vec[theta_minimised,dm2_minimised,...]
	alph (float) is a learning parameter
	"""
	if len(vec)!=len(dvec):
		raise ValueError(f"vec and dvec don't match. len(vec)={len(vec)} != len(dvec)={len(dvec)}")
	n = len(vec)
	x = np.array(vec)*1.0
	x_arr = [vec]
	x_old = np.array([1e3,1e3])
	i=0
	while abs(np.linalg.norm(x)-np.linalg.norm(x_old))>eps:

		x_old = deepcopy(x)
		
		del_f = nabla_f(f,params,x,dvec,order)
		
	
		x = x-alph*del_f
		x_arr.append(x)

		if i>N:
			print("Gradient Method isn't converging")

			H = hessian(f,params,x,dvec,order)
			unc_arr = [np.sqrt(abs(1/H[i,i])) for i in range(n)]
			results = [[x[i],unc_arr[i]] for i in range(n)]
			return results, x_arr
		#print(i)
		i +=1
	
	H = hessian(f,params,x,dvec,order)
	unc_arr = [np.sqrt(abs(1/H[i,i])) for i in range(n)]
	results = [[x[i],unc_arr[i]] for i in range(n)]

	print(f"Gradient i={i}")
	print("Min",x)
	print("NLL(min)=",f(params,x),"\n")
	return results, x_arr





############################################
### Monte Carlo Method related functions####
############################################



def p_boltz(dE,T):
	"""
	Boltzmann probability function 
	"""
	k_b = 1
	return np.exp(-dE/(k_b*T))

def p_acc(dE,T):
	"""
	Acceptance function which says whether a step 
	in MC minimisation is accepted or not.
	k_b=1 so that dE/T ~ 1 is required
	"""
	k_b = 1
	if dE<=0:
		return True
	else:
		return False




def metropolis_alg_sampler(f,params,domain_lims,N,sig,T):
	"""
	Metropolis algorithm with Gaussian proposal function
	with width=sig.
	Finds N points sampled randomly - converging to some minimum.
	Boltzman energy distribution as p_acc
	Produces N points in n-D space.
	minimum - point with lowest value of function
	x_arr = array with all the points/steps taken
	"""
	n = len(domain_lims)
	minimum = 0
	#x_arr = []

	for p in range(N):

		if p==0:
			x = []
			for i in range(n):
				component_init = rand.uniform(domain_lims[i][0],domain_lims[i][1])
				x.append(component_init)
			minimum=deepcopy(x)
			#print(x)
		### Find new vector, more probable x_new ###

		# new proposed vector
		x_proposal = []
		for i in range(len(x)):
			x_comp1 = rand.normalvariate(x[i],sig)
			x_proposal.append(x_comp1)

		#p_acc criteria
		# find the difference between value of functions at old/proposed x
		dE = f(params,x_proposal) - f(params,x)

		if dE<0:
			minimum = deepcopy(x_proposal)
			x = deepcopy(x_proposal)
			#x_arr.append(x)
			#i+=1			 
		else:
			prob_still_accept = p_boltz(dE,T)
			rand_nb = rand.uniform(0,1)
			if rand_nb<prob_still_accept:
				x = deepcopy(x_proposal)
		i+=1
				#x_arr.append(x)
	return minimum#,np.array(x_arr)


def MC_minimiser(f,params,T_arr,domain_lims,N_random,N_steps,sig,zoom,eps):
	"""
	Monte Carlo Minimisation - Simulated Annealing
	Each point of N_random points is used as the initial position for 
	a random walk (with Gaussian proposal function) towards a region with
	lower values of the function. Each step is accepted via p_boltz.
	Lowest value is found for each walk and from each starting point.
	 - this is the global minima.
	Then the whole process is repeated at lower T (annealing).
	If better point is found - becomes a global minima.
	Then zoom into the region of global minima and repeat the process
	until you've found true global minima.

	N_random = number of random points over which MC is sampled at each T
	N_steps = number of steps from the random point towards the minimum
	frac_left_out = fraction of points left out from beginning the loop i.e
				wait to only include points close to the minimum
	domain_lims = [arrays of [min_val,max_val] for each dimension]
	sig = width of Gaussian - proposal function in Metropolis alg.
			depends on the size of features / on your function contour
	"""

	# Array of ranges for each dimension
	ranges = np.array([abs(x[1]-x[0]) for x in domain_lims])
	# Number of dimensions
	n = len(domain_lims)

	#arbitrary initial point
	global_minima = np.array([rand.uniform(domain_lims[i][0],domain_lims[i][1]) for i in range(len(domain_lims))])
	min_old = global_minima*1e3

	#lowest value recorded
	glob_min_arr = []

	a = 0
	while abs(max(np.array(global_minima)-np.array(min_old)))>eps:

		if a==5:
			N_steps = int(N_random*N_steps)
			N_random = 1

		min_old = deepcopy(global_minima)
		# Sort from highest to lowest T
		T_arr = np.sort(np.array(T_arr))[::-1]
		
		#arbitrary initial point inside a zoomed in domain == global minima inside each zoomed in domain (iteration)
		global_minima = [rand.uniform(domain_lims[i][0],domain_lims[i][1]) for i in range(len(domain_lims))]
		iterat = 0
		for T in T_arr:
			#print(f"iter={iterat}/{len(T_arr)}")
			for i in range(N_random):
				#print(f"a={a},iter={iterat}/{len(T_arr)}, {i}/{N_random}")
				
				min_proposal = metropolis_alg_sampler(f,params,domain_lims,N_steps,sig,T)
				#min_avg = [np.mean(x_arr[:,i]) for i in range(n)]
				#if a>5:
					#min_proposal = deepcopy(min_avg)

				dE = f(params,min_proposal)-f(params,global_minima)
				if dE<=0:
					#print(dE)
					global_minima = deepcopy(min_proposal)
			iterat +=1

		glob_min_arr.append(global_minima)
		#Change domain - zoom into the found global minima to repeat the process more locally
		for i in range(len(domain_lims)):
			domain_lims[i] = [global_minima[i]-0.5*ranges[i]/zoom, global_minima[i]+0.5*ranges[i]/zoom]

		# Update zoom-in parameters
		ranges = np.array([abs(x[1]-x[0]) for x in domain_lims])
		sig = sig/float(zoom/1.5)
		T_arr = T_arr/float(zoom)
		print("Montecarlo, zoom=",a,"Iterations:",(a+1)*len(T_arr)*N_random*N_steps)
		print("Min",global_minima)
		print("NLL(min)=",f(params,global_minima))
		print("delta|x_min|=",abs(max(np.array(global_minima)-np.array(min_old))))
		#print(abs(max(np.array(global_minima)-np.array(min_old))))
		#print("Ranges",domain_lims)
		a +=1
		#print("Current Global minimum",global_minima)
		#print("Current NLL(min)=",f(params,global_minima))

	min_values = [f(params,vec) for vec in glob_min_arr]
	index_total_global_min = np.argmin(min_values)
	global_minimum = glob_min_arr[index_total_global_min]
	#print("Global minimum",global_minimum)
	#print("NLL(min)=",f(params,global_minimum))
	#print(a*N_steps*N_random*len(T_arr))
	print(a)
	print("MC TOTAL ITERATIONS:",a*len(T_arr)*N_random*N_steps)
	print("Min",global_minimum)
	print("NLL(min)=",f(params,global_minimum))
	return global_minimum








