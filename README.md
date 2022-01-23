# A-log-likelihood-fit-for-extracting-neutrino-oscillation-parameters
Minimised the negative log-likelihood fit to extract neutrino oscillation parameters from the set of simulated T2K data.  Compared the efficiency and accuracy of Newton, DFP, BFGS, univariate, gradient descent and my own iterative Monte-Carlo minimisation methods.

Hi, these are my coding files for running the project "A log likelihood fit for extracting neutrino oscillation parameters.

You need all the files in the same folder, since they're linked up as libraries for certain functions.


FILES WHICH SERVE AS LIBRARIES OF FUNCTIONS:

differentiation.py
has functions connected to differentation - hessian calculator, derivatives calculator function etc.

functions.py
has various functions in it that are used throughout the files.

minimisers.py
has algorithms for minimisation methods in it


FILES WHICH GIVE RESULTS:

testing_minimisers.py
gives out results of testing the minimisation methods.

part3.py, part4.py, part5.py
are "main" folders where running the code will yield results and show
plots required for the tasks in the script for that project.

Chi2_pval.py
Calculates p-values and reduced chi2 values for the 1D,2D,3D optimisation of NLL.

global_minima.py
calculates errors (via bisection method, where NLL changes for 1/2) on 
results obtained for 1D,2D,3D NLL minimised function.
