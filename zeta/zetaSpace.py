from .spectralEntropy import *
from .singularitySpectrum import *
import numpy as np

def zetaSpace(data,qs=np.linspace(3,7,5), scThresh=1e-4,nqs = 10,**kwargs):
	'''
	========================================================================
	
	Generates an dictionary with the Entropy and Singularity Spectrum Concavity
	
	========================================================================
	Input:
	data - time series (0D+1)
	qs - Hurst exponents extremes
	scThresh - scale threshold of autoMFDFA
	nqs - number of hurst exponents
	**kwargs - PSD parameters
	========================================================================
	Output:
	Dictionary with keywords: 'spectral_entropy' and 'concavity'
	========================================================================
	Wrote by: Rubens A. Sautter (08/2022)
	'''
	_,_, conc = autoMFDFA(data,qs=qs,scThresh=scThresh,nqs=nqs)
	return {"spectral_entropy":spectralEntropy(data,*kwargs), "concavity":conc}
