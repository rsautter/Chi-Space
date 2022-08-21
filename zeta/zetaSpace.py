from .spectralEntropy import *
from .singularitySpectrum import *
import numpy as np

def zetaSpace(data,qs=np.linspace(3,15,10), scThresh=1e-4,nqs = 10,**kwargs):
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
	Dictionary with keywords 'ISE' and 'LDA', where:
	ISE - Inverse Spectral Entropy
	LDA - Logistic Delta Alpha
	========================================================================
	Wrote by: Rubens A. Sautter (08/2022)
	'''
	_,_, lda = autoMFDFA(data,qs=qs,scThresh=scThresh,nqs=nqs)
	return {"ISE":1-spectralEntropy(data,*kwargs), "LDA":lda}
