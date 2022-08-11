from MFDFA import singspect
from MFDFA import  MFDFA
import numpy as np
#import numba
from numba import jit, prange
from scipy.optimize import curve_fit

def quadratic(x,a,b,c):
    return a*(x**2)+b*(x)+c

def getPolynomial2(alpha,falpha):
	return curve_fit(quadratic,alpha,falpha)[0]
		
def singularConcavity(alpha,falpha):
	'''
	========================================================================
	Measures the concavity of the singularity spectrum
	========================================================================
	Input:
	alpha - x values of the singularity spectrum (np.array that must have same lenght of falpha)
	falpha - y values of the singularity spectrum (np.array that must have same lenght of alpha)
	========================================================================
	Output:
	Dictionay with the measures delta_alpha,max_f, delta_f and asymmetry
	========================================================================
	Wrote by: Rubens A. Sautter (02/2022)
	'''
	sol = getPolynomial2(alpha,falpha)
	return -1.0/sol[0]
	

def singularitySpectrumMetrics(alpha,falpha):
	'''
	========================================================================
	Measures of the singularity spectrum
	========================================================================
	Input:
	alpha - x values of the singularity spectrum (np.array that must have same lenght of falpha)
	falpha - y values of the singularity spectrum (np.array that must have same lenght of alpha)
	========================================================================
	Output:
	Dictionay with the measures delta_alpha,max_f, delta_f and asymmetry
	========================================================================
	Wrote by: Rubens A. Sautter (02/2022)
	'''
	maxFa = np.argmax(falpha)
	delta = np.max(alpha)-np.min(alpha)
	assym = np.inf if np.abs(falpha[0]-falpha[len(falpha)-1])<1e-15 else np.abs(falpha[0]-falpha[len(falpha)-1])
    
	return {'delta_alpha':delta,
		'max_f':falpha[maxFa],
		'delta_f': (np.max(falpha)-np.min(falpha)),
		'asymmetry': assym,
		'concavity': singularConcavity(alpha,falpha)
		}

def selectScales(timeSeries,threshold=1e-3):
	'''
	========================================================================
	Select random scales to apply MFDFA, from a set of scales with  
	large Power Spectrum Density values 
	========================================================================
	Input:
	timeSeres - input time series (np.array)
	threshold - determines the minimum PSD of the series (0 to 1)
	========================================================================
	Output:
	scales - set of scales randomly selected
	========================================================================
	Wrote by: Rubens A. Sautter (02/2022)
	'''
	psd = np.fft.fft(timeSeries)
	freq = np.fft.fftfreq(len(timeSeries))
	psd = np.real(psd*np.conj(psd))
	pos = (freq>threshold)
	psd = psd[pos]
	freq = freq[pos]
	maxPSD = np.max(psd)
	psd = psd/maxPSD
	scales = np.abs(1/freq[(psd >threshold)])
	scales = scales.astype(np.int)
	scales = np.unique(scales)
	scales = np.sort(scales)
	return scales

def normalize(d):
	data = d-np.average(d)
	data = data/np.std(data)
	data = np.cumsum(data)
	data = data-np.average(data)
	data = data/np.std(data)
	return data

#@jit(forceobj=True,parallel=True)
def autoMFDFA(timeSeries,qs=np.linspace(3,15,10), scThresh=1e-4,nqs = 10):
	'''
	========================================================================
	Complementary method to measure multifractal spectrum.
	Base MFDFA implementation: https://github.com/LRydin/MFDFA

	(I)	A set of scales is randomly selected
	(II)	The time series is normalized according to its global average and global standard deviation	
	(III)	Cumulative sum of the series
	(IV)	MFDFA is applied over the normalized time-series, with scales from step (I)
	(V)	A quadratic polynomial is fitted for each multifractal spectrum,
			if the first component (x^2) is positive, then the spectrum is discarded.
	=========================================================================
	Input:
	timeSeries - serie of elements (np.array)
	qs - set of hurst exponent ranges
	scThresh - threshold to select DFA scales (see selectScales function)
	nqs - number of hurst exponents measured

	=========================================================================
	Output:
	alphas, falphas - set of multifractal spectrum
	concavity - -1/a, where a is the first term of the quadratic equation (y = ax^2+bx+c ) 
	=========================================================================
	Wrote by: Rubens A. Sautter (02/2022)
	'''
	nSeries = len(qs)
	shape = (nSeries,nqs)
	alphas,falphas = [], []
	data = normalize(timeSeries)
	scales = selectScales(timeSeries,threshold=scThresh)
	for it  in prange(len(qs)):
		qrange = qs[it]
		q = np.linspace(-qrange,qrange,nqs)
		q = q[q != 0.0]
			
		lag,dfa = MFDFA(data, scales, q=q)
		alpha,falpha = singspect.singularity_spectrum(lag,dfa,q=q)
		if np.isnan(alpha).any() or np.isnan(falpha).any():
			continue
		sol = getPolynomial2(alpha,falpha)
		if sol[0]>= 0.0:
			continue
		if (alpha<0.0).any():
			continue
		alphas.append(alpha)
		falphas.append(falpha)
	ralphas,rfalphas = np.ravel(alphas),np.ravel(falphas)
	seq = np.argsort(ralphas)
	if len(ralphas)>2:
		return alphas, falphas, singularConcavity(ralphas[seq],rfalphas[seq])
	else:
		raise Exception("Threshold should be lower! No singularity spectrum found")

