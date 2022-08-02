from MFDFA import singspect
from MFDFA import  MFDFA
import numpy as np
#import numba
from numba import jit, prange
from scipy.optimize import curve_fit

def singularitySpectrumMetrics(alpha,falpha):
	'''
	Wrote by: Rubens A. Sautter (02/2022)
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
	'''
	maxFa = np.argmax(falpha)
	delta = np.max(alpha)-np.min(alpha)
    
	return {'delta_alpha':delta,
		'max_f':falpha[maxFa],
		'delta_f': (np.max(falpha)-np.min(falpha)),
		'asymmetry': np.abs(falpha[0]-falpha[len(falpha)-1])
		}

def selectScales(timeSeries,threshold=1e-4,nScales=10):
	'''
	Wrote by: Rubens A. Sautter (02/2022)
	========================================================================
	Select random scales to apply MFDFA, from a set of scales with  
	large Power Spectrum Density values 
	========================================================================
	Input:
	timeSeres - input time series (np.array)
	threshold - determines the minimum PSD of the series
	nScales - number of scales sampled
	========================================================================
	Output:
	scales - set of scales randomly selected
	========================================================================
	'''
	psd = np.fft.fft(timeSeries)
	freq = np.fft.fftfreq(len(timeSeries))
	psd = np.real(psd*np.conj(psd))
	maxPSD = np.max(psd)
	psd = psd/maxPSD
	scales = np.abs(1/freq[(psd >1e-4) & (freq>1e-20)])
	scales = scales.astype(np.int)
	scales = np.unique(scales)
	np.random.shuffle(scales)
	scales = scales[0:min(nScales,len(scales)-1)]
	return scales
	
def quadratic(x,a,b,c):
    return a*(x**2)+b*(x)+c

#@jit(forceobj=True,parallel=True)
def autoMDFDA(timeSeries,qs=np.array([5,6,7,8,9]), scThresh=1e-4,nScales=50,nqs = 14):
	'''
	Wrote by: Rubens A. Sautter (02/2022)
	========================================================================
	Complementary method to measure multifractal spectrum.
	Original implementation: https://github.com/LRydin/MFDFA

	A quadratic polynomial is fitted for each multifractal spectrum,
	if the first component (x^2) is positive, then the spectrum is discarded.
	The best spectrum is selected from the average end-points (max and min alpha)
	=========================================================================
	Input:
	timeSeries - serie of elements (np.array)
	qs - 
	scThresh - threshold to select DFA scales (see selectScales function)
	nScales - number of scales  (see selectScales function)
	nqs - 

	=========================================================================
	Output:
	alpha, falpha - average singularity spectrum 
	alphas, falphas - set of multifractal spectrum
	weight - Euclidean distance of end-points 
	=========================================================================
	'''
	nSeries = len(qs)
	shape = (nSeries,nqs)
	alphas,falphas,signSum,metrics = np.zeros(shape),np.zeros(shape),np.zeros(nSeries),np.zeros(nSeries)
	for it  in prange(len(qs)):
		qrange = qs[it]
		q = np.linspace(-qrange,qrange,nqs)
		q = q[q != 0.0]
		scales = selectScales(timeSeries,threshold=scThresh,nScales=nScales)
		
		lag,dfa = MFDFA(timeSeries, scales, q=q)
		alpha,falpha = singspect.singularity_spectrum(lag,dfa,q=q)
		if np.isnan(alpha).any() or np.isnan(falpha).any():
			continue
		sol,_ = curve_fit(quadratic,alpha.copy(),falpha.copy())
		print(sol)
		if sol[0]> 0.0:
			lmin,lmax = np.min(falpha),np.max(falpha)
			falpha = -falpha+lmax+lmin
		print(alpha)
		if (alpha<0.0).any():
			continue
		print("Saving")
		index = it
		metrics[index] = singularitySpectrumMetrics(alpha,falpha)['asymmetry']
		alphas[index] = alpha
		falphas[index] = falpha  
	criteria = np.argsort(metrics)
	alphas,falphas,metrics = alphas[criteria],falphas[criteria],metrics[criteria]
	return alphas[0],falphas[0], alphas, falphas, metrics

