from MFDFA import singspect
from MFDFA import  MFDFA
import numpy as np
#import numba
from numba import jit, prange
from scipy.optimize import curve_fit
import pandas as pd

import warnings
warnings.filterwarnings("ignore")



def quadratic(x,a,b,c):
    return a*(x**2)+b*(x)+c

def getPolynomial2(alpha,falpha):
	return curve_fit(quadratic,alpha,falpha)[0]
		
def deltaAlpha(alpha):
	return np.average(np.max(alpha,axis=1)-np.min(alpha,axis=1))
	
def logisticHalfAvg(x,k=7):
	return 1/(1+np.exp(k*(0.5-x)))

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
		'alpha':alpha,
		'falpha':falpha
		}

def getAverageSing(serie):
	'''
	========================================================================
	Retrieves the singularity spectrum with median delta alpha
	========================================================================
	Input:
	serie - time series
	========================================================================
	Output:
	alphas and f(alphas)
	========================================================================
	Wrote by: Rubens A. Sautter (12/2022)
	'''
	a, fa, lda = autoMFDFA(serie,nqs=20)		
	metrics = [singularitySpectrumMetrics(a[i],fa[i]) for i in range(len(a))]
	#metrics = pd.DataFrame(metrics)
	#metrics = metrics.sort_values(by=['delta_alpha'])
	#median = metrics.iloc[metrics.shape[0]//2]
	#a, fa = median["alpha"],median["falpha"]
	
	return np.average(a,axis=0), np.average(fa,axis=0)

def selectScales(timeSeries,threshold=1e-3,nscales=30):
	'''
	========================================================================
	Select random scales to apply MFDFA, from a set of frequencies with  
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
	pos = (freq>1e-13)
	psd = psd[pos]
	freq = freq[pos]
	maxPSD = np.max(psd)
	psd = psd/maxPSD
	scales = 1/np.abs(freq[(psd >threshold)])
	scales = scales.astype(np.int)
	scales = np.unique(scales)
	scales = np.sort(scales)
	return np.random.choice(scales,nscales)

def normalize(d):
	data = d-np.average(d)
	data = data/np.std(data)
	return data
	

#@jit(forceobj=True,parallel=True)
def autoMFDFA(timeSeries,qs=np.arange(5,15,2), scThresh=1e-2, nqs = 10, nsamples=40, nscales=20,magnify=5):
	'''
	========================================================================
	Complementary method to measure multifractal spectrum.
	Base MFDFA implementation: https://github.com/LRydin/MFDFA

	(I)	The time series is normalized according to its global average and global standard deviation
	(II)	A set of scales is randomly selected	
	(III)	MFDFA is applied over the normalized time-series  ('nsamples' times) 
	(IV)	Delta Alpha is measured
	(V)	Singularity spectrum with delta alpha outliers (greater than average + standard deviation) are removed 
	(VI)	logistic function is measured
	=========================================================================
	Input:
	timeSeries - serie of elements (np.array)
	qs - set of hurst exponent ranges
	scThresh - threshold to select DFA scales (see selectScales function)
	nqs - number of hurst exponents measured
	nsamples - number of singularity spectrum samples per hurst exponent set (q)
	nscales - number of random scales
	magnify - logistic function constant - greater values increases the differences between extremes

	=========================================================================
	Output:
	alphas, falphas - set of multifractal spectrum
	LDA -  Logistic of the average delta alpha
	=========================================================================
	Wrote by: Rubens A. Sautter (02/2022)
	'''
	
	# signularity spectra of the series
	alphas,falphas = [], []
	
	# signularity spectra of surrogate series
	salphas,sfalphas = [], []
	
	data = normalize(timeSeries)
	
	deltas = []
	 
	for i in range(nsamples):
		scales = selectScales(data,threshold=scThresh,nscales=nscales)
		for it  in prange(len(qs)):
			qrange = qs[it]
			q = np.linspace(-qrange,qrange,nqs)
			q = q[q != 0.0]
			lag,dfa = MFDFA(data, scales, q=q)
			alpha,falpha = singspect.singularity_spectrum(lag,dfa,q=q)
			if np.isnan(alpha).any() or np.isnan(falpha).any():
				continue
			if (falpha>1.5).any():
				continue
			alphas.append(alpha)
			falphas.append(falpha)
			deltas.append(deltaAlpha([alpha]))
			
	# Remove outlier:
	alphas,falphas = np.array(alphas),np.array(falphas) 
	alphas = alphas[ (deltas < np.average(deltas)+np.std(deltas)) ]
	falphas = falphas[(deltas < np.average(deltas)+np.std(deltas))]
	delta = deltaAlpha(alphas)
	
	'''
	#Surrogate testing
	s = surrogate(data)
	deltas = []
	for i in range(nsamples):
		scales = selectScales(s,threshold=scThresh,nscales=nscales)
		for it  in prange(len(qs)):
			qrange = qs[it]
			q = np.linspace(-qrange,qrange,nqs)
			q = q[q != 0.0]
			lag,dfa = MFDFA(s, scales, q=q)
			alpha,falpha = singspect.singularity_spectrum(lag,dfa,q=q)
			if np.isnan(alpha).any() or np.isnan(falpha).any():
				continue
			if deltaAlpha([alpha])>delta:
				# not well shuffled or bad scales
				continue
			if (falpha>1.5).any():
				continue
			salphas.append(alpha)
			sfalphas.append(falpha)
			deltas.append(deltaAlpha([alpha]))
	ds = deltaAlpha(np.array(salphas))		
	'''
	d = deltaAlpha(np.array(alphas)) 
	if len(alphas)>2:
		return alphas, falphas, logisticHalfAvg(d,magnify) 
	else:
		raise Exception("Threshold should be lower! No singularity spectrum found")

