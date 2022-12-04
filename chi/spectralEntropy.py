import numpy as np
from scipy.signal import welch

def getPSD(data):
	psd = np.fft.fft(data)
	psd = np.real(psd*np.conj(psd))
	freq = np.fft.fftfreq(len(data))
	return freq, psd
	
def getPowerLaw(data):
	freq, psd = getPSD(data)
	posFreq = (freq>0.0)
	beta, _ = np.polyfit(np.log(freq[posFreq]),np.log(psd[posFreq]),deg=1)
	return beta

def spectralEntropy(data, method='fft',start=0,end=None,**kwargs):
	'''
	Wrote by: Rubens A. Sautter (05/2021)
	========================================================================
	Measures the spectral entropy 'data', in range frequency between 'start' and 'end', with a given method
	
	=========================================================================
	Input:
	data - a time series (np.array) - len(data) must be greater than 7
	method - string with the spectrum measure technique ('fft' or 'welch')
	start - index of the minimum spectrum frequency to measure the entropy (must be lesser than len(data) ) 
	end - index of the maximum spectrum frequency to measure the entropy (must be greater than 0 and less than len(data)+1)
	**kwargs - extra arguments which will be passes to fft or welch methods
	
	=========================================================================
	Output:
	h - Spectral entorpy (value between 0 and 1)
	=========================================================================
	'''
	n = len(data)
	if n<8:
		raise Exception("Series must have at least 8 elements. Got: "+str(n)+" elements in the series.")
	psd = []
	if method == 'fft':
		psd = np.fft.fft(data,**kwargs)
		psd = np.real(psd*np.conj(psd))
	elif method == 'welch':
		psd = welch(data,**kwargs)
	else:
		raise Exception("Power Spectrum Density Method unknown: "+str(method))
	if end is None:
		endSpec = len(psd)
	else:
		endSpec = end
	psd = psd[start:endSpec]
	# To avoid log(0) errors
	psd = psd[psd>1e-6]
	psd = psd/np.sum(psd)
	h = np.sum(-psd*np.log(psd))/np.log(len(psd))
	return h
