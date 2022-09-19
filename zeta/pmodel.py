#pmodeltsg.py
#p-model from Meneveau & Sreenevasan, 1987 & Malara et al., 2016
#Author: R.R.Rosa & N. Joshi
#Version: 1.6
#Date: 11/04/2018

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm, genextreme
from sklearn.cluster import KMeans 
import tqdm as tqdm
from scipy.stats import skew, kurtosis
 
 
def pmodel(noValues=4096, p=0.4999, slope=[]):
    noOrders = int(np.ceil(np.log2(noValues)))
    noValuesGenerated = 2**noOrders
    
    y = np.array([1])
    for n in range(noOrders):
        y = next_step_1d(y, p)
    
    if (slope):
        fourierCoeff = fractal_spectrum_1d(noValues, slope/2)
        meanVal = np.mean(y)
        stdy = np.std(y)
        x = np.fft.ifft(y - meanVal)
        phase = np.angle(x)
        x = fourierCoeff*np.exp(1j*phase)
        x = np.fft.fft(x).real
        x *= stdy/np.std(x)
        x += meanVal
    else:
        x = y
    
    return x[0:noValues], y[0:noValues]
 
     
def next_step_1d(y, p):
    y2 = np.zeros(y.size*2)
    sign = np.random.rand(1, y.size) - 0.5
    sign /= np.abs(sign)
    y2[0:2*y.size:2] = y + sign*(1-2*p)*y
    y2[1:2*y.size+1:2] = y - sign*(1-2*p)*y
    
    return y2
 
 
def fractal_spectrum_1d(noValues, slope):
    ori_vector_size = noValues
    ori_half_size = ori_vector_size//2
    a = np.zeros(ori_vector_size)
    
    for t2 in range(ori_half_size):
        index = t2
        t4 = 1 + ori_vector_size - t2
        if (t4 >= ori_vector_size):
            t4 = t2
        coeff = (index + 1)**slope
        a[t2] = coeff
        a[t4] = coeff
        
    a[1] = 0
    
    return a


###################################################
# Complementary by: Rubens A. Sautter(2021)
# Endo-Exo generators:

def generateUniformExo(N=4096):
  # 0.675 < p < 0.8 , 0.65 < beta < 0.75
  p = (0.8-0.675)*np.random.rand()+0.675
  beta = (0.75-0.65)*np.random.rand()+0.65
  #print(p,beta)
  x, dx = pmodel (N, p, beta)
  return np.array(dx), p, beta

def generateUniformEndo(N=4096):
  #  0.525 < p < 0.65 , 0.35 < beta < 0.45
  p = (0.65-0.525)*np.random.rand()+0.525
  beta = (0.45-0.35)*np.random.rand()+0.35
  x, dx = pmodel (N, p, beta)
  return np.array(dx), p, beta
  

####################################################
# Experimental spectral normalization: (R.Sautter-2022)
# 

def specNorm(series,newBeta):
	ft = np.fft.fft(series)
	
	# finding the power-law:
	psd = np.real(ft.copy()*np.conj(ft.copy()))
	freq = np.fft.fftfreq(len(series))
	psd, freq = psd[freq>0.0], freq[freq>0]
	[oldBeta, _] = np.polyfit(np.log(freq),np.log(psd),deg =1)
	oldBeta = -oldBeta
	
	pot = (oldBeta-newBeta)/2
	freq = np.abs(np.fft.fftfreq(len(series)))
	
	print(pot)
	
	# rebuilding the spectrum from the older log-log rule and the new log-log rule
	ft[1:] = ft[1:]*(freq[1:]**pot) 
	
	return np.real(np.fft.ifft(ft))
	
		  
  
  
