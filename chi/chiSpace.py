from .spectralEntropy import *
from .singularitySpectrum import *
from .qqMetric import *
import numpy as np

from pkgutil import extend_path
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns 
from matplotlib.lines import Line2D

def chiSpace(data,qs=np.arange(5,15,2), scThresh=1e-2, nqs = 10, nsamples=40, nscales=20,magnify=5):
	'''
	========================================================================
	
	Generates an dictionary with the Gaussian Distance metric (GQQ) and the Logistic Delta Alpha (LDA)
	
	========================================================================
	Input:
	data - time series (0D+1)
	qs - Hurst exponents extremes
	scThresh - scale threshold of autoMFDFA
	nqs - number of hurst exponents
	nsamples - number of scale samples to compute delta alpha
	magnify - logistic function parameter (for delta alpha normalization)
	========================================================================
	Output:
	Dictionary with keywords 'GQQ' and 'LDA', where:
	GQQ - Gaussian Q-Q Plot distance
	LDA - Logistic Delta Alpha
	========================================================================
	Wrote by: Rubens A. Sautter (08/2022)
	'''
	_,_, lda = autoMFDFA(data,qs=qs,scThresh=scThresh,nqs=nqs,nsamples=nsamples, nscales=nscales,magnify=magnify)
	return {"GQQ":qqGaussianDistance(data), "LDA":lda}
	
def plot(figsize=(12,12)):
	'''
	========================================================================
	Plots the Chi-Space using matplotlib and seaborn, it also shows the histogram of GQQ and LDA
	========================================================================
	figsize - size of of the figure frame
	========================================================================
	Wrote by: Rubens A. Sautter (12/2022)
	'''
	# Loading data
	endo = pkgutil.get_data('Chi-Space.results', 'zEndo.csv')
	print("Got it!")
	endo = pd.DataFrame(StringIO(endo.decode()))
	print("Endo:",endo)
	
	endo = pd.read_csv('results/zEndo.csv')
	exo = pd.read_csv('results/zExo.csv')
	reds = pd.read_csv('results/zRed.csv')
	zL = pd.read_csv('results/zLorenz.csv')
	
	# Plotting
	
	gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1],wspace=0,hspace=0)

	plt.figure(figsize=figsize)

	plt.subplot(gs[0,0])
	sns.kdeplot(data=zL, x="LDA", y="GQQ",fill=True,color='goldenrod',thresh=0.02,alpha = 0.3)
	#plt.scatter(zL["LDA"],zL["GQQ"])
	sns.kdeplot(data=endo, x="LDA", y="GQQ",fill=True,color='cyan',thresh=0.05,alpha = 0.5)
	sns.kdeplot(data=exo, x="LDA", y="GQQ",fill=True,color='blue',thresh=0.005,alpha = 0.7)
	sns.kdeplot(data=reds, x="LDA", y="GQQ",fill=True,color='red',thresh=0.05,alpha = 0.5)

	#wn = plt.scatter(0.0,0.0,s=200,color='k')

	plt.xlim(0.0,1.0)
	plt.ylim(0.0,1.0)
	handles = [Line2D([0], [0], color='cyan', lw=6),
		             Line2D([0], [0], color='blue', lw=6),
		             Line2D([0], [0], color='red', lw=6),
		             Line2D([0], [0], color='goldenrod', lw=6)
		             ]
	plt.legend(handles,['Endogenous p-model','Exogenous p-model','Red Noise','Lorenz Synchronization'],loc=2)
	plt.xlabel('')
	plt.xticks([])
	plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
	plt.ylabel(r"$G_{QQ}$",fontsize=20)


	plt.subplot(gs[0,1])
	ax = sns.histplot(reds,color='red',fill=False,y='GQQ',lw=3.5,bins=10,stat="percent",kde=True,kde_kws={'cut': 3})
	ax.containers[0].remove()
	ax = sns.histplot(zL,color='goldenrod',fill=False,y='GQQ',lw=3.5,bins=10,stat="percent",kde=True,kde_kws={'cut': 3})
	ax.containers[0].remove()
	ax = sns.histplot(exo,color='blue',fill=False,y='GQQ',lw=3.5,bins=10,stat="percent",kde=True,kde_kws={'cut': 3})
	ax.containers[0].remove()
	ax = sns.histplot(endo,color='cyan',fill=False,y='GQQ',lw=3.5,bins=10,stat="percent",kde=True,kde_kws={'cut': 3})
	ax.containers[0].remove()

	plt.ylim(0,1)
	plt.yticks([])
	plt.ylabel('')
	plt.xticks([5,10,15,20,25],["5%","10%","15%","20%","25%"])
	plt.xlabel('')


	plt.subplot(gs[1,0])
	ax = sns.histplot(reds,color='red',fill=False,x='LDA',lw=3.5,bins=10,stat="percent",kde=True,kde_kws={'cut': 3})
	ax.containers[0].remove()
	ax = sns.histplot(zL,color='goldenrod',fill=False,x='LDA',lw=3.5,bins=10,stat="percent",kde=True,kde_kws={'cut': 3})
	ax.containers[0].remove()
	ax = sns.histplot(exo,color='blue',fill=False,x='LDA',lw=3.5,bins=10,stat="percent",kde=True,kde_kws={'cut': 3})
	ax.containers[0].remove()
	ax = sns.histplot(endo,color='cyan',fill=False,x='LDA',lw=3.5,bins=10,stat="percent",kde=True,kde_kws={'cut': 3})
	ax.containers[0].remove()

	plt.xlim(0,1)
	plt.yticks([10,20,30,40,50],["10%","20%","30%","40%","50%"])
	plt.xticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
	plt.ylabel('')
	plt.xlabel(r"$L(\Delta \alpha)$",fontsize=20)
	plt.tight_layout()
	#plt.savefig("Zeta.png",dpi=400)
	#plt.show()
