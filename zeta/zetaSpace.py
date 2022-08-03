from .spectralEntropy import *
from .singularitySpectrum import *
import numpy as np

def zetaSpace(data,qs=np.array([5,6,7,8,9])):
	a, fa, _, _, _ = autoMFDFA(data,qs=qs)
	return {"spectral_entropy":spectralEntropy(data), "delta_alpha":singularitySpectrumMetrics(a,fa)["delta_alpha"]}
