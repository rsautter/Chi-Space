import .spectralEntropy
import numpy as np
import .singularitySpectrum

def zetaSpace(data,qs=np.array([5,6,7,8,9])):
	a, fa, _, _, _ = autoMDFDA(data,qs=qs)
	return {"spectral_entropy":SpectralEntropy(data), "delta_alpha":singularitySpectrumMetrics(a,fa)["delta_alpha"]}
