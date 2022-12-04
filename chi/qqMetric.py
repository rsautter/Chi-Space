import numpy as np
from scipy import stats

def qqGaussianDistance(serie):
    normalized = serie-np.average(serie)
    normalized = normalized/np.std(normalized)
    (quantiles, ordered_values), (slope, clin, sCoefficientDetermination) = stats.probplot(normalized)
    return 1-(sCoefficientDetermination**2)
