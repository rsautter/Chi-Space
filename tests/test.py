import singularitySpectrum 
import numpy as np
import matplotlib.pyplot as plt
import spectralEntropy
import cNoise

pot =13
data = cNoise.cNoise(0.0,[2**pot])
a, f, c  = singularitySpectrum.autoMFDFA(data)
wn=data.copy()
plt.figure()
for i in range(len(a)):
	plt.plot(a[i],f[i],color='b',marker='.')

print("White Noise", c)

print()
data = cNoise.cNoise(2.0,[2**pot])
rn = data.copy()
a, f, c = singularitySpectrum.autoMFDFA(data)
for i in range(len(a)):
	plt.plot(a[i],f[i],color='r',marker='.')
	
print("Red Noise",c)

plt.figure()
plt.plot(wn/np.std(wn),color='b')
plt.plot(rn/np.std(rn),color='r')


plt.figure()
plt.plot(singularitySpectrum.normalize(wn),color='b')
plt.plot(singularitySpectrum.normalize(rn),color='r')

plt.show()
