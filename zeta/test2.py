import singularitySpectrum 
import numpy as np
import matplotlib.pyplot as plt
import spectralEntropy
import pmodel

pot =13
data,_,_ = pmodel.generateUniformEndo(2**pot)
a, f, c  = singularitySpectrum.autoMFDFA(data)
wn=data.copy()
plt.figure()
for i in range(len(a)):
	if i==0:
		plt.plot(a[i],f[i],color='b',marker='.',label='Endogenous')
	else:
		plt.plot(a[i],f[i],color='b',marker='.')

print("Endo", c)

print()
data,_,_ = pmodel.generateUniformExo(2**pot)
rn = data.copy()
a, f, c = singularitySpectrum.autoMFDFA(data)
for i in range(len(a)):
	if i==0:
		plt.plot(a[i],f[i],color='r',marker='.',label='Exogenous')
	else:
		plt.plot(a[i],f[i],color='r',marker='.')
plt.legend()	
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$f(\alpha)$")


print("Exo",c)


plt.figure()
plt.plot(singularitySpectrum.normalize(wn),color='b',label='Endogenous')
plt.plot(singularitySpectrum.normalize(rn),color='r',label='Exogenous')
plt.legend()

plt.show()
