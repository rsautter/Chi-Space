import numpy as np
import matplotlib.pyplot as plt
import spectralEntropy
import cNoise
import pmodel


'''
pot =13
data = cNoise.cNoise(2.0,[2**pot])
out = pmodel.specNorm(data,2.0)

wn=data.copy()
plt.figure()
plt.plot(data)
plt.figure()
plt.plot(out)
plt.title("renorm")

'''
pot =13 
data,_,_ = pmodel.generateUniformEndo(2**pot)

print(spectralEntropy.getPowerLaw(data))

plt.figure()
plt.subplot(1,3,1)
plt.plot(data)
plt.subplot(1,3,2)
out = pmodel.specNorm(data,0.0)
plt.plot(out)
plt.title("renorm White")

plt.subplot(1,3,3)
out = pmodel.specNorm(data,2.0)
plt.plot(out)
plt.title("renorm Red")

plt.show()
