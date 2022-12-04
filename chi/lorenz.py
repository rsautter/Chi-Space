import numpy as np
from scipy.integrate import solve_ivp

def lorenzSynch(mu=5,tmax=500,n=20000):
	'''
	=====================================================================================
	Generates the distance between two Lorenz systems, where one of them is coupled by a parameter mu
	=====================================================================================
	Input:
		mu - coupling parameter
		tmax - max time of integration
		n - number of points in the series
	=====================================================================================
	Output:
		t - time 
		r - euclidian distance between oscillators solutions 
	=====================================================================================
	Wrote by: Valdivia (11/2022)
	'''

	sigma1, beta1, rho1 = 10, 8/3, 28
	sigma2, beta2, rho2 = 10, 8/3, 29

	u10, v10, w10, u20, v20, w20  = 0.1,0.1,0.1, 20,20,2
	def lorenz(t, X, mu):
		"""The Lorenz equations."""
		u1, v1, w1, u2, v2, w2 = X

		up1 = -sigma1*(u1 - v1)
		vp1 = rho1*u1 - v1 - u1*w1
		wp1 = -beta1*w1 + u1*v1

		up2 = -sigma2*(u2 - v2) - mu*(u2-u1)
		vp2 = rho2*u2 - v2 - u2*w2 - mu*(v2-v1)
		wp2 = -beta2*w2 + u2*v2 - mu*(w2-w1)

		return up1, vp1, wp1, up2, vp2, wp2

    # Integrate the Lorenz equations.
	soln = solve_ivp(lorenz, (0, tmax), (u10, v10, w10, u20, v20, w20), args=(mu,),
                     dense_output=True)
	t = np.linspace(0, tmax, n)
	x1, y1, z1, x2, y2, z2 = soln.sol(t)
	r=np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
	return t,r
