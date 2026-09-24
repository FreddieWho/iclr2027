"""Finite-difference check for O04 convex BCE/ranking objective."""
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from readout_refit import objective, convex
rng=np.random.default_rng(8)
z=rng.normal(size=(12,3));y=np.arange(12)%2;w=np.ones(12)/12;t=rng.normal(size=4)
for pairs in (None,np.array([[1,0],[3,2]])):
    _,g=objective(t,z,y,w,.01,pairs)
    numerical=approx_derivative(lambda v:objective(v,z,y,w,.01,pairs)[0],t).ravel()
    np.testing.assert_allclose(g,numerical,atol=1e-7)
_,log=convex(z,y,w,.01)
assert log['success'] and log['grad_inf']<1e-6
print('BCE/ranking finite-difference gradients and convex convergence passed')
