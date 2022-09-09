""" Module to compute projections on the positive simplex or the L1-ball

A positive simplex is a set X = { \mathbf{x} | \sum_i x_i = s, x_i \geq 0 }

The (unit) L1-ball is the set X = { \mathbf{x} | || x ||_1 \leq 1 }

Adrien Gaidon - INRIA - 2011

Adapted for JAX: Christoph Lampert - ISTA - 2021

Licensed under the terms of the MIT license, see LICENSE.md
"""

from jax import jit
from functools import  partial
import numpy as np
import jax.numpy as jnp

def euclidean_proj_simplex(v, s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if np.isclose(v.sum(), s) and jnp.alltrue(v >= 0):
      return v
    else:
      return do_euclidean_proj_simplex(v=v, s=s)

@partial(jit, static_argnums=(1))
def do_euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex

    Solves the optimisation problem (using the algorithm from [1]):

        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 

    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project

    s: int, optional, default: 1,
       radius of the simplex

    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.

    References
    ----------
    [1] Efficient Projections onto the L1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    n, = v.shape  # will raise ValueError if v is not 1-D
    u = jnp.sort(v)[::-1]
    cssv = jnp.cumsum(u)
    ar = jnp.cumsum(jnp.ones_like(u))
    # get the number of > 0 components of the optimal solution
    rho = jnp.asarray(u * jnp.arange(1, n+1) > (cssv - s)).sum()-1
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho+1.)
    # compute the projection by thresholding v using theta
    w = jnp.maximum(v - theta, 0)
    return w


  
def euclidean_proj_l1ball(v, s=1):
    """ Compute the Euclidean projection on a L1-ball

    Solves the optimisation problem (using the algorithm from [1]):

        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s

    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project

    s: int, optional, default: 1,
       radius of the L1-ball

    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s

    Notes
    -----
    Solves the problem by a reduction to the positive simplex case

    See also
    --------
    euclidean_proj_simplex
    """
    #assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = jnp.abs(v)
    # check if v is already a solution
    
    if (u.sum()-s) > 1e-8:
      # v is not already a solution: optimum lies on the boundary (norm == s)
      # project *u* on the simplex
      u = euclidean_proj_simplex(u, s=s)
      # compute the solution to the original problem on v
      u *= jnp.sign(v)
    return u

@jit
def euclidean_proj_intersection(x, c, s=1, k=10): # Dykstra's alternating projection
  x0=x
  p=0
  q=0
  for i in range(k):
    y=euclidean_proj_l1ball(x+p-c, s=s)+c # s-ball around c
    p=x+p-y
    x=euclidean_proj_simplex(y+q)
    q=y+q-x
  return x
