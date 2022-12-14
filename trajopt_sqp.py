import numpy as np
from scipy.optimize import minimize, NonlinearConstraint


def objfun(z, S):

  xs = np.concatenate((np.expand_dims(S.x0, axis=1),
                       np.reshape(z[:S.N * S.n], (S.n, S.N), order='F')), axis=1)
  us = np.reshape(z[S.N * S.n:], (S.c, S.N), order='F')

  f = 0
  g = np.zeros_like(z)
  for i in range(S.N + 1):

    if i < S.N:
      L, Lx, Lxx, Lu, Luu = S.L(i, xs[:, i], us[:, i])

      # control gradients
      uind = S.N * S.n + i * S.c
      g[uind:uind + S.c] = Lu
    else:
      L, Lx, Lxx = S.Lf(xs[:, i])

    f = f + L

    # set state gradients
    if (i > 0):
      xind = (i - 1) * S.n
      g[xind:xind + S.n] = Lx

  return (f, g)


def nonlcon_eq(z, S):

  xs = np.concatenate((np.expand_dims(S.x0, axis=1),
                       np.reshape(z[:S.N * S.n], (S.n, S.N), order='F')), axis=1)
  us = np.reshape(z[S.N * S.n:], (S.c, S.N), order='F')

  ceq = np.zeros(S.n * S.N)

  for i in range(S.N):
    # discrete dynamics
    ind = i * S.n
    ceq[ind:ind + S.n] = xs[:, i + 1] - S.f(i, xs[:, i], us[:, i])[0]

  return ceq


def nonlcon_ineq(z, S):

  xs = np.concatenate((np.expand_dims(S.x0, axis=1),
                       np.reshape(z[:S.N * S.n], (S.n, S.N), order='F')), axis=1)
  us = np.reshape(z[S.N * S.n:], (S.c, S.N), order='F')

  c = np.array([])

  for i in range(S.N):
    ci = S.con(i, xs[:, i], us[:, i])
    if (ci is not None):
      c = np.concatenate((c, ci))

  # add final inequality constraint
  cf = S.con(i, xs[:, i], us[:, i])
  if (cf is not None):
    c = np.concatenate((c, cf))

  return c


def plot_traj(z, S):

  xs = np.concatenate((np.expand_dims(S.x0, axis=1),
                       np.reshape(z[:S.N * S.n], (S.n, S.N), order='F')), axis=1)
  us = np.reshape(z[S.N * S.n:], (S.c, S.N), order='F')
  S.plot_traj(xs, us)


def trajopt_sqp(xs, us, S):
  # Example code for discrete trajectory optimization using direct collocation
  # with SQP

  # @param xs current trajectory guess (n-x-(N+1) matrix)
  # @param us current controls guess (c-x-N matrix)
  # @param S system properties (dynamics, cost, constraints, etc...)

  # @return xs optimized trajectory
  # @return us optimized controls
  # @return cost computed cost
  # @return exitflag -- from fmincon (see fmincon docs)
  # @return output -- from fmincon (see fmincon docs)

  # Authors: Marin Kobilarov, marin(at)jhu.edu,
  #          Adam Polevoy, adam.polevoy@jhuapl.edu

  S.n = xs.shape[0]
  S.c = us.shape[0]
  S.N = us.shape[1]
  S.x0 = xs[:, 0]
  z = np.concatenate((xs[:, 1:].flatten(order='F'), us.flatten(order='F')))

  def objfun_(z_): return objfun(z_, S)

  def nonlcon_eq_(z_): return nonlcon_eq(z_, S)
  c_eq = nonlcon_eq(z, S)
  lb_eq = np.zeros_like(c_eq)
  ub_eq = np.zeros_like(c_eq)
  c_eq = NonlinearConstraint(nonlcon_eq_, lb_eq, ub_eq)

  def nonlcon_ineq_(z_): return nonlcon_ineq(z_, S)
  c_ineq = nonlcon_ineq(z, S)
  use_ineq_constraints = False
  if (len(c_ineq) > 0):
    lb_ineq = np.zeros_like(c_ineq)
    lb_ineq[:] = -np.inf
    ub_ineq = np.zeros_like(c_ineq)
    c_ineq = NonlinearConstraint(nonlcon_ineq_, lb_ineq, ub_ineq)
    use_ineq_constraints = True

  options = {"maxiter": 5000, 'ftol': 1e-3}

  def plot_traj_(z_): return plot_traj(z_, S)
  

  if (use_ineq_constraints):
    res = minimize(objfun_, z, jac=True, constraints=(c_eq, c_ineq),
                   options=options, callback=plot_traj_)
  else:
    res = minimize(objfun_, z, jac=True, constraints=(c_eq),
                   options=options, callback=plot_traj_)

  z = res.x
  cost = res.fun

  xs = np.concatenate((np.expand_dims(S.x0, axis=1),
                       np.reshape(z[:S.N * S.n], (S.n, S.N), order='F')), axis=1)
  us = np.reshape(z[S.N * S.n:], (S.c, S.N), order='F')

  return xs, us, cost
