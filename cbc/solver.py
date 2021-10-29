import numpy as np
import scipy.linalg
import numdifftools as ndt

"""
TODOs
   * Get nfev to work!
   * Change finite_differences_jacobian to a Jacobian class
      * Implement forward and backward steps of a given stepsize
      * Calculate weights for higher orders
      * Combine weights and steps to get Jacobian
   * Add other stop-functions to the solver
      * Let user swap out the convergence rule easily
      * Add extra functions that the user can optionally define to specify when convergence has failed (eg. when parameter has left some target range)
   * Implement Broyden and Newton as separate subclasses of the base solver?
      * Have a forward-facing solver class that automatically figures out whihc base-solver-subclass to invoke
"""


def finite_differences_jacobian(f, x, step, f_x=None, method="central"):
    if np.isscalar(step):
        perturbations = step * np.eye(x.size)
    else:
        if x.shape != np.array(step).shape:
            raise ValueError("step must be scalar, or 1d array of same size as x")
        perturbations = np.diag(step)

    if method == "forward":
        f_x = f_x if f_x is not None else f(x)
        return np.array(
            [(f(x + h) - f_x) / h[i] for i, h in enumerate(perturbations)]
        ).T
    if method == "backward":
        f_x = f_x if f_x is not None else f(x)
        return np.array(
            [(f_x - f(x - h)) / h[i] for i, h in enumerate(perturbations)]
        ).T
    # Method = "central"
    return np.array(
        [(f(x + h) - f(x - h)) / (2 * h[i]) for i, h in enumerate(perturbations)]
    ).T


def newton_update(f, x, f_x, jacobian):
    #step = np.linalg.solve(jacobian, -f_x)
    Q, R = scipy.linalg.qr(jacobian)
    y = np.dot(Q.T, -f_x)
    step = scipy.linalg.solve_triangular(R, y)

    new_x = x + step
    return new_x, step, f(new_x)


def broyden_update(f, x, f_x, jacobian):
    #step = np.linalg.solve(jacobian, -f_x)
    Q, R = scipy.linalg.qr(jacobian)
    y = np.dot(Q.T, -f_x)
    step = scipy.linalg.solve_triangular(R, y)

    new_soln = x + step
    new_f_x = f(new_soln)
    jacobian_step = np.matmul(
        ((new_f_x - f_x) - np.matmul(jacobian, step)) / (np.linalg.norm(step) ** 2),
        step.T,
    )
    return new_soln, step, new_f_x, jacobian + jacobian_step


class Solver:
    def __init__(
        self,
        update_rule,
        jacobian_method,
        stepsize_tol,
        soln_tol,
        max_iter=100,
        printout=True,
    ):
        update_rules = {"newton": newton_update, "broyden": broyden_update}
        jacobian_methods = {
            "central": lambda f, x, h, f_x: finite_differences_jacobian(
                f, x, h, f_x, "central"
            ),
            "forward": lambda f, x, h, f_x: finite_differences_jacobian(
                f, x, h, f_x, "forward"
            ),
            "backward": lambda f, x, h, f_x: finite_differences_jacobian(
                f, x, h, f_x, "backward"
            ),
            "numdifftools": "numdifftools",
        }

        if update_rule.lower() not in update_rules:
            raise ValueError(
                "Update rule must be one of ", ", ".join(update_rules.keys())
            )
        if jacobian_method.lower() not in jacobian_methods:
            raise ValueError(
                "Jacobian method must be one of ", ", ".join(jacobian_methods.keys())
            )
        if max_iter is not None and not isinstance(max_iter, int):
            raise TypeError("max_iter must be an integer")
        self.update_rule = update_rules[update_rule]
        self.jacobian = jacobian_methods[jacobian_method]
        self.max_iter = max_iter
        self.stepsize_tol = stepsize_tol
        self.soln_tol = soln_tol
        if update_rule == "broyden":
            self._nfev_multiplier = 0
        elif jacobian_method == "central":
            self._nfev_multiplier = 2
        else:
            self._nfev_multiplier = 1
        self.printout = printout

    def __call__(self, f, x0, finite_differences_stepsize):
        if x0.ndim != 1:
            raise ValueError("x must be a 1d array")
        if self.jacobian == "numdifftools":
            jacobian_obj = ndt.Jacobian(f, step=finite_differences_stepsize)
            jacobian_func = lambda x, f_x: jacobian_obj(x)
        else:
            jacobian_func = lambda x, f_x: self.jacobian(
                f, x, finite_differences_stepsize, f_x
            )
        step, f_x, i, jacobian = np.inf, f(x0), 0, None

        while not self.convergence_rule(step, f_x, i) and i < self.max_iter:
            i += 1
            if self.printout:
                print("Iteration: ", i)
            if jacobian is None:
                jacobian = jacobian_func(x0, f_x)
                last_jacobian = jacobian
            update_result = self.update_rule(f, x0, f_x, jacobian)
            try:
                x0, step, f_x, jacobian = update_result
            except ValueError:
                x0, step, f_x = update_result
                jacobian = None
            if self.printout:
                print(
                    "Function value:\n",
                    f_x,
                    "\nNew solution:\n",
                    x0,
                    "\nResidual norm: ",
                    np.linalg.norm(f_x),
                    "\nJacobian:\n",
                    (last_jacobian if jacobian is None else jacobian),
                    "\nJacobian condition:\n",
                    np.linalg.cond(last_jacobian if jacobian is None else jacobian),
                    "\n",
                )

       ######################### TODO n_fev is wrong; why?
       ### Split up into an initial fev and a step-fev, with initial fev defined in __init__
        n_fev = (
            (i + 1 + x0.size + (i-1) * self._nfev_multiplier * x0.size)
            if self.jacobian != "numdifftools"
            else -1
        )
        if self._nfev_multiplier == 2:
            n_fev += x0.size
       ######################### TODO n_fev is wrong; why?
        return {
            "nfev": n_fev,
            "nit": i,
            "success": self.convergence_rule(step, f_x, i),
            "x": x0,
            "fun": f_x,
            "fjac": (last_jacobian if jacobian is None else jacobian),
        }

    def convergence_rule(self, step, f_x, iterate=None):
        if iterate == 0:
            return np.linalg.norm(f_x) < self.soln_tol
        return (
            np.linalg.norm(step) < self.stepsize_tol
            or np.linalg.norm(f_x) < self.soln_tol
        )
