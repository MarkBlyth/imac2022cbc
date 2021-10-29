import pprint
from . import splines
from . import discretise
from .system import ParameterException
import numpy as np
import time
import collections


##### DATA STORAGE ######

"""
The inner workings of this aren't particularly important here.

The CBC results are stored in a ContinuationSolution object. This is
basically a 2d named tuple. For each continuation step, we have
information such as the parameter, timestamps, states, etc. The 2d
named tuple lets us get either all the data for a single step (eg.
data = solution[i] gets us a named tuple containing things like
data.par, data.solution), or it gives us a list of a single piece of
data across all the steps (eg. solution.residuals gives us a list of
the residuals for the first, second, ... steps of the continuation.
The class is iterable.
"""

ContinuationStep = collections.namedtuple(
    "ContinuationStep",
    [
        "par",
        "ts",
        "states",
        "target",
        "angletarget",
        "residuals",
        "residualnorm",
        "jacobian",
        "solution",
    ],
)


class ContinuationSolution:
    def __init__(self):
        for attr in ContinuationStep._fields:
            setattr(self, attr, [])

    def __len__(self):
        return len(getattr(self, ContinuationStep._fields[0]))

    def __getitem__(self, key):
        if isinstance(key, int):
            return ContinuationStep(
                *[getattr(self, attr)[key] for attr in ContinuationStep._fields]
            )
        if isinstance(key, str):
            if key not in ContinuationStep._fields:
                raise KeyError(f"{key} not a ContinuationSolution attribute")
            return self.__getattr__(self, key)
        raise TypeError("Key must be int or string")

    def __iter__(self):
        self.current = -1
        return self

    def __next__(self):
        self.current += 1
        if self.current < len(self.par):
            return self.__getitem__(self.current)
        raise StopIteration

    def append(self, solution):
        if not isinstance(solution, ContinuationStep):
            raise TypeError("Appended solution must be a ContinuationStep instance")
        for attr in ContinuationStep._fields:
            getattr(self, attr).append(getattr(solution, attr))


######## PERIODIC ORBIT CONTINUATION ########


def run_a_continuation(
    stepsize,
    this_solver,
    system_runner,
    initial_pars,
    discretisation_size,
    max_steps=np.inf,
    par_min=-np.inf,
    par_max=np.inf,
    min_amplitude=0,
    save_figs=False,
    adapt_discretisor=False,
    verbose=True,
    rediscretise_after=None,
    spline_order=3,
):
    """
    Construct and run a continuation experiment for the chosen
    hyperparameters, and return a list of accepted continuation
    solutions.
    

    Updating spline knots:
            * If rediscretise_after is None, just ignore.
            * If we can call rediscretise_after as a function,
                pass it the last solution. If it returns true, update
                the knots from scratch using the most recent solution.
                If it returns false, don't.
            * If we can't call rediscretise_after as a function,
                assume it's an integer n specifying that we update
                the knots from scratch every n steps. If it's time
                to update the knots according to this, do so from
                scratch, using the most recent solution.
        If, after all of this, we haven't updated the spline knots
        from scratch, but we still want an adaptive discretisor,
        then do an optimization on the current knots instead;
        improve the existing knots, rather than generating new
        ones from scratch.
    

    Two ways to stop the continuation: either when the user gets bored
    and does a keyboard interrupt, or when the system stops doing what
    we want (leaves target parameter range, or fails to show the
    target periodics etc.)
    """

    _, starter_states = system_runner(initial_pars[0])
    discretisor = discretise.ThetaSplines(
        starter_states, discretisation_size, spline_order
    )

    solutions = ContinuationSolution()
    for soln in get_initial_solutions(initial_pars, system_runner, discretisor):
        solutions.append(soln)

    filename = "continuation" + time.strftime("%Y%m%d%H%M%S") + "step"

    try:
        while (
            solutions[-1].par < par_max
            and solutions[-1].par > par_min
            and np.max(solutions[-1].states[0]) - np.min(solutions[-1].states[0])
            > min_amplitude
        ):
            if verbose:
                print("Step ", len(solutions) - 1)

            ## Update the spline knots.
            adapted_thistime = False
            try:
                if rediscretise_after(solutions):
                    discretisor = discretise.ThetaSplines(
                        solutions[-1].states, discretisation_size
                    )
                    adapted_thistime = True
            except TypeError:
                if (
                    rediscretise_after is not None
                    and (len(solutions) - 2) % rediscretise_after == 0
                ):
                    discretisor = discretise.ThetaSplines(
                        solutions[-1].states, discretisation_size
                    )
                    adapted_thistime = True
            if len(solutions) > 2 and adapt_discretisor and not adapted_thistime:
                discretisor.update_discretisation_scheme(solutions[-1].states, verbose)

            # Take a continuation step
            y0 = get_continuation_vector(solutions[-2], discretisor)
            y1 = get_continuation_vector(solutions[-1], discretisor)
            IO_map = get_IO_map(system_runner, discretisor)
            next_soln = get_next_soln(
                y0, y1, stepsize, IO_map, this_solver, discretisor
            )

            # Construct a results object to store everything in
            target = discretisor.undiscretise(next_soln["x"][1:])
            angle_model = discretisor.angle_undiscretise(next_soln["x"][1:])
            par = next_soln["x"][0]
            ts, states = system_runner(par, target)
            residual_norm = np.linalg.norm(next_soln["fun"])
            soln_obj = ContinuationStep(
                par,
                ts,
                states,
                target,
                angle_model,
                next_soln["fun"],
                residual_norm,
                next_soln["fjac"],
                next_soln["x"],
            )
            solutions.append(soln_obj)

            # Print out some useful information
            if verbose:
                del next_soln["fjac"]
                pprint.pprint(next_soln)
                norm = np.linalg.norm(next_soln["x"][1:])
                print(f"Discretisation norm: {norm:.5f}")
                print("\n")
    except (
        KeyboardInterrupt,
        ParameterException,
    ) as e:  ### Put ValueError in here to catch integration problems
        if isinstance(e, ValueError):
            print("Continuation terminated due to integration error")
        elif isinstance(e, ParameterException):
            print("Continuation terminated due to invalid parameter value")
    return solutions


def get_initial_solutions(initial_pars, system_runner, initial_discretisor):
    initial_solns = []
    for par in initial_pars:
        ts, states = system_runner(par)
        discretisation = initial_discretisor.discretise(states)
        target = initial_discretisor.undiscretise(discretisation)
        angletarget = initial_discretisor.angle_undiscretise(discretisation)
        IO_map = get_IO_map(system_runner, initial_discretisor)
        coeff_error = discretisation - IO_map(np.r_[par, discretisation])
        residuals = np.r_[0, coeff_error]
        residualnorm = np.linalg.norm(residuals)
        soln = ContinuationStep(
            par,
            ts,
            states,
            target,
            angletarget,
            residuals,
            residualnorm,
            None,
            np.r_[par, discretisation],
        )
        initial_solns.append(soln)
    return initial_solns


def get_continuation_vector(solution, discretisor):
    discretisation = discretisor.discretise(solution.states)
    return np.r_[solution.par, discretisation]


def get_IO_map(system_runner, discretisor):
    def IO_map(continuation_vector):
        parameter = continuation_vector[0]
        target = discretisor.undiscretise(continuation_vector[1:])
        ts, states = system_runner(parameter, target, 2 * np.pi)
        return discretisor.discretise(states)

    return IO_map


def get_next_soln(y0, y1, stepsize, IO_map, solver, discretisor):
    prediction = y1 + stepsize * (y1 - y0) / np.linalg.norm(y1 - y0)

    def continuation_equations(y_new):
        pseudo_arclength_condition = np.inner(y1 - y0, y_new - prediction)
        coeff_error = y_new[1:] - IO_map(y_new)
        return np.r_[pseudo_arclength_condition, coeff_error]

    return solver(continuation_equations, prediction)
