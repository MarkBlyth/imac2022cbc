#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import cbc
import time
import pickle

### Continuation numerical parameters
INITIAL_PARS = (0.2, 0.18)
STEPSIZE = 0.1
PAR_MIN = 0.02
PAR_MAX = 5 ### Doesn't really matter since we're stepping backwards
MIN_AMPLITUDE = 0.05
### Solver
SOLVER = "newton"  ### newton, broyden
JACOBIAN_METHOD = "forward"  ### central, forward, backward, numdifftools
FINITE_DIFFERENCES_STEPSIZE = 1e-3  ### Can be None for numdifftools
STEP_TOL = 1e-3  ### Newton and Broyden step convergence tolerance
SOLN_TOL = 1e-5  ### Newton and Broyden residual convergence tolerance
MAX_ITER = 1  ### Max. number of solver iterations
### System
KP = 1 #### 
INITIAL_COND = [0, -1.8] # Start the first run from this state (arbitrary)
TRANSIENT_TIME = 50 # Throw away the first 50s
EVALUATION_TIME = 200  # Evaluate for 200s
SAMPLE_RATE = 20 # Take 20 datapoints per second
### Discretisation
DSIZE = 10 # 10 internal knots
ORDER = 3 # Cubic splines
REDISCRETISE_AFTER = None # Get spline knots from scratch only at the start; alternative ideas below
# REDISCRETISE_AFTER = lambda s: s[-1].residualnorm > 1e-2 # Get spline knots from scratch when ||residuals|| of the last solution s > 1e-2
# REDISCRETISE_AFTER = 3 # Get new spline knots from scratch every 3 steps

### Continuation outputs
SAVE_FIGS = False ## Save the system output phase plane after each iteration
ADAPT_DISCRETISOR = True ## Update existing knots after each continuation step
VERBOSE = True ## Say lots of stuff
PICKLE = False ## Save key computed data and simulation parameters


def stickslip(t, x, par, control_force=0):
    """
    RHS of controlled ODE
    """
    k = 1
    m = 1
    c = 0
    u, v = x

    def friction(w):
        fc = -0.5
        fs = 2
        fv = 0.3
        return ( fc + (fs - fc)*np.exp(-np.abs(w)) + fv * np.abs(w) ) * np.sign(w)

    non_friction_force = -c*v  - k*u + control_force
    max_friction_force = -friction(v - par)
    actual_friction = np.sign(max_friction_force) * np.min(np.abs([max_friction_force, non_friction_force]))
    
    u_dot = v
    v_dot = (actual_friction + non_friction_force) / m
    return np.array([u_dot, v_dot])


def stickslip_system_runner(parameter, control_target=None, period=None):
    """
    Simulate the stick-slip oscillator at a given parameter value. If
    a control target is provided, use it with a porportional
    controller. If a period is provided, extend the integration time
    slightly so that we integrate for an integer number of periods.
    Also integrates the squared control action, to quantify
    invasiveness. This is printed but not returned.
    
    Returns a tuple (output times, output states), where output states
    is a list of [position velocity] for each output time.
    """
    if parameter < PAR_MIN or parameter > PAR_MAX:
        print("Received a parameter of {0}, exiting".format(parameter))
        raise KeyboardInterrupt

    end_time = TRANSIENT_TIME + EVALUATION_TIME
    if period is not None:
        period = np.abs(period)
        extra_evaluation_time = period - np.mod(end_time, period)
    else:
        extra_evaluation_time = 0
    end_time += extra_evaluation_time
    total_evaluation_time = EVALUATION_TIME + extra_evaluation_time

    t_span = [0, end_time]
    t_eval = np.linspace(
        TRANSIENT_TIME,
        end_time,
        int(total_evaluation_time * SAMPLE_RATE),
    )

    if control_target is None:
        system = lambda t, x: stickslip(t, x, parameter)
        initial_cond = INITIAL_COND
    else:
        def system(t, x):
            control_term = KP * (control_target(x[:-1]) - x[0])
            ode_rhs = np.array(stickslip(t, x[:-1], parameter, control_term))
            return [*ode_rhs, control_term**2]
        initial_cond = [*INITIAL_COND, 0]

    soln = scipy.integrate.solve_ivp(
        system,
        t_span=t_span,
        t_eval=t_eval,
        y0=initial_cond,
        method="DOP853",
        atol=1e-9,
        rtol=1e-9,
    )
    if control_target is None:
        return soln.t, soln.y
    print("Mean-square control action: ", (soln.y[-1, -1] - soln.y[-1, 0]) /(soln.t[-1] - soln.t[0]))
    return soln.t, soln.y[:-1]


def main():
    # Get a solver
    solver_obj = cbc.Solver(SOLVER, JACOBIAN_METHOD, STEP_TOL, SOLN_TOL, MAX_ITER)
    this_solver = lambda f, x0: solver_obj(f, x0, FINITE_DIFFERENCES_STEPSIZE)

    # Run the continuation
    solutions = cbc.run_a_continuation(
        STEPSIZE,
        this_solver,
        stickslip_system_runner,
        INITIAL_PARS,
        DSIZE,
        min_amplitude=MIN_AMPLITUDE,
        par_min=PAR_MIN,
        par_max=PAR_MAX,
        adapt_discretisor=ADAPT_DISCRETISOR,
        save_figs=SAVE_FIGS,
        verbose=VERBOSE,
        rediscretise_after=REDISCRETISE_AFTER,
        spline_order=ORDER,
    )
    print("Continuation terminated!")

    # Save solutions list
    if PICKLE:
        filename = "stickslip" + time.strftime("%Y%m%d%H%M%S") + ".pickle"
        global_vars = {
            k: v for k, v in globals().items() if isinstance(v, (str, int, float))
        }
        with open(filename, "wb") as picklefile:
            pickle.dump([global_vars, solutions.par, solutions.ts, solutions.states], picklefile)

    # Plot
    print("Running open-loop simulations")
    maxs, mins = [], []
    for par in solutions.par:
        ts, states = stickslip_system_runner(par)
        maxs.append(states[0].max())
        mins.append(states[0].min())
    fig, (ax, ax2) = plt.subplots(2)
    ax.plot(solutions.par, maxs, "b--", alpha=0.5, label="True solution")
    ax.plot(solutions.par, mins, "b--", alpha=0.5)
    ax.plot(
        solutions.par,
        [np.min(states[0]) for states in solutions.states],
        color="k",
    )
    ax.plot(
        solutions.par,
        [np.max(states[0]) for states in solutions.states],
        color="k",
        label="CBC",
    )
    ax.scatter(
        solutions.par,
        [np.min(states[0]) for states in solutions.states],
        color="k",
    )
    ax.scatter(
        solutions.par,
        [np.max(states[0]) for states in solutions.states],
        color="k",
        label="Solution points",
    )
    ax.set_xlabel(r"Treadmill speed $v$")
    ax.set_ylabel(r"min($x$), max($x$)")
    ax.legend()

    ts, states = solutions[0].ts, solutions[0].states
    ax2.plot(ts[ts<100] - ts[0], states[0, ts<100])
    ax2.set_xlabel(r"Time $t$ (a.u.)")
    ax2.set_ylabel(r"Position $x$ (a.u.)")
    plt.show()

if __name__ == "__main__":
    main()
