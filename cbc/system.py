import numpy as np
import scipy.integrate
import warnings

"""
TODO add enough documentation that other users would know what to do with all this
"""


class ParameterException(Exception):
    pass


class System:
    """
    A model of a physical system. The model is governed by a set of
    ODEs. The system is evaluated using some control target and some
    parameter value. The system is first initialised at the provided
    initial condition. Thereafter, it is initialised from the final
    state of the previous run, to model a real system. When a period
    is provided, the integration time is extended so that the final
    state is always at phase=0, which avoids any discontinuity between
    the current and next control target. The system is controlled
    using a controller object, and returned observations are obtained
    by observing the states according to the controller's observations.
    """

    def __init__(
        self,
        initial_cond,
        ode_rhs,
        transient_time,
        evaluation_time,
        sample_rate,
        controller,
        valid_par_range=None,
        method="DOP853",
        atol=1e-9,
        rtol=1e-9,
    ):
        self.initial_cond = initial_cond
        self.ode_rhs = ode_rhs
        self.evaluation_time = evaluation_time
        self.transient_time = transient_time
        self.sample_rate = sample_rate
        self.controller = controller
        self.valid_par_range = valid_par_range
        self.n_runs = 0
        self._method = method
        self._atol = atol
        self._rtol = rtol

    def __call__(self, parameter, control_target=None, period=None):
        """
        Run the system, controlled or uncontrolled.

            parameter : float
                Parameter value at which to run the system

            control_target : func(t)
                Get the target system output at time t. If None, the
                system is ran uncontrolled.

            period : float
                The period of a periodic control target. If set, the
                integration time is extended so that the last
                measurement is taken at control phase = 0
        """
        if self.valid_par_range is not None:
            if parameter < min(self.valid_par_range) or parameter > max(
                self.valid_par_range
            ):
                print("Received a parameter of {0}, exiting".format(parameter))
                raise ParameterException(
                    f"Parameter value {parameter:.3f} is outside of valid range"
                )

        ## If a period is passed, integrate for an integer number of periods
        end_time = self.transient_time + self.evaluation_time
        if period is not None:
            period = np.abs(period)
            extra_evaluation_time = period - np.mod(end_time, period)
        else:
            extra_evaluation_time = 0
        end_time += extra_evaluation_time
        total_evaluation_time = self.evaluation_time + extra_evaluation_time

        t_span = [0, end_time]
        t_eval = np.linspace(
            self.transient_time,
            end_time,
            int(total_evaluation_time * self.sample_rate),
        )

        if control_target is None:
            system = lambda t, x: self.ode_rhs(t, x, parameter)
            initial_cond = self.initial_cond
        else:
            def system(t, x):
                control_term = self.controller(t, x[:-1], control_target)
                uncontrolled = self.ode_rhs(t, x[:-1], parameter)
                return [*(uncontrolled + control_term), np.linalg.norm(control_term)**2]
            # system = lambda t, x: self.ode_rhs(t, x, parameter) + self.controller(
            #     x, control_target
            # )
            initial_cond = [*self.initial_cond, 0]

        soln = scipy.integrate.solve_ivp(
            system,
            t_span=t_span,
            t_eval=t_eval,
            y0=initial_cond,
            atol=self._atol,
            rtol=self._rtol,
            method=self._method,
        )
        self.n_runs += 1
        if control_target is None:
            self.initial_cond = np.squeeze(soln.y[:, -1])
            return soln.t, soln.y
        self.initial_cond = np.squeeze(soln.y[:-1, -1])
        print("Mean-square control action: ", (soln.y[-1, -1] - soln.y[-1, 0]) /(soln.t[-1] - soln.t[0]))
        return soln.t, soln.y[:-1]


class ThetaPLLController:
    """
    Callable class whose function gives the control action for a given
    state and target.
    """

    def __init__(self, kp, controlled_var, observed_var):
        """
        Assume only the i'th variable is measured as the system output;
        this is encoded with observed_var, a vector with the same
        dimension as the state vector, whose i'th entry is one and all
        other entries are zero. Similarly, assume control is applied only
        to the j'th variable; this is encoded with controlled_var, a
        vector with the same dimension as the state vector, whose j'th
        entry is one and all other entries are zero. Control gain is given
        by kp.
        """
        self.kp = kp
        self.observed_var = np.array(observed_var)
        self.controlled_var = np.array(controlled_var)

    def observe(self, states):
        """
        Given an array where each row is a state variable, return the
        row corresponding to observations of the system. TODO this
        method is inefficient. It would be computationally better to
        simply index the data.
        """
        return np.dot(self.observed_var, states)

    def __call__(self, t, state, target):
        try:
            error = target(state) - self.observe(state)
        except TypeError:
            error = target - self.observe(state)
        return self.kp * self.controlled_var * error


class PController:
    """
    Callable class whose function gives the control action for a given
    state and target.
    """

    def __init__(self, kp, controlled_var, observed_var):
        """
        Assume only the i'th variable is measured as the system output;
        this is encoded with observed_var, a vector with the same
        dimension as the state vector, whose i'th entry is one and all
        other entries are zero. Similarly, assume control is applied only
        to the j'th variable; this is encoded with controlled_var, a
        vector with the same dimension as the state vector, whose j'th
        entry is one and all other entries are zero. Control gain is given
        by kp.
        """
        self.kp = kp
        self.observed_var = np.array(observed_var)
        self.controlled_var = np.array(controlled_var)

    def observe(self, states):
        """
        Given an array where each row is a state variable, return the
        row corresponding to observations of the system. TODO this
        method is inefficient. It would be computationally better to
        simply index the data.
        """
        return np.dot(self.observed_var, states)

    def __call__(self, t, state, target):
        try:
            error = target(t) - self.observe(state)
        except TypeError:
            error = target - self.observe(state)
        return self.kp * self.controlled_var * error
