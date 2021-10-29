from scipy.interpolate import BSpline
import scipy.interpolate
import warnings
import numpy as np


class PeriodicSpline:
    """
    OOP interface to periodic spline modelling.
    TODO figure out how best to explain the periodicity stuff
    """

    def __init__(self, knots, order):
        """
        given a set of interior knots, initialise an appropriate set
        of BSpline basis functions, as required to give a periodic
        spline model. Data will be assumed as being distributed across
        the unit period, and therefore the interior knots must be
        within the unit interval. boundary knots (ie. 0, 1) must not
        be provided. knots must be sorted into ascending order. cubic
        spline knots are assumed, meaning at least three interior
        knots must be provided.

            knots : 1d array
                1d array of interior knots, sorted from smallest to
                largest
        """
        self._order = order
        self._BSplines = _get_BSplines(knots, order)
        self._knots = knots

    def fit(self, data_t, data_y, period):
        """
        Compute the BSpline coefficients that specify a periodic
        BSpline curve, for the knot set provided at initialisation.

            data_t, data_y : 1d array
                t data (independent variable), y data (dependent variable)
                to fit a splines model to

            period : float > 0
                Period of the spline curve to be evaluated.

        Returns a list of the coefficients of the fitted spline model. The
        last three coefficients are ommitted, as they are equal to the
        first three coefficients.
        """
        return _fit_bspline_coefficients(self._BSplines, data_t, data_y, period, self._order)

    def eval(self, data_t, coefficients, period):
        """
        Given a set of time points, BSpline coefficients, and a
        period, evaluate the spline curve with that period, at the
        specified time points.

            coefficients : 1d array
                Array of BSpline coefficients, eg. as returned by the
                fit method. No checking is performed on these.

            data_t : 1d array
                Array of timepoints at which to evaluate the spline curve.

            period : float > 0
                Period of the spline curve to be evaluated.
        """
        if np.array(coefficients).ndim == 2:
            return _evaluate_multispline_curve(
                self._BSplines, data_t, coefficients, period, self._order
            )
        return _evaluate_spline_curve(self._BSplines, data_t, coefficients, period, self._order)

    def derivative(self, data_t, coefficients, period):
        data_t = np.array(data_t)
        if data_t.ndim > 1:
            raise ValueError("data_t must be one-dimensional")
        if not float(period) > 0:
            warnings.warn("Negative period encountered")
        # Reduce time data down to within BSpline support
        stacked_ts = np.mod(data_t / period, 1)

        full_coeffs = np.r_[coefficients, coefficients[:self._order]].T
        full_knots = _get_full_knots(self._knots, self._order)
        splrep = (full_knots, full_coeffs, self._order)
        ret = np.array(scipy.interpolate.splev(stacked_ts, splrep, der=1))
        return ret.T.squeeze()

    def __call__(self, data_t, coefficients, period):
        """
        Given a set of time points, BSpline coefficients, and a
        period, evaluate the spline curve with that period, at the
        specified time points.

            coefficients : 1d array
                Array of BSpline coefficients, eg. as returned by the
                fit method. No checking is performed on these.

            data_t : 1d array
                Array of timepoints at which to evaluate the spline curve.

            period : float > 0
                Period of the spline curve to be evaluated.
        """
        return self.eval(data_t, coefficients, period)


def _get_full_knots(interior_knots, order):
    starter_knots = np.hstack((interior_knots[-order:], [1])) - 1
    end_knots = np.hstack(([0], interior_knots[:order])) + 1
    return np.r_[starter_knots, interior_knots, end_knots]


def _get_BSplines(interior_knots, order):
    """
    given a set of interior knots, compute the bspline basis functions
    required to give a periodic spline model. data are assumed to be
    distributed across the unit period, and therefore the interior
    knots must be within the unit interval. boundary knots (ie. 0, 1)
    must not be provided. knots must be sorted into ascending order.
    cubic spline knots are assumed, meaning at least three interior
    knots must be provided.

        interior_knots : 1d array
            1d array of interior knots, sorted from smallest to
            largest
    """
    # Check everything is okay
    interior_knots = np.array(interior_knots)
    if not np.all(np.logical_and(interior_knots >= 0, interior_knots <= 1)):
        raise ValueError("Interior knots must be distributed across the unit period")
    if interior_knots.ndim != 1:
        raise ValueError("Knot array must be one-dimensional")
    if interior_knots.size < order:
        raise ValueError("Must have at least as many interior knots as spline order")
    full_knots = _get_full_knots(interior_knots, order)
    # Produce list of spline elements
    return [
        BSpline.basis_element(full_knots[i : i + order + 2], extrapolate=False)
        for i in range(full_knots.size - (order + 1))
    ]


def _fit_bspline_coefficients(basis_splines, data_t, data_y, period, order):
    """
    Given a list of basis splines, as constructed from the specified
    knots, and an array of data t and y coordinates, compute the
    BSpline coefficients that specify a periodic BSpline curve.

        basis_splines : list
            List of basis spline functions. Assumed to have been
            returned from _get_BSplines, so no further checking
            is performed on this list.

        data_t, data_y : 1d array
            t data (independent variable), y data (dependent variable)
            to fit a splines model to

            period : float > 0
                Period of the spline curve to be evaluated.

    Returns a list of the coefficients of the fitted spline model. The
    last three coefficients are ommitted, as they are equal to the
    first three coefficients.
    """
    # Check everything is okay
    data_t = np.array(data_t)
    if data_t.ndim != 1:
        raise ValueError("data_t must be one-dimensional")
    data_y = np.array(data_y)
    if data_y.ndim != 1:
        raise ValueError("data_y must be one-dimensional")
    if not float(period) > 0:
        warnings.warn("Negative period encountered")
    # Reduce time data down to within BSpline support
    stacked_ts = np.mod(data_t / period, 1)
    # Construct the design matrix
    design_mat = np.zeros((data_t.size, len(basis_splines) - order))
    for i, basis_func in enumerate(basis_splines[:-order]):
        design_mat[:, i] = np.nan_to_num(basis_func(stacked_ts).T)
    for i, basis_func in enumerate(basis_splines[-order:]):
        design_mat[:, i] += np.nan_to_num(basis_func(stacked_ts).T)
    # Return fitted coefficients
    return np.linalg.lstsq(design_mat, data_y.reshape((-1, 1)), rcond=None)[0].squeeze()


def _evaluate_spline_curve(basis_splines, data_t, coefficients, period, order):
    """
    Given a list of basis splines, a set of BSpline coefficients, and
    a period, evaluate the spline curve at the specified time-data
    points.

        basis_splines : list
            List of basis spline functions. Assumed to have been
            returned from _get_BSplines, so no further checking
            is performed on this list.

        coefficients : 1d array
            Array of BSpline coefficients, as returned by
            fit_bspline_coefficients. No further checking is
            performed.

        data_t : 1d array
            Array of timepoints at which to evaluate the spline curve.

        period : float > 0
            Period of the spline curve to be evaluated.
    """
    # Check everything is okay
    data_t = np.array(data_t)
    if data_t.ndim > 1:
        raise ValueError("data_t must be one-dimensional")
    if not float(period) > 0:
        warnings.warn("Negative period encountered")
    # Reduce time data down to within BSpline support
    stacked_ts = np.mod(data_t / period, 1)
    # Append the final BSpline coefficients
    full_coeffs = np.hstack((coefficients, coefficients[:order]))
    # Evaluate!
    eval_mat = np.zeros((full_coeffs.size, stacked_ts.size))
    for i, basis_func in enumerate(basis_splines):
        eval_mat[i, :] = basis_func(stacked_ts)
    eval_mat = np.nan_to_num(eval_mat)
    return np.inner(eval_mat.T, full_coeffs).squeeze()


def _evaluate_multispline_curve(basis_splines, data_t, coefficients, period, order):
    """
    Given a list of basis splines, a set of BSpline coefficients, and
    a period, evaluate the spline curve at the specified time-data
    points.

        basis_splines : list
            List of basis spline functions. Assumed to have been
            returned from _get_BSplines, so no further checking
            is performed on this list.

        coefficients : 2d array
            Array of BSpline coefficients, as returned by
            fit_bspline_coefficients. No further checking is
            performed.

        data_t : 1d array
            Array of timepoints at which to evaluate the spline curve.

        period : float > 0
            Period of the spline curve to be evaluated.
    """
    # Check everything is okay
    data_t = np.array(data_t)
    if data_t.ndim > 1:
        raise ValueError("data_t must be one-dimensional")
    if not float(period) > 0:
        warnings.warn("Negative period encountered")
    # Reduce time data down to within BSpline support
    stacked_ts = np.mod(data_t / period, 1)
    full_coeffs = np.vstack((coefficients, coefficients[:order]))
    # Append the final BSpline coefficients
    if len(basis_splines) != len(full_coeffs[:, 0]):
        raise ValueError(
            "Must have same number of coefficient vectors ({0}) as basis splines ({1})".format(
                len(full_coeffs[:, 0]), len(basis_splines)
            )
        )
    results = np.zeros((stacked_ts.size, len(coefficients[0])))
    for i, dim in enumerate(full_coeffs.T):
        eval_mat = np.zeros((dim.size, stacked_ts.size))
        for j, basis_func in enumerate(basis_splines):
            eval_mat[j, :] = basis_func(stacked_ts)
        eval_mat = np.nan_to_num(eval_mat)
        dim_vals = np.inner(eval_mat.T, dim).squeeze()
        results[:, i] = dim_vals
    return results
