import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from joblib import Parallel, delayed


def acq_max(ac, gp, y_max, bounds, random_state, x_tries, x_seeds, x_before=0, pen=0.0, alpha=1.0, cost_var=None,
            n_warmup=10000, n_iter=10):
    """
    A function to find the maximum of the acquisition function

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running L-BFGS-B from `n_iter` (250) random starting points.

    Parameters
    ----------
    :param ac:
        The acquisition function object that return its point-wise value.

    :param gp:
        A gaussian process fitted to the relevant data.

    :param y_max:
        The current maximum known value of the target function.

    :param bounds:
        The variables bounds to limit the search of the acq max.

    :param random_state:
        instance of np.RandomState random number generator

    :param n_warmup:
        number of times to randomly sample the aquisition function

    :param n_iter:
        number of times to run scipy.minimize

    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """

    # Warm up with random points
    # x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
    #                                size=(n_warmup, bounds.shape[0]))
    if not isinstance(gp, list):
        ys = ac(x_tries, gp=gp, y_max=y_max, x_before=x_before, alpha=alpha, cost_var=cost_var)
    else:
        ys = [ac(x_tries, gp=gp[i], y_max=y_max[i], x_before=x_before, alpha=alpha, cost_var=cost_var) for i in
              range(len(gp))]
        ys = np.average(np.asarray(ys), axis=0)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()
    penalization = pen
    temp1 = np.std(ys)
    if penalization != 0:
        penalization *= temp1
        if penalization < 1e-10:
            penalization = 1e-10
        if not isinstance(gp, list):
            ys = ac(x_tries, gp=gp, y_max=y_max, x_before=x_before, pen=penalization, alpha=alpha, cost_var=cost_var)
        else:
            ys = [ac(x_tries, gp=gp[i], y_max=y_max[i], x_before=x_before, pen=penalization, alpha=alpha,
                     cost_var=cost_var) for i in range(len(gp))]
            ys = np.average(np.asarray(ys), axis=0)
        x_max = x_tries[ys.argmax()]
        max_acq = ys.max()
        while max_acq < 0:
            penalization *= 0.7
            if not isinstance(gp, list):
                ys = ac(x_tries, gp=gp, y_max=y_max, x_before=x_before, pen=penalization, alpha=alpha,
                        cost_var=cost_var)
            else:
                ys = [ac(x_tries, gp=gp[i], y_max=y_max[i], x_before=x_before, pen=penalization, alpha=alpha,
                         cost_var=cost_var) for i in
                      range(len(gp))]
                ys = np.average(np.asarray(ys), axis=0)
            x_max = x_tries[ys.argmax()]
            max_acq = ys.max()

    # Explore the parameter space more throughly
    # x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
    #                                size=(n_iter, bounds.shape[0]))
    if not isinstance(gp, list):
        for x_try in x_seeds:
            # Find the minimum of minus the acquisition function
            res = minimize(
                lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max, x_before=x_before, pen=penalization, alpha=alpha,
                              cost_var=cost_var),
                x_try.reshape(1, -1),
                bounds=bounds,
                method="L-BFGS-B")

            # See if success
            if not res.success:
                continue

            # Store it if better than previous minimum(maximum).
            if max_acq is None or -res.fun >= max_acq:
                x_max = res.x
                max_acq = -res.fun

    else:
        res = Parallel(n_jobs=6)(delayed(minimize)(lambda x: np.average(
            np.asarray([-ac(x.reshape(1, -1), gp=gp[i], y_max=y_max[i], x_before=x_before, pen=penalization,
                            alpha=alpha, cost_var=cost_var)
                        for i in range(len(gp))]), axis=0),
                                                    x_try.reshape(1, -1),
                                                    bounds=bounds,
                                                    method="L-BFGS-B") for x_try in x_seeds)
        id = [res[i].fun for i in range(len(res))].index(min([res[i].fun for i in range(len(res))]))
        if max_acq is None or -res[id].fun >= max_acq:
            x_max = res[id].x
            max_acq = -res[id].fun

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])


class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, xi, kappa_decay=1, kappa_decay_delay=0, pen=0.0, alpha=1.0):

        self.kappa = kappa
        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay

        self.xi = xi

        self._iters_counter = 0

        if kind not in ['ucb', 'ei', 'poi', 'port', 'ei_pm', 'ei_cool', 'ucb_wp', 'ei_wp']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def update_params(self):
        self._iters_counter += 1

        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
            self.kappa *= self._kappa_decay

    def utility(self, x, gp, y_max, x_before=0, pen=0.0, alpha=1.0, cost_var=None):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)
        if self.kind == 'ei_pm':
            return self._ei(x, gp, y_max, self.xi) / self._time_compute(x, x_before, cost_var)
        if self.kind == 'ei_cool':
            return self._ei(x, gp, y_max, self.xi) / np.power(self._time_compute(x, x_before, cost_var), alpha)
        if self.kind == 'ucb_wp':
            return self._ucb(x, gp, self.kappa) - pen * self._time_compute(x, x_before, cost_var)
        if self.kind == 'ei_wp':
            return self._ei(x, gp, y_max, self.xi) - pen * self._time_compute(x, x_before, cost_var)

    @staticmethod
    def _ucb(x, gp, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return mean + kappa * std

    @staticmethod
    def _ei(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        a = (mean - y_max - xi)
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi) / std
        return norm.cdf(z)

    @staticmethod
    def _time_compute(x, x_before, cost_var):
        # for 6 dim
        var_type = np.array([[1, 1, 2, 3, 3, 2, 0], [200, 100, 20, 0, 0, 40, 20]])
        t = var_type[1, 0] * np.abs(x[:, 0] - x_before[0])
        t += var_type[1, 1] * np.abs(x[:, 1] - x_before[1])
        t += var_type[1, 2] / (x[:, 2] + 0.1)
        t += var_type[1, 5] / (x[:, 5] + 0.1)
        t += var_type[1, 6]
        return t


def load_logs(optimizer, logs):
    """Load previous ...

    """
    import json

    if isinstance(logs, str):
        logs = [logs]

    for log in logs:
        with open(log, "r") as j:
            while True:
                try:
                    iteration = next(j)
                except StopIteration:
                    break

                iteration = json.loads(iteration)
                try:
                    optimizer.register(
                        params=iteration["params"],
                        target=iteration["target"],
                    )
                except KeyError:
                    pass

    return optimizer


def ensure_rng(random_state=None):
    """
    Creates a random number generator based on an optional seed.  This can be
    an integer or another random state for a seeded rng, or None for an
    unseeded rng.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state


class Colours:
    """Print in nice colours."""

    BLUE = '\033[94m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    END = '\033[0m'
    GREEN = '\033[92m'
    PURPLE = '\033[95m'
    RED = '\033[91m'
    UNDERLINE = '\033[4m'
    YELLOW = '\033[93m'

    @classmethod
    def _wrap_colour(cls, s, colour):
        return colour + s + cls.END

    @classmethod
    def black(cls, s):
        """Wrap text in black."""
        return cls._wrap_colour(s, cls.END)

    @classmethod
    def blue(cls, s):
        """Wrap text in blue."""
        return cls._wrap_colour(s, cls.BLUE)

    @classmethod
    def bold(cls, s):
        """Wrap text in bold."""
        return cls._wrap_colour(s, cls.BOLD)

    @classmethod
    def cyan(cls, s):
        """Wrap text in cyan."""
        return cls._wrap_colour(s, cls.CYAN)

    @classmethod
    def darkcyan(cls, s):
        """Wrap text in darkcyan."""
        return cls._wrap_colour(s, cls.DARKCYAN)

    @classmethod
    def green(cls, s):
        """Wrap text in green."""
        return cls._wrap_colour(s, cls.GREEN)

    @classmethod
    def purple(cls, s):
        """Wrap text in purple."""
        return cls._wrap_colour(s, cls.PURPLE)

    @classmethod
    def red(cls, s):
        """Wrap text in red."""
        return cls._wrap_colour(s, cls.RED)

    @classmethod
    def underline(cls, s):
        """Wrap text in underline."""
        return cls._wrap_colour(s, cls.UNDERLINE)

    @classmethod
    def yellow(cls, s):
        """Wrap text in yellow."""
        return cls._wrap_colour(s, cls.YELLOW)
