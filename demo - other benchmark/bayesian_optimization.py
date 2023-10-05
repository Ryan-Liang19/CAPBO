import warnings
import numpy as np

from .target_space import TargetSpace
from .event import Events, DEFAULT_EVENTS
from .logger import _get_default_logger
from .util import UtilityFunction, acq_max, ensure_rng

from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.gaussian_process import GaussianProcessRegressor


class Queue:
    def __init__(self):
        self._queue = []

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._queue)

    def __next__(self):
        if self.empty:
            raise StopIteration("Queue is empty, no more objects to retrieve.")
        obj = self._queue[0]
        self._queue = self._queue[1:]
        return obj

    def next(self):
        return self.__next__()

    def add(self, obj):
        """Add object to end of queue."""
        self._queue.append(obj)


class Observable(object):
    """

    Inspired/Taken from
        https://www.protechtraining.com/blog/post/879#simple-observer
    """
    def __init__(self, events):
        # maps event names to subscribers
        # str -> dict
        self._events = {event: dict() for event in events}

    def get_subscribers(self, event):
        return self._events[event]

    def subscribe(self, event, subscriber, callback=None):
        if callback is None:
            callback = getattr(subscriber, 'update')
        self.get_subscribers(event)[subscriber] = callback

    def unsubscribe(self, event, subscriber):
        del self.get_subscribers(event)[subscriber]

    def dispatch(self, event):
        for _, callback in self.get_subscribers(event).items():
            callback(event, self)


class BayesianOptimization(Observable):
    def __init__(self, f, pbounds, random_state=None, verbose=2,
                 bounds_transformer=None):
        """"""
        self._random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._space = TargetSpace(f, pbounds, random_state)

        # queue
        self._queue = Queue()

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-4,
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=self._random_state,
        )

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer
        if self._bounds_transformer:
            self._bounds_transformer.initialize(self._space)

        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)

    @property
    def space(self):
        return self._space

    @property
    def max(self):
        return self._space.max()

    @property
    def res(self):
        return self._space.res()

    def register(self, params, target):
        """Expect observation with known target"""
        self._space.register(params, target)
        self.dispatch(Events.OPTIMIZATION_STEP)

    def probe(self, params, lazy=True, x_before=None):
        """Probe target of x"""
        if lazy:
            self._queue.add(params)
        else:
            if x_before is None:
                try:
                    self._space._params_before = np.concatenate([self._space._params_before,
                                                                 self._space.params[-1, :].reshape(1, -1)])
                except IndexError:
                    self._space._params_before = np.concatenate([self._space._params_before,
                                                                 np.zeros((1, self._space._params_before.shape[1]))])
            else:
                self._space._params_before = np.concatenate([self._space._params_before, x_before.reshape(1, -1)])
            self._space.probe(params)
            self.dispatch(Events.OPTIMIZATION_STEP)

    def suggest(self, utility_function, port, pen=0.0, alpha=1.0, x_before=None, cost_var=None, cost=None):
        """Most promissing point to probe next"""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)

        x_tries = self._random_state.uniform(self._space.bounds[:, 0], self._space.bounds[:, 1],
                                             size=(10000, self._space.bounds.shape[0]))
        x_seeds = self._random_state.uniform(self._space.bounds[:, 0], self._space.bounds[:, 1],
                                             size=(10, self._space.bounds.shape[0]))

        # Linear fitting, used for cost estimation
        if cost_var is not None:
            mat = np.zeros(self._space._params_before.shape)
            params = self._space.params[:self._space._params_before.shape[0], :]
            for i in range(0, cost_var.shape[0]):
                if cost_var[i] == 1:
                    mat[:, i] = np.abs(params[:, i] - self._space._params_before[:, i])
                elif cost_var[i] == 2:
                    mat[:, i] = 1 / (params[:, i] + 0.1)
                elif cost_var[i] == 4:
                    mat[:, i] = params[:, i]
            model = LinearRegression().fit(mat, cost)
            cost_var = np.vstack((cost_var, np.hstack((model.coef_, model.intercept_))))

        else:
            if x_before is None:
                x_before=list(self.res[-1]['params'].values())
            suggestion = acq_max(
                ac=utility_function.utility,
                gp=self._gp,
                y_max=self._space.target.max(),
                bounds=self._space.bounds,
                random_state=self._random_state,
                x_tries=x_tries, x_seeds=x_seeds,
                x_before=x_before,
                pen=pen,
                alpha=alpha,
                cost_var=cost_var
            )
            return self._space.array_to_params(suggestion)

    def _prime_queue(self, init_points):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        for _ in range(init_points):
            self._queue.add(self._space.random_sample())

    def _prime_subscriptions(self):
        if not any([len(subs) for subs in self._events.values()]):
            _logger = _get_default_logger(self._verbose)
            self.subscribe(Events.OPTIMIZATION_START, _logger)
            self.subscribe(Events.OPTIMIZATION_STEP, _logger)
            self.subscribe(Events.OPTIMIZATION_END, _logger)

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 kappa_decay=1,
                 kappa_decay_delay=0,
                 xi=0.0,
                 pen=0.0,
                 alpha=1.0,
                 direct=True,
                 cost_var=None,
                 cost=None,
                 x_before=None,
                 **gp_params):
        """Mazimize your function"""
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points)
        self.set_gp_params(**gp_params)

        util = UtilityFunction(kind=acq,
                               kappa=kappa,
                               xi=xi,
                               kappa_decay=kappa_decay,
                               kappa_decay_delay=kappa_decay_delay,
                               pen=pen,
                               alpha=alpha)
        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                util.update_params()
                if acq == 'port':
                    flag = True
                    [x1, x2, x3, x4, x5, x6] = self.suggest(util, flag, pen)
                else:
                    flag = False
                    x_probe = self.suggest(util, flag, pen, alpha, x_before, cost_var, cost)
                iteration += 1

            if direct:
                self.probe(x_probe, lazy=False, x_before=x_before)
            else:
                return x_probe

            if self._bounds_transformer:
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

        self.dispatch(Events.OPTIMIZATION_END)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        self._space.set_bounds(new_bounds)

    def set_gp_params(self, **params):
        self._gp.set_params(**params)
