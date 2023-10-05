import numpy as np
import pandas as pd
import warnings
import time
from bayes_opt import BayesianOptimization
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import LinearRegression

from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.util import acq_max


def black_box_function(x1, x2, x3, x4, x5, x6):
    alpha = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.06, 10, 0.1, 14]])
    c = np.array([1, 1.2, 3, 3.2])
    p = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                  [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                  [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                  [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    X = [x1, x2, x3, x4, x5, x6] * np.ones([4, 1])
    return np.sum(c * np.exp(-np.sum(alpha * (X - p) ** 2, axis=1)))


def time_compute(x, x_before, type):
    t = 0.0
    for i in range(0, type.shape[1]):
        if type[0, i] == 1:
            t += type[1, i] * np.abs(x[i] - x_before[i])
        elif type[0, i] == 2:
            t += type[1, i] / (x[i] + 0.1)
        elif type[0, i] == 0:
            t += type[1, i]
    return t


for seed in range(0, 50):
    # read existing model data
    pbounds = {'x1': (0, 1), 'x2': (0, 1), 'x3': (0, 1), 'x4': (0, 1), 'x5': (0, 1), 'x6': (0, 1)}
    cost_var = np.array([1, 1, 2, 3, 3, 2, 0])
    var_type = np.array([[1, 1, 2, 3, 3, 2, 0], [200, 100, 20, 0, 0, 40, 20]])
    num_proc = 6
    warnings.filterwarnings('ignore')

    # switch of different method
    method1 = True
    method2 = False
    method3 = False
    method4 = False
    method5 = False
    method6 = False
    random_seed = 1024
    sam_num = 50
    penalize = np.array([10, 8, 6, 4, 2, 0])
    t1 = time.time()

    # method 1: pure Bayesian optimization only: EI
    if method1:
        optimizer1 = BayesianOptimization(f=black_box_function, pbounds=pbounds, verbose=2, random_state=seed+random_seed)
        t_step = []
        ymax = []
        t_step.append(0)
        ymax.append(0)
        optimizer1.maximize(init_points=1, n_iter=0, acq='ei', xi=0.0)
        ymax.append(optimizer1.max['target'])
        t_step.append(time_compute(list(optimizer1.res[-1]['params'].values()), list([0, 0, 0, 0, 0, 0]), var_type))
        for i in range(1, 21):
            optimizer1.maximize(init_points=1, n_iter=0, acq='ei', xi=0.0)
            ymax.append(optimizer1.max['target'])
            t_step.append(t_step[-1] + time_compute(list(optimizer1.res[-1]['params'].values()),
                                                    list(optimizer1.res[-2]['params'].values()), var_type))
        for i in range(21, 150):
            optimizer1.maximize(init_points=0, n_iter=1, acq='ei', xi=0.0)
            ymax.append(optimizer1.max['target'])
            t_step.append(t_step[-1] + time_compute(list(optimizer1.res[-1]['params'].values()),
                                                    list(optimizer1.res[-2]['params'].values()), var_type))
        ymax = np.asarray(ymax)

    # method 2: Bayesian optimization based on EI per minute
    if method2:
        optimizer2 = BayesianOptimization(f=black_box_function, pbounds=pbounds, verbose=2, random_state=seed+random_seed)
        t_step = []
        ymax = []
        t_step.append(0)
        ymax.append(0)
        optimizer2.maximize(init_points=1, n_iter=0, acq='ei', xi=0.0)
        ymax.append(optimizer2.max['target'])
        t_step.append(time_compute(list(optimizer2.res[-1]['params'].values()), list([0, 0, 0, 0, 0, 0]), var_type))
        for i in range(1, 21):
            optimizer2.maximize(init_points=1, n_iter=0, acq='ei', xi=0.0)
            ymax.append(optimizer2.max['target'])
            t_step.append(t_step[-1] + time_compute(list(optimizer2.res[-1]['params'].values()),
                                                    list(optimizer2.res[-2]['params'].values()), var_type))
        for i in range(21, 150):
            optimizer2.maximize(init_points=0, n_iter=1, acq='ei_pm', xi=0.0)
            ymax.append(optimizer2.max['target'])
            t_step.append(t_step[-1] + time_compute(list(optimizer2.res[-1]['params'].values()),
                                                    list(optimizer2.res[-2]['params'].values()), var_type))
        ymax = np.asarray(ymax)

    # method 3: Bayesian optimization based on EI-cool
    if method3:
        optimizer6 = BayesianOptimization(f=black_box_function, pbounds=pbounds, verbose=2, random_state=seed+random_seed)
        t_step = []
        ymax = []
        t_step.append(0)
        ymax.append(0)
        optimizer6.maximize(init_points=1, n_iter=0, acq='ei', xi=0.0)
        ymax.append(optimizer6.max['target'])
        t_step.append(time_compute(list(optimizer6.res[-1]['params'].values()), list([0, 0, 0, 0, 0, 0]), var_type))
        t_max = 15000
        for i in range(1, 21):
            optimizer6.maximize(init_points=1, n_iter=0, acq='ei', xi=0.0)
            ymax.append(optimizer6.max['target'])
            t_step.append(t_step[-1] + time_compute(list(optimizer6.res[-1]['params'].values()),
                                                    list(optimizer6.res[-2]['params'].values()), var_type))
        t_init = t_step[-1]
        for i in range(21, 150):
            alpha = (t_max - t_step[-1]) / (t_max - t_init)
            if alpha < 0:
                alpha = 0
            optimizer6.maximize(init_points=0, n_iter=1, acq='ei_cool', xi=0.0, alpha=alpha)
            ymax.append(optimizer6.max['target'])
            t_step.append(t_step[-1] + time_compute(list(optimizer6.res[-1]['params'].values()),
                                                    list(optimizer6.res[-2]['params'].values()), var_type))
        ymax = np.asarray(ymax)

    # method 4: ACBO (multi-armed bandits)
    if method4:
        optimizer7 = BayesianOptimization(f=black_box_function, pbounds=pbounds, verbose=2, random_state=seed+random_seed)
        t_step = []
        ymax = []
        t_step.append(0)
        ymax.append(0)
        optimizer7.maximize(init_points=1, n_iter=0, acq='ei', xi=0.0)
        ymax.append(optimizer7.max['target'])
        t_step.append(time_compute(list(optimizer7.res[-1]['params'].values()), list([0, 0, 0, 0, 0, 0]), var_type))
        flag = 0
        D1 = np.array(list(optimizer7.res[-1]['params'].values())).reshape(1, -1)
        D1_obj = np.array([optimizer7.res[-1]['target']])
        for i in range(1, 21):
            optimizer7.maximize(init_points=1, n_iter=0, acq='ei', xi=0.0)
            ymax.append(optimizer7.max['target'])
            t_step.append(t_step[-1] + time_compute(list(optimizer7.res[-1]['params'].values()),
                                                    list(optimizer7.res[-2]['params'].values()), var_type))
            D1 = np.concatenate([D1, np.array(list(optimizer7.res[-1]['params'].values())).reshape(1, -1)], axis=0)
            D1_obj = np.concatenate([D1_obj, [optimizer7.res[-1]['target']]])
        D2 = D1.copy()
        D2_obj = D1_obj.copy()
        gpr1 = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-4, normalize_y=True, n_restarts_optimizer=10, random_state=seed+random_seed)
        gpr2 = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-4, normalize_y=True, n_restarts_optimizer=10, random_state=seed+random_seed)
        for i in range(21, 150):
            gpr1.fit(D1, D1_obj)
            x_seed = np.random.rand(2000, 6)
            y_seed1 = gpr1.sample_y(x_seed, random_state=seed+random_seed+i)
            gpr2.fit(D2, D2_obj)
            y_seed2 = gpr2.sample_y(x_seed, random_state=seed+random_seed+i)
            if (np.max(y_seed1) >= np.max(y_seed2)):
                optimizer7.maximize(init_points=0, n_iter=1, acq='ei', xi=0.0)
                print('EI used!')
                D1 = np.concatenate([D1, np.array(list(optimizer7.res[-1]['params'].values())).reshape(1, -1)], axis=0)
                D1_obj = np.concatenate([D1_obj, [optimizer7.res[-1]['target']]])
            else:
                optimizer7.maximize(init_points=0, n_iter=1, acq='ei_pm', xi=0.0)
                print('EIpu used!')
                D2 = np.concatenate([D2, np.array(list(optimizer7.res[-1]['params'].values())).reshape(1, -1)], axis=0)
                D2_obj = np.concatenate([D2_obj, [optimizer7.res[-1]['target']]])
            ymax.append(optimizer7.max['target'])
            t_step.append(t_step[-1] + time_compute(list(optimizer7.res[-1]['params'].values()),
                                                    list(optimizer7.res[-2]['params'].values()), var_type))
        ymax = np.asarray(ymax)

    # method 5: Kriging believer
    if method5:
        optimizer_master = BayesianOptimization(f=black_box_function, pbounds=pbounds, verbose=2,
                                                random_state=seed + random_seed)
        y = []
        t_step = []
        t_pass = []
        cost = np.empty(shape=(0))
        y = []
        ymax = []
        t_step.append(0)
        ymax.append(0)
        pending = np.zeros([num_proc, 6])
        pending_before = np.zeros([num_proc, 6])
        pending_t = np.zeros(num_proc)
        pending_record = np.zeros(num_proc)
        gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-4, normalize_y=True, n_restarts_optimizer=10,
                                       random_state=seed + random_seed)
        for i in range(num_proc):
            pending[i, :] = optimizer_master.maximize(init_points=1, n_iter=0, acq='ei', xi=0.0, direct=False)
            pending_t[i] = time_compute(pending[i, :], list([0, 0, 0, 0, 0, 0]), var_type)
            pending_record[i] = pending_t[i]
        for i in range(num_proc, 21):
            proc_next = np.argmin(pending_t)
            optimizer_master.probe(pending[proc_next, :], lazy=False, x_before=pending_before[proc_next, :])
            cost = np.concatenate([cost, [pending_record[proc_next]]])
            y.append(optimizer_master.res[-1]['target'])
            ymax.append(optimizer_master.max['target'])
            t_one = pending_t[proc_next]
            t_step.append(t_step[-1] + t_one)
            pending_t -= t_one
            pending_before[proc_next, :] = pending[proc_next, :]
            pending[proc_next, :] = optimizer_master.maximize(init_points=1, n_iter=0, acq='ei', xi=0.0, direct=False)
            pending_t[proc_next] = time_compute(pending[proc_next, :],
                                                list(optimizer_master.res[-1]['params'].values()), var_type)
            pending_record[proc_next] = pending_t[proc_next]
        for ii in range(i + 1, 150 + num_proc):
            proc_next = np.argmin(pending_t)
            optimizer_master.probe(pending[proc_next, :], lazy=False, x_before=pending_before[proc_next, :])
            cost = np.concatenate([cost, [pending_record[proc_next]]])
            y.append(optimizer_master.res[-1]['target'])
            ymax.append(optimizer_master.max['target'])
            t_one = pending_t[proc_next]
            t_step.append(t_step[-1] + t_one)
            pending_t -= t_one
            gpr.fit(optimizer_master._space._params, optimizer_master._space._target)
            for j in range(num_proc):
                if j != proc_next:
                    optimizer_master._space._params = np.vstack((optimizer_master._space._params, pending[j]))
                    kb = gpr.predict(pending[j][None])
                    optimizer_master._space._target = np.hstack((optimizer_master._space._target, kb))
            pending_before[proc_next, :] = pending[proc_next, :]
            pending[proc_next, :] = list(optimizer_master.maximize(init_points=0, n_iter=1, acq='ei',
                                                                   xi=0.0, direct=False, cost_var=cost_var, cost=cost,
                                                                   x_before=pending[proc_next]).values())
            optimizer_master._space._params = np.delete(optimizer_master._space._params, range(-num_proc + 1, 0), 0)
            optimizer_master._space._target = np.delete(optimizer_master._space._target, range(-num_proc + 1, 0), 0)
            pending_t[proc_next] = time_compute(pending[proc_next, :],
                                                list(optimizer_master.res[-1]['params'].values()), var_type)
            pending_record[proc_next] = pending_t[proc_next]

    # method 6: Monte Carlo acq
    if method6:
        optimizer_master = BayesianOptimization(f=black_box_function, pbounds=pbounds, verbose=2,
                                                random_state=seed + random_seed)
        y = []
        t_step = []
        t_pass = []
        cost = np.empty(shape=(0))
        ymax = []
        t_step.append(0)
        ymax.append(0)
        gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-4, normalize_y=True, n_restarts_optimizer=10,
                                       random_state=seed + random_seed)
        pending = np.zeros([num_proc, 6])
        pending_before = np.zeros([num_proc, 6])
        pending_t = np.zeros(num_proc)
        pending_record = np.zeros(num_proc)
        fake_set = []
        for i in range(sam_num):
            utility_function = UtilityFunction(kind='ei', kappa=3, xi=0.0)
            gpr_fake = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-4, normalize_y=True,
                                                n_restarts_optimizer=10, random_state=seed + random_seed)
            fake_set.append(gpr_fake)
        for i in range(num_proc):
            pending[i, :] = optimizer_master.maximize(init_points=1, n_iter=0, acq='ei', xi=0.0, direct=False)
            pending_t[i] = time_compute(pending[i, :], list([0, 0, 0, 0, 0, 0]), var_type)
            pending_record[i] = pending_t[i]
        for i in range(num_proc, 21):
            proc_next = np.argmin(pending_t)
            optimizer_master.probe(pending[proc_next, :], lazy=False, x_before=pending_before[proc_next, :])
            cost = np.concatenate([cost, [pending_record[proc_next]]])
            y.append(optimizer_master.res[-1]['target'])
            ymax.append(optimizer_master.max['target'])
            t_one = pending_t[proc_next]
            t_step.append(t_step[-1] + t_one)
            pending_t -= t_one
            pending_before[proc_next, :] = pending[proc_next, :]
            pending[proc_next, :] = optimizer_master.maximize(init_points=1, n_iter=0, acq='ei', xi=0.0, direct=False)
            pending_t[proc_next] = time_compute(pending[proc_next, :],
                                                list(optimizer_master.res[-1]['params'].values()), var_type)
            pending_record[proc_next] = pending_t[proc_next]
        x_tries = optimizer_master._random_state.uniform(optimizer_master._space.bounds[:, 0],
                                                         optimizer_master._space.bounds[:, 1],
                                                         size=(21, optimizer_master._space.bounds.shape[0]))
        for ii in range(i + 1, 150 + num_proc):
            proc_next = np.argmin(pending_t)
            optimizer_master.probe(pending[proc_next, :], lazy=False, x_before=pending_before[proc_next, :])
            cost = np.concatenate([cost, [pending_record[proc_next]]])
            y.append(optimizer_master.res[-1]['target'])
            ymax.append(optimizer_master.max['target'])
            t_one = pending_t[proc_next]
            t_step.append(t_step[-1] + t_one)
            pending_t -= t_one
            gpr.fit(optimizer_master._space._params, optimizer_master._space._target)
            y_seed = gpr.sample_y(pending, n_samples=sam_num)
            fake_y_max = []
            x_tries = optimizer_master._random_state.uniform(optimizer_master._space.bounds[:, 0],
                                                             optimizer_master._space.bounds[:, 1],
                                                             size=(10000, optimizer_master._space.bounds.shape[0]))
            x_seeds = optimizer_master._random_state.uniform(optimizer_master._space.bounds[:, 0],
                                                             optimizer_master._space.bounds[:, 1],
                                                             size=(10, optimizer_master._space.bounds.shape[0]))
            for j in range(sam_num):
                temp_params = np.vstack((optimizer_master._space._params, pending))
                temp_target = np.hstack((optimizer_master._space._target, y_seed[:, j]))
                fake_set[j].fit(temp_params, temp_target)
                fake_y_max.append(max(temp_target))
            pending_before[proc_next, :] = pending[proc_next, :]
            mat = np.zeros(optimizer_master._space._params.shape)
            for i in range(0, cost_var.shape[0]):
                if cost_var[i] == 1:
                    mat[:, i] = np.abs(
                        optimizer_master._space._params[:, i] - optimizer_master._space._params_before[:, i])
                elif cost_var[i] == 2:
                    mat[:, i] = 1 / (optimizer_master._space._params[:, i] + 0.1)
            model = LinearRegression().fit(mat, cost)
            cost_varfit = np.vstack((cost_var, np.hstack((model.coef_, model.intercept_))))
            suggestion = acq_max(ac=utility_function.utility, gp=fake_set, y_max=fake_y_max,
                                 bounds=optimizer_master._space.bounds, random_state=optimizer_master._random_state,
                                 x_tries=x_tries, x_seeds=x_seeds,
                                 x_before=list(optimizer_master.res[-1]['params'].values()),
                                 cost_var=cost_varfit)
            pending[proc_next, :] = suggestion
            pending_t[proc_next] = time_compute(pending[proc_next, :],
                                                list(optimizer_master.res[-1]['params'].values()), var_type)
            pending_record[proc_next] = pending_t[proc_next]

    t2 = time.time()
    print('Time spent: {0}'.format(t2 - t1))
    df = pd.read_excel('sensitivity_6dim_time.xlsx', header=0)
    df[seed] = t_step
    pd.DataFrame(df).to_excel('sensitivity_6dim_time.xlsx', index=False, header=True)
    df = pd.read_excel('sensitivity_6dim_result.xlsx', header=0)
    df[seed] = ymax
    pd.DataFrame(df).to_excel('sensitivity_6dim_result.xlsx', index=False, header=True)
