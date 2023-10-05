import torch
import pandas as pd
import time
from gp_utils import BoTorchGP
from functions import BraninFunction, Hartmann6D, Hartmann4D, Ackley4D, Michalewicz2D, Perm10D, Hartmann3D, Alpine4D, Ackley5D, mixed2D, mixed4D, mixed6D, gls4D
from snake import RandomTSP, SnAKe
from bayes_op import UCBwLP, oneExpectedImprovement, oneProbabilityOfImprovement, EIperUnitCost, TruncatedExpectedImprovement
from temperature_env import NormalDropletFunctionEnv
from scipy.spatial import distance_matrix
import numpy as np
import sys
import os
from bayes_opt import BayesianOptimization
from GaussianModel import NormalMix
from scipy.integrate import odeint
from scipy.interpolate import interp1d

'''
This script was used to get the synchronous experiment results on synthetic benchmarks.

To reproduce any run, type:

python experiment_async 'method' 'function_number' 'run_number' 'budget' 'epsilon' 'cost_func' 

Where:

method - 'SnAKe', 'EI', 'UCB', 'PI', 'Random', 'EIpu', 'TrEI'
function number - integer between 0 and 5
run number - any integer, in experiments we used 1-10 inclusive
budget - integer in [15, 50, 100, 250]
epsilon - integer [0, 0.1, 1.0], alternatively modify the script to set epsilon = 'lengthscale' for ell-SnAKe
cost_func - 1, 2, 3 corresponding to 1-norm, 2-norm, inf-norm
'''

method = 'SnAKe'
function_number = 2
run_num = 0
budget = 130
epsilon = 0.1
cost_func = 2
gamma = 1

# epsilon = 'lengthscale'

# for function_number in range(0, 6):
if 1:
    for run_num in range(0, 50):
    # if 1:
        t1 = time.time()
        print(method, function_number, run_num, budget, gamma, cost_func)
        # budget = 250
        # Make sure problem is well defined
        # assert method in ['SnAKe', 'EI', 'UCB', 'PI', 'Random', 'EIpu', 'TrEI'], 'Method must be string in [SnAKe, EI, UCB, PI, Random]'
        # assert function_number in range(6), \
        #     'Function must be integer between 0 and 5'
        # assert budget in [15, 50, 100, 250], \
        #     'Budget must be integer in [15, 50, 100, 250]'
        # assert epsilon in [0, 0.1, 0.25, 1, 'lengthscale'], \
        #     'Epsilon must be in [0, 0.1, 0.25, 1, lengthscale]'
        # assert cost_func in [1, 2, 3], \
        #     'Cost function must be integer in [1, 2, 3] (where 3 corresponds to infinity norm)'

        # Define function name
        functions = [BraninFunction(), Hartmann3D(), Hartmann6D(), Ackley4D(), Michalewicz2D(), Perm10D(), Alpine4D(), Ackley5D(), mixed2D(), mixed4D(), mixed6D(), gls4D()]
        func = functions[function_number]

        # We start counting from zero, so set budget minus one
        # budget = budget - 1

        # Define cost function
        if cost_func == 1:
            cost_function = lambda x, y: distance_matrix(x, y, p = 1)
            cost_name = '1norm'
        elif cost_func == 2:
            cost_function = lambda x, y: distance_matrix(x, y, p = 2)
            cost_name = '2norm'
        elif cost_func == 3:
            cost_function = lambda x, y: distance_matrix(x, y, p = float('inf'))
            cost_name = 'inftynorm'

        # Define seed, sample initalisation points
        # seed = run_num + function_number * 335
        seed = run_num + 1024
        torch.manual_seed(seed)
        np.random.seed(seed)

        initial_temp = np.random.uniform(size = (1, func.t_dim)).reshape(1, -1)

        dim = func.t_dim
        if func.x_dim is not None:
            dim = dim + func.x_dim

        x_train = np.random.uniform(0, 1, size = (max(int(budget / 5), 10 * dim), dim))
        y_train = []
        for i in range(0, x_train.shape[0]):
            y_train.append(func.query_function(x_train[i, :].reshape(1, -1), run_num))

        y_train = np.array(y_train)

        # Train and set educated guess of hyper-parameters
        gp_model = BoTorchGP(lengthscale_dim = dim)

        gp_model.fit_model(x_train, y_train)
        gp_model.optim_hyperparams()

        hypers = gp_model.current_hyperparams()

        print('Initial hyper-parameters:', hypers)
        # Define Normal BayesOp Environment without delay
        env = NormalDropletFunctionEnv(func, budget, max_batch_size = 1)

        # Choose the correct method
        if method == 'SnAKe':
            mod = SnAKe(env, merge_method = 'e-Point Deletion', merge_constant = epsilon, cost_function = cost_function, initial_temp = initial_temp, \
                hp_update_frequency = 25)
        elif method == 'EI':
            mod = oneExpectedImprovement(env, initial_temp = initial_temp, hp_update_frequency = 25)
        elif method == 'UCB':
            mod = UCBwLP(env, initial_temp = initial_temp, hp_update_frequency = 25)
        elif method == 'PI':
            mod = oneProbabilityOfImprovement(env, initial_temp = initial_temp, hp_update_frequency = 25)
        elif method == 'Random':
            mod = RandomTSP(env, initial_temp = initial_temp)
        elif method == 'EIpu':
            mod = EIperUnitCost(env, initial_temp = initial_temp, cost_constant = gamma)
        elif method == 'TrEI':
            mod = TruncatedExpectedImprovement(env, initial_temp = initial_temp)

        mod.set_hyperparams(constant = hypers[0], lengthscale = hypers[1], noise = hypers[2], mean_constant = hypers[3], \
                    constraints = True)

        # def black_box_function(x1, x2, x3):
        #     alpha = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
        #     c = np.array([1, 1.2, 3, 3.2])
        #     p = np.array([[0.3689, 0.1170, 0.2673], [0.4699, 0.4387, 0.7470],
        #                   [0.1091, 0.8732, 0.5547], [0.3810, 0.5743, 0.8828]])
        #     X = [x1, x2, x3] * np.ones([4, 1])
        #     return np.sum(c * np.exp(-np.sum(alpha * (X - p) ** 2, axis=1)))
        #
        # pbounds = {'x1': (0, 1), 'x2': (0, 1), 'x3': (0, 1)}

        # def black_box_function(x1, x2, x3, x4):
        #     x1 = 10 * x1 - 10
        #     x2 = 10 * x2 - 20 / 3
        #     x3 = 10 * x3 - 10 / 3
        #     x4 = 10 * x4 - 0
        #     y1 = np.abs(x1 * np.sin(x1) + 0.1 * x1)
        #     y2 = np.abs(x2 * np.sin(x2) + 0.1 * x2)
        #     y3 = np.abs(x3 * np.sin(x3) + 0.1 * x3)
        #     y4 = np.abs(x4 * np.sin(x4) + 0.1 * x4)
        #     return -(y1 + y2 + y3 + y4)
        #
        # pbounds = {'x1': (0, 1), 'x2': (0, 1), 'x3': (0, 1), 'x4': (0, 1)}

        # def black_box_function(x1, x2, x3, x4, x5):
        #     x1 = 30 * x1 - 25
        #     x2 = 30 * x2 - 20
        #     x3 = 30 * x3 - 15
        #     x4 = 30 * x4 - 10
        #     x5 = 30 * x5 - 5
        #     a = 20
        #     b = 0.2
        #     c = 2 * np.pi
        #     return (a * np.exp(-b * np.sqrt(1 / 5 * (x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 + x5 ** 2))) +
        #             np.exp(1 / 5 * (np.cos(c * x1) + np.cos(c * x2) + np.cos(c * x3) + np.cos(c * x4) + np.cos(c * x5))) -
        #             a - np.exp(1))
        #
        # pbounds = {'x1': (0, 1), 'x2': (0, 1), 'x3': (0, 1), 'x4': (0, 1), 'x5': (0, 1)}


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

        pbounds = {'x1': (0, 1), 'x2': (0, 1), 'x3': (0, 1), 'x4': (0, 1), 'x5': (0, 1), 'x6': (0, 1)}

        # dim = 2
        # num = 5
        # df = pd.read_excel(r'C:\Users\60494\PycharmProjects\Bayesian\GaussianModel_2dim.xlsx', sheet_name=run_num, header=None)
        # df = df.values
        # mu = df[0, :]
        # mu = np.reshape(mu[~np.isnan(mu)], (dim, num))
        # sigma = df[1, :]
        # sigma = np.reshape(sigma[~np.isnan(sigma)], (num, dim, dim))
        # b = df[2, :]
        # b = np.reshape(b[~np.isnan(b)], (-1, 1))
        # pbounds = {'x1': (0, 1), 'x2': (0, 1)}
        #
        # def black_box_function(x1, x2, noise=True):
        #     X = np.array([x1, x2])
        #     if noise:
        #         return 10 * (NormalMix(X, mu, sigma, b, num=5) + np.random.normal(loc=0.0, scale=0.005))
        #     else:
        #         return NormalMix(X, mu, sigma, b, num=5)

        # dim = 4
        # num = 12
        # df = pd.read_excel(r'C:\Users\60494\PycharmProjects\Bayesian\GaussianModel_4dim.xlsx', sheet_name=run_num, header=None)
        # df = df.values
        # mu = df[0, :]
        # mu = np.reshape(mu[~np.isnan(mu)], (dim, num))
        # sigma = df[1, :]
        # sigma = np.reshape(sigma[~np.isnan(sigma)], (num, dim, dim))
        # b = df[2, :]
        # b = np.reshape(b[~np.isnan(b)], (-1, 1))
        # pbounds = {'x1': (0, 1), 'x2': (0, 1), 'x3': (0, 1), 'x4': (0, 1)}
        #
        #
        # def black_box_function(x1, x2, x3, x4, noise=True):
        #     X = np.array([x1, x2, x3, x4])
        #     if noise:
        #         return 10 * (NormalMix(X, mu, sigma, b, num=12) + np.random.normal(loc=0.0, scale=0.005))
        #     else:
        #         return NormalMix(X, mu, sigma, b, num=12)

        # dim = 6
        # num = 20
        # df = pd.read_excel(r'C:\Users\60494\PycharmProjects\Bayesian\GaussianModel_6dim.xlsx', sheet_name=run_num, header=None)
        # df = df.values
        # mu = df[0, :]
        # mu = np.reshape(mu[~np.isnan(mu)], (dim, num))
        # sigma = df[1, :]
        # sigma = np.reshape(sigma[~np.isnan(sigma)], (num, dim, dim))
        # b = df[2, :]
        # b = np.reshape(b[~np.isnan(b)], (-1, 1))
        # pbounds = {'x1': (0, 1), 'x2': (0, 1), 'x3': (0, 1), 'x4': (0, 1), 'x5': (0, 1), 'x6': (0, 1)}
        #
        # def black_box_function(x1, x2, x3, x4, x5, x6, noise=True):
        #     X = np.array([x1, x2, x3, x4, x5, x6])
        #     if noise:
        #         return 10 * (NormalMix(X, mu, sigma, b, num=20) + np.random.normal(loc=0.0, scale=0.005))
        #     else:
        #         return NormalMix(X, mu, sigma, b, num=20)


        # def dy_dt(c, t, T, P):
        #     A = 1e7 * np.array([0.4, 12, 0.14, 5.6, 0.18, 84])
        #     Ea = np.array([41.2, 52.4, 35.2, 58.4, 34.6, 52.1])
        #     k = A * np.exp(-Ea / 8.314 / (T + 273.15) * 1000)
        #     H_index = np.array([4.264, 4.438, 4.623, 4.792, 4.96])
        #     T_step = np.array([40, 50, 60, 70, 80])
        #     f1 = interp1d(T_step, H_index, kind='linear')
        #     H = f1(T)
        #     cH = H * 0.1 * P
        #     return np.array([-k[0] * c[0] * cH,
        #                      k[0] * c[0] * cH - k[1] * c[1] * cH,
        #                      k[1] * c[1] * cH - k[2] * c[2] * cH - k[3] * c[2] * c[3],
        #                      k[2] * c[2] * cH - k[3] * c[2] * c[3] - k[4] * c[3] * cH,
        #                      k[3] * c[2] * c[3] - k[5] * c[4] * cH,
        #                      k[4] * c[3] * cH + 2 * k[5] * c[4] * cH])
        #
        #
        # def black_box_function(x1, x2, x3, x4, noise=True):
        #     x1 = 40 * x1 + 40  # T belong to [40, 80]
        #     x2 = 2 * x2 + 1  # P belong to [1, 3]
        #     x3 = 19.999 * x3 + 0.001  # R belong to [0.001, 20]
        #     x4 = 0.49 * x4 + 0.01  # C belong to [0.01, 0.5]
        #     y0 = [x4, 0, 0, 0, 0, 0]
        #     t = np.arange(0, x3, 0.001)
        #     y = odeint(dy_dt, y0, t, args=(x1, x2))
        #     ans = y[-1][-1] + noise * np.random.normal(loc=0.0, scale=0.003)
        #     if ans >= 0.8 * x4:
        #         return ans / x3
        #     else:
        #         return 0

        # pbounds = {'x1': (0, 1), 'x2': (0, 1), 'x3': (0, 1), 'x4': (0, 1)}
        optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds, verbose=2, random_state=seed)
        for i in range(20):
            optimizer.maximize(init_points=1, n_iter=0, acq='ei', xi=0.0)
            mod.X.append(list(optimizer.res[-1]['params'].values()))
            mod.Y.append([optimizer.res[-1]['target']])
        X, Y = mod.run_optim(verbose = True, random_state=run_num)

        print(X)
        print(np.array(Y))

        if epsilon == 'lengthscale':
            epsilon = 'l'

        if method == 'EaS':
            folder_inputs = 'experiment_results/' + f'{epsilon}-EaS/' + func.name + f'/budget{budget + 1}/' + cost_name + '/inputs/'
            folder_outputs = 'experiment_results/' + f'{epsilon}-EaS/' + func.name + f'/budget{budget + 1}/' + cost_name + '/outputs/'
            file_name = f'run_{run_num}'
        elif method == 'Random':
            folder_inputs = 'experiment_results/' + f'Random/' + func.name + f'/budget{budget + 1}/' + cost_name + '/inputs/'
            folder_outputs = 'experiment_results/' + f'Random/' + func.name + f'/budget{budget + 1}/' + cost_name + '/outputs/'
            file_name = f'run_{run_num}'
        elif method == 'EIpu':
            folder_inputs =  'experiment_results/' + str(gamma) + method + '/' + func.name + '/' + f'/budget{budget + 1}/inputs/'
            folder_outputs =  'experiment_results/' + str(gamma) + method + '/' + func.name + '/' + f'/budget{budget + 1}/outputs/'
            file_name = f'run_{run_num}'
        else:
            folder_inputs =  'experiment_results/' + method + '/' + func.name + '/' + f'/budget{budget + 1}/inputs/'
            folder_outputs =  'experiment_results/' + method + '/' + func.name + '/' + f'/budget{budget + 1}/outputs/'
            file_name = f'run_{run_num}'

        # create directories if they exist
        os.makedirs(folder_inputs, exist_ok = True)
        os.makedirs(folder_outputs, exist_ok = True)

        print(X.shape[0])
        print(np.array(Y).shape[0])

        np.save(folder_inputs + file_name, X)
        np.save(folder_outputs + file_name, np.array(Y))


        def time_compute(x, x_before, type, noise=True):
            t = 0.0
            for i in range(0, type.shape[1]):
                if type[0, i] == 1:
                    t += type[1, i] * np.abs(x[i] - x_before[i])
                elif type[0, i] == 2:
                    t += type[1, i] / (x[i] + 0.1)
                elif type[0, i] == 4:
                    t += type[1, i] * x[i]
                elif type[0, i] == 0:
                    t += type[1, i]
            if noise:
                t += np.random.normal(loc=0.0, scale=5)
            return t

        var_type = np.array([[1, 1, 2, 3, 3, 2, 0], [200, 100, 20, 0, 0, 40, 20]])
        t_step = [0]
        t_step.append(time_compute(X[0, :], list([0, 0, 0, 0, 0, 0]), var_type))
        for i in range(1, X.shape[0]):
            t_step.append(time_compute(X[i, :], X[i - 1, :], var_type) + t_step[-1])

        y = np.asarray(Y)
        y = np.concatenate([[[0]], y], axis=0)
        ymax = -99999
        for i in range(y.shape[0]):
            if y[i, 0] > ymax:
                ymax = y[i, 0]
            else:
                y[i, 0] = ymax
        t2 = time.time()
        print('Time spent: {0}'.format(t2 - t1))
        df = pd.read_excel(r'C:\Users\60494\PycharmProjects\Bayesian\sensitivity_4dim_time.xlsx', header=0)
        df[run_num] = t_step
        pd.DataFrame(df).to_excel(r'C:\Users\60494\PycharmProjects\Bayesian\sensitivity_4dim_time.xlsx',
                                  index=False, header=True)
        df = pd.read_excel(r'C:\Users\60494\PycharmProjects\Bayesian\sensitivity_4dim_result.xlsx', header=0)
        df[run_num] = y.reshape(-1)
        pd.DataFrame(df).to_excel(r'C:\Users\60494\PycharmProjects\Bayesian\sensitivity_4dim_result.xlsx',
                                  index=False, header=True)
