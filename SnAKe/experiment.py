import torch
from gp_utils import BoTorchGP
from functions import BraninFunction, Hartmann6D, Hartmann4D, Ackley4D, Michalewicz2D, Perm10D, Hartmann3D
from snake import RandomTSP, SnAKe
from bayes_op import UCBwLP, oneExpectedImprovement, oneProbabilityOfImprovement, EIperUnitCost, TruncatedExpectedImprovement
from temperature_env import NormalDropletFunctionEnv
from scipy.spatial import distance_matrix
import numpy as np
import sys
import os

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

# method = str(sys.argv[1])
# function_number = int(float(sys.argv[2]))
# run_num = int(sys.argv[3])
# budget = int(sys.argv[4])
# epsilon = float(sys.argv[5])
# cost_func = int(sys.argv[6])
method = 'SnAKe'
function_number = 2
run_num = 0
budget = 250
epsilon = 0.1
cost_func = 2
gamma = 1

# epsilon = 'lengthscale'

# for function_number in range(0, 6):
if 1:
    # for run_num in range(1, 11):
    if 1:
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
        functions = [BraninFunction(), Hartmann3D(), Hartmann6D(), Ackley4D(), Michalewicz2D(), Perm10D()]
        func = functions[function_number]

        # We start counting from zero, so set budget minus one
        budget = budget - 1

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
        seed = run_num + function_number * 335
        torch.manual_seed(seed)
        np.random.seed(seed)

        initial_temp = np.random.uniform(size = (1, func.t_dim)).reshape(1, -1)

        dim = func.t_dim
        if func.x_dim is not None:
            dim = dim + func.x_dim

        x_train = np.random.uniform(0, 1, size = (max(int(budget / 5), 10 * dim), dim))
        y_train = []
        for i in range(0, x_train.shape[0]):
            y_train.append(func.query_function(x_train[i, :].reshape(1, -1)))

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

        X, Y = mod.run_optim(verbose = True)

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