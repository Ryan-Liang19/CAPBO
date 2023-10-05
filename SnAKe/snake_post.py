import numpy as np
import pandas as pd


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


for run_num in range(50):
    path = r'C:\Users\60494\PycharmProjects\Bayesian\SnAKe-main\experiment_results\SnAKe\Hartmann6D\budget131\inputs'
    X = np.load(path + r'\run_' + str(run_num) + '.npy')

    pbounds = {'x1': (0, 1), 'x2': (0, 1), 'x3': (0, 1), 'x4': (0, 1), 'x5': (0, 1), 'x6': (0, 1)}
    var_type = np.array([[1, 1, 2, 3, 3, 2, 0], [200, 100, 20, 0, 0, 40, 20]])
    t_step = [0]
    t_step.append(time_compute(X[0, :], list([0, 0, 0, 0, 0, 0]), var_type))
    for i in range(1, X.shape[0]):
        t_step.append(time_compute(X[i, :], X[i - 1, :], var_type) + t_step[-1])

    df = pd.read_excel(r'C:\Users\60494\PycharmProjects\Bayesian\sensitivity_4dim_time.xlsx', header=0)
    df[run_num] = t_step
    pd.DataFrame(df).to_excel(r'C:\Users\60494\PycharmProjects\Bayesian\sensitivity_4dim_time.xlsx',
                              index=False, header=True)
