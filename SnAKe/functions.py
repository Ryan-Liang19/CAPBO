import numpy as np
import math
import torch
import pandas as pd
from GaussianModel import NormalMix
from scipy.integrate import odeint
from scipy.interpolate import interp1d
# from summit.benchmarks import SnarBenchmark
# from summit.utils.dataset import DataSet

'''
In this script we include all the Benchmark function which we will evaluate.
'''

class ConvergenceTest():
    def __init__(self):
        self.x_dim = None
        self.t_dim = 1
        self.grid_search = False
    
    def draw_new_function(self):
        pass

    def query_function(self, temp):
        return np.sin(10 * temp) + np.exp(-(temp - 0.775) ** 2 / 0.1) / 3

class TwoDSinCosine():
    def __init__(self, random = False):
        self.t_dim = 2
        self.x_dim = None
        self.random = random
        self.optimum = 1.15371572971344
        self.draw_new_function()
        self.name = 'SineCosine 2D'
        self.grid_search = False

    def draw_new_function(self):
        if self.random:
            self.mu1 = np.random.uniform()
            self.mu2 = np.random.uniform()
        else:
            self.mu1 = 0
            self.mu2 = 0
        pass

    def query_function(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        return np.sin(5 * (x1 - self.mu1)) * np.cos(5 * (x2 - self.mu2)) * np.exp((x1-0.5)**2 / 2) * np.exp((x2-0.5)**2 / 2)

    def query_function_torch(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        return torch.sin(5 * (x1 - self.mu1)) * torch.cos(5 * (x2 - self.mu2)) * torch.exp((x1-0.5)**2 / 2) * torch.exp((x2-0.5)**2 / 2)

class BraninFunction():
    def __init__(self, t_dim = 2):
        self.t_dim = t_dim
        if self.t_dim == 2:
            self.x_dim = None
        else:
            self.x_dim = 2 - self.t_dim
        # optimum calculated using gradient methods (see code at the bottom)
        self.optimum = 1.0473939180374146
        self.grid_search = False

        self.name = 'Branin2D'
    
    def draw_new_function(self):
        pass

    def query_function(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        x1bar = 15 * x1 - 5
        x2bar = 15 * x2

        s1 = (x2bar - 5.1 * x1bar**2 / (4 * math.pi**2) + 5 * x1bar / math.pi - 6)**2
        s2 = (10 - 10 / (8 * math.pi)) * np.cos(x1bar) - 44.81

        return -(s1 + s2) / 51.95
    
    def query_function_torch(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        x1bar = 15 * x1 - 5
        x2bar = 15 * x2

        s1 = (x2bar - 5.1 * x1bar**2 / (4 * math.pi**2) + 5 * x1bar / math.pi - 6)**2
        s2 = (10 - 10 / (8 * math.pi)) * torch.cos(x1bar) - 44.81

        return -(s1 + s2) / 51.95

class Hartmann3D():
    def __init__(self, t_dim = 3):
        self.t_dim = t_dim
        if self.t_dim == 3:
            self.x_dim = None
        else:
            self.x_dim = 3 - self.t_dim
        # taken from website: https://www.sfu.ca/~ssurjano/hart3.html
        self.optimum = 3.8627797869493365
        self.grid_search = False

        self.name = 'Hartmann3D'

        self.A = np.array( \
            [[3, 10, 30], \
                [0.1, 10, 35], \
                    [3, 10, 30], \
                        [0.1, 10, 35]])
        
        self.P = 1e-4 * np.array( \
            [[3689, 1170, 2673], \
                [4699, 4387, 7470], \
                    [1091, 8732, 5547], \
                        [381, 5743, 8828]])

        self.alpha = np.array([1, 1.2, 3, 3.2])

    def draw_new_function(self):
        pass
    
    def query_function(self, x, seed=0):
        S1 = 0
        for i in range(0, 4):
            S2 = 0
            for j in range(0, 3):
                S2 += self.A[i, j] * (x[:, j] - self.P[i, j])**2
            S1 += self.alpha[i] * np.exp(-S2)
        return S1

class Hartmann4D():
    def __init__(self, t_dim = 4):
        self.t_dim = t_dim
        if self.t_dim == 4:
            self.x_dim = None
        else:
            self.x_dim = 4 - self.t_dim
        # taken from website: https://www.sfu.ca/~ssurjano/hart3.html
        self.optimum = 3.7298407554626465
        self.grid_search = False

        self.name = 'Hartmann4D'

        self.A = np.array( \
            [[10, 3, 17, 3.5, 1.7, 8], \
                [0.05, 10, 17, 0.1, 8, 14], \
                    [3, 3.5, 1.7, 10, 17, 8], \
                        [17, 8, 0.05, 10, 0.1, 14]])
        
        self.P = 1e-4 * np.array( \
            [[1312, 1696, 5569, 124, 8283, 5886], \
                [2329, 4135, 8307, 3736, 1004, 9991], \
                    [2348, 1451, 3522, 2883, 3047, 6650], \
                        [4047, 8828, 8732, 5743, 1091, 381]])

        self.alpha = np.array([1, 1.2, 3, 3.2])

    def draw_new_function(self):
        pass
    
    def query_function(self, x):
        S1 = 0
        for i in range(0, 4):
            S2 = 0
            for j in range(0, 4):
                S2 += self.A[i, j] * (x[:, j] - self.P[i, j])**2
            S1 += self.alpha[i] * np.exp(-S2)
        return S1

    def query_function_torch(self, x):
        A = torch.tensor(self.A)
        P = torch.tensor(self.P)
        alpha = torch.tensor(self.alpha)

        S1 = 0
        for i in range(0, 4):
            S2 = 0
            for j in range(0, 4):
                S2 += A[i, j] * (x[:, j] - P[i, j])**2
            S1 += alpha[i] * torch.exp(-S2)
        return S1

class Hartmann6D():
    def __init__(self, t_dim = 6):
        self.t_dim = t_dim
        if self.t_dim == 6:
            self.x_dim = None
        else:
            self.x_dim = 6 - self.t_dim
        # taken from website: https://www.sfu.ca/~ssurjano/hart6.html
        self.optimum = 3.322368011391339
        self.grid_search = False

        self.name = 'Hartmann6D'

        self.A = np.array( \
            [[10, 3, 17, 3.5, 1.7, 8], \
                [0.05, 10, 17, 0.1, 8, 14], \
                    [3, 3.5, 1.7, 10, 17, 8], \
                        [17, 8, 0.05, 10, 0.1, 14]])
        
        self.P = 1e-4 * np.array( \
            [[1312, 1696, 5569, 124, 8283, 5886], \
                [2329, 4135, 8307, 3736, 1004, 9991], \
                    [2348, 1451, 3522, 2883, 3047, 6650], \
                        [4047, 8828, 8732, 5743, 1091, 381]])

        self.alpha = np.array([1, 1.2, 3, 3.2])

    def draw_new_function(self):
        pass
    
    def query_function(self, x, seed):
        S1 = 0
        for i in range(0, 4):
            S2 = 0
            for j in range(0, 6):
                S2 += self.A[i, j] * (x[:, j] - self.P[i, j])**2
            S1 += self.alpha[i] * np.exp(-S2)
        return S1

class Michalewicz2D():
    # taken from website: https://www.sfu.ca/~ssurjano/michal.html
    def __init__(self, t_dim = 2):
        self.t_dim = t_dim
        if self.t_dim == 2:
            self.x_dim = None
        else:
            self.x_dim = 2 - self.t_dim
        # calculated below
        self.optimum = 0.6754469275474548
        self.grid_search = False

        self.name = 'Michaelwicz2D'

        self.m = 10

    def draw_new_function(self):
        pass
    
    def query_function(self, x):
        x = x * np.pi
        S1 = np.sin(x[:, 0]) * (np.sin(x[:, 0] / np.pi))**(2*self.m)
        S2 = np.sin(x[:, 1]) * (np.sin(2 * x[:, 1] / np.pi))**(2*self.m)
        return S1 + S2
    
    def query_function_torch(self, x):
        x = x * 4
        S1 = torch.sin(x[:, 0]) * (torch.sin(x[:, 0] / np.pi))**(2*self.m)
        S2 = torch.sin(x[:, 1]) * (torch.sin(2 * x[:, 1] / np.pi))**(2*self.m)
        return S1 + S2

class Perm8D():
    def __init__(self, t_dim = 8):
        self.t_dim = t_dim
        if self.t_dim == 8:
            self.x_dim = None
        else:
            self.x_dim = 8 - self.t_dim
        # taken from website: https://www.sfu.ca/~ssurjano/permdb.html
        self.optimum = 0
        self.grid_search = False

        self.name = 'Perm8D'

        self.beta = 0.5

    def draw_new_function(self):
        pass
    
    def query_function(self, x):
        x = (x - 0.5) * 16
        S1 = 0
        for i in range(1, 1 + 8):
            S2 = 0
            for j in range(1, 1 + 8):
                S2 += (j**i + self.beta) * ((x[:, j-1] / j)**i - 1)
            S1 += S2**2
        return - S1 / 10**13

class Perm10D():
    def __init__(self, t_dim = 10):
        self.t_dim = t_dim
        if self.t_dim == 10:
            self.x_dim = None
        else:
            self.x_dim = 10 - self.t_dim
        # taken from website: https://www.sfu.ca/~ssurjano/permdb.html
        self.optimum = 0
        self.grid_search = False

        self.name = 'Perm10D'

        self.beta = 10

    def draw_new_function(self):
        pass
    
    def query_function(self, x):

        x = (x - 0.5) * 20

        S1 = 0
        for i in range(1, 1 + 10):
            S2 = 0
            for j in range(1, 1 + 10):
                S2 += (j**i + self.beta) * ((x[:, j-1] / j)**i - 1)
            S1 += S2**2
        
        out = - S1 / (10 ** 21)
        return out.astype(float)

class Ackley4D():
    def __init__(self, t_dim = 4):
        self.t_dim = t_dim
        if self.t_dim == 4:
            self.x_dim = None
        else:
            self.x_dim = 4 - self.t_dim
        # taken from website: https://www.sfu.ca/~ssurjano/ackley.html
        self.optimum = 0
        self.grid_search = False

        self.name = 'Ackley4D'

        self.a = 20
        self.b = 0.2
        self.c = 2 * np.pi

    def draw_new_function(self):
        pass
    
    def query_function(self, x):
        # new optimum
        #shift = np.array([0.4, 0.5, 0.45, 0.55])
        # first reparametrise x
        x = (x - 0.45) * (2 * 2)
        s1 = np.sum(x**2, axis = 1) / 4
        s2 = np.sum(np.cos(self.c * x), axis = 1) / 4
        return self.a * np.exp(-self.b * np.sqrt(s1)) + np.exp(s2) - self.a - np.exp(1)
    
    def query_function_torch(self, x):
        # first reparametrise x
        x = (x - 0.45) * (2 * 2)
        s1 = torch.sum(x**2, axis = 1) / 4
        s2 = torch.sum(torch.cos(self.c * x), axis = 1) / 4
        return self.a * torch.exp(-self.b * torch.sqrt(s1)) + torch.exp(s2) - self.a - np.exp(1)


class Alpine4D():
    def __init__(self, t_dim=4):
        self.t_dim = t_dim
        if self.t_dim == 4:
            self.x_dim = None
        else:
            self.x_dim = 4 - self.t_dim
        self.optimum = 0
        self.grid_search = False

        self.name = 'Alpine4D'

    def draw_new_function(self):
        pass

    def query_function(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        x1 = 10 * x1 - 10
        x2 = 10 * x2 - 20 / 3
        x3 = 10 * x3 - 10 / 3
        x4 = 10 * x4 - 0

        y1 = np.abs(x1 * np.sin(x1) + 0.1 * x1)
        y2 = np.abs(x2 * np.sin(x2) + 0.1 * x2)
        y3 = np.abs(x3 * np.sin(x3) + 0.1 * x3)
        y4 = np.abs(x4 * np.sin(x4) + 0.1 * x4)
        return -(y1 + y2 + y3 + y4)

    def query_function_torch(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        x1 = 10 * x1 - 10
        x2 = 10 * x2 - 20 / 3
        x3 = 10 * x3 - 10 / 3
        x4 = 10 * x4 - 0

        y1 = torch.abs(x1 * torch.sin(x1) + 0.1 * x1)
        y2 = torch.abs(x2 * torch.sin(x2) + 0.1 * x2)
        y3 = torch.abs(x3 * torch.sin(x3) + 0.1 * x3)
        y4 = torch.abs(x4 * torch.sin(x4) + 0.1 * x4)
        return -(y1 + y2 + y3 + y4)


class Ackley5D():
    def __init__(self, t_dim=5):
        self.t_dim = t_dim
        if self.t_dim == 5:
            self.x_dim = None
        else:
            self.x_dim = 5 - self.t_dim
        self.optimum = 0
        self.grid_search = False

        self.name = 'Ackley5D'

    def draw_new_function(self):
        pass

    def query_function(self, x, seed):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        x5 = x[:, 4]

        x1 = 30 * x1 - 25
        x2 = 30 * x2 - 20
        x3 = 30 * x3 - 15
        x4 = 30 * x4 - 10
        x5 = 30 * x5 - 5

        a = 20
        b = 0.2
        c = 2 * np.pi
        return (a * np.exp(-b * np.sqrt(1 / 5 * (x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 + x5 ** 2))) +
                np.exp(1 / 5 * (np.cos(c * x1) + np.cos(c * x2) + np.cos(c * x3) + np.cos(c * x4) + np.cos(c * x5))) -
                a - np.exp(1))

    def query_function_torch(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        x5 = x[:, 4]

        x1 = 30 * x1 - 25
        x2 = 30 * x2 - 20
        x3 = 30 * x3 - 15
        x4 = 30 * x4 - 10
        x5 = 30 * x5 - 5

        a = 20
        b = 0.2
        c = 2 * torch.pi
        return (a * torch.exp(-b * torch.sqrt(1 / 5 * (x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 + x5 ** 2))) +
                torch.exp(1 / 5 * (torch.cos(c * x1) + torch.cos(c * x2) + torch.cos(c * x3) + torch.cos(c * x4) + torch.cos(c * x5))) -
                a - torch.exp(1))


class mixed2D():
    def __init__(self, t_dim=2):
        self.t_dim = t_dim
        if self.t_dim == 2:
            self.x_dim = None
        else:
            self.x_dim = 2 - self.t_dim
        self.optimum = 0
        self.grid_search = False

        self.name = 'mixed2D'

    def draw_new_function(self):
        pass

    def query_function(self, x, seed):
        dim = 2
        num = 5
        df = pd.read_excel(r'C:\Users\60494\PycharmProjects\Bayesian\GaussianModel_2dim.xlsx', sheet_name=seed, header=None)
        df = df.values
        mu = df[0, :]
        mu = np.reshape(mu[~np.isnan(mu)], (dim, num))
        sigma = df[1, :]
        sigma = np.reshape(sigma[~np.isnan(sigma)], (num, dim, dim))
        b = df[2, :]
        b = np.reshape(b[~np.isnan(b)], (-1, 1))

        return 10 * np.array([NormalMix(x, mu, sigma, b, num=5) + np.random.normal(loc=0.0, scale=0.005)])


    def query_function_torch(self, x, seed):
        dim = 2
        num = 5
        df = pd.read_excel(r'C:\Users\60494\PycharmProjects\Bayesian\GaussianModel_2dim.xlsx', sheet_name=seed, header=None)
        df = df.values
        mu = df[0, :]
        mu = np.reshape(mu[~np.isnan(mu)], (dim, num))
        sigma = df[1, :]
        sigma = np.reshape(sigma[~np.isnan(sigma)], (num, dim, dim))
        b = df[2, :]
        b = np.reshape(b[~np.isnan(b)], (-1, 1))

        return np.array([NormalMix(x, mu, sigma, b, num=5) + np.random.normal(loc=0.0, scale=0.005)])


class mixed4D():
    def __init__(self, t_dim=4):
        self.t_dim = t_dim
        if self.t_dim == 4:
            self.x_dim = None
        else:
            self.x_dim = 4 - self.t_dim
        self.optimum = 0
        self.grid_search = False

        self.name = 'mixed4D'

    def draw_new_function(self):
        pass

    def query_function(self, x, seed):
        dim = 4
        num = 12
        df = pd.read_excel(r'C:\Users\60494\PycharmProjects\Bayesian\GaussianModel_4dim.xlsx', sheet_name=seed, header=None)
        df = df.values
        mu = df[0, :]
        mu = np.reshape(mu[~np.isnan(mu)], (dim, num))
        sigma = df[1, :]
        sigma = np.reshape(sigma[~np.isnan(sigma)], (num, dim, dim))
        b = df[2, :]
        b = np.reshape(b[~np.isnan(b)], (-1, 1))

        return 10 * np.array([NormalMix(x, mu, sigma, b, num=12) + np.random.normal(loc=0.0, scale=0.005)])


    def query_function_torch(self, x, seed):
        dim = 4
        num = 12
        df = pd.read_excel(r'C:\Users\60494\PycharmProjects\Bayesian\GaussianModel_4dim.xlsx', sheet_name=seed, header=None)
        df = df.values
        mu = df[0, :]
        mu = np.reshape(mu[~np.isnan(mu)], (dim, num))
        sigma = df[1, :]
        sigma = np.reshape(sigma[~np.isnan(sigma)], (num, dim, dim))
        b = df[2, :]
        b = np.reshape(b[~np.isnan(b)], (-1, 1))

        return np.array([NormalMix(x, mu, sigma, b, num=12) + np.random.normal(loc=0.0, scale=0.005)])


class mixed6D():
    def __init__(self, t_dim=6):
        self.t_dim = t_dim
        if self.t_dim == 6:
            self.x_dim = None
        else:
            self.x_dim = 6 - self.t_dim
        self.optimum = 0
        self.grid_search = False

        self.name = 'mixed6D'

    def draw_new_function(self):
        pass

    def query_function(self, x, seed):
        dim = 6
        num = 20
        df = pd.read_excel(r'C:\Users\60494\PycharmProjects\Bayesian\GaussianModel_6dim.xlsx', sheet_name=seed, header=None)
        df = df.values
        mu = df[0, :]
        mu = np.reshape(mu[~np.isnan(mu)], (dim, num))
        sigma = df[1, :]
        sigma = np.reshape(sigma[~np.isnan(sigma)], (num, dim, dim))
        b = df[2, :]
        b = np.reshape(b[~np.isnan(b)], (-1, 1))

        return 10 * np.array([NormalMix(x, mu, sigma, b, num=20) + np.random.normal(loc=0.0, scale=0.005)])


    def query_function_torch(self, x, seed):
        dim = 6
        num = 20
        df = pd.read_excel(r'C:\Users\60494\PycharmProjects\Bayesian\GaussianModel_6dim.xlsx', sheet_name=seed,
                           header=None)
        df = df.values
        mu = df[0, :]
        mu = np.reshape(mu[~np.isnan(mu)], (dim, num))
        sigma = df[1, :]
        sigma = np.reshape(sigma[~np.isnan(sigma)], (num, dim, dim))
        b = df[2, :]
        b = np.reshape(b[~np.isnan(b)], (-1, 1))

        return 10 * np.array([NormalMix(x, mu, sigma, b, num=20) + np.random.normal(loc=0.0, scale=0.005)])


class gls4D():
    def __init__(self, t_dim=4):
        self.t_dim = t_dim
        if self.t_dim == 4:
            self.x_dim = None
        else:
            self.x_dim = 4 - self.t_dim
        self.optimum = 0
        self.grid_search = False

        self.name = 'gls4D'

    def draw_new_function(self):
        pass

    def query_function(self, x, seed):
        def dy_dt(c, t, T, P):
            A = 1e7 * np.array([0.4, 12, 0.14, 5.6, 0.18, 84])
            Ea = np.array([41.2, 52.4, 35.2, 58.4, 34.6, 52.1])
            k = A * np.exp(-Ea / 8.314 / (T + 273.15) * 1000)
            H_index = np.array([4.264, 4.438, 4.623, 4.792, 4.96])
            T_step = np.array([40, 50, 60, 70, 80])
            f1 = interp1d(T_step, H_index, kind='linear')
            H = f1(T)
            cH = H * 0.1 * P
            return np.array([-k[0] * c[0] * cH,
                             k[0] * c[0] * cH - k[1] * c[1] * cH,
                             k[1] * c[1] * cH - k[2] * c[2] * cH - k[3] * c[2] * c[3],
                             k[2] * c[2] * cH - k[3] * c[2] * c[3] - k[4] * c[3] * cH,
                             k[3] * c[2] * c[3] - k[5] * c[4] * cH,
                             k[4] * c[3] * cH + 2 * k[5] * c[4] * cH]).reshape(-1)

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        x1 = 40 * x1 + 40
        x2 = 2 * x2 + 1
        x3 = 19.999 * x3 + 0.001
        x4 = 0.49 * x4 + 0.01
        y0 = [x4[0], 0, 0, 0, 0, 0]
        t = np.arange(0, x3, 0.001)
        y = odeint(dy_dt, y0, t, args=(x1, x2))
        ans = y[-1][-1] + np.random.normal(loc=0.0, scale=0.003)
        if ans >= 0.8 * x4:
            return np.array([ans / x3[0]])
        else:
            return np.array([0])


    def query_function_torch(self, x, seed):
        def dy_dt(c, t, T, P):
            A = 1e7 * torch.tensor([0.4, 12, 0.14, 5.6, 0.18, 84])
            Ea = torch.tensor([41.2, 52.4, 35.2, 58.4, 34.6, 52.1])
            k = A * torch.exp(-Ea / 8.314 / (T + 273.15) * 1000)
            H_index = torch.tensor([4.264, 4.438, 4.623, 4.792, 4.96])
            T_step = torch.tensor([40, 50, 60, 70, 80])
            f1 = interp1d(T_step, H_index, kind='linear')
            H = f1(T)
            cH = H * 0.1 * P
            return torch.tensor([-k[0] * c[0] * cH,
                                 k[0] * c[0] * cH - k[1] * c[1] * cH,
                                 k[1] * c[1] * cH - k[2] * c[2] * cH - k[3] * c[2] * c[3],
                                 k[2] * c[2] * cH - k[3] * c[2] * c[3] - k[4] * c[3] * cH,
                                 k[3] * c[2] * c[3] - k[5] * c[4] * cH,
                                 k[4] * c[3] * cH + 2 * k[5] * c[4] * cH])

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        x1 = 40 * x1 + 40
        x2 = 2 * x2 + 1
        x3 = 19.999 * x3 + 0.001
        x4 = 0.49 * x4 + 0.01
        y0 = [x4, 0, 0, 0, 0, 0]
        t = torch.arange(0, x3, 0.001)
        y = odeint(dy_dt, y0, t, args=(x1, x2))
        ans = y[-1][-1] + np.random.normal(loc=0.0, scale=0.003)
        if ans >= 0.8 * x4:
            return ans / x3
        else:
            return 0


class Schekel2D():
    def __init__(self, t_dim = 2, n_optims = 2):
        self.t_dim = t_dim
        if self.t_dim == 2:
            self.x_dim = None
        else:
            self.x_dim = 2 - self.t_dim
        # taken from website: https://www.sfu.ca/~ssurjano/shekel.html
        self.optimum = -11

        self.name = 'Schekel2D'
        self.grid_search = True

        self.num_of_optims = n_optims
        self.beta = np.array([10, 10, 2, 4, 4, 6, 3, 7, 5, 5])
        self.C = np.array([[2, 6.7, 8, 6, 3, 2, 5, 8, 6, 7], \
            [9, 2, 8, 6, 7, 9, 3, 1, 2, 3.6], \
            [4, 1, 8, 6, 3, 2, 5, 8, 6, 7], \
            [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]])

    def draw_new_function(self):
        pass
    
    def query_function(self, x):
        # new optimum
        #shift = np.array([0.4, 0.5, 0.45, 0.55])
        # first reparametrise x
        x = x * 10
        S1 = 0
        for i in range(self.num_of_optims):
            S2 = 0
            for j in range(2):
                S2 = S2 + (x[:, j] - self.C[i, j])**2
            S1 = S1 + 1 / (S2 + self.beta[i])
        return 10 * S1
    
    def query_function_torch(self, x):
        # first reparametrise x
        x = x * 10
        S1 = 0
        for i in range(self.num_of_optims):
            S2 = 0
            for j in range(2):
                S2 = (x[:, j] - self.C[j, i])**2
            S1 = S1 + 1 / (S2 + self.beta[i])
        return S1

class MultiSchekel2D():
    def __init__(self, t_dim = 2, n_optims = [2, 3, 2]):
        self.t_dim = t_dim
        if self.t_dim == 2:
            self.x_dim = None
        else:
            self.x_dim = 2 - self.t_dim
        # taken from website: https://www.sfu.ca/~ssurjano/shekel.html
        self.optimum = -11
        # search grid
        self.grid_search = True
        sobol_gen = torch.quasirandom.SobolEngine(2)
        self.grid_to_search = sobol_gen.draw(1000).double()
        # two objectives
        self.num_of_objectives = 3

        self.name = 'Schekel2D'

        self.num_of_optims = n_optims
        self.beta = [0, 0, 0]
        self.C = [0, 0, 0]
        # first params
        self.beta[0] = np.array([9, 9, 2, 4, 4, 6, 3, 7, 5, 5])
        self.C[0] = np.array([[2, 6.7, 8, 6, 3, 2, 5, 8, 6, 7], \
            [9, 2, 8, 6, 7, 9, 3, 1, 2, 3.6], \
            [4, 1, 8, 6, 3, 2, 5, 8, 6, 7], \
            [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]])
        # second params
        self.beta[1] = np.array([10, 8, 8, 4, 4, 6, 3, 7, 5, 5])
        self.C[1] = np.array([[7, 6, 5, 6, 3, 2, 5, 8, 6, 7], \
            [3.8, 9.9, 5, 6, 7, 9, 3, 1, 2, 3.6], \
            [9, 0.1, 5, 6, 3, 2, 5, 8, 6, 7], \
            [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]])
        # third params
        # second params
        self.beta[2] = np.array([7, 9, 8, 4, 4, 6, 3, 7, 5, 5])
        self.C[2] = np.array([[4, 3, 5, 6, 3, 2, 5, 8, 6, 7], \
            [8.5, 4, 5, 6, 7, 9, 3, 1, 2, 3.6], \
            [9, 0.1, 5, 6, 3, 2, 5, 8, 6, 7], \
            [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]])

    def draw_new_function(self):
        pass
    
    def query_function1(self, x):
        # new optimum
        #shift = np.array([0.4, 0.5, 0.45, 0.55])
        # first reparametrise x
        x = x * 10
        S1 = 0
        for i in range(self.num_of_optims[0]):
            S2 = 0
            for j in range(2):
                S2 = S2 + (x[:, j] - self.C[0][i, j])**2
            S1 = S1 + 1 / (S2 + self.beta[0][i])
        return 10 * S1
    
    def query_function2(self, x):
        # new optimum
        #shift = np.array([0.4, 0.5, 0.45, 0.55])
        # first reparametrise x
        x = x * 10
        S1 = 0
        for i in range(self.num_of_optims[1]):
            S2 = 0
            for j in range(2):
                S2 = S2 + (x[:, j] - self.C[1][i, j])**2
            S1 = S1 + 1 / (S2 + self.beta[1][i])
        return 10 * S1
    
    def query_function3(self, x):
        # new optimum
        #shift = np.array([0.4, 0.5, 0.45, 0.55])
        # first reparametrise x
        x = x * 10
        S1 = 0
        for i in range(self.num_of_optims[2]):
            S2 = 0
            for j in range(2):
                S2 = S2 + (x[:, j] - self.C[2][i, j])**2
            S1 = S1 + 1 / (S2 + self.beta[2][i])
        return 10 * S1
    
    def query_function(self, x):
        q1 = self.query_function1(x)
        q2 = self.query_function2(x)
        q3 = self.query_function3(x)
        return [q1, q2, q3]

class ScheckelSingle(MultiSchekel2D):
    def __init__(self, t_dim=2, n_optims=[2, 3, 2], obj_to_query = 0):
        super().__init__(t_dim, n_optims)
        self.obj_to_query = obj_to_query
    
    def query_function_torch(self, x):
        if self.obj_to_query == 0:
            return super().query_function1(x)
        elif self.obj_to_query == 1:
            return super().query_function2(x)
        else:
            return super().query_function3(x)

class YpacaraiLake(MultiSchekel2D):
    def __init__(self, t_dim=2, n_optims=[2, 3, 2]):
        super().__init__(t_dim, n_optims)
        self.grid_to_search = torch.tensor(np.load('lake_grid.npy'))
        self.name = 'YpacaraiLake'

class YpacaraiLakeSingleObjective(ScheckelSingle):
    def __init__(self, t_dim=2, n_optims=[2, 3, 2], obj_to_query = 0):
        self.obj_to_query = obj_to_query
        super().__init__(t_dim, n_optims, obj_to_query)
        self.grid_to_search = torch.tensor(np.load('lake_grid.npy'))
        self.name = 'YpacaraiLakeSingleObjective'
        self.num_of_objectives = 1
    
    def query_function(self, x):
        if self.obj_to_query == 0:
            return super().query_function1(x)
        elif self.obj_to_query == 1:
            return super().query_function2(x)
        else:
            return super().query_function3(x)

# class SnAr():
#     def __init__(self):
#         self.t_dim = 3
#         self.x_dim = 1
#         self.grid_search = False
#
#         self.name = 'SnarBenchmark'
#
#         self.snar_bench = SnarBenchmark()
#
#     def draw_new_function(self):
#         pass
#
#     def query_function(self, x):
#         x = x.reshape(1, -1)
#         temp = x[:, 0] * 80 + 40
#         conc_dfnb = x[:, 1] * 0.4 + 0.1
#         residence_time = x[:, 2] * 1.5 + 0.5
#         equiv_pldn = x[:, 3] * 4 + 1
#
#         values = {
#             ("tau", "DATA"): [residence_time],
#             ("equiv_pldn", "DATA"): [equiv_pldn],
#             ("conc_dfnb", "DATA"): [conc_dfnb],
#             ("temperature", "DATA"): [temp],
#         }
#
#         conditions = DataSet(values)
#         experiments = self.snar_bench.run_experiments(conditions, computation_time = False)
#         return experiments['sty'][0] / 10000 - experiments['e_factor'][0] / 10

def find_optimum(func, n_starts = 25, n_epochs = 100):
    # find dimension
    if func.x_dim is not None:
        dim = func.x_dim + func.t_dim
    else:
        dim = func.t_dim
    # define bounds
    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
    # random multistart
    X = torch.rand(n_starts, dim)
    X.requires_grad = True
    optimiser = torch.optim.Adam([X], lr = 0.01)

    for i in range(n_epochs):
        # set zero grad
        optimiser.zero_grad()
        # losses for optimiser
        losses = - func.query_function_torch(X)
        loss = losses.sum()
        loss.backward()
        # optim step
        optimiser.step()

        # make sure we are still within the bounds
        for j, (lb, ub) in enumerate(zip(*bounds)):
            X.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself
    
    final_evals = func.query_function_torch(X)
    best_eval = torch.max(final_evals)
    best_start = torch.argmax(final_evals)
    best_input = X[best_start, :].detach()

    return best_input, best_eval

# this last part is used to find the optimum of functions using gradient methods, if optimum is not available online
if __name__ == '__main__':
    func = ScheckelSingle(obj_to_query = 0)
    best_input, best_eval = find_optimum(func, n_starts = 100000, n_epochs = 1000)
    print(float(best_input.detach()[0]))
    print(float(best_input.detach()[1]))
    print(float(best_eval.detach()))