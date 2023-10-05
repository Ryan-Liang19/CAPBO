import numpy as np
# import matplotlib.pyplot as plt


def Normal(X, mu, sigma):  # 多元正态分布概率密度函数
    return np.exp(-0.5 * np.dot(np.dot((X - mu).T, np.linalg.inv(sigma)), (X - mu))) / (2 * np.pi) / np.linalg.norm(sigma) ** 0.5


def NormalMix(X, mu, sigma, b, num):
    res = 0
    X = X.reshape((-1, 1))
    for i in range(0, num):
        res += b[i] * Normal(X, mu[:, i:i+1], sigma[i, :, :])
    return res[0][0]


def generate(dim, num):
    mu = np.random.rand(dim, num)
    sigma = np.random.rand(num, dim, dim)
    b = np.random.rand(num, 1)
    b = b / b.sum()
    for i in range(0, num):
        A = np.random.rand(dim, dim) - 0.5 * np.ones([dim, dim]) + 0.5 * np.eye(dim)
        B = np.dot(A, A.transpose())
        C = B + B.T
        C = C / C.max() / 10
        for j in range(0, dim):
            for k in range(0, dim):
                if j != k:
                    # C[j, k] /= 2
                    C[j, k] /= 1.1
                else:
                    C[j, k] += 0.01
        sigma[i, :, :] = C
    return mu, sigma, b


def pplot(mu, sigma, b, num):                # 只能在dim=2时使用
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    x, y = np.meshgrid(x, y)
    z = np.zeros([100, 100])
    for i in range(0, 100):
        for j in range(0, 100):
            temp = np.array([x[i][j], y[i][j]])
            z[i][j] = NormalMix(temp, mu, sigma, b, num)

    fig = plt.figure()
    ax = plt.gca(projection='3d')
    ax.plot_surface(x, y, z)


if __name__ == '__main__':
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    x, y = np.meshgrid(x, y)
    mu, sigma, b = generate(dim=2, num=20)
    z = np.zeros([100, 100])
    for i in range(0, 100):
        for j in range(0, 100):
            temp = np.array([x[i][j], y[i][j]])
            z[i][j] = NormalMix(temp, mu, sigma, b, num=20)

    fig = plt.figure()
    ax = plt.gca(projection='3d')
    ax.plot_surface(x, y, z)
