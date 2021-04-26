# generate fOU process
# dX(t) = alpha * (mu - tanh(X(t))) * dt + sigma dW(t)

# generate mean-reverting tanh tranformed fbm
import multiprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm
import sdepy
from fbm import FBM
from mpmath import tanh

nlength = 1000  # trading days
N = 10000
t = 1
dt = t / nlength
times = np.linspace(start=0, stop=t, num=nlength)  # numpy.ndarray (nlength,)


def rho(nlength):
    alpha = np.random.uniform(0.5, 2)  # speed of mean-reverse
    mu = np.random.uniform(-1, 1)  # mean
    sigma = np.random.uniform(0, (1-np.abs(mu) / 2))  # volatility
    hurst = np.random.uniform(0.12, 0.15)  # hurst exponent

    fbm = FBM(n=nlength-1, hurst=hurst, length=t, method='cholesky')
    fbm_sample = fbm.fbm()

    W = fbm_sample.reshape(nlength, 1)

    sdepy_W = sdepy.process(t=times, x=W)

    @sdepy.integrate
    def X_process(t, x, alpha=1, mu=1, sigma=1):
        return {
            'dt': alpha*(mu - ((2 / np.pi) * np.arctan((np.pi / 2) * x))),
            'dw': sigma,
        }

    X = X_process(paths=1, x0=0, alpha=alpha, mu=mu,
                  sigma=sigma, dw=sdepy_W)(times)
    arctanX = (2 / np.pi) * np.arctan((np.pi / 2) * X)
    arctanX_reshape = np.array(arctanX).reshape(1, nlength)
    data = np.concatenate([arctanX_reshape[0], [alpha, mu, sigma, hurst]])
    pd.DataFrame(data).transpose().to_csv(
        f'rho_arctan_data{np.random.uniform(0, 100000)}.csv', index=False, header=False)

    arctan_X = np.array(X).reshape(1, nlength)
    data_X = np.concatenate([arctan_X[0], [alpha, mu, sigma, hurst]])
    pd.DataFrame(data_X).transpose().to_csv(
        f'rho_arctan_data_raw{np.random.uniform(0, 100000)}.csv', index=False, header=False)


num_cores = multiprocessing.cpu_count()


def generate_rho(i):
    rho(nlength)


if __name__ == "__main__":
    for i in tqdm(range(6250)):
        p = multiprocessing.Pool(num_cores)
        p.map(generate_rho, range(num_cores))
        p.close()
