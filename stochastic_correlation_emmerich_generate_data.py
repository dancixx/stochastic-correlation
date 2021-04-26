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
N = 500000
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
            'dt': (alpha*(mu - np.tanh(x))) / (1 - (np.tanh(x))**2),
            'dw': sigma / (np.sqrt(1 - (np.tanh(x))**2)),
        }

    X = X_process(paths=1, x0=0, alpha=alpha, mu=mu,
                  sigma=sigma, dw=sdepy_W)(times)

    tanhX = np.tanh(X)
    tanh_reshape = np.array(tanhX).reshape(1, nlength)
    data = np.concatenate([tanh_reshape[0], [alpha, mu, sigma, hurst]])
    pd.DataFrame(data).transpose().to_csv(
        f'rho_data_emmerich{np.random.uniform(0, 1000)}.csv', index=False, header=False)

    tanh_X = np.array(X).reshape(1, nlength)
    data_X = np.concatenate([tanh_X[0], [alpha, mu, sigma, hurst]])
    pd.DataFrame(data_X).transpose().to_csv(
        f'rho_data_emmerich_raw{np.random.uniform(0, 1000)}.csv', index=False, header=False)


num_cores = multiprocessing.cpu_count()


def generate_rho(i):
    rho(nlength)


if __name__ == "__main__":
    for i in tqdm(range(6250)):
        p = multiprocessing.Pool(num_cores)
        p.map(generate_rho, range(num_cores))
        p.close()
