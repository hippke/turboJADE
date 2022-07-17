import numpy as np
from numba import jit, prange
from tqdm import tqdm
import random


MAX_INT = 4294967295
GAUSS_CONST = 4 * np.exp(-0.5) / np.sqrt(2)


# Random numbers from getrandbits are 10x faster than single numpy calls
@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def rand_int(limit=None):
    r = random.getrandbits(32)
    if limit is not None:
        r = (r / MAX_INT) * limit
    return int(r)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def rand_float():
    return rand_int() / MAX_INT


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def rand_cauchy(mu=0, sigma=1):
    return mu + sigma * np.tan(np.pi * (rand_normal() - 0.5))


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def rand_normal(mu=0, sigma=1):
    while True:
        u2 = 1 - rand_float()
        z = GAUSS_CONST * (rand_float() - 0.5) / u2
        if z * z / 4 <= -np.log(u2):
            break
    return mu + z * sigma


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def clip(min_, max_, val):
    return max(min_, min(max_, val))


@jit(cache=True, nopython=True, fastmath=True, parallel=True)
def eval_func(pop, fit, func):
    for idx in prange(pop.shape[0]):
        fit[idx] = func(pop[idx])
    return fit


# Thousands separator for ncall integers in tqdm progress bar 
def int_commas(x):
    result = ""
    while x >= 1000:
        x, r = divmod(x, 1000)
        result = ",%03d%s" % (r, result)
    return "%d%s" % (x, result)


@jit(cache=True, nopython=True, fastmath=True, parallel=False)
def generate_trial_func(pop, trial, limits, n_pop, n_dim, p, cr, f, cr_arr, f_arr, fit):
    k = int(p * n_pop)
    k_high = np.argsort(fit)[: -k - 1 : -1]
    pbest_i = k_high[rand_int(k)]
    for i in range(n_pop):
        r1 = rand_int(n_pop)
        r2 = rand_int(n_pop)
        while r1 == i:
            r1 = rand_int(n_pop)
        while r2 == r1 or r2 == i:
            r2 = rand_int(n_pop)
        always = rand_int(n_dim)
        cr_i = clip(0, 1, rand_normal(cr, 0.1))
        f_i = clip(0, 1, rand_cauchy(f, 0.1))
        cr_arr[i] = cr_i
        f_arr[i] = f_i
        for d in range(n_dim):
            if rand_normal() < cr_i or d == always:
                v = (
                    pop[i, d]
                    + f_i * (pop[pbest_i, d] - pop[i, d])
                    + f_i * (pop[r1, d] - pop[r2, d])
                )
            else:
                v = pop[i, d]
            trial[i, d] = clip(limits[0][d, 0], limits[0][d, 1], v)


class turboJADE:
    def __init__(self, func, limits, n_dim, n_pop=250, c=0.01, converge=True, batch_id=1, progress=True):
        self.counter = 0
        self.c = c
        self.n_pop = n_pop
        self.progress = progress
        self.batch_id = batch_id
        self.converge = converge
        self.converge_iters = 200

        self.p = 0.1
        self.cr = 0.5
        self.f = 0.5
        self.cr_arr = np.empty(n_pop)
        self.f_arr = np.empty(n_pop)

        self.func = func
        self.f = 0.7
        self.cr = 0.95
        limits = [limits] * n_dim
        self.limits = np.array(limits)
        n_dim = len(self.limits)
        self.pop = np.empty((n_pop, n_dim))
        self.trial = np.empty((n_pop, n_dim))
        for dim in range(n_dim):
            lower, upper = limits[0][dim]
            range_ = upper - lower
            for ind in range(n_pop):
                self.pop[ind, dim] = np.random.random_sample() * range_ + lower
        self.fit = np.zeros(n_pop)
        self.trial_fit = np.zeros(n_pop)
        eval_func(self.pop, self.fit, self.func)

    def selection(self):
        pop = self.pop
        trial = self.trial
        fit = self.fit
        trial_fit = self.trial_fit
        cr_arr = self.cr_arr
        f_arr = self.f_arr
        num_cr = 0.0
        den_cr = 0.0
        num_f = 0.0
        den_f = 0.0
        for i in range(self.pop.shape[0]):
            if trial_fit[i] > fit[i]:
                fit[i] = trial_fit[i]
                pop[i] = trial[i]
                num_cr += cr_arr[i]
                den_cr += 1.0
                num_f += f_arr[i] ** 2
                den_f += f_arr[i]
        if den_cr > 0:
            self.cr = (1.0 - self.c) * self.cr + self.c * (num_cr / den_cr)
        if den_f > 0:
            self.f = (1.0 - self.c) * self.f + self.c * (num_f / den_f)

    def run(self, n_it=1000):
        trend = []
        self.k_high = None
        if self.progress:
            pbar = tqdm(total=n_it, position=self.batch_id)
        for iteration in range(n_it):
            generate_trial_func(
                self.pop,
                self.trial,
                self.limits,
                self.pop.shape[0],
                self.pop.shape[1],
                self.p,
                self.cr,
                self.f,
                self.cr_arr,
                self.f_arr,
                self.fit,
            )
            eval_func(self.trial, self.trial_fit, self.func)
            self.selection()
            if self.progress:
                if iteration % 100 == 0:
                    pbar.update(100)
                    ncall = len(self.pop) * iteration
                    pbar.set_postfix({'ncall': int_commas(ncall), "fit": str(round(min(self.fit), 1))})
            best_in_iter = min(self.fit)
            trend.append(best_in_iter)

            if self.converge:
                look_back = iteration - self.converge_iters
                if look_back < 0:
                    look_back = 0
                if best_in_iter == trend[look_back] and iteration > self.converge_iters:
                    break
        if self.progress:
            pbar.close()
        besti = np.argmax(self.fit)
        evals = (1 + iteration) * self.pop.shape[0]
        return self.pop[besti], trend, evals
