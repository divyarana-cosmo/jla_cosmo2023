import numpy as np
import matplotlib.pyplot as plt

def model(x, m, b):
    return m * x + b

def log_likelihood(theta, x, y, icov):
    m, b = theta
    mod = model(x, m, b)
    Delta = y - mod
    chisq = np.dot(Delta, np.dot(icov, Delta))
    return -0.5 * chisq


def log_prior(theta):
    m, b = theta
    if -2.0 < m < 2 and 3 < b < 6 : # uniform priors
        return 0.0
    return -np.inf


def log_probability(theta, x, y, icov):
    lp = log_prior(theta)
    if not np.isfinite(lp): # check if the parameters are outside the priors --> to reject them
        return -np.inf
    return lp + log_likelihood(theta, x, y, icov)



import emcee

# Generate mock data and covariance for demonstration purposes
m_true = -0.9594
b_true = 4.294

N = 50
x = np.sort(10 * np.random.rand(N))
yerr = 0.1 + 0.5 * np.random.rand(N)
y = m_true * x + b_true
y += yerr * np.random.randn(N)


cov = np.diag(yerr * yerr) # covariance
icov = np.linalg.inv(cov) # taking the inverse


nwalkers = 32 # number of walkers
ndim = 2 # number of parameters in the model

# initializing the walkers
p_m  = np.random.uniform(-1, 1, nwalkers)
p_c  = np.random.uniform(4, 5.0, nwalkers)
pos = np.transpose([p_m, p_c]) # assigning random positions with in the priors

# initializing the sampler and running the chains
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, icov))
sampler.run_mcmc(pos, 5000, progress=True); # running with 5000 steps


#discarding 1000 steps
flat_samples = sampler.get_chain(discard=1000, flat=True) 
chisq = -2*sampler.get_log_prob(discard=1000, flat=True)
#here we are saving the chains with chisq as last column
mat = np.zeros((len(flat_samples[:,0]), len(flat_samples[0,:]) + 1))
mat[:,:-1] = flat_samples
mat[:, -1] = chisq
np.savetxt('test_chains.dat', mat)

# getting the contours
import corner
labels = ["m", "b"]
fig = corner.corner(flat_samples, labels=labels, levels=[0.68, 0.95]);

plt.savefig('test_corner.png')


