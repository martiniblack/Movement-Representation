import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.io import loadmat  # loading data from matlab

# Parameters
nbStates = 10  # Number of radial basis functions
nbData = 200  # Length of a trajectory
polDeg = 1  # Degree of polynomial

# Load handwriting data
letter = 'S'  # choose a letter in the alphabet
datapath = 'pbdlib/data/2Dletters/'
data = loadmat(datapath + '%s.mat' % letter)
demos = [d['pos'][0][0].T for d in data['demos'][0]]  # cleaning awful matlab data
data = np.array(demos[1])
t = np.linspace(0, 1, nbData)

# LWR with radial basis functions and local polynomial fitting
MuRBF = np.linspace(t[0], t[-1], nbStates)  # Set centroids equally spread in time
SigmaRBF = 0.005  # Set constant shared bandwidth
phi = np.zeros((nbStates, nbData))

for i in range(nbStates):
    tc = t - np.tile(MuRBF[i], (1, nbData))
    phi[i, :] = np.exp(-0.5 * np.sum(np.divide(tc, SigmaRBF) * tc, axis=0))  # Eq.(2)

# Locally weighted regression
X = np.zeros((nbData, polDeg + 1))
for d in range(polDeg):
    X[:, d] = t.conj().T ** d
Y = data.conj()


# Compute reward function
def rewardEval(p):
    r = np.zeros((1, p.shape[1]))
    for i in range(p.shape[1]):
        A = p[:, i].reshape(polDeg + 1, Y.shape[1], nbStates)
        Yr = np.zeros_like(Y)
        for j in range(nbStates):
            W = np.diag(phi[j, :])  # Eq.(4)
            Yr = Yr + np.dot(np.dot(W, X), A[:, :, j])  # Yr is the calculated position from the X, W, A
        r[:, i] = - np.sum(np.absolute(Yr - Y))  # The reward is the negative distance between Yr and GT
    return r


# Parameters for CEM
nbVar = (polDeg + 1) * Y.shape[1] * nbStates  # Dimension of data points
nbEpisods = 10000  # Number of exploration iterations
nbE = 100  # Number of initial points (for the first iteration)
nbPointsRegr = 50  # Number of points with highest rewards considered at each iteration (importance sampling)
minSigma = np.eye(nbVar) * 1  # Minimum exploration covariance matrix

Sigma = np.eye(nbVar) * 50  # Initial exploration noise
p = np.empty((nbVar, 0))  # Storing tested parameters (initialized as empty)
r = np.empty((1, 0))  # Storing associated rewards (initialized as empty)

# Initialise the A parameter
A = np.zeros(nbVar) + 3

# EM-based stochastic optimization
for i in range(0, nbEpisods):
    # Generate noisy data with variable exploration noise
    D, V = LA.eig(Sigma)
    pNoisy = np.tile(A.reshape(nbVar, 1), (1, nbE)) + np.dot(np.dot(V, np.diag(D ** 0.5)), np.random.randn(nbVar, nbE))
    nbE = 1  # nbE=1 for the next iterations

    # Compute associated rewards
    rNoisy = rewardEval(pNoisy)

    # Add new points to dataset
    p = np.append(p, pNoisy, axis=1)
    r = np.append(r, rNoisy, axis=1)

    # Keep the nbPointsRegr points with highest rewards
    rSrt, idSrt = np.flip(np.sort(r), 1), np.squeeze(np.flip(np.argsort(r), 1))
    nbP = min(idSrt.shape[0], nbPointsRegr)
    pTmp = p[:, idSrt[:nbP]]
    rTmp = np.squeeze(rSrt[:, :nbP])

    # Compute error term
    eTmp = pTmp - np.tile(A.reshape(nbVar, 1), (1, nbP))

    # CEM update of mean and covariance (exploration noise)
    A = np.mean(pTmp, axis=1).reshape(nbVar, 1)
    Sigma0 = (eTmp.dot(eTmp.conj().T)) / nbP

    # Add minimal exploration noise
    Sigma = Sigma0 + minSigma

A = A.reshape(polDeg + 1, Y.shape[1], nbStates)
Yr = np.zeros_like(Y)
for i in range(nbStates):
    W = np.diag(phi[i, :])  # Eq.(4)
    Yr = Yr + np.dot(np.dot(W, X), A[:, :, i])

plt.figure()
plt.plot(Yr[:, 0], Yr[:, 1], '-r', alpha=1)
plt.plot(Y[:, 0], Y[:, 1], '.b', alpha=0.3)
plt.show()
