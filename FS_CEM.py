import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.io import loadmat  # loading data from matlab
from scipy import fftpack  # calculate the fft

# Parameters
nbData = 200  # Number of data points in a trajectory
nbStates = 10  # Number of basis functions

# Load handwriting data
letter = 'S'  # choose a letter in the alphabet
datapath = '2Dletters/'
data = loadmat(datapath + '%s.mat' % letter)
demos = [d['pos'][0][0].T for d in data['demos'][0]]  # cleaning awful matlab data
data = np.array(demos[1])
t = np.linspace(0, 1, nbData)
x = np.reshape(data, nbData * 2)

# Compute reward function
def rewardEval(p):
    r = np.zeros((1, p.shape[1]))
    for i in range(p.shape[1]):
        w = p[:, i]
        xr = np.dot(Psi, w) # Eq.(29)
        r[:, i] = - np.sum(np.absolute(xr - x))  # The reward is the negative distance between xr and x
    return r


# Compute basis functions Psi and activation weights w
phi = np.zeros((nbData, nbStates))
for i in range(nbStates):
    xTmp = np.zeros((1, nbData))
    xTmp[:, i] = 1
    phi[:, i] = fftpack.idct(xTmp)  # Discrete cosine transform

Psi = np.kron(phi, np.eye(2))  # Eq.(27)

# Parameters for CEM
nbVar = 2 * nbStates  # Dimension of datapoints
nbEpisods = 2000  # Number of exploration iterations
nbE = 100  # Number of initial points (for the first iteration)
nbPointsRegr = 50  # Number of points with highest rewards considered at each iteration (importance sampling)
minSigma = np.eye(nbVar) * 1  # Minimum exploration covariance matrix

Sigma = np.eye(nbVar) * 50  # Initial exploration noise
p = np.empty((nbVar, 0))  # Storing tested parameters (initialized as empty)
r = np.empty((1, 0))  # Storing associated rewards (initialized as empty)

# Initialise the w parameter
w = np.zeros(nbVar) + 3

# EM-based stochastic optimization
for i in range(0, nbEpisods):
    # Generate noisy data with variable exploration noise
    D, V = LA.eig(Sigma)
    pNoisy = np.tile(w.reshape(nbVar, 1), (1, nbE)) + np.dot(np.dot(V, np.diag(D ** 0.5)), np.random.randn(nbVar, nbE))
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
    eTmp = pTmp - np.tile(w.reshape(nbVar, 1), (1, nbP))

    # CEM update of mean and covariance (exploration noise)
    w = np.mean(pTmp, axis=1).reshape(nbVar, 1)
    Sigma0 = (eTmp.dot(eTmp.conj().T)) / nbP

    # Add minimal exploration noise
    Sigma = Sigma0 + minSigma

xr = np.dot(Psi, w)   # Eq.(29)

xr = xr.reshape(nbData, 2) # Reshape the data to 2D coordination
x = x.reshape(nbData, 2) # Reshape the data to 2D coordination

plt.figure()
plt.plot(xr[:, 0], xr[:, 1], '-r', alpha=1)
plt.plot(x[:, 0], x[:, 1], '.b', alpha=0.3)
plt.show()
