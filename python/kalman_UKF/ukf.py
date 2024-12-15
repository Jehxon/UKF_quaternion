import numpy as np
from matplotlib import pyplot as plt

class UKF:
  def __init__(self, evolutionModel, evolutionProcessCovariance, measurementsObservationFunctions, measurementsCovariances):
    # Evolution model
    self.evolutionModel = evolutionModel
    self.evolutionProcessCovariance = evolutionProcessCovariance
    # Measurements processes
    self.measurementsObservationFunctions = measurementsObservationFunctions
    self.measurementsCovariances = measurementsCovariances
    # State
    self.nx = evolutionProcessCovariance.shape[0]
    self.x = np.ones(self.nx)
    self.P = np.eye(self.nx)
    
    alpha = 0.001
    beta = 2
    kappa = 0
    self.lba = alpha*alpha*(self.nx+kappa) - self.nx
    # Sigma points
    self.ns = 2*self.nx+1
    self.X = np.empty((self.nx, self.ns))
    # Weights
    self.weightsState = np.empty(self.ns)
    self.weightsCov = np.empty(self.ns)
    self.weightsState[0] = self.lba/(self.nx+self.lba)
    self.weightsCov[0] = self.lba/(self.nx+self.lba)+(1-alpha*alpha+beta)
    self.weightsState[1:] = 0.5/(self.nx+self.lba)
    self.weightsCov[1:] = 0.5/(self.nx+self.lba)
    
    self.initialized = False
  
  def getState(self):
    return self.x
  
  def getCovariance(self):
    return self.P
  
  def init(self, x0, P0):
    self.x = x0.astype(np.float64)
    self.P = P0.astype(np.float64)
    self.generateSigmaPoints()
    self.initialized = True
    
  def generateSigmaPoints(self):
    M = np.linalg.cholesky(self.P)
    a = np.sqrt(self.nx+self.lba)
    self.X[:,0] = self.x
    for i in range(1,self.nx+1):
      self.X[:,i] = self.x + M[:,i-1]*a
    for i in range(self.nx+1, self.ns):
      self.X[:,i] = self.x - M[:,i-1-self.nx]*a
    
  def predict(self, *args, **kwargs):
    if not self.initialized:
      return
    self.generateSigmaPoints()
    # Propagate sigma points throught the function
    for i in range(self.ns):
      self.X[:,i] = self.evolutionModel(self.X[:,i], *args, **kwargs)
    # Compute mean and covariance
    xhat = np.zeros(self.nx)
    for i in range(self.ns):
      xhat += self.weightsState[i]*self.X[:,i]
    Phat = np.zeros((self.nx, self.nx))
    for i in range(self.ns):
      d = (self.X[:,i]-xhat).reshape(-1,1)
      Phat += self.weightsCov[i]*np.dot(d, d.T)
    Phat += self.evolutionProcessCovariance
    # Update filter estimates
    self.x[:] = xhat
    self.P[:] = Phat
  
  def update(self, observationFunctionId, measurement, *args, **kwargs):
    if not self.initialized:
      return
    self.generateSigmaPoints()
    g = self.measurementsObservationFunctions[observationFunctionId]
    nm = measurement.shape[0]
    # Propagate sigma points throught the function
    Y = np.hstack([g(self.X[:,i], *args, **kwargs).reshape(-1,1) for i in range(self.ns)])
    # Compute mean and covariance
    yhat = np.zeros(nm)
    for i in range(self.ns):
      yhat += self.weightsState[i]*Y[:,i]
    # Covariance
    Pyy = np.zeros((nm, nm))
    # Cross-covariance
    Pxy = np.zeros((self.nx, nm))
    for i in range(self.ns):
      ds = (self.X[:,i] - self.x).reshape(-1,1)
      dm = (Y[:,i]-yhat).reshape(-1,1)
      Pyy += self.weightsCov[i]*np.dot(dm, dm.T)
      Pxy += self.weightsCov[i]*np.dot(ds, dm.T)
    Pyy += self.measurementsCovariances[observationFunctionId]
    # Update estimates
    K = np.dot(Pxy, np.linalg.inv(Pyy))
    self.x += np.dot(K, measurement-yhat)
    self.P -= np.dot(np.dot(K,Pyy),K.T)


def tests():
  def model(x):
    return np.array([x[0] + np.exp(-x[1]), x[1] + np.exp(-x[0])])
  def measurement(x):
    return np.array([x[0] + 2*x[1]*x[1], x[0]*x[1]])
  
  numSteps = 20
  x0 = np.array([0,0])
  P0 = 0.1*np.eye(2)
  Pv = 0.1*np.eye(2)
  Pn = 0.1*np.eye(2)
  L = 2
  
  v = Pv@(np.random.random((L, numSteps))-0.5)
  n = Pn@(np.random.random((L, numSteps))-0.5)
  
  x = np.zeros((L, numSteps))
  y = np.zeros((L, numSteps))
  
  x[:,0] = x0
  y[:,0] = measurement(x0) + v[:,0]
  
  for i in range(1, numSteps):
    x[:,i] = model(x[:,i-1]) + v[:,i]
    y[:,i] = measurement(x[:,i]) + n[:,i]
    
  ukf = UKF(model, Pv, [measurement], [Pn])
  ukf.init(x0, P0)
  
  xEst = np.zeros((L,numSteps))
  PEst = np.zeros((L,L,numSteps))
  xEst[:,0] = ukf.x
  PEst[:,:,0] = ukf.P
  
  for i in range(1, numSteps):
    ukf.predict()
    ukf.update(0, y[:,i])
    xEst[:,i] = ukf.x
    PEst[:,:,i] = ukf.P
    print(ukf.x)
    print(ukf.P)
  
  plt.plot(x[0,:], x[1,:], '-r')
  plt.plot(xEst[0], xEst[1], '.b')
  plt.show()
  

if __name__ == "__main__":
  tests()
  