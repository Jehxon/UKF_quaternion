from quaternions import Quaternion
import numpy as np
from kalman_UKF.ukf import UKF
from collections import deque

GRAVITY = 9.81
GRAVITY_ERROR = 0.1
gravity_measure_acc_buffer_time = 0.5 # seconds needed of acceleration norm being approximately GRAVITY to call a gravity measurement

gyroStdDev = 0.005
accStdDev = 0.1
magStdDev = 0.7

gyro_freq = 20.0
acc_freq = 10.0

class OrientationEstimateInitSteps(np.uint8):
    NONE = np.uint8(0)
    ACC = np.uint8(1 << 0)
    MAG = np.uint8(1 << 1)
    WAITING_FOR_INIT = ACC #| MAG
    RUNNING = np.uint8(1 << 2)
    
class OrientationEstimator:
  def __init__(self):
    self.lastTs = dict()
    self.estimate = Quaternion()
    self.initialized = OrientationEstimateInitSteps.NONE
    
    gVar = gyroStdDev*gyroStdDev
    aVar = accStdDev*accStdDev
    mVar = magStdDev*magStdDev
    
    processCov = np.diag([gVar, gVar, gVar, gVar])
    accCov = np.diag([aVar]*3)
    magCov = np.diag([mVar]*3)
    self.ukf = UKF(self.ukfModel, processCov, [self.ukfAccMeasure, self.ukfMagMeasure], [accCov, magCov])
    self.ukf.init(
      np.array([1,0,0,0]),
      np.diag([1,1,1,1]))
    self.accBuffer = deque(maxlen=int(np.ceil(gravity_measure_acc_buffer_time * acc_freq)))
  
  def getOrientation(self):
    x = self.ukf.getState()
    return Quaternion(x[0], x[1], x[2], x[3])
  
  def normalizeState(self):
    self.ukf.x[:4] = self.ukf.x[:4] / np.linalg.norm(self.ukf.x[:4])
  
  def ukfModel(self, x, w, dt: float):
    w = Quaternion(0, w[0], w[1], w[2])
    q = Quaternion(x[0], x[1], x[2], x[3])
    qdot = 0.5 * q * w
    q += qdot*dt
    return np.concatenate([q.asArray(), x[4:]])
  
  def ukfAccMeasure(self, x):
    q = Quaternion(x[0], x[1], x[2], x[3])
    yhat = q.applyInverseTo([0,0,1])
    return yhat
    
  def ukfMagMeasure(self, x, m):
    q = Quaternion(x[0], x[1], x[2], x[3])
    yhat = q.applyInverseTo([1,0,0])
    #yhat = q.applyInverseTo([0,1,0])
    #print(f"yhat mag:{yhat}")
    #print(f"mag measure:{m}")
    #print(f"diff norm:{np.linalg.norm(yhat-m)}")
    return yhat
  
  def initUkf(self):
    if(self.initialized != OrientationEstimateInitSteps.WAITING_FOR_INIT): return
    x0 = np.concatenate((self.estimate.asArray(),[]))
    P0 = np.diag([0.1,0.1,0.1,0.1])
    self.ukf.init(x0, P0)
    self.initialized |= OrientationEstimateInitSteps.RUNNING
    print("UKF initialized !")
  
  def callback(f):
    def wrapper(self, ts, *args):
      if(self.lastTs.get(f) is None):
        self.lastTs[f] = ts
        return
      dt = ts - self.lastTs[f]
      self.lastTs[f] = ts
      f(self, dt, *args)
    return wrapper
  
  @callback
  def PushGyro(self, dt, wx, wy, wz):
    # print(f"gyro dt={dt}")
    w = Quaternion(0, wx, wy, wz)
    qdot = 0.5 * self.estimate * w
    self.estimate += qdot * dt
    self.estimate = self.estimate.normalized()
    self.ukf.predict(np.array([wx, wy, wz]), dt)
    self.normalizeState()
  
  @callback
  def PushAcc(self, dt, ax, ay, az):
    # print(f"acc dt={dt}")
    self.accBuffer.append(np.array([ax, ay, az]))
    accNorm = np.linalg.norm(self.accBuffer, axis=1)
    if(len(self.accBuffer) != self.accBuffer.maxlen or not np.all(np.isclose(accNorm, GRAVITY, atol=GRAVITY_ERROR))):
    #   print(f"accNorm={accNorm}")
      return
    a = np.mean(self.accBuffer, axis=0)
    n = np.linalg.norm(a)
    if(not (self.initialized & OrientationEstimateInitSteps.ACC)):
      up = self.estimate.applyTo(a/n)
      measure = Quaternion.FromVectors(up, [0,0,1])
      self.estimate = (measure * self.estimate).normalized()
      self.initialized |= OrientationEstimateInitSteps.ACC
      self.initUkf()
    else:
      print(f"Acc measure: {a}")
      self.ukf.update(0, a/n)
      # Normalize orientation part of state vector for filter stability
      self.normalizeState()
    
    