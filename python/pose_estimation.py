from quaternions import Quaternion
import numpy as np
from kalman_UKF.ukf import UKF
from collections import deque

GRAVITY = 9.81
GRAVITY_ERROR = 0.1
gravity_measure_acc_buffer_time = 0.5 # seconds needed of acceleration norm being approximately GRAVITY to call a gravity measurement

gyroStdDev = 0.005
accStdDev = 0.1

imu_freq = 10.0

class PoseEstimateInitSteps(np.uint8):
    NONE = np.uint8(0)
    ACC = np.uint8(1 << 0)
    MAG = np.uint8(1 << 1)
    WAITING_FOR_INIT = ACC #| MAG
    RUNNING = np.uint8(1 << 2)
    
class PoseEstimator:
  def __init__(self):
    self.orientationEstimate = Quaternion()
    self.initialized = PoseEstimateInitSteps.NONE
    self.lastGyro = np.zeros(3)
    self.lastAcc = np.zeros(3)
    self.lastGyroTs = None
    self.lastAccTs = None
    self.lastGyroUpdateTs = None
    self.lastAccUpdateTs = None
    
    gVar = gyroStdDev*gyroStdDev
    aVar = accStdDev*accStdDev
    
    processCov = np.diag([gVar, gVar, gVar, gVar, 0, 0, 0, aVar, aVar, aVar])
    accCov = np.diag([aVar]*3)
    self.ukf = UKF(self.ukfModel, processCov, [self.ukfAccMeasure], [accCov])
    self.ukf.init(
      np.array([1,0,0,0, 0,0,0, 0,0,0]),
      np.diag([1,1,1,1, 0.01,0.01,0.01, 0.1,0.1,0.1]))
    self.accBuffer = deque(maxlen=int(np.ceil(gravity_measure_acc_buffer_time * imu_freq)))
  
  def getOrientation(self):
    x = self.ukf.getState()
    return Quaternion(x[0], x[1], x[2], x[3])
  
  def getPosition(self):
    x = self.ukf.getState()
    return x[4:7]
  
  def getVelocity(self):
    x = self.ukf.getState()
    return x[7:10]
  
  def normalizeOrientation(self):
    self.ukf.x[:4] = self.ukf.x[:4] / np.linalg.norm(self.ukf.x[:4])
  
  def ukfModel(self, x, w, a, dtw: float, dta: float):
    w = Quaternion(0, w[0], w[1], w[2])
    q = Quaternion(x[0], x[1], x[2], x[3])
    qdot = 0.5 * q * w

    p = x[4:7].copy()
    v = x[7:10].copy()
    a_world = q.applyTo(a) # a is measured in body frame, we want to update v in world frame
    dv = a_world - np.array([0, 0, GRAVITY]) # Remove gravity vector

    q += qdot*dtw
    p += v*dta
    v += dv*dta
    return np.concatenate([q.asArray(), p, v])
  
  def ukfAccMeasure(self, x):
    q = Quaternion(x[0], x[1], x[2], x[3])
    yhat = q.applyInverseTo([0,0,1])
    return yhat
  
  def initUkf(self):
    if(self.initialized != PoseEstimateInitSteps.WAITING_FOR_INIT): return
    x0 = np.concatenate((self.orientationEstimate.asArray(),[0,0,0, 0,0,0]))
    P0 = np.diag([0.1,0.1,0.1,0.1, 0.01,0.01,0.01, 0.1,0.1,0.1])
    self.ukf.init(x0, P0)
    self.initialized |= PoseEstimateInitSteps.RUNNING
    print("UKF initialized !")
  
  def PushGyro(self, ts, wx, wy, wz):
    self.lastGyro = np.array([wx, wy, wz])
    self.lastGyroTs = ts
    self.tryPushProprio()
  
  def PushAcc(self, ts, ax, ay, az):
    self.lastAcc = np.array([ax, ay, az])
    self.lastAccTs = ts
    self.tryPushProprio()

  def tryPushProprio(self):
    push_ok = True
    if(self.lastAccTs is None or self.lastGyroTs is None):
      push_ok = False
    if(self.lastGyroUpdateTs is None):
      self.lastGyroUpdateTs = self.lastGyroTs
      push_ok = False
    if(self.lastAccUpdateTs is None):
      self.lastAccUpdateTs = self.lastAccTs
      push_ok = False
    if(not push_ok): return
    if(min(self.lastAccTs, self.lastGyroTs) <= max(self.lastGyroUpdateTs, self.lastAccUpdateTs)):
      return
    dt_gyro = self.lastGyroTs - self.lastGyroUpdateTs
    dt_acc = self.lastAccTs - self.lastAccUpdateTs
    self.PushProprio(self.lastGyro, dt_gyro, self.lastAcc, dt_acc)
    self.lastGyroUpdateTs = self.lastGyroTs
    self.lastAccUpdateTs = self.lastAccTs
      

  def PushProprio(self, w, dtw, a, dta):
    print(f"Pushed proprio with gyroDt={dtw}, accDt={dta}")
    self.ukf.predict(w, a, dtw, dta)
    self.normalizeOrientation()
    self.accMeasurement(a)

  
  def accMeasurement(self, m):
    self.accBuffer.append(m)
    accNorm = np.linalg.norm(self.accBuffer, axis=1)
    if(len(self.accBuffer) != self.accBuffer.maxlen or not np.all(np.isclose(accNorm, GRAVITY, atol=GRAVITY_ERROR))):
    #   print(f"accNorm={accNorm}")
      return
    a = np.mean(self.accBuffer, axis=0)
    n = np.linalg.norm(a)
    if(not (self.initialized & PoseEstimateInitSteps.ACC)):
      up = self.orientationEstimate.applyTo(a/n)
      measure = Quaternion.FromVectors(up, [0,0,1])
      self.orientationEstimate = (measure * self.orientationEstimate).normalized()
      self.initialized |= PoseEstimateInitSteps.ACC
      self.initUkf()
    else:
      # print(f"Acc measure: {a}")
      self.ukf.update(0, a/n)
      # Normalize orientation part of state vector for filter stability
      self.normalizeOrientation()
    
    