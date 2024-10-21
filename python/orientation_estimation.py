from server_sensor_data import SensorServer
from quaternions import Quaternion
import numpy as np
from kalman_UKF.ukf import UKF
import json
from collections import deque

GRAVITY = 9.81
GRAVITY_ERROR = 0.05

class OrientationEstimator:
  def __init__(self):
    self.estimate = Quaternion()
    self.truthValue = Quaternion()
    self.lastTs = dict()
    
    gyroStdDev = 0.005
    accStdDev = 0.02
    magStdDev = 0.7
    
    gVar = gyroStdDev*gyroStdDev
    aVar = accStdDev*accStdDev
    mVar = magStdDev*magStdDev
    
    processCov = np.diag([gVar, gVar, gVar, gVar])
    accCov = np.diag([aVar]*3)
    magCov = np.diag([mVar]*3)
    self.ukf = UKF(processCov, self.ukfModel, [accCov, magCov], [self.ukfAccMeasure, self.ukfMagMeasure])
    self.ukf.init(
      np.array([1,0,0,0]),
      np.diag([1,1,1,1]))
    self.initialized = 0
    self.accBuffer = deque(maxlen=10) # unused for now
  
  def getOrientation(self):
    x = self.ukf.getState()
    return Quaternion(x[0], x[1], x[2], x[3])
  
  def normalizeState(self):
    self.ukf.x[:4] = self.ukf.x[:4] / np.linalg.norm(self.ukf.x[:4])
  
  def ukfModel(self, x, w, dt):
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
    print(f"diff norm:{np.linalg.norm(yhat-m)}")
    return yhat
  
  def initUkf(self):
    x0 = np.concatenate((self.estimate.asArray(),[]))
    P0 = np.diag([0.1,0.1,0.1,0.1])
    self.ukf.init(x0, P0)
    self.initialized |= 1<<2
    print("init ukf !")
    
    
  def callback(f):
    def wrapper(self, data):
      data = json.loads(data)
      ts = data["Timestamp"] * 0.001
      x = float(data["x"])
      y = float(data["y"])
      z = float(data["z"])
      if(self.lastTs.get(f) is None):
        self.lastTs[f] = ts
        return
      dt = ts - self.lastTs[f]
      self.lastTs[f] = ts
      f(self, x, y, z, dt)
    return wrapper
  
  @callback
  def gyroCb(self, wx, wy, wz, dt):
    #print(f"gyro dt={dt}")
    w = Quaternion(0, wx, wy, wz)
    qdot = 0.5 * self.estimate * w
    self.estimate += qdot * dt
    self.estimate = self.estimate.normalized()
    self.ukf.predict(np.array([wx, wy, wz]), dt)
  
  @callback
  def accCb(self, ax, ay, az, dt):
    #print(f"acc dt={dt}")
    a = np.array([ax, ay, az])
    accNorm = np.linalg.norm(a)
    if(not np.isclose(accNorm, GRAVITY, atol=GRAVITY_ERROR)):
      return
    #print("Acc measure !")
    up = self.estimate.applyTo(a/accNorm)
    measure = Quaternion.FromVectors(up, [0,0,1])
    self.estimate = (measure * self.estimate).normalized()
    self.initialized |= 1 << 0
    
    self.ukf.update(0, a/accNorm)
    # Normalize orientation part of state vector for filter stability
    self.normalizeState()
  
  @callback
  def magnetoCb(self, mx, my, mz, dt):
    #print(f"magneto dt={dt}")
    # Only update if we know where 'up' is, hence if we have had an accelerometer data before
    if(not self.initialized & (1 << 0)):
      return
    
    m = np.array([mx, my, mz])
    magNorm = np.linalg.norm(m)
    north = self.estimate.applyTo(m/magNorm)
    # This 'north' is not always in the right plane... Let's force it orthogonal to up :
    east = np.cross(north, [0,0,1])
    east = east/np.linalg.norm(east)
    measure = Quaternion.FromVectors(east, [1,0,0])
    self.estimate = (measure * self.estimate).normalized()
    self.initialized |= 1 << 1
    if(self.initialized == 1<<0 | 1<<1):
      self.initUkf()
    # Call ukf update : same idea
    # Force measure to be east, orthogonal to ukf 'up' estimate
    q = self.getOrientation()
    north = q.applyTo(m)
    east = np.cross(north, [0,0,1])
    # We reconstruct a vector with norm 1 but in the east direction
    east = east/np.linalg.norm(east)
    eastBody = q.applyInverseTo(east)
    self.ukf.update(1, eastBody, eastBody)
    #self.ukf.update(1, m/np.linalg.norm(m), m/np.linalg.norm(m))
    # Normalize orientation part of state vector for filter stability
    #print(f"ukf state norm:{np.linalg.norm(self.ukf.x)}")
    self.normalizeState()
    #print(f"ukf mean sigma point norm:{np.mean([np.linalg.norm(self.ukf.X[:,i]) for i in range(self.ukf.ns)])}")
    #print(f"state cov:{self.ukf.getCovariance()}")
  
    
  def orientationCb(self, data):
    data = json.loads(data)
    # Convert from NED to ENU
    r = -np.deg2rad(float(data["pitch"]))
    p = np.deg2rad(float(data["roll"]))
    y = -np.deg2rad(float(data["azimuth"]))
    print(f"orientation cb : r={r},p={p},y={y}")
    self.truthValue = Quaternion.FromEulerZYX(r, p, y)
  
  def bindCallbacksToServer(self, sensorServer):
    sensorServer.gyroCb.append(self.gyroCb)
    sensorServer.accCb.append(self.accCb)
    sensorServer.magnetoCb.append(self.magnetoCb)
    sensorServer.orientationCb.append(self.orientationCb)
    
if __name__ == "__main__":
    estimator = OrientationEstimator()
    sensorServer = SensorServer()
    estimator.bindCallbacksToServer(sensorServer)
    sensorServer.start()
    