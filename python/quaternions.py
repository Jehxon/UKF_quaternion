import numpy as np

class Quaternion:
  def __init__(self, w = 1.0, x = 0.0, y = 0.0, z = 0.0):
    self.w = w
    self.x = x
    self.y = y
    self.z = z
  
  def __eq__(self, q):
    if isinstance(q, Quaternion):
      return self.w == q.w and self.x == q.x and self.y == q.y and self.z == q.z
    else:
      return False
  
  def __mul__(self, q):
    if isinstance(q, Quaternion):
      return Quaternion(
      self.w * q.w - self.x * q.x - self.y * q.y - self.z * q.z,
      self.w * q.x + self.x * q.w + self.y * q.z - self.z * q.y,
      self.w * q.y + self.y * q.w + self.z * q.x - self.x * q.z,
      self.w * q.z + self.z * q.w + self.x * q.y - self.y * q.x
      )
    elif isinstance(q, (int, float)):
      return Quaternion(
      self.w * q,
      self.x * q,
      self.y * q,
      self.z * q
      )
    else:
      raise TypeError(f'Cannot multiply a Quaternion with {type(q)}')
  
  def __rmul__(self, q):
    if isinstance(q, (int, float)):
      return self.__mul__(q)
    else:
      raise TypeError(f'Cannot right-multiply a Quaternion with {type(q)}')
  
  def __truediv__(self, v):
    return self.__mul__(1/v)
  
  def __add__(self, q):
    return Quaternion(
    self.w + q.w,
    self.x + q.x,
    self.y + q.y,
    self.z + q.z
    )
  
  def asArray(self):
    return np.array([self.w, self.x, self.y, self.z])
  
  def norm2(self):
    return self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
  
  def norm(self) -> float:
    return self.norm2() ** 0.5
  
  def normalized(self):
    norm = self.norm()
    return self / norm
    
  def __str__(self):
    return f"(w={self.w},x={self.x},y={self.y},z={self.z})"
  
  def __repr__(self):
    return f"Quaternion({self.w},{self.x},{self.y},{self.z})"
  
  def conjugate(self):
    return Quaternion(
    self.w,
    -self.x,
    -self.y,
    -self.z
    )
  
  def inverse(self):
    return self.conjugate() / self.norm2()
  
  def applyTo(self, v):
    x = Quaternion(0, v[0], v[1], v[2])
    # Ensure normalized quat
    q = self.normalized()
    vr = q * x * q.conjugate()
    return np.array([vr.x, vr.y, vr.z])
  
  def applyInverseTo(self, v):
    return self.conjugate().applyTo(v)
  
  def toMatrix(self):
    r00 = 2*(self.w*self.w + self.x*self.x) - 1
    r01 = 2*(self.x*self.y - self.w*self.z)
    r02 = 2*(self.x*self.z + self.w*self.y)

    r10 = 2*(self.x*self.y + self.w*self.z)
    r11 = 2*(self.w*self.w + self.y*self.y) - 1
    r12 = 2*(self.y*self.z - self.w*self.x)

    r20 = 2*(self.x*self.z - self.w*self.y)
    r21 = 2*(self.y*self.z + self.w*self.x)
    r22 = 2*(self.w*self.w + self.z*self.z) - 1
     
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
    return np.array([[r00, r01, r02],
                     [r10, r11, r12],
                     [r20, r21, r22]])

  def toEulerZYX(self):
    sqw = self.w*self.w
    sqx = self.x*self.x
    sqy = self.y*self.y
    sqz = self.z*self.z
    norm = np.sqrt(sqw + sqx + sqy + sqz)
    pole_result = (self.x * self.z) + (self.y * self.w)
    if (pole_result > (0.5 * norm)): # singularity at north pole
        ry = np.pi/2 #heading/yaw?
        rz = 0 #attitude/roll?
        rx = 2 * np.arctan2(self.x, self.w) #bank/pitch?
        return (rx, ry, rz)
    if (pole_result < (-0.5 * norm)): # singularity at south pole
        ry = -np.pi/2
        rz = 0
        rx = -2 * np.arctan2(self.x, self.w)
        return (rx, ry, rz)
    r11 = 2*(self.x*self.y + self.w*self.z)
    r12 = sqw + sqx - sqy - sqz
    r21 = -2*(self.x*self.z - self.w*self.y)
    r31 = 2*(self.y*self.z + self.w*self.x)
    r32 = sqw - sqx - sqy + sqz
    rx = np.arctan2(r31, r32)
    ry = np.arcsin (r21)
    rz = np.arctan2(r11, r12)
    return (rx, ry, rz)
    
  def FromEulerZYX(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return Quaternion(qw, qx, qy, qz)
  
  @staticmethod
  def FromVectors(u, v):
    # Returns a quaternion q such that q*u*qconj / ||u|| = v / ||v||
    # The chosen quaternion will be the only one to not apply torque to u (see https://math.stackexchange.com/questions/2251214/calculate-quaternions-from-two-directional-vectors)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    cross = np.cross(u, v)
    return 1 / (2*nu*nv) * Quaternion(
      nu*nv + u.T@v,
      cross[0],
      cross[1],
      cross[2]
    )
    


def tests():
  q1 = Quaternion()
  print(q1)
  print(f"norm of q1 is : {q1.norm()} (should be 1)")
  
  q2 = Quaternion(6.28426, 0.53668, -0.0499, 4.99386)
  print(q2)
  print(f"norm of q2 is : {q2.norm()}")
  print(2*q2)
  print(f"2*q2 = {2*q2} of norm {(2*q2).norm()}")
  print(f"q2*2 == 2*q2 : {2*q2 == q2*2} (should be true)")
  q2n = q2.normalized()
  print(f"q2.normalized is : {q2n}, of norm {q2n.norm()} (should be 1)")
  
  print(f"q1*q2 = {q1*q2}, q2*q1 = {q2*q1} (should be the same as q2)")
  print(f"q1 == q2 : {q1== q2} (should be false)")
  print(f"q2 == q2n : {q2 == q2n} (should be false)")
  print(f"q2*q2.inverse() : {q2*q2.inverse()}")
  
  q3 = Quaternion(-0.8426, 0.3668, -0.499, 0.386)
  print(f"q3 : {q3}")
  print(f"q2*q3 : {q2*q3}")
  print(f"q3*q2 : {q3*q2}")
  print(f"q2*q3 == q3*q2 : {q2*q3 == q3*q2} (should be false)")
  
  print(f"3.2*q1*q2*q3 = {3.2*q1*q2*q3}")
  
  v = np.random.random(3)
  print(f"v={v}")
  print(f"q2*v : {q2.applyTo(v)}")
  print(f"q1*v == v : {q1.applyTo(v) == v} (should be true)")
  
  print(f"q1.toEuler = {q1.toEulerZYX()}")
  q4 = Quaternion(-0.242, 0.354, -0.857, 0.286)
  print(f"q4={q4}, euler are {q4.toEulerZYX()}")
  
  print(f"FromEuler(1.1, -0.3, -1.8).toEuler : {Quaternion.FromEulerZYX(1.1, -0.3, -1.8).toEulerZYX()}")

if __name__ == "__main__":
  tests()