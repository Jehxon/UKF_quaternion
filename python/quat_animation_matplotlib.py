import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import threading
from server_sensor_data import SensorServer
from quaternions import Quaternion
from orientation_estimation import OrientationEstimator
from arrow3d import Arrow3D

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

estimate_arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', linewidth=4)
# true_arrow_prop_dict = dict(mutation_scale=1, arrowstyle='-[', linewidth=3)

x_arrow = Arrow3D(0,0,0, 1,0,0, **estimate_arrow_prop_dict, color='red')
y_arrow = Arrow3D(0,0,0, 0,1,0, **estimate_arrow_prop_dict, color='green')
z_arrow = Arrow3D(0,0,0, 0,0,1, **estimate_arrow_prop_dict, color='blue')
ax.add_artist(x_arrow)
ax.add_artist(y_arrow)
ax.add_artist(z_arrow)


# true_x_arrow = Arrow3D(0,0,0, 1,0,0, **true_arrow_prop_dict, color='pink')
# true_y_arrow = Arrow3D(0,0,0, 0,1,0, **true_arrow_prop_dict, color='cyan')
# true_z_arrow = Arrow3D(0,0,0, 0,0,1, **true_arrow_prop_dict, color='magenta')
# ax.add_artist(true_x_arrow)
# ax.add_artist(true_y_arrow)
# ax.add_artist(true_z_arrow)

estimator = OrientationEstimator()
sensorServer = SensorServer()
sensorServer.gyroCb.append(estimator.PushGyro)
sensorServer.accCb.append(estimator.PushAcc)
# sensorServer.magnetoCb.append(estimator.magnetoCb)
# sensorServer.orientationCb.append(estimator.orientationCb)
serverThread = threading.Thread(target=sensorServer.start)

def init():
    return x_arrow, y_arrow, z_arrow,

def animate(i):
    #q = estimator.estimate
    q = estimator.getOrientation()
    x_vector = q.applyTo([1,0,0])
    y_vector = q.applyTo([0,1,0])
    z_vector = q.applyTo([0,0,1])
    x_arrow.setDirection(*x_vector)
    y_arrow.setDirection(*y_vector)
    z_arrow.setDirection(*z_vector)
    
    # qt = estimator.truthValue
    # x_vector = qt.applyTo([1,0,0])
    # y_vector = qt.applyTo([0,1,0])
    # z_vector = qt.applyTo([0,0,1])
    # true_x_arrow.setDirection(*x_vector)
    # true_y_arrow.setDirection(*y_vector)
    # true_z_arrow.setDirection(*z_vector)
    return x_arrow, y_arrow, z_arrow,# true_x_arrow, true_y_arrow, true_z_arrow,

anim = animation.FuncAnimation(fig, animate, init_func=init, interval=1000/24, blit=True, cache_frame_data=False)

serverThread.start()
plt.show()