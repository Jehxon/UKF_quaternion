import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import patches
import threading
from server_sensor_data import SensorServer
from quaternions import Quaternion
from orientation_estimation import OrientationEstimator

fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_xlabel('East')
ax.set_ylabel('North')

arrow = patches.FancyArrow(0,0,1,0)
ax.add_patch(arrow)

estimator = OrientationEstimator()
sensorServer = SensorServer()
estimator.bindCallbacksToServer(sensorServer)
serverThread = threading.Thread(target=
sensorServer.start)

def init():
    return arrow,

def animate(i):
    q = estimator.getOrientation()
    vector = q.applyTo([0,1,0])
    vec2D = vector[:2]
    arrow.set_data(dx=vec2D[0], dy=vec2D[1])
    return arrow,

anim = animation.FuncAnimation(fig, animate, init_func=init, interval=1000/24, blit=True, cache_frame_data=False)

serverThread.start()
plt.show()