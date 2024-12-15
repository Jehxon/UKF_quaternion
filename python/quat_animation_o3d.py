import numpy as np
import threading
import open3d as o3d
import time
from server_sensor_data import SensorServer
from orientation_estimation import OrientationEstimator

FPS = 60

estimator = OrientationEstimator()
sensorServer = SensorServer()
sensorServer.gyroCb.append(estimator.PushGyro)
sensorServer.accCb.append(estimator.PushAcc)

serverThread = threading.Thread(target=sensorServer.start)
serverThread.start()

class Visualizer:
    def __init__(self):
        self.current_orientation = np.eye(3)
        self.vehicle = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("UKF Orientation estimation")
        self.vis.get_render_option().line_width = 5
        self.vis.add_geometry(self.vehicle)
        self.vis.get_view_control().set_zoom(1.2)
        self.vis.get_view_control().set_lookat([0, 0, 0])
        self.vis.get_view_control().set_front([1, 1, 1])
        self.vis.get_view_control().set_up([0, 0, 1])

        self.shutdown = False
        self._registerKeyCallbacks()


    def _registerKeyCallbacks(self):
        self._registerKeyCallback(["Ā", "\x1b"], self._quit)
    
    def _quit(self, vis):
        self.shutdown = True

    def _registerKeyCallback(self, keys, callback):
        for key in keys:
            self.vis.register_key_callback(ord(key), callback)

    def refresh(self):
        new_orientation = estimator.getOrientation().toMatrix()
        self.vehicle.rotate(new_orientation @ self.current_orientation.T)
        self.current_orientation = new_orientation
        self.vis.update_geometry(self.vehicle)

    def run(self):
        while(not self.shutdown):
            self.refresh()
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(1.0/60)
        self.vis.destroy_window()

visu = Visualizer()
visu.run()
