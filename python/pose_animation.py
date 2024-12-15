import numpy as np
import threading
import open3d as o3d
import time
from server_sensor_data import SensorServer
from pose_estimation import PoseEstimator

FPS = 24

estimator = PoseEstimator()
sensorServer = SensorServer()
sensorServer.gyroCb.append(estimator.PushGyro)
sensorServer.accCb.append(estimator.PushAcc)

serverThread = threading.Thread(target=sensorServer.start)
serverThread.start()

class Visualizer:
    def __init__(self):
        self.current_position = np.zeros(3)
        self.current_orientation = np.eye(3)
        self.vehicle = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.lines = o3d.geometry.LineSet()
        self.lines_index = [[0,0]] # Prevent Open3d warning
        self.lines_pts = []

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("UKF Pose estimation")
        self.vis.get_render_option().line_width = 5
        self.vis.add_geometry(self.vehicle)
        self.vis.add_geometry(self.lines)

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
        new_position = estimator.getPosition()
        new_orientation = estimator.getOrientation().toMatrix()
        # print(f"new_position={new_position}")
        self.vehicle.translate(new_position - self.current_position)
        self.vehicle.rotate(new_orientation @ self.current_orientation.T)
        self.current_position = new_position
        self.current_orientation = new_orientation

        # Update trail
        n_pts = len(self.lines_pts)
        if(n_pts > 1):
            self.lines_index.append([n_pts-1, n_pts])
        self.lines_pts.append(self.current_position)
        self.lines.points = o3d.utility.Vector3dVector(self.lines_pts)
        self.lines.lines = o3d.utility.Vector2iVector(self.lines_index)
        self.lines.paint_uniform_color([0.1,0.2,1])

        self.vis.update_geometry(self.vehicle)
        self.vis.update_geometry(self.lines)

    def run(self):
        while(not self.shutdown):
            self.refresh()
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(1.0/FPS)
        self.vis.destroy_window()

visu = Visualizer()
visu.run()
