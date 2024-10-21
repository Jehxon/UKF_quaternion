import asyncio
import websockets
import socket
from base64 import b64decode
import json


class SensorServer:
  
  def __init__(self):
    self.accCb = []
    self.gyroCb = []
    self.magnetoCb = []
    self.orientationCb = []
    self.geolocCb = []
  
  def get_ip(self):
      s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      try:
          # doesn't even have to be reachable
          s.connect(('10.255.255.255', 1))
          IP = s.getsockname()[0]
      except Exception:
          IP = '127.0.0.1'
      finally:
          s.close()
      return IP
  
  async def echo(self, websocket, path):
    async for message in websocket:
        if path == '/accelerometer':
            data = await websocket.recv()
            for cb in self.accCb:
              cb(data)

        if path == '/gyroscope':
            data = await websocket.recv()
            for cb in self.gyroCb:
              cb(data)

        if path == '/magnetometer':
            data = await websocket.recv()
            for cb in self.magnetoCb:
              cb(data)

        if path == '/orientation':
            data = await websocket.recv()
            for cb in self.orientationCb:
              cb(data)

        if path == '/geolocation':
            data = await websocket.recv()
            for cb in self.geolocCb:
              cb(data)

        if path == '/stepcounter':
            data = await websocket.recv()
            print(f"stepcounter: {data}")

        if path == '/thermometer':
            data = await websocket.recv()
            print(f"thermometer: {data}")

        if path == '/lightsensor':
            print("connected")
            data = await websocket.recv()
            print(f"lightsensor: {data}")

        if path == '/proximity':
            data = await websocket.recv()
            print(f"proximity: {data}")

  # Contribution by Evan Johnston
  async def main(self):
    async with websockets.serve(self.echo, '0.0.0.0', 5000, max_size=1_000_000_000):
        await asyncio.Future()
    
  def start(self):
    hostname = socket.gethostname()
    IPAddr = self.get_ip()
    print("Your Computer Name is: " + hostname)
    print("Your Computer IP Address is: " + IPAddr)
    print(
    "* Enter {0}:5000 in the app.\n* Press the 'Set IP Address' button.\n* Select the sensors to stream.\n* Update the 'update interval' by entering a value in ms.".format(IPAddr))
    asyncio.run(self.main())


if __name__ == "__main__":
  s = SensorServer()
  s.start()
