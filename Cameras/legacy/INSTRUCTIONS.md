# Camera Folder Instructions

_Written by Dmitrii Kapranov (previous group)_

`Device_Tracking_2_16.py` is the latest object tracking program the previous group developed. It
uses OpenCV libraries to find the position of different colored spheres and prints it in the
terminal. The program is written to alternate between two Azure Kinect cameras, so make sure
both cameras are connected if you want to test the file.

Requires: `pykinect_azure`, `opencv-python`, `numpy`, `pyzmq`. Publishes 3D coordinates over a
ZeroMQ `PUB` socket on port 5555 (see `../Data Processing/legacy/Subscription_Ball_Test_11_4_2.py` for a
matching subscriber).

If you have more questions about this file, the previous group's notes say to reach out to
Chris Powers, as he was the main developer of the program.
