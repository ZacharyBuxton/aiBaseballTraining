import zmq
import json
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

# ==================== CONFIGURATION ====================
CAMERA_REFERENCE_POINTS = {
    0: np.array([0.0, 0.0, 0.0]),      # Camera 0 reference position [X, Y, Z] in meters
    1: np.array([0.5, 0.0, 0.0])       # Camera 1 reference position [X, Y, Z] in meters (example: 0.5m offset in X)
}

GLOBAL_ORIGIN = np.array([0.0, 0.0, 0.0])
MIN_DISTANCE_THRESHOLD = 0.05

class QuaternionCalculator:
    def __init__(self):
        pass

    def calculate_relative_position(self, camera_id, position):
        camera_offset = CAMERA_REFERENCE_POINTS.get(camera_id, np.array([0.0, 0.0, 0.0]))
        relative_position = position - camera_offset
        return relative_position

    def position_to_quaternion(self, position_vector, reference_up=np.array([0, 0, 1])):
        distance = np.linalg.norm(position_vector)

        if distance < MIN_DISTANCE_THRESHOLD:
            return np.array([0.0, 0.0, 0.0, 1.0])

        direction = position_vector / distance
        reference_direction = np.array([0, 0, 1])
        rotation_axis = np.cross(reference_direction, direction)
        axis_length = np.linalg.norm(rotation_axis)
        dot_product = np.dot(reference_direction, direction)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)

        if axis_length < 1e-6:
            if dot_product > 0:
                return np.array([0.0, 0.0, 0.0, 1.0])
            else:
                if abs(reference_direction[0]) < 0.9:
                    rotation_axis = np.array([1, 0, 0])
                else:
                    rotation_axis = np.array([0, 1, 0])
                angle = np.pi
        else:
            rotation_axis = rotation_axis / axis_length

        rotation = R.from_rotvec(rotation_axis * angle)

        # Get quaternion [x, y, z, w]
        quaternion = rotation.as_quat()

        return quaternion

    def process_coordinate(self, camera_id, color_name, x, y, z, timestamp):
        camera_position = np.array([x, y, z])
        global_position = self.calculate_relative_position(camera_id, camera_position)

        # Calculate quaternion from position
        position_quaternion = self.position_to_quaternion(global_position)

        # Prepare output data
        result = {
            'camera_id': camera_id,
            'color': color_name,
            'timestamp': timestamp,
            'camera_position': camera_position.tolist(),
            'global_position': global_position.tolist(),
            'distance_from_origin': float(np.linalg.norm(global_position)),
            'quaternion': position_quaternion.tolist(),
            'euler_angles_deg': R.from_quat(position_quaternion).as_euler('xyz', degrees=True).tolist(),
        }

        return result

# ==================== SUBSCRIBER ====================
class CoordinateSubscriber:
    def __init__(self, host="localhost", port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{host}:{port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        print(f"Subscribed to coordinates on {host}:{port}")

        self.calculator = QuaternionCalculator()
        self.message_count = 0
        self.last_print_time = time.time()

    def run(self, callback=None, print_interval=1.0):
        print("\nWaiting for coordinate data...")
        print("=" * 80)

        try:
            while True:
                message_str = self.socket.recv_string()
                message = json.loads(message_str)

                # Extract data
                camera_id = message['camera_id']
                color_name = message['color']
                pos = message['position']
                timestamp = message['timestamp']

                # Process and calculate quaternion
                result = self.calculator.process_coordinate(
                    camera_id, color_name,
                    pos['x'], pos['y'], pos['z'],
                    timestamp
                )

                # Call callback if provided
                if callback is not None:
                    callback(result)

                # Print results periodically
                self.message_count += 1
                current_time = time.time()

                if current_time - self.last_print_time >= print_interval:
                    self.print_result(result)
                    self.last_print_time = current_time

        except KeyboardInterrupt:
            print("\n\nShutting down subscriber...")
        finally:
            self.close()

    def print_result(self, result):
        """Print formatted result"""
        print(f"\n[Camera {result['camera_id']}] {result['color'].upper()}")
        print(f"  Camera Position:  [{result['camera_position'][0]:7.3f}, "
              f"{result['camera_position'][1]:7.3f}, {result['camera_position'][2]:7.3f}] m")
        print(f"  Global Position:  [{result['global_position'][0]:7.3f}, "
              f"{result['global_position'][1]:7.3f}, {result['global_position'][2]:7.3f}] m")
        print(f"  Distance:         {result['distance_from_origin']:7.3f} m")

        quat = result['quaternion']
        print(f"  Quaternion:       [x:{quat[0]:7.4f}, y:{quat[1]:7.4f}, "
              f"z:{quat[2]:7.4f}, w:{quat[3]:7.4f}]")

        euler = result['euler_angles_deg']
        print(f"  Euler (XYZ):      [roll:{euler[0]:7.2f}°, pitch:{euler[1]:7.2f}°, "
              f"yaw:{euler[2]:7.2f}°]")

        print("-" * 80)

    def close(self):
        """Clean up ZMQ resources"""
        self.socket.close()
        self.context.term()
        print(f"\nProcessed {self.message_count} messages total.")

# ==================== MAIN ====================
if __name__ == "__main__":
    print("=" * 80)
    print("QUATERNION CALCULATOR SUBSCRIBER")
    print("=" * 80)
    print("\nCamera Reference Points:")
    for cam_id, pos in CAMERA_REFERENCE_POINTS.items():
        print(f"  Camera {cam_id}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m")
    print(f"\nGlobal Origin: [{GLOBAL_ORIGIN[0]:.3f}, {GLOBAL_ORIGIN[1]:.3f}, {GLOBAL_ORIGIN[2]:.3f}] m")
    print("=" * 80)

    subscriber = CoordinateSubscriber(host="localhost", port=5555)

    # Run subscriber (prints every 1 second)
    subscriber.run(callback=None, print_interval=1.0)
