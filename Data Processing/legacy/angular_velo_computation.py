import json
import numpy as np
from scipy.interpolate import PchipInterpolator
import csv

# -----------------------------
# Load JSON objects
# -----------------------------
t_list = []
quaternions = []

with open("changing_outputs.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        t_list.append(data["ts_ns_mono"])
        quaternions.append(data["quaternion_wxyz"])  # [w, x, y, z]

t_list = np.array(t_list, dtype=np.float64)
quaternions = np.array(quaternions, dtype=np.float64)

# Normalize timestamps to seconds
t_list = (t_list - t_list[0]) * 1e-9

# -----------------------------
# PCHIP interpolators for each component
# -----------------------------
quat_itp = [PchipInterpolator(t_list, quaternions[:, i]) for i in range(4)]
quat_dot_itp = [itp.derivative() for itp in quat_itp]

# Evaluate the quaternion interpolators on a time grid
print_times = np.linspace(t_list[0], t_list[-1], 20)  # 20 sample points

print("\nInterpolated quaternion components:")
for t in print_times:
    qw = quat_dot_itp[0](t)
    qx = quat_dot_itp[1](t)
    qy = quat_dot_itp[2](t)
    qz = quat_dot_itp[3](t)
    print(f"t = {t:.6f} s:  qw={qw:.6f}, qx={qx:.6f}, qy={qy:.6f}, qz={qz:.6f}")

# -----------------------------
# Compute angular velocity
# -----------------------------
omega_list = []

for t in t_list:
    # Evaluate quaternion and derivative at this time
    q = np.array([itp(t) for itp in quat_itp])        # [w,x,y,z]
    qdot = np.array([itp(t) for itp in quat_dot_itp]) # [w_dot,x_dot,y_dot,z_dot]

    # Renormalize quaternion
    q = q / np.linalg.norm(q)

    # Construct W matrix
    w, x, y, z = q
    W = np.array([
        [-x,  w,  z, -y],
        [-y, -z,  w,  x],
        [-z,  y, -x,  w]
    ])

    # Angular velocity
    omega = 2 * W @ qdot
    omega_list.append(omega)

omega_array = np.array(omega_list)  # shape (N,3)

with open('cam_ang_velo.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time','omega_x','omega_y','omega_z'])
    for t, w in zip(t_list, omega_array):
        writer.writerow([t, w[0], w[1], w[2]])


print(f"\nAngular velocity written to: cam_ang_velo.csv")
print("First 5 rows:")
for t, omega in zip(t_list[:5], omega_array[:5]):
    print(f"t = {t:.6f} s, ω = {omega}")
