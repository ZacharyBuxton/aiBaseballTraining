% Dmitrii Kapranov
% ECE 4972
% Cross Correlation
% Last Edited: 04/26/2026

clc; clear; close all;

% Load csv files
% IMPORTANT: Update file path every time you run the code
cam=readtable('C:\Users\Dmitrii Kapranov\OneDrive - Villanova University\Documents\CAPSTONE\cam_ang_velo.csv');
imu=readtable('C:\\reports\\ICM45686_and_ADXL375_test_data_2025-12-03_14-51-28.csv');

% Access time
t_cam=cam.time;
t_imu=imu.time;

% Access camera angular velocity
wx_cam=cam.omega_x;
wy_cam=cam.omega_y;
wz_cam=cam.omega_z;

% Access IMU angular velocity
wx_imu=imu.omega_x;
wy_imu=imu.omega_y;
wz_imu=imu.omega_z;

% Compute angular velocity magnitudes
cam_mag=sqrt(wx_cam.^2+wy_cam.^2+wz_cam.^2);
imu_mag=sqrt(wx_imu.^2+wy_imu.^2+wz_imu.^2);

% Find delay between streams
[~,~,D]=alignsignals(cam_mag,imu_mag);
t_del=t_imu(D);

% Adjusted camera time by calculated delay
t_adj=t_cam+t_del;

% Plot aligned data streams
figure
hold on
stem(t_adj,cam_mag)
stem(t_imu,imu_mag)
title('Camera and IMU Data Aligned')
xlabel('Time (s)')
ylabel('Angular Velocity Magnitude (rad/s)')

% Calculate % difference between aligned values
imu_alg=imu_mag(D:D+size(t_cam,1)-1,1);
diff=abs(cam_mag-imu_alg);
p_diff=diff./imu_alg;
avg_pdiff=mean(p_diff);
