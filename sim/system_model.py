import numpy as np  


# System Constants
NUM_STATES = 6  # [x, y, z, vx, vy, vz]

# Noise / process config
process_noise_std = 0.0025
process_noise_variance = np.sqrt(process_noise_std)
measurement_noise_std = 0.01
measurement_noise_variance = np.sqrt(measurement_noise_std)

# Rates
imu_hz = 50        # IMU update/logging rate (can be different from integrator)
pf_dt = 1 / imu_hz
ranging_hz = 10

# Firefly Algo Rates
firefly_imu_hz = 50        # IMU update/logging rate (can be different from integrator)
firefly_pf_dt = 1 / firefly_imu_hz
firefly_ranging_hz = 1

# Beacons
CENTER = np.array([0.0,  0.0,  1.0]) # central of drone movement(0,0,1)
BEACONS = np.array([
    [-12.0,    12.0,  0.0],  
    [-12.0,   -12.0,  12.0],
    [12.0,  -12.0,  7.0],
], dtype=float)

# Beacons
FIREFLY_BEACONS = np.array([
    [-12.0,    12.0,  0.0],  
    #[-12.0,   -12.0,  12.0],
], dtype=float)

# Indoor Space Definition
X_LIM = [ -15.0, 15.0 ]
Y_LIM = [ -15.0, 15.0 ]
Z_LIM = [   0.0, 15.0 ]

A = np.array([
                [1, 0, 0, pf_dt,      0,      0],
                [0, 1, 0,     0,  pf_dt,      0],
                [0, 0, 1,     0,      0,  pf_dt],
                [0, 0, 0,     1,      0,      0],
                [0, 0, 0,     0,      1,      0],
                [0, 0, 0,     0,      0,      1]
            ])

B = np.array([
                 [0.5*pf_dt**2,            0,            0],
                 [           0, 0.5*pf_dt**2,            0],
                 [           0,            0, 0.5*pf_dt**2],
                 [       pf_dt,            0,            0],
                 [           0,        pf_dt,            0],
                 [           0,            0,        pf_dt]
             ])

A_FIREFLY = np.array([
                [1, 0, 0, firefly_pf_dt,      0,      0],
                [0, 1, 0,     0,  firefly_pf_dt,      0],
                [0, 0, 1,     0,      0,  firefly_pf_dt],
                [0, 0, 0,     1,      0,      0],
                [0, 0, 0,     0,      1,      0],
                [0, 0, 0,     0,      0,      1]
            ])

B_FIREFLY = np.array([
                 [0.5*firefly_pf_dt**2,            0,            0],
                 [           0, 0.5*firefly_pf_dt**2,            0],
                 [           0,            0, 0.5*firefly_pf_dt**2],
                 [       firefly_pf_dt,            0,            0],
                 [           0,        firefly_pf_dt,            0],
                 [           0,            0,        firefly_pf_dt]
             ])
