a = 6378137.0
f = 1.0/298.257223563
b = a*(1-f)
c = a ** 2 / b
e2 = (a ** 2 - b ** 2) / (b ** 2)
omega_e = 7.292115e-5
GM = 3.986004418e14
m = omega_e ** 2 * a ** 2 * b / GM
gamma_a = 9.7833267715
gamma_b = 9.8321863685

gnss_file = "GNSS.txt"
imu_file = "IMU.txt"

gnss_hz = 1.0
imu_hz = 400.0

lat_lon_plotting_bounds = [15.42, 15.4305, 46.9835, 46.9765, ]
min_plotting_time = 221000.0
max_plotting_time = 222100.0

still_gyr_threshold = 0.8
still_acc_threshold = 0.6
min_still_time = 10.0
still_time_margin = 5.0

coarse_alignment_time = 30.0
coarse_alignment_iterations = 30
coarse_alignment_iteration_alpha = 0.5

phi_iteration_epsilon = 1e-5
h_iteration_epsilon = 1e-5
