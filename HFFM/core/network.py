import math
import numpy as np
import matplotlib.pyplot as plt

SEED = 49
global_random_gn=np.random.default_rng(SEED)

def draw_network(servers, users, area_size):
    plt.figure(figsize=(6, 6))
    plt.scatter(servers[:, 0], servers[:, 1], marker='s', label='Edge Servers', s=50)
    plt.scatter(users[:, 0], users[:, 1], marker='o', alpha=0.6, label='Users', s=30)
    # dynamically compute the plotting bounds so nothing gets clipped:
    all_x = np.concatenate([servers[:, 0], users[:, 0]])
    all_y = np.concatenate([servers[:, 1], users[:, 1]])
    margin = 0.05 * max(area_size, all_x.ptp(), all_y.ptp())  # 5% of the data‐range
    xmin, xmax = all_x.min() - margin, all_x.max() + margin
    ymin, ymax = all_y.min() - margin, all_y.max() + margin
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title('Random Deployment (all points guaranteed visible)')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.gca().set_aspect('equal', 'box')
    plt.legend()
    # === 4) Save to file ===
    plt.savefig('deployment.png', dpi=150, bbox_inches='tight')
    plt.show()


def norm2_power2(rayleigh_faded_channel_matrix):
    return rayleigh_faded_channel_matrix[:, 0] ** 2 + rayleigh_faded_channel_matrix[:, 1] ** 2

def calculate_transmission_rate(sender, receiver, transmit_power):
    N0 = -174  # -174dBm/hz
    bandwidth = 300 #Khz
    # Calculate transmitter to receiver channel gain------------
    channels = calculate_complex_channel_gain(sender, receiver, global_random_gn.standard_normal((1, 2)))
    transmit_powers = norm2_power2(channels) * transmit_power

    # calculate noise -----------------------------------------
    # bandwidth is khz, we convert it to hz via multiplying it by 1000
    # N0 is dBm/hz, we convert it to mW/hz
    # n0 is mW
    n0 = (10 ** (N0 / 10)) * bandwidth * 1000

    # Calculate SNIR
    SNIRs = transmit_powers / (n0)

    # Calculate data rate
    data_rate = bandwidth * np.log2(1 + SNIRs)  # kb/s
    # data_rates *= Environment.bit_per_hertz  # kbps
    # data_rates *= Environment.virtual_clock.time_unit  # kb/ms
    return data_rate[0]


def calculate_complex_channel_gain(sender, receiver, complex_fading_matrix):
    beta_0 = -30  # -30db
    d_0 = 1  # 1m
    alpha = 3
    rayleigh_faded_channel_matrix = complex_fading_matrix

    transmitter_receiver_distance = math.sqrt(np.power(receiver[0] - sender[0], 2)
                                               + np.power(receiver[1] - sender[1], 2))

    # transmit power in db
    clear_transmit_power = beta_0 - 10 * alpha * np.log10(transmitter_receiver_distance / d_0)
    # convert to watt
    clear_transmit_power = np.sqrt(10 ** (clear_transmit_power / 10))
    # applying rayleigh fading
    rayleigh_faded_channel_matrix *= clear_transmit_power

    return rayleigh_faded_channel_matrix