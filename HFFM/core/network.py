import math
import numpy as np
import matplotlib.pyplot as plt

SEED = 49
global_random_gn=np.random.default_rng(SEED)

def norm2_power2(rayleigh_faded_channel_matrix):
    return rayleigh_faded_channel_matrix[:, 0] ** 2 + rayleigh_faded_channel_matrix[:, 1] ** 2

def calculate_transmission_rate(sender, receiver, transmit_power, bandwidth = 1000):     #bandwidth is Khz
    N0 = -174  # -174dBm/hz

    # Calculate transmitter to receiver channel gain------------
    channels = calculate_complex_channel_gain(sender, receiver, global_random_gn.standard_normal((1, 2)))
    transmit_powers = norm2_power2(channels) * transmit_power

    n0 = (10 ** (N0 / 10)) * bandwidth

    # Calculate SNIR
    SNIRs = transmit_powers / (n0)

    # Calculate data rate
    data_rate = bandwidth * np.log2(1 + SNIRs)  # kb/s
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