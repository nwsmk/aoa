import random
import json
import numpy as np
from paho.mqtt import client as mqtt_client
from numpy import linalg as la
from matplotlib import pyplot as plt
from matplotlib import cm
from numpy import unravel_index
from findpeaks import findpeaks
from scipy.optimize import curve_fit
from pyargus.directionEstimation import *


broker = 'localhost'
port = 1883
topic = "silabs/aoa/iq_report/ble-pd-540F572DEE1D/ble-pd-842E1431BB77"
client_id = f'python-mqtt-{random.randint(0, 100)}'

t_ref_interval = 1e-6
t_sw_interval = 2e-6

# constants
c = 3.0e8
f = 2.4e9

# antennas
Nx = 4
Ny = 4
rx = 1.0e-3 * np.array([57.7, 21.7, -21.7, -57.7])
ry = 1.0e-3 * np.array([57.7, 21.7, -21.7, -57.7])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Azimuth (degrees)')
ax.set_ylabel('Elevation (degrees)')
ax.set_zlabel('MUSIC spectrum (dB)')

num_ref = 8
M = 1
N = 16

def phase_objective(t, fdev, b):
    return 2 * np.pi * (fdev) * t + b


def cal_phase(i_sample, q_sample):
    sample_len = len(i_sample)
    phase = []
    for i in range(sample_len):
        i_ = i_sample[i]
        q_ = q_sample[i]

        if (i_ > 0) and (q_ >= 0):
            phase.append(np.arctan(q_/i_))
        elif (i_ < 0):
            phase.append(np.arctan(q_/i_) + np.pi)
        elif (i_ > 0) and (q_ < 0):
            phase.append(np.arctan(q_/i_) + (2 * np.pi))
        elif (i_ == 0) and (q_ > 0):
            phase.append(np.pi / 2)
        elif (i_ == 0) and (q_ < 0):
            phase.append(np.pi * 3 / 2)



# calculate phase difference
def cal_phase_diff(i_samples, q_samples):
    phases = np.unwrap(np.arctan2(-1*q_samples, i_samples))
    phases_diff = np.diff(phases)
    phases_diff = np.mean(phases_diff)
    return phases_diff


def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        raw_payload = json.loads(msg.payload.decode())
        raw_sample = np.array(raw_payload['samples'])
        raw_sample_len = len(raw_sample)
        raw_i_idx = np.arange(0, raw_sample_len, 2)
        raw_q_idx = np.arange(1, raw_sample_len, 2)
        raw_i_sample = raw_sample[raw_i_idx]
        raw_q_sample = raw_sample[raw_q_idx]

        phase_all = np.unwrap(np.arctan2(-1 * raw_q_sample, raw_i_sample))
        # plt.plot(range(len(phase_all)), phase_all, marker='x')
        # plt.show()

        ref_i_sample = raw_i_sample[0:num_ref]
        ref_q_sample = raw_q_sample[0:num_ref]
        ref_phase = phase_all[0:num_ref]
        ref_phase_diff = np.mean(np.diff(ref_phase)) * 2

        ref_t = np.arange(num_ref) * 1e-6
        popt, _ = curve_fit(phase_objective, ref_t, ref_phase)
        f_dev, beta = popt
        # print('f_dev = %.5f, beta = %.5f \n' % (f_dev, beta))

        sw_i_sample = raw_i_sample[num_ref:]
        sw_q_sample = raw_q_sample[num_ref:]
        sw_amplitude = np.sqrt(np.square(sw_i_sample) + np.square(sw_q_sample))
        sw_phase = phase_all[num_ref:]

        sw_sample_len = len(sw_phase)

        sw_cal_phase = []
        for i in range(sw_sample_len):
            cal_phase = sw_phase[i] - (i * ref_phase_diff)
            sw_cal_phase.append(cal_phase)
        sw_cal_phase_arr = np.hstack(sw_cal_phase)

        num_cycle = np.floor_divide(sw_sample_len, N)
        sw_amp = []
        sw_pha = []
        for i in range(num_cycle):
            sw_amp.append(sw_amplitude[i * N : (i + 1) * N])
            sw_pha.append(sw_cal_phase_arr[i * N : (i + 1) * N])
        sw_amp = np.vstack(sw_amp)
        sw_pha = np.vstack(sw_pha)

        X = np.multiply(sw_amp, np.exp(1j * sw_pha))
        X = np.matrix(np.transpose(X))

        Rxx = np.matmul(X, X.H) / num_cycle
        w, v = la.eig(Rxx)
        d_id = np.argsort(w)
        E = v[0:, d_id]
        En = E[0:, 0:-M]

        # antenna steering vector
        el_search = np.deg2rad(np.arange(0, 90, 5))
        az_search = np.deg2rad(np.arange(-180, 180, 5))
        Z = np.zeros((len(el_search), len(az_search)))
        for p in range(len(el_search)):
            el = el_search[p]
            for q in range(len(az_search)):
                az = az_search[q]
                A = []
                for m in range(Nx):
                    for k in range(Ny):
                        a = np.exp(1j * 2 * np.pi * f / c * (
                                    rx[m] * np.sin(el) * np.cos(az) + ry[k] * np.sin(el) * np.sin(az)))
                        A.append(a)
                A_arr = np.hstack(A)
                A_mat = np.matrix(A_arr)
                A_mat = A_mat.T

                m1 = np.matmul(A_mat.H, En)
                m2 = np.matmul(En.H, A_mat)
                m12 = np.matmul(m1, m2)
                Z[p, q] = 1 / m12[0,0]

        Z_array = np.matrix(Z)
        Z_array = np.multiply(Z_array, np.conjugate(Z_array))

        # Make data.
        Xax = el_search
        Yax = az_search
        Xax, Yax = np.meshgrid(Xax, Yax)
        Z = 10 * np.log10(Z_array / num_cycle)
        Z = np.transpose(Z)

        # result = unravel_index(Z.argmax(), Z.shape)
        # print("Elevation = %.2f, Azimuth = %.2f \n" % (np.rad2deg(el_search[result[0]]), np.rad2deg(az_search[result[1]])))
        # print(f"azimuth = {np.rad2deg(az_search[result[0]])}, elevation = {np.rad2deg(el_search[result[1]])}")

        # Plot the surface.
        ax.plot_surface(np.rad2deg(Xax), np.rad2deg(Yax), Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
        ax.set_xlabel('elevation')
        ax.set_ylabel('azimuth')
        plt.pause(0.005)
        ax.cla()

    client.subscribe(topic)
    client.on_message = on_message


def run():

    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()


if __name__ == '__main__':
    run()