import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, sin

from app.ode import solve, integrate_rk4

g = 9.81
m1 = 0.1  # kg, mass of the rod
m2 = 0.5  # kg, mass of the wheel
L = 0.22  # m, length of the rod
r = 0.75  # m, radius of the wheel
Ib = m1 * L ** 2 / 3  # kg m^2, inertia of the rod about its end
J = m2 * r ** 2  # kg m^2, wheel inertia about its spin axis


def get_control_lqr(state, t, dt):
    K = np.array([150, 50, -10, -20])

    return - K @ np.array(state)


theta_i = 0.0
U_max = 0.55


def get_control_pid(state, t, dt):
    Kp = 150.0
    Kd = 50.0
    Ki = 12.0

    K_phi = 10.0
    K_phi_dot = 20.0

    theta, theta_dot, phi, phi_dot = state

    global theta_i
    theta_i += theta * dt

    u_theta = Kp * theta + Kd * theta_dot + Ki * theta_i
    u_wheel = K_phi * phi + K_phi_dot * phi_dot

    u = -u_theta + u_wheel
    if u > U_max:
        return U_max
    elif u < -U_max:
        return -U_max
    else:
        return u


def get_control_nn(state, t, dt):
    # TODO: forward pass from exported ONNX model
    ...


def derivate(state, step, t, dt):
    theta, theta_dot, phi, phi_dot = state
    u = get_control_pid(state, t, dt)

    theta_ddot = (u - (m1 / 2 + m2) * L * g * sin(-theta)) / (Ib + m2 * L ** 2 + J)
    phi_ddot = u / J
    return [theta_dot, theta_ddot, phi_dot, phi_ddot]


theta_0 = pi / 12
theta_dot_0 = 0.0
phi_0 = 0.0
phi_dot_0 = 0.0

init_state = [theta_0, theta_dot_0, phi_0, phi_dot_0]
T_max = 20.0  # seconds
dt = 0.01


def main():
    times = np.arange(0.0, T_max, dt)
    states = solve(init_state, times[:-1], integrate_rk4, derivate)

    theta = states[:, 0]
    theta_dot = states[:, 1]
    phi_dot = states[:, 2]

    plt.figure()

    plt.plot(times, theta, label=r"$\theta$ [deg]")
    plt.plot(times, theta_dot, label=r"$\dot{\theta}$ [deg/s]")
    plt.plot(times, phi_dot, label=r"$\dot{\phi}$ [deg/s]")

    plt.xlabel("time [s]")
    plt.ylabel("angle / rate")
    plt.title("FIP — θ, θ̇, φ̇")

    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
