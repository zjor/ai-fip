# Motor torque estimation

## Parameters of the physical model

### Wheel

| Name          | Units          | Value      | Description                                                                               |
|---------------|----------------|------------|-------------------------------------------------------------------------------------------|
| m_load        | g              | 2.9        | Mass of a bolt and a nut used to increase moment of inertia of the wheel                  |
| m_wheel_total | g              | 110        | Mass of the wheel with bolts and nuts (loads)                                             |
| m_wheel_naked | g              | 86.8       | Mass of the wheel without loads                                                           |
| R             | mm             | 112        | Radius of the wheel                                                                       |
| J_wheel_naked | $kg\cdot{m^2}$ | 0.00054    | Let wheel is a ring of the mass $m$, $r, \frac{1}{3}r$ => $J = \frac{1}{2}m(r_1^2+r_2^2)$ |
| r_load        | mm             | 105        | Distance between the load and the center of the wheel                                     |
| J_loads       | $kg\cdot{m^2}$ | 0.00026    | Moment of inertia of the loads around the center of the wheel                             |
| J_wheel       | $kg\cdot{m^2}$ | **0.0008** | Total moment of inertia of the wheel                                                      |

### Pendulum

| Name       | Units          | Value      | Description                                                   |
|------------|----------------|------------|---------------------------------------------------------------|
| m_motor    | g              | 292        | Mass of a stepper motor plus mounting bolts                   |
| T_motor    | $H\cdot{m}$    | 0.42       | Maximum torque of the stepper motor, model: 17HS4401          |
| m_rod      | g              | 167        | Mass of the pendulum rod with the circuit board               |
| L          | mm             | 225        | Length of the rod                                             |
| J_rod      | $kg\cdot{m^2}$ | 0.0028     | Moment of inertia of the rod around the pivot                 |
| J_motor    | $kg\cdot{m^2}$ | 0.0149     | Moment of inertia of the motor around the pivot of the rod    |
| J_pendulum | $kg\cdot{m^2}$ | **0.0176** | Total moment of inertia of the pendulum (~x22 of the wheel!!) |

## Torque estimation

$$
M = mg\cdot{sin({\theta})}\cdot{l_c}, l_c \approx l
$$
Momentum needed to hold a pendulum at the angle $\theta = \pi/6$ is $\approx0.5 H\cdot{m}$


