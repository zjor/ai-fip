# Double inverted pendulum

The double pendulum is a classic example in classical mechanics and consists of two pendulums attached end-to-end. To derive the Euler-Lagrange equations for a double pendulum, we follow these steps:

### Step 1: Define the system

Consider a double pendulum with two masses \( m_1 \) and \( m_2 \), and two lengths \( l_1 \) and \( l_2 \). Let \(\theta_1\) be the angle of the first pendulum with the vertical, and \(\theta_2\) be the angle of the second pendulum with the vertical.

### Step 2: Kinetic and Potential Energies

First, we need to express the kinetic and potential energies of the system.

**Kinetic Energy \( T \):**

The positions of the masses $m_1$ and $m_2$ are given by:
$x_1 = l_1 \sin \theta_1$
$y_1 = -l_1 \cos \theta_1$

$x_2 = l_1 \sin \theta_1 + l_2 \sin \theta_2$
$y_2 = -l_1 \cos \theta_1 - l_2 \cos \theta_2$

The velocities of the masses are obtained by differentiating these positions with respect to time:
\[ \dot{x}_1 = l_1 \cos \theta_1 \dot{\theta}_1 \]
\[ \dot{y}_1 = l_1 \sin \theta_1 \dot{\theta}_1 \]

\[ \dot{x}_2 = l_1 \cos \theta_1 \dot{\theta}_1 + l_2 \cos \theta_2 \dot{\theta}_2 \]
\[ \dot{y}_2 = l_1 \sin \theta_1 \dot{\theta}_1 + l_2 \sin \theta_2 \dot{\theta}_2 \]

The kinetic energy \( T \) is:
$T = \frac{1}{2} m_1 (\dot{x}_1^2 + \dot{y}_1^2) + \frac{1}{2} m_2 (\dot{x}_2^2 + \dot{y}_2^2)$

Substitute the velocities:
$T = \frac{1}{2} m_1 (l_1^2 \dot{\theta}_1^2) + \frac{1}{2} m_2 \left[ (l_1 \cos \theta_1 \dot{\theta}_1 + l_2 \cos \theta_2 \dot{\theta}_2)^2 + (l_1 \sin \theta_1 \dot{\theta}_1 + l_2 \sin \theta_2 \dot{\theta}_2)^2 \right]$

**Potential Energy \( V \):**

The potential energy \( V \) is due to the height of the masses:
\[ V = m_1 g y_1 + m_2 g y_2 \]
\[ V = -m_1 g l_1 \cos \theta_1 - m_2 g (l_1 \cos \theta_1 + l_2 \cos \theta_2) \]

### Step 3: Lagrangian

The Lagrangian \( L \) is:
\[ L = T - V \]

### Step 4: Euler-Lagrange Equations

The Euler-Lagrange equations are:
\[ \frac{d}{dt} \left( \frac{\partial L}{\partial \dot{\theta}_i} \right) - \frac{\partial L}{\partial \theta_i} = 0 \]

We need to compute these for \(\theta_1\) and \(\theta_2\).

**For \(\theta_1\):**

\[ \frac{d}{dt} \left( \frac{\partial L}{\partial \dot{\theta}_1} \right) - \frac{\partial L}{\partial \theta_1} = 0 \]

**For \(\theta_2\):**

\[ \frac{d}{dt} \left( \frac{\partial L}{\partial \dot{\theta}_2} \right) - \frac{\partial L}{\partial \theta_2} = 0 \]

### Detailed Calculations

Performing these detailed calculations by hand can be quite lengthy, but here is the result:

1. The Euler-Lagrange equation for \(\theta_1\) is:
$(m_1 + m_2) l_1 \ddot{\theta}_1 + m_2 l_2 \ddot{\theta}_2 \cos(\theta_1 - \theta_2) + m_2 l_2 \dot{\theta}_2^2 \sin(\theta_1 - \theta_2) + (m_1 + m_2) g \sin \theta_1 = 0$

2. The Euler-Lagrange equation for \(\theta_2\) is:
$l_2 \ddot{\theta}_2 + l_1 \ddot{\theta}_1 \cos(\theta_1 - \theta_2) - l_1 \dot{\theta}_1^2 \sin(\theta_1 - \theta_2) + g \sin \theta_2 = 0$

These coupled differential equations describe the motion of a double pendulum. Solving them analytically is very difficult, and numerical methods are often used to simulate the behavior of a double pendulum.
