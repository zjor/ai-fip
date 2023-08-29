# AI-controlled Flywheel Inverted Pendulum

The goal is to build a flywheel inverted pendulum (FIP) model stabilized by a neural network trained by deep reinforcement learning method. Literally, the network will not know anything about the physics of the phenomena and about it's own physical "body", it will use a method of trial and error in order to figure out how to get to upright position.


## Milestones

- [ ] Get enough understanding of a physics model
- [ ] Simulate free-fall system with spontaneous rotation of a wheel
- [ ] Implement a balancing algorithm in a simulation
- [ ] Implement a swing-up algorithm in a simulation
- [ ] Build a physical model with stepper motor for rotating a wheel
- [ ] Try other controller to stabilize the system (TBD)
- [ ] Learn DRL enough to apply to the system
- [ ] Pre-train a NN in a simulation and use it inside a real device
- [ ] Allow NN to tune itself in a real device


## References

- [DESIGN AND CONTROL OF A FLYWHEEL INVERTED PENDULUM SYSTEM'16](design_and_control_of_a_flywheel_inverted_pendulum_system.pdf)
- [Design of a Flywheel-Controlled Inverted Pendulum'19](design_of_a_flywheel-controlled_inverted_pendulum_2019.pdf)
- [Flywheel Inverted Pendulum Design for Evaluation of Swing-Up Energy-Based Strategies'20](flywheel_inverted_pendulum_design_for_evaluation_of_swing-up_energy-based_strategies_2020.pdf)
- [Global Stabilization of a Reaction Wheel Pendulum'20](global_stabilization_of_a_reaction_wheel_pendulum_2020.pdf)
- [Inertia Wheel Inverted Pendulum'19](inertia_wheel_inverted_pendulum_2019.pdf)
- [Nonlinear control of the Reaction Wheel Pendulum'01](nonlinear_control_of_the_reaction_wheel_pendulum_2001.pdf)
- [Robust Control of the FIP System Considering Parameter Uncertainty'21](robust_control_of_the_fip_system_considering_parameter_uncertainty_2021.pdf)
- [Two-Axis Reaction Wheel Inverted Pendulum'17](two-axis_reaction_wheel_inverted_pendulum_2017.pdf)