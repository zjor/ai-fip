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

- [Flywheel Inverted Pendulum Design for Evaluation of Swing-Up Energy-Based Strategies](articles/flywheel-inverted-pendulum.pdf)