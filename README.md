# Reinforcement Learning framework for security optimisation

# What is RL framework for security optimisation?
This framework coded in C++ addresses an IoT device survival problem. An IoT device is underfed by an Energy Harvester and is obliged to make security decisions to stay alive and still try to maximise packet security (see systemmodel.png).

# Machine Learning for IoT device survival
The RL algorithms coded in C++ in this framework are:
- Double Q-learning - a less biased version of Q-learning that requires more memory due to the use of two Q-tables instead of one but that tends to learn faster
- Expected SARSA - an off-policy algorithm, variant of SARSA that has an update rule that accounts for the expected value of the action a_{n+1}, i.e., considers both the probability of selecting an action and the action value of that same action.

# How to play with the framework
The framework has all the code bulked in one .c file. It has several parameters related to the system model that can be tuned. Double Q-learning or Expected SARSA can be chosen as the learning algorithm. Learning time can be tuned and the framework outputs files where the learned policies, the current Q-tables and the number of visits to each state-action pair are displayed for convergence control. All policies learned can be simulated and consequently, any relevant data to the problem can be retrieved.

