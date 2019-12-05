# Online-learning-framework

# What Online-learning-framework?
Online-learning-framework is a framework in C++ that addresses the iot survival problem with the implementation of several RL algorithms in the same framework, for fair comparison. (see systemmodel.png and iot-survival repository).

# Machine Learning for IoT device survival
Several RL algorithms are coded in C++ in this framework:
- Q-learning - an off policy RL method that explores the action value function
- Double Q-learning - a less biased version of Q-learning that requires more memory and tends to learn faster
- SARSA - an on-policy learning method
- Expected SARSA - an off-policy variant of SARSA with an update rule that accounts for the expected value of the action a_{n+1}
- n step SARSA - an on-policy learning method that makes updates to Q-tables every n time slots
