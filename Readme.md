Code for implementing a type of Policy gradients algorithm Reinforce to solve the CartPole-v1 environment

train_policy_gradient.py - Contains code for learning the optimal policy using policy gradient REINFORCE algorithm. 
The Policy is parametrized by a 2 layer (fully-connected layers with 128 neurons each) neural network defined in utils.py
Learned model is stored as policy_model.pt and can be tested from a function inside the tester code

You can see the training curves:
![picture alt](plot1.png)

utils.py - contains code for defining the policy model
Also contains code for testing the model policy_model.pt learned by policy gradient

To see how good the policy_model, just run test_model.py

![picture alt](test.gif)
