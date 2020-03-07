import gym
import utils
import numpy as np

def test_model_pg():
	env = gym.make('CartPole-v1')
	env = gym.wrappers.Monitor(env, directory='recording', force=True)

	tester = utils.Tester()  # Construct Testing Object

	env.reset()

	for i_episode in range(5):
	    state = env.reset()
	    for t in range(2000):
	        env.render()
	        # print(state)
	        action = tester.policy_gradient_test(state)
	        state, reward, done, info = env.step(action)
	        if done:
	            print("Episode finished after {} timesteps".format(t+1))
	            break
	
	env.stats_recorder.save_complete()
	env.stats_recorder.done = True            
	env.close()

def main():
	test_model_pg()

if __name__ == '__main__':
	main()