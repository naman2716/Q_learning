import gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make("MountainCar-v0")
##print(env.action_space.n)
env.reset()


LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 4000
SHOW_EVERY = 1000
STATS_EVERY = 100

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

DISCRET_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRET_OS_SIZE

#print(discrete_os_win_size)


epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_dacay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


q_table = np.random.uniform(low = -2 , high = 0 , size = (DISCRET_OS_SIZE + [env.action_space.n]))


def get_discrete_states(state):
	discrete_state = (state - env.observation_space.low)/discrete_os_win_size
	return tuple(discrete_state.astype(np.int))



for episode in range(EPISODES):
	episode_reward = 0
	discrete_state = get_discrete_states(env.reset())
	done = False

	if episode % SHOW_EVERY == 0:
		render = True
		print(episode)

	else:
		render = False


	while not done:

		if np.random.random() > epsilon:
			action = np.argmax(q_table[discrete_state])

		else:
			action = np.random.randint(0 , env.action_space.n)


		action = np.argmax(q_table[discrete_state]) #always go rigth
		new_state , reward , done , _ = env.step(action)
		env.step(action)
		episode_reward += reward

		new_discrete_state = get_discrete_states(new_state)

		if episode % SHOW_EVERY == 0:
			env.render()

		print(reward , new_state)
		env.render()

		#new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward = DISCOUNT * max_future_q)

		if not done:
			max_future_q = np.max(q_table[new_discrete_state])

			current_q = q_table[discrete_state + (action,)]  # q value for the action we have taken

			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

			q_table[discrete_state + (action , )] = new_q

		elif new_state[0] >= env.goal_position:
			q_table[discrete_state + (action , )] = 0

		discrete_state = new_discrete_state

	if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
		epsilon -= epsilon_dacay_value


	ep_rewards.append(episode_reward)
	if not episode % STATS_EVERY:
		average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
		aggr_ep_rewards['ep'].append(episode)
		aggr_ep_rewards['avg'].append(average_reward)
		aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
		aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
		print('Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')

	if episode % 10 == 0:
		np.save("qtables/{episode}-qtable.npy", q_table)


plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.show()

env.close()





