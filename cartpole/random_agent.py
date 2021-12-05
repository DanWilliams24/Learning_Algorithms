import gym
#random agent

# get the sample environment from AI gym library
env = gym.make("CartPole-v1")
# run 20 episodes (trials)
for i_episode in range(20):
    observation = env.reset()  # what is recieved from the enviornment
    for t in range(100):
        # renders view to programmer. Not required to run, just a GUI
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print("This is the observation: {}".format(observation))
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print(reward)
            print(observation)
            print(info)
            break
env.close()

