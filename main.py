# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import gym


# Press the green button in the gutter to run the script.
# get the sample environment from AI gym library
env = gym.make("FrozenLake-v0")
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

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
