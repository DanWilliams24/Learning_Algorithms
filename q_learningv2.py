import numpy as np
import math
from random import random
import gym
import matplotlib.pyplot as plt
from RL_Tester import Agent_Tester
'''Homework: (Frozen Lake)
- have your grid marking different states
- start from state s
- want to reach g without falling into a hole (game over)
- if you reach the goal -> reward 1'''



class ReinforcementLearner:
    def __init__(self, num_episodes=10000, max_steps_per_episode=100, exploration_decay_rate=0.001, data_collection_interval=100, output_allowed=True):
        self.env = gym.make("FrozenLake-v0")
        state_space_size = self.env.observation_space.n
        action_space_size = self.env.action_space.n
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.current_state = 0
        self.states = self.env.observation_space
        self.actions = self.env.action_space
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.output_allowed = output_allowed
        self.learning_rate = 0.1
        self.discount_rate = 0.99999999999999
        self.exploration_rate = 1.0
        self.smart_exploration_k = 1
        self.max_exploration_rate = 1
        self.min_exploration_rate = 0.001
        self.exploration_decay_rate = exploration_decay_rate
        self.boost = False
        # these are for stats
        self.data_collection_interval = data_collection_interval  # the frequency at which we evaluate success rate
        self.truth_count = 0  # the number of successful episodes over an interval
        self.truth_rates = []  # a list of % successful episodes per interval over course of learning
        self.steps_per_episodes = []  # list of the number of actual steps taken each episode
        self.exploration_values = []


        print("{} by {} q-table defined with states {} and actions {}.".format(state_space_size, action_space_size,
                                                                               self.states, self.actions))
    def displayQTable(self):
        print("Q-TABLE ============================== Q-TABLE")
        for i in range(len(self.q_table)):
            for j in range(len(self.q_table[i])):
                print(self.q_table[i][j], end=" ")
            if (i+1) % 4 == 0:
                print("\n")
            else:
                print("  |  ", end="")
        print()

    def learn(self):
        # for each episode
        for current_episode in range(self.num_episodes):
            # start the episode
            self.begin_episode()
            # the more we explore, the more we decay the exploration rate, making it less likely
            # lets make this decay a bit smarter. After we pass a 25% exploration rate
            # lets only decay when we win the round
            # if self.exploration_rate > 0.25:
                #self.decay_exploration_rate()


            # lets make this decay deterministic via mathematical function. Here we use the Sigmoid function
            # where the input is the current episode number. We use this episode number to get the
            # related exploration rate for that value. This will decrease the rate smoothly during training,
            # hopefully allowing the agent to learn more through exploration while also not spending too much time
            # in exploration.

            if self.exploration_rate > self.min_exploration_rate and not self.boost:
                self.sigmoid_decay_exploration(current_episode)
                #print(self.exploration_rate)
            elif self.exploration_rate <= self.min_exploration_rate:
                self.boost = True
                self.exploration_rate = 1

            if self.boost:
                self.sigmoid_decay_exploration2(current_episode)
                print(self.exploration_rate)

            #self.exploration_fluctuator_method(current_episode)

            #self.exploration_sigmoid_flux_method(current_episode)

            # initially this was inside the episode loop, but it seems that it makes more sense to decrease
            # the exploration rate over episodes rather than individual steps. This prevents the
            # exploration rate from dropping off rather quickly, preventing the agent from learning enough
            # about its env

            # collect accuracy data for later stats
            if current_episode % self.data_collection_interval == 0:
                self.truth_rates.append(self.truth_count/self.data_collection_interval)
                self.truth_count = 0
                self.exploration_values.append(self.exploration_rate)

    def update_policy(self, action, new_state, reward):
        discount = self.discount_rate
        learning = self.learning_rate
        current_q = self.q_table[self.current_state][action]
        new_state_q_values = self.q_table[new_state]
        # old policy
        #self.q_table[self.current_state][action] = current_q + learning * (reward +
                                                                                 #discount * max(new_state_q_values) -
                                                                                 #self.q_table[self.current_state][action])
        self.q_table[self.current_state][action] = current_q + learning*(reward + discount*max(new_state_q_values)-self.q_table[self.current_state][action])
        self.output("updated policy for [{}][{}] to value {}".format(self.current_state, action,
                                                               self.q_table[self.current_state][action]))
        self.current_state = new_state

    def sigmoid(self, x, n=0, k=1):
        # Numerically-stable sigmoid function.
        if x >= 0:
            z = math.exp(-k * (x - (n / 2)))
            return 1 / (1 + z)
        else:
            z = math.exp(k * (x - (n / 2)))
            return z / (1 + z)


    def sigmoid_decay_exploration(self, episode):
        self.exploration_rate = self.sigmoid(episode, self.num_episodes, -self.exploration_decay_rate)

    def sigmoid_decay_exploration2(self, episode):
        self.exploration_rate = self.sigmoid(episode, self.num_episodes+episode, -self.exploration_decay_rate)

    def decay_cos(self,x):
        self.exploration_rate = abs(math.cos(x+(self.num_episodes/4)))


    def decay_exploration_rate(self):
        current_rate = self.exploration_rate
        min_rate = self.min_exploration_rate
        decay_rate = self.exploration_decay_rate
        if current_rate > min_rate:
            self.exploration_rate -= decay_rate


    def begin_episode(self):
        # for each step in an episode
        self.current_state = 0
        self.env.reset()
        for current_step in range(self.max_steps_per_episode):
            self.output(action="render")
            e = self.exploration_rate
            done = False
            if random() < e:
                # explore the env (take random action)
                random_action = self.env.action_space.sample()

                new_state, reward, done, _ = self.env.step(random_action)

                # update q-table with findings
                self.update_policy(random_action, new_state, reward)


            else:
                # exploit the env (take directed action based on q-table)
                target_action = np.argmax(self.q_table[self.current_state])
                new_state, reward, done, _ = self.env.step(target_action)


                self.current_state = new_state

                # self.update_policy(target_action, new_state, reward)

                # Should the q-table be updated when we choose to exploit the env?
                # My logic would dictate that exploiting the env means to use the agents prior knowledge to
                # take actions, which means that there is no reason for the agent to learn in this step.

            # if action results in episode over state, end for loop
            if done:

                # collect stats for later
                self.truth_count += reward
                # store number of time-steps needed (for episodes won)
                if reward == 1:
                    self.steps_per_episodes.append(current_step+1)

                self.output("Episode Won: {}".format(reward == 1))
                self.output(current_step)

                break

        return

    '''
    Export the q-table model as a csv file
    '''
    def toFile(self, accuracy):
        filename = "frozenlake_agent_model.csv"
        flattened_table = self.q_table.flatten()
        flattened_table = np.insert(flattened_table, 0, accuracy)
        flattened_table.tofile(filename, sep=',')
        return

    @staticmethod
    def analyze_linear_regression(axis, x, y, slope_multiplier=1):
        m, b = np.polyfit(x, y, 1)
        axis.plot(x, [i * m + b for i in x])
        mid_x = x[int(len(x) / 2)]
        _, x_max = axis.get_xlim()
        _, y_max = axis.get_ylim()
        rate = "({})"
        if slope_multiplier == 100:
            rate = rate.format("%")
        else:
            rate = ""
        axis.annotate('Slope {}: {}'.format(rate, round(m*slope_multiplier, 6)), xy=(mid_x, mid_x * m + b),
                        xytext=(x_max / (x_max/25), (y_max - y_max / 4)),
                        arrowprops=dict(facecolor='black', shrink=0.0005, width=0.1, headwidth=1.2))


    def output(self, text="", action=""):
        if self.output_allowed:
            if action == "render":
                self.env.render()
            else:
                print(text)



    def display_stats(self):
        fig, axs = plt.subplots(2)

        # plot the truth rate (tr)
        axs[0].title.set_text('Truth Rate During Training')
        base_x_axis_tr = list(range(len(self.truth_rates)))
        x_tr = [i * self.data_collection_interval for i in base_x_axis_tr]
        y_tr = self.truth_rates
        axs[0].plot(x_tr, y_tr)
        axs[0].set(xlabel='# of Episodes', ylabel='Success Rate')


        x_e = list(range(len(self.exploration_values)))
        y_e = self.exploration_values
        axs[0].plot(x_tr, y_e)


        # plot the steps per successful episodes (spe)
        axs[1].title.set_text('Steps During Successful Episodes')
        x_spe = list(range(len(self.steps_per_episodes)))
        y_spe = self.steps_per_episodes
        axs[1].plot(x_spe, y_spe)
        axs[1].set(xlabel='# of Successful Episodes', ylabel='Steps During Episode')

        # plot linear regressions for both subplots
        ReinforcementLearner.analyze_linear_regression(axs[0], x_tr, y_tr, slope_multiplier=100)
        ReinforcementLearner.analyze_linear_regression(axs[1], x_spe, y_spe)

        fig.suptitle('Stats from Training')
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    learner = ReinforcementLearner(num_episodes=20000, max_steps_per_episode=100, exploration_decay_rate=0.002, output_allowed=False)
    learner.displayQTable()
    learner.learn()
    print(learner.truth_rates)

    see_stats = input("See Model Performance Stats? [y/n] ")
    percentage_correct = 0
    if see_stats == "y" or see_stats == "yes":
        agent = Agent_Tester(learner.q_table, 10000, output_allowed=False)
        current_model_results = agent()
        print("The model has an accuracy of {} across {} episodes".format(current_model_results, agent.num_episodes))

        best_model_results = []
        try:
            best_model_results = np.fromfile("frozenlake_agent_model.csv", sep=",")[0]
            if best_model_results > current_model_results:
                print(" This new model is {}% slower than what is currently saved. ".format(round(best_model_results - current_model_results, 6) * 100))
            elif current_model_results > best_model_results:
                print(" This new model is {}% faster than what is currently saved. ".format(round(
                    current_model_results - best_model_results, 6) * 100))
            else:
                print(" These models are equally accurate based on saved readings.")
        except FileNotFoundError:
            print("There is no saved model on file. ")

        learner.display_stats()
    output = input("Save model to file? [y/n] ")
    if output == "y" or output == "yes":
        learner.toFile(current_model_results)















'''

env = gym.make("FrozenLake-v0")

    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    print(env.action_space.high)

    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n

    q_table = np.zeros((state_space_size, action_space_size))

    # run 20 episodes (trials)
    for i_episode in range(20):
        observation = env.reset()  # what is recieved from the enviornment
        for t in range(100):
            # renders view to programmer. Not required to run, just a GUI
            env.render()

            #choose


            action = env.action_space.sample()




            observation, reward, done, info = env.step(action)
            print("This is the reward: {}".format(reward))
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                print(reward)
                print(observation)
                print(info)
                break
if __name__ == '__main__':
    # run 20 episodes (trials)
    print(q_table)
    cont = input()
    for i_episode in range(mum_episodes):
        observation = env.reset()  # what is recieved from the enviornment
        for t in range(max_steps_per_episode):
            # renders view to programmer. Not required to run, just a GUI
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print("This is the reward: {}".format(reward))
            print(reward)
            print(observation)
            print(info)


            if done:
                print("Episode finished after {} timesteps".format(t+1))
                print(reward)
                print(observation)
                print(info)

                break
    env.close()

'''