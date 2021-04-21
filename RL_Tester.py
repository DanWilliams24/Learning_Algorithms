import numpy as np
import csv
import pandas as pd
import time
import os
import gym

class Agent_Tester:
    '''
    We have learned quite a bit about the q-learning agent from this excercise. I enumerate through these points below.
    1. Based on the reward system, the Agent will maximize its long term rewards such that
     if it is more rewarding not to win, it will not. I found that when I gave the agent
     a 0.5 reward for staying alive (I thought this would cause it to do a better job at avoiding holes),
     I instead found that the agent learned instead to never reach the goal position, as by staying alive for
     the entire duration of the episode, it could maximize its reward versus getting one slightly larger reward of 1 once.
    2. The exploration rate must be such that it does not spend too much time exploring,
     while also not spending too little time either. Over-exploring results in aggregately random results across episodes.
     Under-exploring leads to the agent being unable to gain any new knowledge about its environment.
    3.
    '''


    def __init__(self, q_table="",  num_episodes=10, delay=1, output_allowed=True):
        self.q_table = []

        if type(q_table) == str:
            if q_table == "":
                raise ValueError("A valid filename is needed to run Agent Testing")
            df = np.fromfile(q_table, sep=",")[1:].reshape((16, 4))
            self.q_table = df.tolist()
        elif type(q_table) == np.ndarray:
            if len(q_table) == 0:
                raise ValueError("A populated Q-table is needed to run Agent Testing")
            self.q_table = q_table
        else:
            raise ValueError("A valid filename or a populated Q-table is needed to run Agent Testing")

        self.env = gym.make("FrozenLake-v0")
        self.current_state = 0
        self.num_episodes = num_episodes
        self.delay = delay
        self.output_allowed = output_allowed
        if not self.output_allowed:
            self.delay = 0
        else:
            self.delay = delay


    def __call__(self):
        self.output(action="clear")
        truth_count = 0
        # run 20 episodes (trials)
        for i_episode in range(self.num_episodes):
            self.print_episode_title(i_episode+1)
            observation = self.env.reset()
            self.current_state = observation
            for t in range(100):
                # renders view to programmer. Not required to run, just a GUI
                self.output(action="clear")
                self.output(action="render")
                self.output("t = {}".format(t))
                # exploit the env (take directed action based on q-table)
                target_action = np.argmax(self.q_table[self.current_state])
                new_state, reward, done, _ = self.env.step(target_action)

                self.current_state = new_state
                time.sleep(0.5*self.delay)

                if done:
                    self.output(action="clear")
                    self.output("Episode finished after {} timesteps".format(t + 1))
                    self.output("Episode Won? {}".format(reward == 1))
                    if reward == 1:
                        truth_count += 1
                    time.sleep(2.5*self.delay)
                    self.output(action="clear")
                    break
        return truth_count/self.num_episodes

    def output(self, text="", action=""):
        if self.output_allowed:
            if action == "clear":
                self.clear()
            if action == "render":
                self.env.render()
            else:
                print(text)


    def print_episode_title(self, number):
        self.output("============")
        self.output("Episode {}".format(number))
        self.output("============")
        time.sleep(2.5*self.delay)
        self.output(action="clear")

    @staticmethod
    def clear():
        # for windows
        if os.name == 'nt':
            _ = os.system('cls')

        # for mac and linux(here, os.name is 'posix')
        else:
            _ = os.system('clear')



if __name__ == '__main__':
    trials = int(input("How many episodes should be run? "))
    fast_mode = int(input("Fast Mode On/Off? [1/0] "))
    my_agent = Agent_Tester("frozenlake_agent_model.csv", trials, abs(fast_mode-1), not bool(fast_mode))
    percentage_correct = my_agent()
    print("The model has an accuracy of {} across {} episodes".format(round(percentage_correct), my_agent.num_episodes))

