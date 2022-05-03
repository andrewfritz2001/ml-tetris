"""
    All network parameters/constants should go in to a config file. 
    Then we can load that file in as a json and use it to initialize
    the game as well as the nueral nets.
    *
    Let's us make easy adjustments to the network without having to dive
    too deeply in to the code.
"""
import tensorflow as tf
import numpy as np

from tetris import Tetris
from deep_q_learning_agent import DeepQLearningAgent

import timeit

EPISODES = 15

def train():
    env = Tetris()
    model = DeepQLearningAgent()

    running = True
    total_games = 0
    total_steps = 0
    
    rewards = []
    while running:
        # Train the network
        scores = []
        game_times = []
        steps = 0 

        for _ in range(EPISODES):
            env.reset()
            start = timeit.default_timer()
            game_state = env.next_states()
            prev_state = None
            done = False
            total_reward = 0
            while not done:
                # must give act() a list of tuples like (action,state)
                # game state looks like this: (x-pos,rots):[bumpiness,holes,etc] ]
                action, state = model.act(game_state)

                # give step() an action in the form (x-pos,numrotations)
                # step returns a state 
                game_state, reward, done = env.step(action)
                
                
                model.experience_log.add((prev_state, reward, state, done))
                prev_state = state
                steps += 1
                total_reward += reward
            
            end = timeit.default_timer()
            game_times.append(end-start)
            rewards.append(total_reward)
            scores.append(env.score)
            model.learn()
        
        sum_scores = 0
        for score in scores:
            sum_scores += score
        avg_score = sum_scores/len(scores)

        sum_times = 0
        for time in game_times:
            sum_times += time
        avg_time = sum_times/len(game_times)

        # Display the results of x episodes that were trained
        print("=======================")
        print("+ Total Games: ", len(rewards))
        print("+ Total Steps: ", steps)
        print("+ Epsilon: ", model.epsilon)
        print("+ Avg. Score: ", avg_score)
        print("+ Avg. Time/Game: ", avg_time)
        # running = False




if __name__ == "__main__":
    train()