import tensorflow as tf
import numpy as np

from tetris import Tetris
from deep_q_learning_agent import DeepQLearningAgent

EPISODES = 10

"""
    TODO:
    Anytime env.some_func() is called, we need to make sure that the
    game engine is giving the neural network the correct inputs
    Then, we should be ok, but hard to tell at the moment.
"""


def train():
    env = Tetris()
    model = DeepQLearningAgent()

    running = True
    total_games = 0
    total_steps = 0
    
    while running:
        # Train the network
        rewards = []
        scores = []
        steps = 0 

        for _ in range(EPISODES):
            game_state = env.reset()
            prev_state = None
            done = False
            total_reward = 0
            while not done:
                action, state = model.act(game_state)
                game_state, reward, done, info = env.step(action)
                model.experiences.add((prev_state, reward, state, done))
                prev_state = state
                steps += 1
                total_reward += reward
            
            rewards.append(total_reward)
            scores.append(env.score)
            model.learn()
        
        # Display the results of x episodes that were trained
        print("+ Total Games: ", len(scores))
        print("+ Total Steps: ", steps)

        running = False



if __name__ == "__main__":
    train()