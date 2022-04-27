import tensorflow as tf
import numpy as np

from tetris import Tetris
from deep_q_learning_agent import DeepQLearningAgent

EPISODES = 250

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
            env.reset()
            game_state = env.next_states()
            prev_state = None
            done = False
            total_reward = 0
            while not done:
                print(game_state)
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
            
            rewards.append(total_reward)
            scores.append(env.score)
            model.learn()
        
        # Display the results of x episodes that were trained
        print("+ Total Games: ", len(scores))
        print("+ Total Steps: ", steps)

        running = False



if __name__ == "__main__":
    train()