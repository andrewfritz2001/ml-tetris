import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import random

"""
    Hyperparameters for training
"""
DISCOUNT_RATE = 1
EPSILON = 0.05
DECAY = 0.99
MIN_EPSILON = 0.0005 # we always want at least a little bit of randomness


"""
    Class serves as a container to log the previous x actions made by the 
    neural network.
"""
class ExperienceLog:
    def __init__(self, buffer_size=10000):
        self.buffer = [None] * buffer_size
        self.buffer_size = buffer_size
        self.insert_pos = 0

    def add(self, experience):
        if self.insert_pos >= self.buffer_size:
            self.insert_pos = 0
        self.buffer[self.insert_pos] = experience
        self.insert_pos += 1

    def sample(self, size):
        return random.sample(self.buffer[0:self.size()],size)
    
    def size(self):
        sum = 0
        for i in self.buffer:
            if i:
                sum += 1
        return sum



"""
    The Deep-Q-Learning agent we will be using to play tetris
"""
class DeepQLearningAgent:
    def __init__(self, state_size=4):  
        self.state_size = state_size    
        self.discount_rate =   DISCOUNT_RATE
        self.epsilon = EPSILON
        self.min_epsilon = MIN_EPSILON
        self.decay = DECAY
        self.experience_log = ExperienceLog()
        self.model = keras.Sequential([
                layers.Dense(64, activation="relu", input_dim=self.state_size),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(1, activation="linear")
            ])

        # Using Mean Squared Error (MSE) as our loss function. adam=gradient descent
        self.model.compile(optimizer='adam', loss='mse',metrics=['mean_squared_error'])
        self.model.summary()
        # keras.utils.plot_model(self.model, IMAGE_PATH, show_shapes=True) --- nn visualization
        # self.tensorboard = keras.callbacks.TensorBoard(logdir=LOG_DIR,histogram_freq=1000,write_graph=True,write_images=True)


    """
        Given a list of state-action pairs, returns the state-action pair with the highest rating
        If a random num (0,1) is lower than epsilon, then we act pick a random state-action pair
    """
    def act(self, possible_states):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(possible_states)
        
        max_rating = None
        best_state = None
        ratings = self.predict_ratings([state for action,state in possible_states])

        for i, (action, state) in enumerate(possible_states):
            rating = ratings[i]
            if not max_rating or rating > max_rating:
                max_rating = rating
                best_state = (action,state)

        return best_state


    """
        Given a list of states, returns the rating/score given by the neural network
        for each state
    """
    def predict_ratings(self, states):
        y = np.array(states)
        predictions = self.model.predict(y)
        return [predict[0] for predict in predictions]


    """
        Where the model learns from previous actions/experiences and adjusts the weights
        batch_size (500 by default) previous experiences are taken from the experience_log
        and used to train the neural network
    """
    def learn(self, batch_size=500, epochs=1):
        # not enough samples yet to learn
        if self.experience_log.size() < batch_size:
            return

        experiences = self.experience_log.sample(batch_size)
        print(experiences)
        x = []
        y = []

        ratings = self.predict_ratings([x[2] for x in experiences])

        for i, (prev_state, reward, next_state, done) in enumerate(experiences):
            if not done:
                rating = ratings[i]
                q = reward + self.discount_rate * rating
            else:
                q = reward
            x.append(prev_state)
            y.append(q)

        self.model.fit(np.array(x), np.array(y), batch_size=len(x), verbose=0, epochs=epochs) # ,callbacks=[self.tensorboard]
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        print(self.epsilon)