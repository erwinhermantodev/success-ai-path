import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration-exploitation tradeoff
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.epsilon_min = 0.01  # Minimum epsilon value
        self.batch_size = 32
        self.model = self.build_model()
        self.target_model = self.build_model()

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = np.array(random.sample(self.memory, self.batch_size))
        states = np.vstack(minibatch[:, 0])
        actions = minibatch[:, 1].astype(int)
        rewards = minibatch[:, 2]
        next_states = np.vstack(minibatch[:, 3])
        dones = minibatch[:, 4]

        targets = self.model.predict(states)
        targets[np.arange(self.batch_size), actions] = rewards + self.gamma * np.max(self.target_model.predict(next_states), axis=1) * (1 - dones)

        self.model.fit(states, targets, epochs=1, verbose=0)

    def target_train(self):
        self.target_model.set_weights(self.model.get_weights())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Example usage:
state_size = 4
action_size = 2
agent = DQNAgent(state_size, action_size)
