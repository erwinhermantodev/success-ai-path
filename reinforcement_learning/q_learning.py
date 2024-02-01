import numpy as np

# Environment setup
num_states = 6
num_actions = 3
Q = np.zeros((num_states, num_actions)) # Q-value table
alpha = 0.1 # Learning rate
gamma = 0.9 # Discount factor
epsilon = 0.1 # Exploration-exploitation tradeoff

# Q-learning algorithm
def q_learning(state, action, reward, next_state):
    max_next_action_value = np.max(Q[next_state])
    Q[state, action] += alpha * (reward + gamma * max_next_action_value - Q[state, action])
    
# Q-learning loop (example with a simple environment)
for _ in range(1000):
    state = np.random.randint(0, num_states) # Initial state
    while state != 5: # Terminal state
        action = np.argmax(Q[state]) if np.random.rand() > epsilon else np.random.randint(0, num_actions)
        next_state = (state + action) % num_states # Simple transition model
        reward = 1 if next_state == 5 else 0
        q_learning(state, action, reward, next_state)
        state = next_state
        
# Result Q-value table
print("Q-value table:")
print(Q)
        