import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA = 0.9 # discount factor
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON =  0.05 # final value of epsilon
EPSILON_DECAY_STEPS = 100 # decay period

replay_buffer = []
BATCH_SIZE = 50
REPLAY_SIZE = 5000
LEARNING_RATE = 0.001
HIDDEN_NODES = 20

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# TODO: Define Network Graph
w_initializer, b_initializer = tf.random_normal_initializer(0.0, 0.3), tf.constant_initializer(0.1)
hidden_layer = tf.layers.dense(state_in, HIDDEN_NODES, tf.nn.tanh, kernel_initializer=w_initializer, bias_initializer=b_initializer)
q_values = tf.layers.dense(hidden_layer, ACTION_DIM, kernel_initializer=w_initializer, bias_initializer=b_initializer)
q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)

# TODO: Loss/Optimizer Definition
loss = tf.reduce_sum(tf.square(target_in - q_action))
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action


# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= epsilon / EPSILON_DECAY_STEPS
    if epsilon < FINAL_EPSILON:
        epsilon = FINAL_EPSILON

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))

        nextstate_q_values = q_values.eval(feed_dict={
            state_in: [next_state]
        })

        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated
        # if not done:
        # 	target = reward + GAMMA * np.max(nextstate_q_values)
        # else:
        # 	target = reward

        # # Do one training step
        # session.run([optimizer], feed_dict={
        #     target_in: [target],
        #     action_in: [action],
        #     state_in: [state]
        # })

        replay_buffer.append((state, action, reward, next_state, done))
        if len(replay_buffer) > REPLAY_SIZE:
            replay_buffer.pop(0)
        if len(replay_buffer) > BATCH_SIZE:
            minibatch = random.sample(replay_buffer, BATCH_SIZE)
            state_batch = [value[0] for value in minibatch]
            action_batch = [value[1] for value in minibatch]
            reward_batch = [value[2] for value in minibatch]
            next_state_batch = [value[3] for value in minibatch]
            target_batch = []
            nextstate_q_batch = q_values.eval(feed_dict={state_in:next_state_batch})
            for i in range(BATCH_SIZE):
                terminated = minibatch[i][4]
                if terminated:
                    target_batch.append(reward_batch[i])
                else:
                    target_batch.append(reward_batch[i] + GAMMA*np.max(nextstate_q_batch[i]))

        # Do one training step
            session.run([optimizer], feed_dict={
                state_in: state_batch,
                target_in: target_batch,
                action_in: action_batch
                })

        # Update
        state = next_state
        if done:
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
