import tensorflow as tf
import numpy as np
import random
from collections import deque
from game.track import *
from game.car import *

###todo get dqn to shoot input ####


#####notes should create a dual network to identify cars and then signal to change direction?? #####
'''
grab by id 
need grid design previous network didn't work due to state not being trained on

'''

###################################


class Environment:
    def __init__(self):
        self.reward = 0
        self.state = None
        self.action_space = 100
        self.done = False
        self.track = None
        self.state = None
        self.agent = None
        self.carCt = 0

    #todo create grid system
    def identify_state(self, obs):
        #get car position
        pass

    def run_once(self):
        pass

    def step(self, action):
        if self.track.car != None:
            self.track.car.control(action)

        return self.state, self.reward, self.done, 1

    def checkProgress(self, car):
        #if car is in green no reward
        pass
        #if car is in white(road) add reward or postpone

    def reset(self):
        self.agent = Car(self.carCt)
        self.track = track()
        self.track.drawTrack()
        self.track.update_screen()
        self.state = self.track.save_image()
        print(self.state)
        return self.state



##create dqn
input_height = 88
input_width = 80
input_channels = 1
conv_n_maps = [32,64,64]
conv_kernel_sizes = [(8,8),(4,4),(3,3)]
conv_strides = [4,2,1]
conv_paddings = ["SAME"] * 3
conv_activation = [tf.nn.relu] * 3
n_hidden_in = 64 * 11 * 10 #conv3 has 64 maps of 11 * 10 each
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = 4  #4 discrete actions are available
initializer = tf.contrib.layers.variance_scaling_initializer()


def q_network(X_state, name):
    prev_layer = X_state
    with tf.variable_scope(name) as scope:
        for n_maps, kernel_size, strides, padding, activation in zip(
            conv_n_maps, conv_kernel_sizes, conv_strides,
            conv_paddings, conv_activation):
            prev_layer = tf.layers.conv2d(prev_layer, filters=n_maps, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation, kernel_initializer=initializer)


        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
        hidden = tf.layers.dense(last_conv_layer_flat, n_hidden, activation=hidden_activation, kernel_initializer=initializer)
        outputs = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)

    #kernel and bias
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                              for var in trainable_vars}
    return outputs, trainable_vars_by_name



X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels], name="x_state")


online_q_values, online_vars = q_network(X_state, name="q_networks/online")
target_q_values, target_vars = q_network(X_state, name="q_networks/target")

copy_ops = [target_var.assign(online_vars[var_name])
            for var_name, target_var in target_vars.items()]


copy_online_to_target = tf.group(*copy_ops)
X_action = tf.placeholder(tf.int32, shape=[None], name="x_action")
q_value = tf.reduce_sum(target_q_values * tf.one_hot(X_action, n_outputs), axis = 1, keep_dims=True)

y = tf.placeholder(tf.float32, shape=[None, 1], name="y")
error = tf.abs(y - q_value)
clipped_error = tf.clip_by_value(error, 0.0, 1.0)
linear_error = 2 * (error - clipped_error)

loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

learning_rate = 0.001
momentum = 0.95

global_step = tf.Variable(0, trainable=False, name='global_step')
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)

training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

replay_memory_size = 50000
replay_memory = deque([], maxlen=replay_memory_size)


def sample_memories(batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    cols = [[],[],[],[],[]]
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)

    cols = [np.array(col) for col in cols]
    return (cols[0], cols[1], cols[2].reshape(-1,1), cols[3], cols[4].reshape(-1, 1))


eps_min = 0.1
eps_max = 1.0
eps_decay_steps = 20000


def preprocess_obs(obs):
    #shrink image data
    img = obs[::2, ::2]
    img = img.mean(axis=2)
    return img.reshape(88, 80, 1)


def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)

    else:
        return np.argmax(q_values) #optimal action


###todo create new environment
env = Environment()
### hyperparameters ###

learning_rate = 0.001
n_steps = 400000
training_start = 1000
training_interval = 100
save_steps = 1000
copy_steps = 10000
discount_rate = 0.99
skip_start = 90 #skip start of every game
batch_size = 50
iteration = 0 #game iterations
step = 0 #step
checkpoint_path = "./my_dqn.ckpt"
done = True

with tf.Session() as sess:
    tf.global_variables_initializer()
    while iteration < n_steps:
        ###todo implement q learning algorithm
        if done:
            obs = env.reset()
            env.track.update_input()
            env.track.update_screen()
            #preprocess state to pass to q network
            state = preprocess_obs(obs)
            # feeding state to q network
            q_values = online_q_values.eval(feed_dict={X_state: [state]})
            action = epsilon_greedy(q_values, step)
            # online dqn plays
            obs, reward, done, info = env.step(action)

            next_state = obs.reshape(88, 80, 1)

            # lets memorize what just happened
            replay_memory.append((state, action, reward, next_state, 1.0 - done))
            state = next_state
            iteration += 1
        if iteration < training_start or iteration % training_interval != 0:
            continue
            # only train after warmup period and at regular intervals
            # sample memories and use the target dqn to produce the target q-value
        X_state_val, X_action_val, rewards, X_next_state_val, continues = (sample_memories(batch_size))
        next_q_values = target_q_values.eval(feed_dict={X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * discount_rate * max_next_q_values
        # train the online dqn
        training_op.run(feed_dict={X_state: X_state_val, X_action: X_action_val, y: y_val})
        # regularly copy the online dqn to the target dqn
        if step % copy_steps == 0:
            print("copied the online dqn to the target dqn")
            copy_online_to_target.run()
        # and save regularly
        if step % save_steps == 0:
            saver.save(sess, checkpoint_path)

        step += 1


