#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from vizdoom import *
import itertools as it
import pickle
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
from lasagne.init import HeUniform, Constant
from lasagne.layers import Conv2DLayer, InputLayer, DenseLayer, MaxPool2DLayer, get_output, \
    get_all_params, get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify
from lasagne.objectives import squared_error
from lasagne.updates import rmsprop
import theano
from theano import tensor
from tqdm import trange


GPU = False
if GPU:
	print "Trying to run under GPU "
	try: theano.config.device = 'gpu'
	except:pass
	theano.config.floatX = 'float32'
else:
	print "Running with CPU"



# Q-learning settings
#learning_rate = 0.00025
learning_rate = 0.0001
discount_factor = 0.33
#discount_factor = 0.99
epochs = 5
learning_steps_per_epoch = 2000 #2000
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100 #100

# Other parameters
frame_repeat = 12
resolution = (40, 60) #(30, 45) x = 2/3*y, y = z
episodes_to_watch = 10

# Configuration file path
#config_file_path = "../../examples/config/simpler_basic.cfg"
#config_file_path = "../../examples/config/deadly_corridor.cfg"
#config_file_path = "../../examples/config/health_gathering.cfg"
#config_file_path = "../../examples/config/deathmatch.cfg"
config_file_path = "../../examples/config/deepdeathmatch.cfg"
#config_file_path = "../../examples/config/basic.cfg"



# Converts and downsamples the input image
def preprocess(img):
    img = img[0]
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img



def create_network(available_actions_count):
    # Create the input variables
    s1 = tensor.tensor4("States")
    a = tensor.vector("Actions", dtype="int32")
    q2 = tensor.vector("Next State's best Q-Value")
    r = tensor.vector("Rewards")
    isterminal = tensor.vector("IsTerminal", dtype="int8")

    dqn = InputLayer(shape=[None, 1, resolution[0], resolution[1]], input_var=s1)

    dqn = Conv2DLayer(dqn, num_filters=32, filter_size=[8, 8],
                      nonlinearity=rectify, W=HeUniform("relu"),
                      b=Constant(.1), stride=4)

    dqn = Conv2DLayer(dqn, num_filters=64, filter_size=[4, 4],
                      nonlinearity=rectify, W=HeUniform("relu"),
                      b=Constant(.1), stride=2)

    dqn = DenseLayer(dqn, num_units=512, nonlinearity=rectify, W=HeUniform("relu"),
                     b=Constant(.1))


    dqn = DenseLayer(dqn, num_units=available_actions_count, nonlinearity=None)

    q = get_output(dqn)

    target_q = tensor.set_subtensor(q[tensor.arange(q.shape[0]), a], r + discount_factor * (1 - isterminal) * q2)
    loss = squared_error(q, target_q).mean()

    params = get_all_params(dqn, trainable=True)
    updates = rmsprop(loss, params, learning_rate)

    print "Compiling the network ..."
    function_learn = theano.function([s1, q2, a, r, isterminal], loss, updates=updates, name="learn_fn")
    function_get_q_values = theano.function([s1], q, name="eval_fn")
    function_get_best_action = theano.function([s1], tensor.argmax(q), name="test_fn")
    print "Network compiled."

    def simple_get_best_action(state):
        return function_get_best_action(state.reshape([1, 1, resolution[0], resolution[1]]))

    return dqn, function_learn, function_get_q_values, simple_get_best_action





# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print "Initializing doom..."
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()
    print "Doom initialized."
    return game

print "======================================"
print "Training finished. It's time to watch!"

# Create Doom instance
game = initialize_vizdoom(config_file_path)
n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]

net, learn, get_q_values, get_best_action = create_network(len(actions))


params = pickle.load(open('../../examples/python/weightsdumps/weightsDay57DDM6.dump', "r"))
set_all_param_values(net, params)


for _ in range(episodes_to_watch):
    game.new_episode()
    
    while not game.is_episode_finished():

        state = preprocess(game.get_state().image_buffer)
        best_action_index = get_best_action(state)

        game.set_action(actions[best_action_index])
        for _ in range(frame_repeat):
            game.advance_action()

    sleep(1.0)
 