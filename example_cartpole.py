import chainer
from chainer import Variable, optimizers, flag
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import numpy as np
import os
import cPickle as pc
import gan_rl_fitter as fitter
import json

class EnvNet(Chain):
    def __init__(self, env, rand_sz, layer_sz):
        action_size = env.action_space.n
        observation_size = env.observation_space.shape[0]

        super(EnvNet, self).__init__(
            ipt=L.StatefulGRU(action_size + rand_sz, layer_sz),
            out=L.Linear(layer_sz, observation_size + 2),
        )
        self.rand_sz = rand_sz
        self.act_size = action_size
        self.spec = env.spec

    def reset_state(self):
        self.ipt.reset_state()

    def __call__(self, X):
        # generate random values
        R = np.random.randn(X.data.shape[0], self.rand_sz)
        R = Variable(R.astype("float32"))

        # attach random to the inputs
        h = F.concat([R, X])
        #h = R

        h = self.ipt(h)
        y = self.out(h)

        obs = y[:, :-2]
        rew = y[:,[-2]]
        fin = F.sigmoid(y[:, [-1]])

        y = F.concat([obs, rew, fin])

        return y

    def predict(self, X):
        X = fitter.tv(X)
        Y = self(X)
        return Y.data

class DiscriminatorNet(Chain):
    def __init__(self, env, layer_sz):
        action_size = env.action_space.n
        observation_size = env.observation_space.shape[0]

        super(DiscriminatorNet, self).__init__(
            ipt=L.StatefulGRU(action_size + observation_size + 2, layer_sz),  # the first linear layer
            out=L.Linear(layer_sz, 1),  # the feed-forward output layer
        )

    def reset_state(self):
        self.ipt.reset_state()

    def __call__(self, X, Y):
        h = F.concat((X, Y))
        #h = Y

        h = self.ipt(h)
        y = F.sigmoid(self.out(h))

        return y

class RNNAgent(Chain):
    def __init__(self, env, layer_sz):
        action_size = env.action_space.n
        observation_size = env.observation_space.shape[0]

        super(RNNAgent, self).__init__(
            ipt=L.StatefulGRU(observation_size, layer_sz),
            out=L.Linear(layer_sz, action_size),
        )
        self.noise_probability = 0.0 # probability to output noise
        self.action_size = action_size

    def reset_state(self):
        self.ipt.reset_state()

    def __call__(self, X):
        # generate random values

        h = self.ipt(X)
        y = self.out(h)

        # prior knowledge: outputs are probabilities of discrete actions
        y = F.softmax(y)

        return y

    def next(self, X):
        X = fitter.tv(X)
        Y = self(X).data

        if self.noise_probability > 0.0:
            I = np.random.rand(*Y.shape) <= self.noise_probability
            Nz = np.random.rand(*Y.shape)
            Y[I] = Nz[I]
            # renormalize everything again
            Y = Y / np.sum(Y, axis=1)

        Y_formatted = np.argmax(Y, axis=1)

        return Y_formatted, Y

import gym
env = gym.make('CartPole-v0')
#env.monitor.start("cartpole", force=True)


layer_sz = 64 # size of NN of all environments / agents
rnd_sz = 2 # amount of randomness per agent
state_size = 4 # size of state of the environment
act_size = 1  # this encodes actions

fitter.train_gan_rl(
    CreateGenerator = lambda: EnvNet(env, rnd_sz, layer_sz),
    CreateDiscriminator = lambda: DiscriminatorNet(env, layer_sz),
    CreateActor = lambda: RNNAgent(env, layer_sz),
    Environment = env,
    project_folder="results",
    noise_decay_schedule=[1.0, 0.5, 0.25, 0.1, 0.0, 0.0],
    N_real_samples=32,
    N_GAN_batches=1024,
    N_GAN_samples=256,
    GAN_tr_lr=0.01,
    GAN_tr_mm=0.3,
    GAN_training_iter=512,
    evaluation_only=False
)
