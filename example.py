"""
Example of learning with GANs. See ExampleEnv for description of environment used.

As you run this script sufficiently long, the results of evaluation of trained agent
will appear in "results" folder, in stats.json. There you will see average reward
for training and validation GANs. For the environment defined in this file, best
possible average reward is ~0.63. Most of the time final agent's performance is close
to this value and is somewhere in 0.58 - 0.62.

Important: the GAN fitting procedure needs some further improvement, so quality
of environment model varies from run to run.
"""

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

class GeneratorNet(Chain):
    def __init__(self, x_sz, rand_sz, layer_sz, output_sz):
        super(GeneratorNet, self).__init__(
            ipt=L.StatefulGRU(x_sz + rand_sz, layer_sz),
            out=L.Linear(layer_sz, output_sz + 2),
        )
        self.rand_sz = rand_sz
        self.act_size = x_sz
        self.spec = fitter.EnvSpec(4)

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
        #h = F.dropout(h)
        y = self.out(h)

        # prior knowledge: environment observation is one - hot vector
        obs = F.softmax(y[:, :-2])
        # prior knowledge: reward is in [0,1]
        rew = F.sigmoid(y[:,[-2]])
        fin = F.sigmoid(y[:, [-1]])

        y = F.concat([obs, rew, fin])

        return y

    def predict(self, X):
        X = fitter.tv(X)
        Y = self(X)
        return Y.data


class DiscriminatorNet(Chain):
    def __init__(self, action_size, observation_size, layer_sz):
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

def onehot(I, mx):
    Z = np.zeros((len(I), mx))
    Z[range(len(I)), I] = 1.0
    return Z




class ExampleEnv():
    """
    Example [very simple debug] environment compatible with OpenAI gym.
    Agent is in discrete field of fixed size, agent's output is the probability
    to move left or right in the next step.
    Agent is rewarded for making a move to the right.
    Optimal strategy:
    1. go right untill agent is on the boundary
    2. oscilate on the boundary: go left / right / left / right ...
    """

    def __init__(self, size):
        self.size = size
        self.player_position = None
        self.act_size = 1
        self.spec = fitter.EnvSpec(4)

    def observe(self):
        result = np.zeros(self.size)

        if not self.player_position is None:
            result[self.player_position] = 1.0

        return result

    def reset(self):
        """

        :return: observation - initial state of the environment
        """
        # random inital postion of player

        self.player_position = np.random.randint(self.size)

        return self.observe()

    def step(self, actarr):
        """

        :param action: action that agent takes
        :return: observation, reward, done (is episode finished?), info
        """

        action = actarr[0]

        p = self.player_position
        old_p = p

        right_probability = np.clip(action, 0.0, 1.0)

        if np.random.rand() < right_probability:
            action = +1
        else:
            action = -1

        p = np.clip(p+action, 0, self.size-1)

        self.player_position = p

        # stupid reward: going to right rewards
        reward = max(0, p - old_p)

        #done = p == (self.size -1)
        done = False
        # observation, reward, done (is episode finished?), info
        return self.observe(), reward, done, None

class RNNAgent(Chain):
    def __init__(self, obs_size, layer_sz, act_sz):
        super(RNNAgent, self).__init__(
            ipt=L.StatefulGRU(obs_size, layer_sz),
            out=L.Linear(layer_sz, act_sz),
        )
        self.noise_probability = 0.0 # probability to output noise
        self.action_size = act_sz

    def reset_state(self):
        self.ipt.reset_state()

    def __call__(self, X):
        # generate random values

        h = self.ipt(X)
        y = self.out(h)

        # prior knowledge: output should be in [0, 1]
        y = F.sigmoid(y)

        return y

    def next(self, X):
        X = fitter.tv(X)
        Y = self(X).data

        if self.noise_probability > 0.0:
            I = np.random.rand(*Y.shape) <= self.noise_probability
            Nz = np.random.rand(*Y.shape)
            Y[I] = Nz[I]

        return Y, Y

# ground truth agent. Achieves around 0.62 average reward
class DummyAgent(Chain):
    def __init__(self, x_sz, layer_sz, act_sz):
        self.x_sz = x_sz

    def reset_state(self):
        pass

    def __call__(self, X):
        return None

    def next(self, X):
        X = np.argmax(X, axis=1)
        Y = X != self.x_sz-2
        return [Y*1.0]

layer_sz = 64 # size of NN of all environments / agents
rnd_sz = 2 # amount of randomness per agent
state_size = 4 # size of state of the environment
act_size = 1  # this encodes actions

fitter.train_gan_rl(
    CreateGenerator = lambda: GeneratorNet(act_size, rnd_sz, layer_sz, state_size),
    CreateDiscriminator = lambda: DiscriminatorNet(act_size, state_size, layer_sz),
    CreateActor = lambda: RNNAgent(state_size, layer_sz, act_size),
    Environment = ExampleEnv(state_size),
    project_folder="results",
    noise_decay_schedule=[1.0, 0.0],
    N_real_samples=128,
    N_GAN_batches=1024,
    N_GAN_samples=256,
    GAN_tr_lr=0.01,
    GAN_tr_mm=0.3,
    GAN_training_iter=2 ** 10,
    evaluation_only=False
)
