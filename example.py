import chainer
from chainer import Variable, optimizers, flag
from chainer import Link, Chain, ChainList
import chainer.functions as CF
import chainer.links as L
import numpy as np
import os
import cPickle as pc
import environment_fitter as fitter
import json

def empty_vector_init(size, action_size, observation_size):

    # generates empty action, observation, reward, finish vectors
    A = np.zeros((size, action_size))
    R = np.zeros(size)
    F = np.zeros(size)
    O = np.zeros((size, observation_size))

    A, O, R, F = fitter.tv(A), fitter.tv(O), fitter.tv(R), fitter.tv(F)

    return A, O, R, F


class EnvironmentGRNN(Chain):
    def __init__(self, action_size, random_size, layer_sz, observation_size):
        super(EnvironmentGRNN, self).__init__(
            ipt=L.StatefulGRU(action_size + random_size, layer_sz),
            out=L.Linear(layer_sz, observation_size+2),
        )

        self.rand_sz = random_size
        self.action_size = action_size
        self.obs_size = observation_size


    def reset_state(self, size = 1):
        self.ipt.reset_state()
        A, _, _, _ = empty_vector_init(size, self.action_size, self.obs_size)
        O, _, _ = self(A)
        return O

    def __call__(self, X):
        # generate random values
        R = np.random.randn(X.data.shape[0], self.rand_sz)
        R = Variable(R.astype("float32"))

        # attach random to the inputs
        h = CF.concat([R, X])
        #h = R

        h = self.ipt(h)
        y = self.out(h)

        # prior knowledge: environment observation is one - hot vector
        observation = CF.softmax(y[:, :-2])
        # prior knowledge: reward is in [0,1]
        finished = CF.sigmoid(y[:, -2])

        reward = CF.sigmoid(y[:, -1])

        return observation, reward, finished

    def predict(self, X):
        X = fitter.tv(X)
        O, R, F = self(X)
        return O.data, R.data, F.data


class DiscriminatorNet(Chain):
    def __init__(self, action_size, observation_size, layer_sz):
        super(DiscriminatorNet, self).__init__(
            ipt=L.StatefulGRU(action_size + observation_size+ 2, layer_sz),  # the first linear layer
            out=L.Linear(layer_sz, 1),  # the feed-forward output layer
        )
        self.action_size = action_size
        self.obs_size = observation_size

    # initial condition is treated specially
    def reset_state(self, O):
        """
        This resets the discriminator net, and calculates the probability that init_obs,
        which is output of geneartor_net at reset / initial obs., is a fake initial state
        :param init_obs:
        :return:
        """
        self.ipt.reset_state()
        size = len(O.data)
        A, _, R, F = empty_vector_init(size, self.action_size, self.obs_size)

        return self(A, O, R, F)

    def __call__(self, A, O, R, F):
        """

        :param X: actions
        :param Y: tuple of observation, reward, finish
        :return:
        """
        h = CF.concat((
            A,
            O,
            CF.array.expand_dims.expand_dims(R, 1),
            CF.array.expand_dims.expand_dims(F, 1))
        )

        #h = Y

        h = self.ipt(h)
        y = CF.sigmoid(self.out(h))

        return y

def onehot(I, mx):
    Z = np.zeros((len(I), mx))
    Z[range(len(I)), I] = 1.0
    return Z

class ExampleEnv():
    """
    Example [very simple debug] environment compatible with OpenAI gym.
    Agent is in discrete field of fixed size, agent's output is the probability
    in the next step to move left or right.
    Agent is rewarded for making a move to the right.
    Optimal strategy:
    1. go right untill agent is on the boundary
    2. oscilate on the boundary: go left / right / left / right ...
    """

    def __init__(self, size):
        self.obs_size = size
        self.action_size = 1
        self.player_position = None

    def observe(self):
        result = np.zeros(self.obs_size)

        if not self.player_position is None:
            result[self.player_position] = 1.0

        return result

    def reset(self):
        """

        :return: observation - initial state of the environment
        """
        # random inital postion of player

        self.player_position = np.random.randint(self.obs_size)

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

        p = np.clip(p + action, 0, self.obs_size - 1)

        self.player_position = p

        # stupid reward: going to right rewards
        reward = max(0, p - old_p)

        # observation, reward, done (is episode finished?), info
        return self.observe(), reward, False, None

class RNNAgent(Chain):
    def __init__(self, x_sz, layer_sz, act_sz):
        super(RNNAgent, self).__init__(
            ipt=L.StatefulGRU(x_sz, layer_sz),
            out=L.Linear(layer_sz, act_sz),
        )
        self.noise_probability = 0.0 # probability to output noise

    def reset_state(self):
        self.ipt.reset_state()

    def __call__(self, X):
        # generate random values

        h = self.ipt(X)
        y = self.out(h)

        # prior knowledge: output should be in [0, 1]
        y = CF.sigmoid(y)

        return y

    def next(self, X):
        X = fitter.tv(X)
        Y = self(X).data

        if self.noise_probability > 0.0:
            I = np.random.rand(*Y.shape) <= self.noise_probability
            Nz = np.random.rand(*Y.shape)
            Y[I] = Nz[I]

        return Y

# ground truth agent. Achieves aroun 0.62 average reward
class DummyAgent(Chain):
    def __init__(self, x_sz, layer_sz, act_sz):
        self.x_sz = x_sz

    def reset_state(self):
        pass

    def __call__(self, X):
        return None

    def next(self, X):
        X = np.argmax(X, axis=1)
        Y = X != self.x_sz-1
        return [Y*1.0]

# this collects data about real environment
def generate_data(agent, env, N, max_steps):

    I, A, O, R, F = [], [], [], [], []

    a, o, r, f = empty_vector_init(N, env.action_size, env.obs_size)
    I = o.data # Initial observation

    Rv = 0.0
    Ri = 1

    # zero arrays of max_steps length:
    for i in range(max_steps):
        a, o, r, f = empty_vector_init(N, env.action_size, env.obs_size)
        A.append(a.data) # action input
        O.append(o.data)
        R.append(r.data)
        F.append(f.data*0.0 + 1.0)

    # record N episodes
    for episode in range(N):

        agent.reset_state()
        observation = env.reset()

        # play
        for i in range(max_steps):
            act = agent.next([observation])[0]
            observation, reward, done, info = env.step(act)

            Rv += reward
            Ri += 1

            R[i][episode] = reward

            if done:
                break

            A[i][episode] = act
            O[i][episode] = observation
            F[i][episode] = 0.0

    return I, A, O, R, F, Rv / Ri


# warning: below function should be used during training only - agent(observ) returns Chainer tensor
def evaluate_on_diff_env(diff_env, n_sample_traj, agent, max_steps):
    # this function is used to sample n traj using GAN version of environment
    R = 0.0

    # reset agent
    agent.reset_state()

    # get initial observation
    observations = diff_env.reset_state(n_sample_traj)

    for i in range(max_steps):
        act = agent(observations)
        observations, rewards, finished = diff_env(act)
        # below is used that rewards are zero at the end of episode
        R += CF.sum(rewards) / (-len(rewards) * max_steps)

    return R

layer_sz = 64 # size of NN of all environments / agents
rnd_sz = 2 # amount of randomness per agent
state_size = 4 # size of state of the environment
act_size = 1  # this encodes actions
N_real_samples = 128 # number of samples from real environment
GAN_training_iter = 2048 # grad. descent number of iterations to train GAN
N_GAN_samples = 256 # number of trajectories for single agent update sampled from GAN
N_GAN_batches = 1024 # number of batches to train agent on
max_steps = 4 # maximum size of an episode
project_folder = "results" # here environments / agents will be stored

evaluation_only = False # when true, agent is loaded and evaluated on the environment

# initialization with random agent

env = ExampleEnv(state_size)
idx = 0

files = ['train.bin', 'validate.bin']
agent_file = 'agent.bin'
perf_file = 'stats.json'

agent_file = os.path.join(project_folder, agent_file)
perf_file = os.path.join(project_folder, perf_file)
files = [os.path.join(project_folder, f) for f in files]

# load agent if possible
if os.path.exists(agent_file):
    agent = pc.load(open(agent_file))
    print "Loaded agent:", agent
else:
    agent = RNNAgent(state_size, layer_sz, act_size)

optA = optimizers.Adam(alpha=0.001, beta1=0.3)
optA.setup(agent)


# main training loop
for noise_p in [1.0, 0.0]:

    idx += 1

    envs, perf = {}, {}

    # load environments if already trained some
    for fname in files:
        if not os.path.exists(fname):
            envs[fname] = {'G':EnvironmentGRNN(act_size, rnd_sz, layer_sz, state_size), 'D':None}
        else:
            envs[fname] = pc.load(open(fname))

    # set temporarily noise probability != 0
    agent.noise_probability = noise_p

    # fit differentiable environment for training and testing
    for fname in files:
        I, A, O, R, F, Rmean = generate_data(agent, env, N_real_samples, max_steps)
        perf[fname] = float(Rmean) # shows the performance of agent on real environment

        if not evaluation_only:

            # extend the data
            if envs[fname]['D'] is None:
                envs[fname]['D'] = {'I':I, 'A':A, 'O':O, 'R':R, 'F':F}
            else:
                for i in range(max_steps):
                    for elm, v in [('I',I), ('A',A), ('O',O), ('R',R), ('F', F)]:
                        envs[fname][elm][i] = np.concatenate([envs[fname][elm][i], v[i]])

            fitter.FitStochastic(
                envs[fname]['G'],
                DiscriminatorNet(act_size, state_size, layer_sz),
                envs[fname]['D'],
                0.01,
                0.3,
                GAN_training_iter,
                "Example generated [state, reward] sequences:")

            pc.dump(envs[fname], open(fname, 'w'))

    # set noise probability to zero for training
    agent.noise_probability = 0

    print "Performance:", perf
    json.dump(perf, open(perf_file, 'w'))

    if evaluation_only:
        continue

    # train the agent with SGD
    for reps in range(N_GAN_batches):

        # reset the agent
        agent.cleargrads()
        # train
        R = evaluate_on_diff_env(envs[files[0]]['G'], N_GAN_samples, agent, max_steps)
        R.backward()
        optA.update()

        Rv = evaluate_on_diff_env(envs[files[1]]['G'], N_GAN_samples, agent, max_steps)

        print 'Avg. reward: training GAN = ', -R.data, 'testing GAN = ', -Rv.data

    # save trained agent
    pc.dump(agent, open(agent_file, 'w'))
