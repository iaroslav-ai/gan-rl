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
        self.max_steps = 4

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
    in the next step to move left or right.
    Agent is rewarded for making a move to the right.
    Optimal strategy:
    1. go right untill agent is on the boundary
    2. oscilate on the boundary: go left / right / left / right ...
    """

    def __init__(self, size):
        self.size = size
        self.player_position = None
        self.act_size = 1
        self.max_steps = 4

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

        self.player_position = np.random.randint(self.size-1)

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

        done = p == (self.size -1)
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
        Y = X != self.x_sz-2
        return [Y*1.0]

# collect the data about env
def generate_data(agent, env, N):

    O, A, R, E = [], [], [], []

    for episode in range(N):

        agent.reset_state()
        observation = env.reset()

        done = False
        obvs, acts, rews, ends = [observation], [np.zeros(env.act_size)], [0.0], [0.0]

        # play
        for i in range(env.max_steps):
            act = agent.next([observation])[0]

            if done: # if episode is done - pad the episode to have max_steps length
                reward = 0.0
            else:
                observation, reward, done, info = env.step(act)

            rews.append(reward)
            obvs.append(observation)
            acts.append(act)
            ends.append(done*1.0)


        O.append(obvs)
        A.append(acts)
        R.append(rews)
        E.append(ends)

    X, Y = [], []

    # convert data to X, Y format
    for i in range(env.max_steps):
        x, y = [], []

        for o, a, r, e in zip(O, A, R, E):
            x.append(a[i])
            y.append(np.array(o[i].tolist() + [r[i], e[i]]))

        X.append(np.array(x))
        Y.append(np.array(y))

    return X, Y, np.mean(np.array(R)[:,1:])

# warning: below function should be used during training only - agent(observ) returns Chainer tensor
def evaluate_on_diff_env(env, n_sample_traj, agent, max_steps):
    # this function is used to sample n traj using GAN version of environment
    R = 0.0

    # reset environment
    env.reset_state()
    agent.reset_state()

    # get initial observation
    observations = env(fitter.tv(np.zeros((n_sample_traj, env.act_size))))[:, :-2]

    for i in range(env.max_steps):
        act = agent(observations)
        obs_rew = env(act)
        rewards = obs_rew[:, -2]
        ends = obs_rew[:, -1]
        observations = obs_rew[:, :-2]
        R += F.sum(rewards * (1.0 - ends)) / (-len(rewards) * max_steps)

    return R

layer_sz = 64 # size of NN of all environments / agents
rnd_sz = 2 # amount of randomness per agent
state_size = 4 # size of state of the environment
act_size = 1  # this encodes actions
N_real_samples = 128 # number of samples from real environment
GAN_training_iter = 2048 # grad. descent number of iterations to train GAN
N_GAN_samples = 256 # number of trajectories for single agent update sampled from GAN
N_GAN_batches = 1024 # number of batches to train agent on
project_folder = "results" # here environments / agents will be stored

evaluation_only = False # when true, agent is loaded and evaluated on the environment

# initialization with random agent

env = ExampleEnv(state_size)
noise_decay_schedule = [1.0, 0.0]

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

optA = optimizers.Adam(alpha=0.001, beta1=0.9)
optA.setup(agent)


# main training loop
for noise_p in noise_decay_schedule:

    idx += 1

    envs, perf = {}, {}

    # load environments if already trained some
    for fname in files:
        if not os.path.exists(fname):
            envs[fname] = {'G':GeneratorNet(act_size, rnd_sz, layer_sz, state_size), 'X':None, 'Y':None}
        else:
            envs[fname] = pc.load(open(fname))

    # set temporarily noise probability != 0
    agent.noise_probability = noise_p

    # fit differentiable environment for training and testing
    for fname in files:
        X, Y, Rmean = generate_data(agent, env, N_real_samples)
        perf[fname] = float(Rmean) # shows the performance of agent on real environment

        if not evaluation_only:

            # extend the data
            if envs[fname]['X'] is None:
                envs[fname]['X'] = X
                envs[fname]['Y'] = Y
            else:
                for i in range(env.max_steps):
                    for elm, v in [('X', X), ('Y', Y)]:
                        envs[fname][elm][i] = np.concatenate([envs[fname][elm][i], v[i]])

            fitter.FitStochastic(
                envs[fname]['G'],
                DiscriminatorNet(act_size, state_size, layer_sz),
                (envs[fname]['X'],envs[fname]['Y']),
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
        R = evaluate_on_diff_env(envs[files[0]]['G'], N_GAN_samples, agent)
        R.backward()
        optA.update()

        Rv = evaluate_on_diff_env(envs[files[1]]['G'], N_GAN_samples, agent)

        print 'Avg. reward: training GAN = ', -R.data, 'testing GAN = ', -Rv.data

    # save trained agent
    pc.dump(agent, open(agent_file, 'w'))
