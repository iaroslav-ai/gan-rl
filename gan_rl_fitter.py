import chainer
from chainer import Variable, optimizers, flag
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import numpy as np
import cPickle as pc
import os
import json

# miscellaneous class
class EnvSpec():
    def __init__(self, max_steps):
        self.timestep_limit = max_steps

### ASSUMPTION: INITIAL CONDITION IS IMPORTANT TO GET RIGHT
w_init = 4.0 # wegiht of objective for initial output of RNN ~size of sequence

class DOBJ(Chain):
    def __init__(self):
        super(DOBJ, self).__init__()

    def __call__(self, X, Yt, D, G):
        D.reset_state()

        r = 0.0
        mg = w_init
        for x, yt in zip(X, Yt):
            t = D(x, yt)
            r += F.mean_squared_error(t, t*0.0 + 1.0)*mg
            mg = 1.0

        D.reset_state()
        G.reset_state()

        mg = w_init
        for x, yt in zip(X, Yt):
            f = D(x, G(x))
            r += F.mean_squared_error(f, f * 0.0)*mg
            mg = 1.0

        return r


class GOBJ(Chain):
    def __init__(self):
        super(GOBJ, self).__init__()

    def __call__(self, X, D, G):
        D.reset_state()
        G.reset_state()

        r = 0.0

        mg = w_init
        for x in X:
            f = D(x, G(x))
            r += F.mean_squared_error(f, f*0.0 + 1.0)*mg
            mg = 1.0

        return r


def tv(x, v = flag.OFF):
    return Variable(np.array(x).astype('float32'), volatile=v)

from time import time

def FitStochastic(G, D, XY, learning_rate, momentum, iters, pr_caption=None):

    X, Y = XY

    X, Y = tv(X), tv(Y)

    objD, objG = DOBJ(), GOBJ()

    optG = optimizers.Adam(alpha=learning_rate, beta1=momentum)
    optD = optimizers.Adam(alpha=learning_rate, beta1=momentum)

    optG.setup(G)
    optD.setup(D)

    st = time()

    for i in range(iters):

        D.zerograds()
        loss = objD(X, Y, D, G)
        loss.backward()
        optD.update()

        G.zerograds()
        loss = objG(X, D, G)
        loss.backward()
        optG.update()




        if i % 10 == 0:

            #for p in G.params():
            #    p.data = p.data + 0.5*np.random.randn(*p.data.shape).astype('float32')

            print "iter:", i, "/", iters, "iter. time:", time()-st

            if not pr_caption is None:
                print pr_caption

            st = time()

            G.reset_state()

            Yp = [G(x) for x in X]

            for j in range(10):

                Yi = [np.round(yp.data[j].astype('float64'), 1).tolist() for yp in Yp]

                print Yi
            """"""





# collect the data about env
def generate_data(agent, env, N):

    O, A, R, E = [], [], [], []

    for episode in range(N):

        print "Real ep. ", episode

        agent.reset_state()
        observation = env.reset()

        done = False
        obvs, acts, rews, ends = [observation], [np.zeros(agent.action_size)], [0.0], [0.0]

        # play
        for i in range(env.spec.timestep_limit):
            act, act_unformatted = agent.next([observation])
            act = act[0]
            act_unformatted = act_unformatted[0]

            if done: # if episode is done - pad the episode to have max_steps length
                reward = 0.0
            else:
                observation, reward, done, info = env.step(act)

            rews.append(reward)
            obvs.append(observation)
            acts.append(act_unformatted)
            ends.append(done*1.0)


        O.append(obvs)
        A.append(acts)
        R.append(rews)
        E.append(ends)

    X, Y = [], []

    # convert data to X, Y format
    for i in range(env.spec.timestep_limit):
        x, y = [], []

        for o, a, r, e in zip(O, A, R, E):
            x.append(a[i])
            y.append(np.array(o[i].tolist() + [r[i], e[i]]))

        X.append(np.array(x))
        Y.append(np.array(y))

    return X, Y, np.mean(np.array(R)[:,1:])

# warning: below function should be used during training only, eg agent(observ) returns Chainer tensor
def evaluate_on_diff_env(env, n_sample_traj, agent):
    # this function is used to sample n traj using GAN version of environment
    R = 0.0

    # reset environment
    env.reset_state()
    agent.reset_state()

    # get initial observation
    observations = env(tv(np.zeros((n_sample_traj, env.act_size))))[:, :-2]

    for i in range(env.spec.timestep_limit):
        act = agent(observations)
        obs_rew = env(act)
        rewards = obs_rew[:, -2]
        ends = obs_rew[:, -1]
        observations = obs_rew[:, :-2]
        R += F.sum(rewards * (1.0 - ends)) / (-len(rewards) * env.spec.timestep_limit)

    return R


def train_gan_rl(CreateGenerator, CreateDiscriminator, CreateActor,
                 Environment,
                 project_folder,
                 noise_decay_schedule,
                 N_real_samples, N_GAN_batches, N_GAN_samples,
                 GAN_tr_lr, GAN_tr_mm, GAN_training_iter,
                 evaluation_only):

    """

    This function performs reinforcement learning of an actor on environment using
    combination of generative adversarial neural network model or real environment
    and gradient descent.

    :param CreateGenerator:
    :param CreateDiscriminator:
    :param CreateActor:
    :param Environment:
    :param project_folder:
    :param noise_decay_schedule:
    :param N_real_samples:
    :param N_GAN_batches:
    :param N_GAN_samples:
    :param GAN_tr_lr:
    :param GAN_tr_mm:
    :param GAN_training_iter:
    :param evaluation_only:
    :return:
    """

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
        agent = CreateActor()

    optA = optimizers.Adam(alpha=0.001, beta1=0.9)
    optA.setup(agent)

    # main training loop
    for noise_p in noise_decay_schedule:

        idx += 1

        envs, perf = {}, {}

        # load environments if already trained some
        for fname in files:
            if not os.path.exists(fname):
                envs[fname] = {'G': CreateGenerator(), 'X': None, 'Y': None}
            else:
                envs[fname] = pc.load(open(fname))

        # set temporarily noise probability != 0
        agent.noise_probability = noise_p

        # fit differentiable environment for training and testing
        for fname in files:
            X, Y, Rmean = generate_data(agent, Environment, N_real_samples)

            perf[fname] = float(Rmean)  # shows the performance of agent on real environment

            if not evaluation_only:

                # extend the data
                if envs[fname]['X'] is None:
                    envs[fname]['X'] = X
                    envs[fname]['Y'] = Y
                else:
                    for i in range(Environment.spec.timestep_limit):
                        for elm, v in [('X', X), ('Y', Y)]:
                            envs[fname][elm][i] = np.concatenate([envs[fname][elm][i], v[i]])

                FitStochastic(
                    envs[fname]['G'],
                    CreateDiscriminator(),
                    (envs[fname]['X'], envs[fname]['Y']),
                    GAN_tr_lr,
                    GAN_tr_mm,
                    GAN_training_iter,
                    "Example generated [state_0, state_1, ..., reward, finished] sequences:")

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