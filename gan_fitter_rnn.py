import chainer
from chainer import Variable, optimizers, flag
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import numpy as np
import cPickle as pc

### ASSUMPTION: INITIAL CONDITION IS IMPORTANT TO GET RIGHT
w_init = 100.0 # wegiht of objective for initial output of RNN

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


        if i % 100 == 0:
            print "iter:", i, "iter. time:", time()-st

            if not pr_caption is None:
                print pr_caption

            st = time()

            G.reset_state()

            Yp = [G(x) for x in X]

            for j in range(10):

                Yi = [np.round(yp.data[j].astype('float64'), 1).tolist() for yp in Yp]

                print Yi

            res = np.zeros((10, 10))

            for rep in range(10):
                for mrep in range(10):

                    G.reset_state()

                    v = []

                    for x, yt in zip(X, Y):
                        yp = G(x)
                        v.append(np.mean(abs(yt.data - yp.data)))

                    res[rep, mrep] = np.mean(v)

            tr_perf = np.mean(np.min(res, axis=1))
            print "Error estimate:", tr_perf


