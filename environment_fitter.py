import chainer
from chainer import Variable, optimizers, flag
from chainer import Link, Chain, ChainList
import chainer.functions as CF
import numpy as np
import cPickle as pc

### ASSUMPTION: INITIAL CONDITION IS IMPORTANT TO GET RIGHT
w_init = 4.0 # wegiht of objective for initial output of RNN ~size of sequence

def diff_to(v, c):
    return CF.mean_squared_error(v, v * 0.0 + c)

class DOBJ(Chain):
    def __init__(self):
        super(DOBJ, self).__init__()

    def __call__(self, I, A, O, R, F, D, G):

        rw = diff_to(D.reset_state(I), 1.0)*w_init

        for a, o, r, f in zip(A, O, R, F):
            rw += diff_to(D(a, o, r, f), 1.0)

        n_trj = len(I.data)
        rw += diff_to(D.reset_state(G.reset_state(n_trj)), 0.0) * w_init

        for a in A:
            o, r, f = G(a)
            rw += diff_to(D(a, o, r, f), 0.0)

        return rw


class GOBJ(Chain):
    def __init__(self):
        super(GOBJ, self).__init__()

    def __call__(self, A, D, G):
        n_trj = len(A[0].data)

        rw = diff_to(D.reset_state(G.reset_state(n_trj)), 1.0)*w_init

        for a in A:
            o, r, f = G(a)
            rw += diff_to(D(a, o, r, f), 1.0)

        return rw

def tv(x, v = flag.OFF):
    return Variable(np.array(x).astype('float32'), volatile=v)

from time import time

def FitStochastic(G, D, X, learning_rate, momentum, iters, pr_caption=None):

    I, A, O, R, F = X['I'], X['A'], X['O'], X['R'], X['F']
    I, A, O, R, F = tv(I), tv(A), tv(O), tv(R), tv(F)
    # tv it all

    objD, objG = DOBJ(), GOBJ()

    optG = optimizers.Adam(alpha=learning_rate, beta1=momentum)
    optD = optimizers.Adam(alpha=learning_rate, beta1=momentum)

    optG.setup(G)
    optD.setup(D)

    st = time()

    for i in range(iters):

        D.zerograds()
        loss = objD(I, A, O, R, F, D, G)
        loss.backward()
        optD.update()

        G.zerograds()
        loss = objG(A, D, G)
        loss.backward()
        optG.update()


        if i % 100 == 0:
            print "iter:", i, "iter. time:", time()-st

            if not pr_caption is None:
                print pr_caption

            st = time()

            def fm(v, j):
                f = np.round(v.data[j].astype(np.float64), 1).tolist()
                return f



            init_state = G.reset_state(len(A[0].data))

            Yp = [G(x) for x in A]

            for j in range(10):
                Yi = [(fm(o,j), fm(r,j), fm(f,j)) for o, r, f in Yp]
                print [fm(init_state,j)] + Yi
