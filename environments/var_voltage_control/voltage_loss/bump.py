import numpy as np



def bump_loss(vs):
    def _bump_loss(v):
        if np.abs(v) < 1:
            return np.exp( - 1 / (1 - v**4) )
        elif 1 < v < 3:
            return np.exp( - 1 / (1 - ( v - 2 )**4 ) )
        else:
            return 0.0
    return np.array([_bump_loss(v) for v in vs])