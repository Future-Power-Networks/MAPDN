import torch as th



class ReinforcementLearning(object):
    def __init__(self, name, args):
        self.name = name
        self.args = args
        self.device = th.device( "cuda" if th.cuda.is_available() and self.args.cuda else "cpu" )

    def __str__(self):
        print (self.name)

    def __call__(self):
        raise NotImplementedError()

    def get_loss(self):
        raise NotImplementedError()
