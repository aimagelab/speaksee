from torch import nn

class CaptioningModel(nn.Module):
    def __init__(self):
        super(CaptioningModel, self).__init__()

    def init_weights(self):
        raise NotImplementedError

    def forward(self, *input):
        raise NotImplementedError

    def init_state(self, b_s, device):
        raise NotImplementedError

    def test(self, *input):
        raise NotImplementedError