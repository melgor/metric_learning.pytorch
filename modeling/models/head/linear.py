from torch import nn

from utils.init_methods import weights_init_kaiming
from modeling.models import registry


@registry.HEADS.register()
class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Linear, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.fc.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.fc(x)
        return x

    @property
    def weight(self):
        return self.fc.weight
