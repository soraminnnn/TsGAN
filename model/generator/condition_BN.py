import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class ConditionalBatchNorm1d(nn.BatchNorm1d):
    """Conditional Batch Normalization"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(ConditionalBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input, weight, bias, **kwargs):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(input, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()
        weight = weight.unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).expand(size)
        return weight * output + bias

class LinearConditionalBatchNorm1d(ConditionalBatchNorm1d):
    def __init__(self, conditionDim, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(LinearConditionalBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Linear(conditionDim,num_features)
        self.biases = nn.Linear(conditionDim,num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, c, **kwargs):

        weight = self.weights(c)
        bias = self.biases(c)

        return super(LinearConditionalBatchNorm1d, self).forward(
                     input, weight, bias)

class CategoricalConditionalBatchNorm1d(ConditionalBatchNorm1d):
    def __init__(self, num_classes, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(CategoricalConditionalBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, c, **kwargs):
        weight = self.weights(c)
        bias = self.biases(c)
        return super(CategoricalConditionalBatchNorm1d, self).forward(input, weight, bias)



