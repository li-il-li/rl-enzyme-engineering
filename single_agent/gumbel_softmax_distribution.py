import torch
import torch.nn.functional as F
from torch.distributions import Distribution

class GumbelSoftmaxDistribution(Distribution):
    
    arg_constraints = {}

    def __init__(self, logits, temperature=1.0):
        super().__init__()
        self.logits = logits
        self.temperature = temperature

    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.logits)))
        y_soft = F.softmax((self.logits + gumbel_noise) / self.temperature, dim=-1)
        
        # Straight-through estimator
        index = y_soft.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(self.logits).scatter_(-1, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        
        return ret

    def log_prob(self, value):
        # Compute log probability using softmax
        return (F.log_softmax(self.logits, dim=-1) * value).sum(-1)

    def entropy(self):
        return -(F.softmax(self.logits, dim=-1) * F.log_softmax(self.logits, dim=-1)).sum(-1)