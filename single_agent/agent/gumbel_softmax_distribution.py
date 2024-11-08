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
    

class GumbelTopKDistribution(Distribution):
    arg_constraints = {}
    
    def __init__(self, logits, k=1, temperature=1.0):
        """
        Args:
            logits: Unnormalized log probabilities
            k: Number of top categories to sample
            temperature: Temperature parameter for Gumbel-Softmax
        """
        super().__init__()
        self.logits = logits
        self.k = k
        self.temperature = temperature
        
    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)
    
    def rsample(self, sample_shape=torch.Size()):
        # Generate Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.logits)))
        perturbed_logits = (self.logits + gumbel_noise) / self.temperature
        
        # Get top-k values and indices
        values, indices = torch.topk(perturbed_logits, k=self.k, dim=-1)
        
        # Create a zero tensor and scatter the top-k values
        y_soft = torch.zeros_like(self.logits)
        y_soft.scatter_(-1, indices, F.softmax(values, dim=-1))
        
        # Create hard one-hot vectors for top-k
        y_hard = torch.zeros_like(self.logits)
        y_hard.scatter_(-1, indices, 1.0 / self.k)  # Uniform distribution over top-k
        
        # Straight-through estimator
        ret = y_hard - y_soft.detach() + y_soft
        return ret
    
    def log_prob(self, value):
        """
        Compute log probability for the selected top-k elements.
        For simplicity, we assume uniform probability distribution over the selected elements.
        """
        # Get the indices of non-zero elements in value
        selected_indices = torch.nonzero(value, as_tuple=True)
        
        # Get the corresponding logits
        selected_logits = self.logits[selected_indices]
        
        # Compute log probability as mean of selected logits
        # normalized by the partition function
        log_partition = torch.logsumexp(self.logits, dim=-1)
        return (selected_logits.sum(-1) / self.k) - log_partition
    
    def entropy(self):
        """
        Compute entropy of the top-k distribution.
        This is an approximation based on softmax probabilities.
        """
        probs = F.softmax(self.logits, dim=-1)
        log_probs = F.log_softmax(self.logits, dim=-1)
        
        # Sort probabilities in descending order
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        sorted_log_probs, _ = torch.sort(log_probs, dim=-1, descending=True)
        
        # Take top-k probabilities and compute their entropy
        topk_probs = sorted_probs[..., :self.k]
        topk_log_probs = sorted_log_probs[..., :self.k]
        
        return -(topk_probs * topk_log_probs).sum(-1)