"""Copyright (c) Meta Platforms, Inc. and affiliates."""
import math
import torch
from torch.func import vmap
from geoopt.manifolds import PoincareBall as geoopt_PoincareBall


class PoincareBall(geoopt_PoincareBall):


    def metric_normalized(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Normalizes a vector U on the tangent space of X according to G^{-1/2}U."""
        return u / self.lambda_x(x, keepdim=True)

    def random_base(self,batch_size, dim, std = 0.03):
        """Sample points from the Normal distribution on the Poincare ball centered in the origin.
        
        Args:
            batch_size (int): Number of points to sample
            dim (int): Dimension of the ball            
        Returns:
            torch.Tensor: sampled points in the ball [batch_size, dim]
        """
        # Sample directions uniformly on the sphere
        #z = torch.randn(batch_size, dim) 
        #z_norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-7)
        #z_unit = z / z_norm
        
        # Sample radii uniformly in [0,1]
        #r = torch.rand(batch_size, 1).pow(1.0 / dim)
        #return r * z_unit

        # Gaussian sampling 
        mean = torch.zeros(dim, device=self.k.device)
        #std = 0.3

        samples = []
        for _ in range(batch_size):
            x = self.wrapped_normal(dim, mean=mean, std=std)
            samples.append(x)

        return torch.stack(samples, dim=0)
        

    def base_logprob(self, x, std=0.3):
        #raise NotImplementedError
        """ 
        Compute the log-probability of points X under the wrapped distribution
        log p(x) = -d/2 * log(2*pi*sigma^2) - d_p(x0, x)^2/(2*sigma^2) + (d-1) * (log(d_p(x0, x)) - log(sinh(d_p(x_0, x))))
        """
        d = x.shape[-1]
        device = x.device
        dtype = x.dtype
        const = torch.log(2 * torch.pi * torch.tensor(std * std, device=device, dtype=dtype))
        mean = torch.zeros_like(x)
        dist = self.dist(mean, x)
        logprob = -d/2*const - dist**2/(2*std*std) +(d-1)*(torch.log(dist + 1e-9) - torch.log(torch.sinh(dist)+ 1e-9))      
        return logprob

    

