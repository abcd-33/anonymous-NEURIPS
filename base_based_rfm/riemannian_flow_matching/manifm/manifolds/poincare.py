import torch
import math
from manifm.manifolds.base import BaseManifold

class PoincareBallManifold(BaseManifold):
    """
    Class for operations on the Poincaré ball model of hyperbolic space.
    
    The Poincaré ball B^n is the set of points in R^n with norm less than 1:
    B^n = {x in R^n : ||x|| < 1}, with the Mobius addition.
    """
    
    def __init__(self, eps=1e-7):
        """Initialize the Poincaré ball manifold.
        
        Args:
            eps (float): Small constant for numerical stability
        """
        super().__init__(eps)

    def lambda_x(self, x):
        """Compute the conformal factor λ(x) = 2/(1-||x||²).
        
        Args:
            x (torch.Tensor): Point in the Poincaré ball
            
        Returns:
            torch.Tensor: Conformal factor at x
        """
        r2 = torch.sum(x * x, dim=-1, keepdim=True)  # ||x||²
        return 2.0 / (1.0 - r2).clamp_min(self.eps)

    def mobius_add(self, x, y):
        """Compute Möbius addition x ⊕ y.
        
        Args:
            x (torch.Tensor): First point in the ball
            y (torch.Tensor): Second point in the ball
            
        Returns:
            torch.Tensor: Result of Möbius addition
        """
        xy = (x * y).sum(dim=-1, keepdim=True)
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        
        numerator = (1.0 + 2.0 * xy + y2) * x + (1.0 - x2) * y
        denominator = 1.0 + 2.0 * xy + x2 * y2
        return numerator / denominator.clamp_min(self.eps)

    def mobius_neg(self, x):
        """Compute Möbius inverse (additive opposite).
        
        Args:
            x (torch.Tensor): Point in the ball
            
        Returns:
            torch.Tensor: Möbius inverse of x
        """
        return -x
    
    def gyration(self, a, b, w):
        """Compute the gyration map gyr[a,b] acting on w.
        
        Args:
            a (torch.Tensor): First point in the ball
            b (torch.Tensor): Second point in the ball
            w (torch.Tensor): Vector to gyrate
            
        Returns:
            torch.Tensor: Result of gyration
        """
        ab = self.mobius_add(a, b)
        bw = self.mobius_add(b, w)
        a_bw = self.mobius_add(a, bw)
        return self.mobius_add(self.mobius_neg(ab), a_bw)

    def project_to_tangent(self, x, v):
        """Project vector v onto the tangent space at x.
        
        In the Poincaré ball, the entire R^n is tangent space at x,
        so we don't need an orthogonal projection.
        
        Args:
            x (torch.Tensor): Point in the ball
            v (torch.Tensor): Vector to project
            
        Returns:
            torch.Tensor: Projected vector
        """
        return v

    def exp_map(self, x0, v):
        """Compute the exponential map at x0 in direction v.
        
        Args:
            x0 (torch.Tensor): Base point in the ball
            v (torch.Tensor): Tangent vector at x0
            
        Returns:
            torch.Tensor: Point reached by following the geodesic
        """
        lam = self.lambda_x(x0)
        norm_v = torch.norm(v, dim=-1, keepdim=True)
        half_r = 0.5 * lam * norm_v
        direction = v / (norm_v + self.eps)
        inside = torch.tanh(half_r) * direction
        return self.mobius_add(x0, inside)

    def log_map(self, x0, x1, tol=1e-6):
        """Compute the logarithmic map from x0 to x1.
        
        Args:
            x0 (torch.Tensor): Base point in the ball
            x1 (torch.Tensor): Target point in the ball
            tol (float): Tolerance for numerical stability
            
        Returns:
            torch.Tensor: Tangent vector v at x0 such that exp_{x0}(v) = x1
        """
        y = self.mobius_add(-x0, x1)
        norm_y = torch.norm(y, dim=-1, keepdim=True)
        lam = self.lambda_x(x0)
        r = norm_y.clamp(max=1 - self.eps)
        
        # Compute atanh(r)/r safely
        ratio_exact = torch.atanh(r) / r
        ratio_series = 1.0 + (r**2)/3.0 + (r**4)/5.0 
        ratio = torch.where(r < tol, ratio_series, ratio_exact)
        
        factor = (2.0 / lam) * ratio
        return factor * y

    def wrap(self, samples):
        """Map points from R^n to the Poincaré ball.
        
        Args:
            samples (torch.Tensor): Points in R^n
            
        Returns:
            torch.Tensor: Points in the Poincaré ball
        """
        x0 = torch.zeros_like(samples)
        return self.exp_map(x0, samples)

    def unwrap(self, points):
        """Map points from the Poincaré ball to R^n.
        
        Args:
            points (torch.Tensor): Points in the Poincaré ball
            
        Returns:
            torch.Tensor: Points in R^n
        """
        x0 = torch.zeros_like(points)
        return self.log_map(x0, points)

    def geodesic(self, x0, x1, t):
        """Compute geodesic from x0 to x1 parametrized by t in [0,1].
        
        Args:
            x0 (torch.Tensor): Start point in the ball
            x1 (torch.Tensor): End point in the ball
            t (torch.Tensor): Parameter in [0,1]
            
        Returns:
            torch.Tensor: Point along the geodesic at parameter t
        """
        v = self.log_map(x0, x1)
        return self.exp_map(x0, t * v)

    def geodesic_velocity(self, x0, x1, t):
        """Compute velocity of geodesic at time t.
        
        Args:
            x0 (torch.Tensor): Start point in the ball
            x1 (torch.Tensor): End point in the ball
            t (torch.Tensor): Parameter in [0,1]
            
        Returns:
            torch.Tensor: Velocity vector at time t
        """
        x_t = self.geodesic(x0, x1, t)
        v_t = self.log_map(x_t, x1)
        delta = 1e-2
        return v_t / (1.0 - t + delta)

    def sample(self, batch_size, dim=2, device="cpu"):
        """Sample points uniformly in the Euclidean sense in the Poincaré ball.
        
        Args:
            batch_size (int): Number of points to sample
            dim (int): Dimension of the ball
            device (torch.device): Device to place the tensor on
            
        Returns:
            torch.Tensor: Uniformly sampled points in the ball
        """
        # Sample directions uniformly on the sphere
        z = torch.randn(batch_size, dim, device=device)
        z_norm = z.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        z_unit = z / z_norm

        # Sample radii uniformly in [0,1]
        r = torch.rand(batch_size, 1, device=device).pow(1.0 / dim)
        return r * z_unit

    def distance(self, x, y, eps=1e-15):
        """Compute the hyperbolic distance using the atanh formula.
        Uses a hybrid approach:
        - For small distances (dist = ||x-y||/sqrt((1-||x||^2)(1-||y||^2)) < 1 - eps),
        uses 2*atanh(x) = log1p(x) - log1p(-x) for high precision.
        - For larger distances or x >= 1 - eps, uses acosh(1 + 2*||x-y||^2/denom)
        to handle distances beyond the atanh domain.

        Args:
            x (Tensor): Tensor of shape (..., dim) with ||x|| < 1.
            y (Tensor): Tensor of shape (..., dim) with ||y|| < 1.
            eps (float): Small threshold for switching and numerical stability.
        Returns:
            Tensor of shape (...) giving the hyperbolic distances.
        """
        sq_norm_x = torch.sum(x * x, dim=-1) 
        sq_norm_y = torch.sum(y * y, dim=-1)
        sq_diff = torch.sum((x - y) ** 2, dim=-1)
        denom = (1 - sq_norm_x) * (1 - sq_norm_y)
        denom = torch.clamp(denom, min=eps)

        # mask for different formula
        r = torch.sqrt( sq_diff / (denom + sq_diff) )
        dist = torch.empty_like(r)

        # stable small‐distance branch:
        small_mask = r < (1 - eps)
        r_small = r[small_mask]
        dist[small_mask] = torch.log1p(r_small) - torch.log1p(-r_small)

        # fallback acosh branch
        cosh_arg = 1 + 2 * sq_diff / denom
        cosh_arg = torch.clamp(cosh_arg, min=1.0)
        dist[~small_mask] = torch.acosh(cosh_arg[~small_mask])

        return dist
        

# For backward compatibility
poincare = PoincareBallManifold()
wrap_poincare = poincare.wrap
unwrap_poincare = poincare.unwrap
project_to_tangent = poincare.project_to_tangent
geodesic_on_poincare = poincare.geodesic
geodesic_velocity = poincare.geodesic_velocity
log_map_poincare = poincare.log_map
exp_map_poincare = poincare.exp_map
sample_poincare = poincare.sample

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data.toys import CheckerboardEuclidean
    import torch.utils.data as data
    import numpy as np

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize Poincaré ball manifold
    poincare = PoincareBallManifold()

    # 1. Test wrap/unwrap identity
    print("\n1. Testing wrap/unwrap identity...")
    # Create a grid of points in R^2
    x = torch.linspace(-2, 2, 20)
    y = torch.linspace(-2, 2, 20)
    X, Y = torch.meshgrid(x, y)
    points_2d = torch.stack([X.flatten(), Y.flatten()], dim=1)

    # Wrap to Poincaré ball and unwrap back
    points_hyp = poincare.wrap(points_2d)
    points_2d_back = poincare.unwrap(points_hyp)

    # Compute error
    error = torch.norm(points_2d - points_2d_back, dim=-1).mean()
    print(f"Mean error after wrap->unwrap: {error:.6f}")

    # 2. Test checkerboard mapping
    print("\n2. Testing checkerboard mapping...")
    # Create checkerboard dataset
    dataset = CheckerboardEuclidean(dataset_size=5000)
    dataloader = data.DataLoader(dataset, batch_size=5000)
    for batch in dataloader:
        checkerboard_points = batch
        break

    # Map checkerboard to Poincaré ball
    checkerboard_hyp = poincare.wrap(checkerboard_points)
    checkerboard_back = poincare.unwrap(checkerboard_hyp)

    # Compute error
    error = torch.norm(checkerboard_points - checkerboard_back, dim=-1).mean()
    print(f"Mean error for checkerboard wrap->unwrap: {error:.6f}")

    # 3. Visualizations
    print("\n3. Creating visualizations...")
    
    # 3.1 Plot original grid points and their wrap->unwrap
    plt.figure(figsize=(12, 5))
    
    # Original grid
    plt.subplot(121)
    plt.scatter(points_2d[:, 0], points_2d[:, 1], c='blue', s=10, label='Original')
    plt.scatter(points_2d_back[:, 0], points_2d_back[:, 1], c='red', s=5, label='Wrap->Unwrap')
    plt.title('Grid Points in R²')
    plt.legend()
    plt.grid(True)
    
    # Checkerboard
    plt.subplot(122)
    plt.scatter(checkerboard_points[:, 0], checkerboard_points[:, 1], c='blue', s=10, label='Original')
    plt.scatter(checkerboard_back[:, 0], checkerboard_back[:, 1], c='red', s=5, label='Wrap->Unwrap')
    plt.title('Checkerboard Points in R²')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3.2 Plot points in Poincaré disk
    plt.figure(figsize=(12, 5))
    
    # Grid points in Poincaré disk
    plt.subplot(121)
    plt.scatter(points_hyp[:, 0], points_hyp[:, 1], c='blue', s=10)
    plt.title('Grid Points in Poincaré Disk')
    plt.grid(True)
    circle = plt.Circle((0, 0), 1, fill=False, color='black')
    plt.gca().add_artist(circle)
    plt.axis('equal')
    
    # Checkerboard points in Poincaré disk
    plt.subplot(122)
    plt.scatter(checkerboard_hyp[:, 0], checkerboard_hyp[:, 1], c='blue', s=10)
    plt.title('Checkerboard Points in Poincaré Disk')
    plt.grid(True)
    circle = plt.Circle((0, 0), 1, fill=False, color='black')
    plt.gca().add_artist(circle)
    plt.axis('equal')
    plt.show()

