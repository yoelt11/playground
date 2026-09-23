import numpy as np
import torch
from typing import Tuple, List

class Poisson2D:
    """2D Poisson problem on (0,1)^2 with manufactured solution."""
    def __init__(self, resolution=50):
        self.res = resolution
        self.x = np.linspace(0, 1, resolution)
        self.y = np.linspace(0, 1, resolution)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.grid_points = np.stack([self.X.ravel(), self.Y.ravel()], axis=-1)

    def u_exact(self, points):
        x, y = points[:, 0], points[:, 1]
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def f_exact(self, points):
        x, y = points[:, 0], points[:, 1]
        return 2 * (np.pi**2) * np.sin(np.pi * x) * np.sin(np.pi * y)

    def get_boundary_points(self, num_pts=100):
        pts = []
        for i in range(num_pts):
            pts.append([i/num_pts, 0])
            pts.append([i/num_pts, 1])
            pts.append([0, i/num_pts])
            pts.append([1, i/num_pts])
        return np.array(pts)

def gaussian_rbf(r, epsilon=1.0):
    return np.exp(-(epsilon * r)**2)

def laplacian_gaussian_rbf(r_vec, epsilon=1.0):
    """Laplacian of Gaussian RBF in 2D: Delta(exp(-eps^2 r^2))"""
    r2 = np.sum(r_vec**2, axis=-1)
    a = epsilon**2
    return (4 * (a**2) * r2 - 4 * a) * np.exp(-a * r2)

class RBFKansaSolver:
    def __init__(self, epsilon=1.5):
        self.epsilon = epsilon
        self.centers = None
        self.weights = None

    def fit(self, centers, f_vals, boundary_pts, u_bnd_vals):
        self.centers = centers
        K = len(centers)
        B = len(boundary_pts)
        
        # To fix the garbage baseline, we move from a simple Least Squares
        # over (K+B) points to a strictly constrained solve.
        # We use K centers. We need K equations.
        # We use a subset of boundary points as strict constraints.
        
        # 1. Build the physics operator matrix A (K x K): L = -Delta
        # (Bugfix: previously used +Delta, which solved Delta u = f and
        # recovered approximately -u_exact for this manufactured problem.)
        A = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                dist_vec = centers[i] - centers[j]
                A[i, j] = -laplacian_gaussian_rbf(dist_vec, self.epsilon)
        
        # 2. Build the Boundary matrix B_mat (B x K)
        B_mat = np.zeros((B, K))
        for i in range(B):
            for j in range(K):
                dist = np.linalg.norm(boundary_pts[i] - centers[j])
                B_mat[i, j] = gaussian_rbf(dist, self.epsilon)
        
        # 3. Use a weighted system to enforce BCs strongly
        # Weight for BCs is typically much higher than interior.
        lambda_b = 1e4 
        
        M = np.zeros((K + B, K))
        rhs = np.zeros(K + B)
        
        # Interior: L(u) = f
        for i in range(K):
            for j in range(K):
                M[i, j] = A[i, j]
            rhs[i] = f_vals[i]
            
        # Boundary: u = u_bnd (weighted)
        for i in range(B):
            for j in range(K):
                M[K + i, j] = lambda_b * B_mat[i, j]
            rhs[K + i] = lambda_b * u_bnd_vals[i]
            
        self.weights = np.linalg.lstsq(M, rhs, rcond=None)[0]

    def predict(self, points):
        if self.weights is None:
            raise ValueError("Solver not fitted")
        
        preds = []
        for p in points:
            dists = np.linalg.norm(self.centers - p, axis=1)
            val = np.sum(self.weights * gaussian_rbf(dists, self.epsilon))
            preds.append(val)
        return np.array(preds)

    def compute_residual(self, points, f_vals):
        res_signed = []
        for i in range(len(points)):
            p = points[i]
            lap_val = 0
            for j in range(len(self.centers)):
                dist_vec = p - self.centers[j]
                lap_val += self.weights[j] * laplacian_gaussian_rbf(dist_vec, self.epsilon)
            # R = -Delta(u) - f
            res_signed.append(-lap_val - f_vals[i])
        return np.array(res_signed)
