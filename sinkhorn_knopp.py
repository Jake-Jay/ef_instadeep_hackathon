import torch
import torch.nn as nn

class SinkhornKnopp(nn.Module):
    """
    Sinkhorn-Knopp estimation of optimal transport plans.
    
    Compute optimal transport `T : [N, M]` between two point clouds
    `x : [N, d]` and `y : [M, d]`. 
    """

    def __init__(self, temp=.2, n_it=10, p=1):
        super().__init__()
        self.temp = temp
        self.n_it = n_it
        self.p = p

    def forward(self, cdist, A=None, B=None):
        """ 
        Optimal transport plan sending A to a neighbour of B.  

        The Gibbs kernel `Q : [N, M]` is defined by exponentiating the cross-
        distance matrix `cdist`, scaled to a standard deviation 
        proportional to the inverse temperature parameter.
        
            Q = torch.exp(-cdist / temp * cdist.std())

        In order to match the source and target single-point densities 
        `A : [N]` and `B : [M]`, Lagrange multipliers `U` and `V` 
        (left and right single-point densities) are iterated upon. 
        
        The returned transport plan `T = U[:,None] * Q * V` matches `A` 
        and almost `B` (both defaulting to uniform distributions on N and M).
        """
        temp, n_it = self.temp, self.n_it
        
        #--- Boundary conditions
        N, M = cdist.shape
        if isinstance(A, type(None)): A = torch.ones(N) / N
        if isinstance(B, type(None)): B = torch.ones(M) / M

        #--- Gibbs density
        C = (cdist - cdist.min()) / (cdist.std())
        C = C ** self.p if not self.p == 1 else C
        Q = torch.exp(- C / temp)
        Q = Q / Q.sum()

        #--- Sinkhorn-Knopp --- 
        with torch.no_grad():
            T = 0. + Q.detach()
            U, V = torch.ones(N), torch.ones(M)
            for i in range(n_it):
                V *= (B / T.sum([0]))
                T = U[:,None] * Q * V
                U *= (A / T.sum([1]))
                T = U[:,None] * Q * V       
        return T
        