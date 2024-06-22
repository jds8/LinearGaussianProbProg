#!/usr/bin/env python3

from typing import Optional, Tuple, List

import torch


class Node:
    def __init__(
            self,
            id: int,
            dim: int,
            parent1: Optional,
            parent2: Optional,
    ):
        self.id = id
        self.dim = dim
        self.parent1 = parent1
        self.parent2 = parent2


class GraphicalModel(dict):
    def __init__(self, *args, **kwargs):
        """
        Store dictionary of {Gaussian: Node}
        """
        super().__init__(*args, **kwargs)
        self.next_id = 0

    def register_gaussian(
            self,
            gaussian,
            parent1: Optional = None,
            parent2: Optional = None,
            register: bool = True,
    ):
        if register:
            dim = gaussian.mean.shape[0]
            self[gaussian] = Node(self.next_id, dim, parent1, parent2)
            self.next_id += 1

    def contains(self, fs: List, g):
        return any([self[f].id == self[g].id for f in fs])

    def get_offsets(self, prior, likelihood) -> int:
        """
        Given p(y|x,z)p(x|z), find offset of prior x and z
        into likelihood's [x,z]
        """
        x_dim = prior.mean.shape[0]  # N

        assert isinstance(likelihood, ConditionalGaussian), \
        f"Likelihood term should be conditional"
        if self[likelihood.lt.x].id == self[prior].id or \
           self[prior].id == self[likelihood].parent1.id:
            return 0, x_dim
        return x_dim, 0

        # What if z not in likelihood?
        # This *should* only be the case when z is
        # a parent of x as this means conditional ind.
        # There, we find the path to the
        # most recent ancestor. If that path contains
        # x, then we don't need z.
        # self.path_to_most_recent_ancestor(prior, prior.x)


class Gaussian:
    def __init__(
        self,
        gm: GraphicalModel,
        mean: torch.Tensor,
        covariance: torch.Tensor,
        *args,
        **kwargs,
    ):
        self.mean = mean
        self.covariance = covariance
        self.conditional_variance = covariance
        gm.register_gaussian(self, **kwargs)


class LinearTransformation(Gaussian):
    def __init__(
        self,
        gm: GraphicalModel,
        A: torch.Tensor,
        b: torch.Tensor,
        x: Gaussian,
        *args,
        **kwargs,
    ):
        unconditional_mean = torch.mm(A, x.mean) + b
        unconditional_covariance = torch.mm(A, torch.mm(x.covariance, A.T))
        super().__init__(
            gm=gm,
            mean=unconditional_mean,
            covariance=unconditional_covariance,
            parent1=x,
            **kwargs,
        )
        self.A = A
        self.b = b
        self.x = x

    def get_wrt(self, y: Gaussian):
        if y == self.x:
            return (self.A, self.b)
        assert hasattr(self, 'x'), "Error: argument to get_wrt is independent of self"
        C, d = self.x.get_wrt(y)
        return torch.mm(self.A, C), torch.mm(self.A, d) + self.b

    def get_transform_wrt(self, gm: GraphicalModel, y: Gaussian):
        if y == self.x:
            return self
        C, d = self.get_wrt(y)
        linear_transform = LinearTransformation(gm, C, d, y, register=False)
        return linear_transform

    def conditional_mean(self, y: torch.tensor):
        return torch.mm(self.A, y) + self.b


class ConditionalGaussian(Gaussian):
    def __init__(
        self,
        gm: GraphicalModel,
        lt: LinearTransformation,
        covariance: torch.Tensor,
        *args,
        **kwargs
    ):
        super().__init__(
            gm=gm,
            mean=lt.mean,
            covariance=lt.covariance + covariance,
            parent1=lt,
            **kwargs,
        )
        self.conditional_variance = covariance
        self.lt = lt

    def get_wrt(self, y: Gaussian):
        return self.lt.get_wrt(y)

    def conditioned_on(self, prior: Gaussian):
        # prior: p(x|z)=N(Az+d,Q)
        # likelihood: p(y∣x,z)=N(C[x z]+b,R)
        assert gm.contains([self.lt.x], prior) or gm.contains([prior, prior.x], self.lt.x)

        # assume y is dim Dx1
        # assume x is dim Nx1
        # assume z is dim Mx1
        y_dim = self.lt.A.shape[0]  # D
        x_dim = prior.mean.shape[0]  # N

        # gets the indices of self.likelihood.x corresponding to prior
        x_offset, z_offset = gm.get_offsets(prior, self)

        # 1/Σ
        C = self.lt.A[:, x_offset:x_offset+x_dim]  # DxN
        R_inv = self.conditional_variance.pinverse()  # DxD
        CRC = torch.mm(torch.mm(C.T, R_inv), C)  # NxN
        Q_inv = prior.conditional_variance.pinverse()  # NxN
        precision = CRC + Q_inv  # NxN
        # Σ
        covariance = precision.pinverse()  # NxN

        y_scale = torch.mm(self.lt.A.T[x_offset:x_offset+x_dim, :], R_inv)  # NxD
        y_shift = -torch.mm(y_scale, self.lt.b)  # Nx1
        # multiple dispatch would be helpful here
        if isinstance(prior, ConditionalGaussian):
            z_dim = prior.lt.x.mean.shape[0]  # M
            dim = y_dim + z_dim  # D+M

            z_shift = torch.mm(Q_inv, prior.lt.b)  # Nx1
            zx_scale_prior = torch.mm(Q_inv, prior.lt.A)  # NxM

            # if xz have an interaction, i.e. if a path
            # from x to z goes through y, then compute
            # xz interaction from x^TC^TR^{-1}Cz
            zx_scale_lik = torch.mm(
                self.lt.A.T[x_offset:x_offset+x_dim, :],  # NxD
                torch.mm(
                    R_inv,  # DxD
                    self.lt.A[:, z_offset:z_offset+z_dim]  # DxM
                )
            )  # NxM

            z_scale = zx_scale_prior  # NxM
            if zx_scale_lik.nelement():
                z_scale += zx_scale_lik  # NxM

            yz_scale = torch.zeros(y_dim, dim)  # Nx(D+M)
            yz_scale[:, :y_dim] = y_scale  # add NxD part
            yz_scale[:, y_dim:] = z_scale  # add NxM part
            yz_shift = y_shift + z_shift  # Nx1

            # create new Gaussian representing the
            # vector concatenation of
            # prior.x and likelihood
            g = ConcatenatedGaussian(gm, self, prior.lt.x)
            temp = LinearTransformation(gm, A=yz_scale, b=yz_shift, x=g, register=False)
        else:
            # note sure if I should create a new variable here as likelihood
            # still depends on likelihood.x here which is now on the LHS
            # of the condition bar...
            temp = LinearTransformation(gm, A=y_scale, b=y_shift, x=self, register=False)

        # posterior_mean=μ=Σ(CR^{-1}(y-b)+Q^{−1}(Az+d))=Σ*temp
        posterior_mean_wrt_temp = LinearTransformation(
            gm,
            A=covariance,
            b=0,
            x=temp,
            register=False
        )
        posterior_mean = posterior_mean_wrt_temp.get_transform_wrt(
            gm,
            temp.x
        )
        output = ConditionalGaussian(gm, posterior_mean, covariance)
        return output

    def conditional_mean(self, y: torch.Tensor):
        return self.lt.conditional_mean(y)


class ConcatenatedGaussian(Gaussian):
    def __init__(
        self,
        gm: GraphicalModel,
        f: ConditionalGaussian,
        g: Gaussian,
        *args,
        **kwargs,
    ):
        self.f_dim = f.mean.shape[0]
        self.g_dim = g.mean.shape[0]
        mean = torch.concat([f.mean, g.mean])
        covariance = self.compute_covariance(f, g)
        super().__init__(
            gm,
            mean,
            covariance,
            parent1=f,
            parent2=g,
        )

    def compute_covariance(self, f: ConditionalGaussian, g: Gaussian):
        variance = torch.zeros(self.f_dim + self.g_dim, self.f_dim + self.g_dim)
        variance[:self.f_dim, :self.f_dim] = f.covariance
        variance[self.f_dim:, self.f_dim:] = g.covariance

        transform = f.lt.get_transform_wrt(gm, g)
        fg_covariance = torch.mm(transform.A, g.covariance)

        variance[:self.f_dim, self.f_dim:] = fg_covariance
        variance[self.f_dim:, :self.f_dim] = fg_covariance
        return variance


if __name__ == '__main__':
    gm = GraphicalModel()

    A = torch.tensor([[0.8990]])
    Q = torch.tensor([[1.2858]])
    C = torch.tensor([[0.6941]])
    R = torch.tensor([[0.3740]])
    b = torch.tensor([[0.]])

    y1 = torch.tensor([-0.6240])
    x0 = torch.tensor([-0.1485])

    xt = Gaussian(gm, torch.zeros(1, 1), Q)
    xts = [xt]
    yts = []
    for i in range(10):
        yt = ConditionalGaussian(gm, LinearTransformation(gm, C, b, xt), R)
        yts.append(yt)
        xt = ConditionalGaussian(gm, LinearTransformation(gm, A, b, xt), Q)
        xts.append(xt)
    x_given_y = yts[1].conditioned_on(xts[1])
    obs = torch.stack([y1, x0])
    print(x_given_y.conditional_mean(obs), x_given_y.conditional_variance)

    # Expected:
    # mean: tensor([-0.6108])
    # cov: tensor([[0.4841]])
