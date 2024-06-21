#!/usr/bin/env python3

from typing import Optional, Tuple, List

import torch


class Node:
    def __init__(self, id: int, parents: Optional[List]):
        self.id = id
        self.parents = parents


class GraphicalModel(dict):
    def __init__(self, *args, **kwargs):
        """
        Store dictionary of {Gaussian: Node}
        """
        super().__init__(*args, **kwargs)
        self.next_id = 0
        self.id_to_gaussian = []  # also map from id to Gaussian

    def register_gaussian(
            self,
            gaussian,
            parents: Optional = None,
            register: bool = True,
    ):
        if register:
            parent_nodes = [self[parent] for parent in parents] if parents else None
            self[gaussian] = Node(self.next_id, parent_nodes)
            self.id_to_gaussian[self.next_id] = gaussian
            self.next_id += 1

    def get_indices(self, prior, likelihood):
        pass


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
        assert A.shape[0] == A.shape[1]
        unconditional_mean = torch.mm(A, x.mean) + b
        unconditional_covariance = torch.mm(A, torch.mm(x.covariance, A.T))
        super().__init__(
            gm=gm,
            mean=unconditional_mean,
            covariance=unconditional_covariance,
            parents=(x,),
            **kwargs,
        )
        self.A = A
        self.b = b
        self.x = x

    def get_wrt(self, y: Gaussian):
        if y == self.x:
            return (self.A, self.b)
        assert isinstance(self.x, LinearTransformation), "Error: argument to get_A_wrt is independent of self"
        C, d = self.x.get_wrt(y)
        return torch.mm(self.A, C), torch.mm(self.A, d) + self.b

    def get_transform_wrt(self, gm: GraphicalModel, y: Gaussian):
        if y == self.x:
            return self
        C, d = self.get_wrt(y)
        linear_transform = LinearTransformation(gm, C, d, y, register=False)
        return linear_transform


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
            parents=(lt,),
            **kwargs,
        )
        self.conditional_variance = covariance
        self.lt = lt


class ConcatenatedGaussian(Gaussian):
    def __init__(
        self,
        gm: GraphicalModel,
        f: ConditionalGaussian,
        g: Gaussian,
        *args,
        **kwargs,
    ):
        mean = torch.mean([f.mean, g.mean])
        covariance = self.compute_covariance(f, g)
        super().__init__(
            gm,
            mean,
            covariance,
            parents=(f, g)
        )
        self.f_dim = f.mean.shape[0]
        self.g_dim = g.mean.shape[0]

    def compute_covariance(self, f: ConditionalGaussian, g: Gaussian):
        variance = torch.zeros(self.f_dim + self.g_dim, self.f_dim + self.g_dim)
        variance[:self.f_dim, :self.f_dim] = f.covariance
        variance[self.f_dim:, self.f_dim:] = g.covariance

        transform = f.get_transform_wrt(g)
        fg_covariance = torch.mm(transform.A, g.covariance)

        variance[:self.f_dim, self.f_dim:] = fg_covariance
        variance[self.f_dim:, :self.f_dim] = fg_covariance
        return variance


def posterior(
    gm: GraphicalModel,
    prior: Gaussian,
    likelihood: ConditionalGaussian
) -> ConditionalGaussian:
    # prior: p(x|z)=N(Ax+d,Q)
    # likelihood: p(y∣x)=N(Cx+b,R)
    # gets the indices of likelihood.x corresponding to prior
    idx = gm.get_indices(prior, likelihood)
    # 1/Σ
    precision = torch.mm(
        torch.mm(
            likelihood.A,
            likelihood.covariance.pinverse()
        ),
        likelihood.A
    ) + prior.covariance.pinverse()
    # Σ
    covariance = precision.pinverse()

    y_scale = torch.mm(likelihood.A, likelihood.conditional_covariance.pinverse())
    y_shift = torch.mm(y_scale, likelihood.b)
    # multiple dispatch would be helpful here
    if isinstance(prior, LinearTransformation):
        # create new Gaussian representing the
        # vector concatenation of
        # prior.x and likelihood
        Q_inv = prior.Q.pinverse()
        y_dim = likelihood.A.shape[0]
        dim = y_dim + prior.A.shape[0]
        z_scale = torch.mm(Q_inv, prior.conditional_covariance)
        z_shift = torch.mm(Q_inv, prior.b)

        yz_scale = torch.zeros(dim, dim)
        yz_scale[:y_dim, :y_dim] = y_scale
        yz_scale[y_dim:, y_dim:] = z_scale
        yz_shift = torch.concat([y_shift, z_shift])

        g = ConcatenatedGaussian(gm, likelihood, prior.x)
        temp = LinearTransformation(gm, A=yz_scale, b=yz_shift, x=g, register=False)
    else:
        # note sure if I should create a new variable here as likelihood
        # still depends on likelihood.x here which is now on the LHS
        # of the condition bar...
        temp = LinearTransformation(gm, A=y_scale, b=y_shift, x=likelihood, register=False)

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
        posterior_mean_wrt_temp.x
    )
    output = ConditionalGaussian(gm, posterior_mean, covariance)
    return output
