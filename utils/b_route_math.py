"""Effective LoRA coordinates and projections for the B route experiment.

All small linear algebra is FP64 on CPU. Model residuals remain differentiable
FP32 tensors on the model device; none of these helpers touch global RNGs.
"""
import math

import numpy as np
import torch
from scipy.optimize import nnls


ARMS = ('N', 'P', 'M', 'S', 'C', 'F', 'P-rot', 'C-rot')
RTOL = 1e-8


def norm(values):
    return math.sqrt(sum(float(v.detach().double().square().sum()) for v in values))


def finite(*values):
    if any(not torch.isfinite(v).all() for v in values):
        raise ValueError('Nonfinite route tensor')


def column_basis(matrix):
    matrix = matrix.detach().double().cpu()
    finite(matrix)
    if not matrix.numel():
        return matrix.new_zeros((matrix.shape[0], 0))
    u, s, _ = torch.linalg.svd(matrix, full_matrices=False)
    rank = int((s > s[0]*RTOL).sum()) if len(s) and s[0] > 0 else 0
    return u[:, :rank]


class EffectiveCoordinates:
    def __init__(self, a_factors, scaling):
        self.keys = list(a_factors)
        self.encode_maps, self.decode_maps, self.info = {}, {}, []
        for key in self.keys:
            a = a_factors[key].detach().double().cpu()
            scale = float(scaling[key])
            finite(a)
            if not math.isfinite(scale) or scale == 0 or a.ndim != 2:
                raise ValueError('Invalid LoRA factor/scaling')
            u, s, _ = torch.linalg.svd(a, full_matrices=False)
            keep = s > (s[0]*RTOL if len(s) and s[0] > 0 else 0.)
            uk, sk = u[:, keep], s[keep]
            self.encode_maps[key] = scale * uk * sk
            self.decode_maps[key] = (uk / sk).T / scale
            self.info.append(dict(module=key, a_rank=int(keep.sum()),
                                  discarded_a_energy=float(s[~keep].square().sum()),
                                  a_svd_rtol=RTOL))

    def encode(self, residual):
        return {k: v @ self.encode_maps[k].to(v) for k, v in residual.items()}

    def decode(self, coordinates):
        return {k: v @ self.decode_maps[k].to(v) for k, v in coordinates.items()}


def rotated_sources(donors, seed, event, null_index):
    """One common left orthogonal rotation per module, independent of labels."""
    result, checks = {}, []
    for index, (key, source) in enumerate(donors.items()):
        d = source.detach().double().cpu()
        rng = np.random.default_rng(np.random.SeedSequence([seed, event, index, null_index, 20261008]))
        q, r = np.linalg.qr(rng.standard_normal((d.shape[1], d.shape[1])))
        q *= np.where(np.diag(r) < 0., -1., 1.)[None, :]
        rotation = torch.from_numpy(q)
        changed = torch.einsum('ab,jbr->jar', rotation, d)
        original_matrix = d.permute(1, 0, 2).reshape(d.shape[1], -1)
        changed_matrix = changed.permute(1, 0, 2).reshape(d.shape[1], -1)
        gram = original_matrix.T @ original_matrix
        error = float((gram-changed_matrix.T@changed_matrix).norm())/max(float(gram.norm()), 1e-15)
        if error > 1e-10:
            raise ValueError('Random rotation failed Gram preservation')
        checks.append(dict(module=key, null_index=null_index, gram_relative_error=error))
        result[key] = changed
    return result, checks


def cone_projection(design, target):
    """NNLS with normalized columns and an independently checked KKT residual."""
    d = design.detach().double().cpu()
    y = target.detach().double().cpu()
    scales = torch.linalg.vector_norm(d, dim=0)
    live = scales > 0
    coefficients = d.new_zeros(d.shape[1])
    if not bool(live.any()) or not y.numel() or float(y.norm()) <= 1e-15:
        return d @ coefficients, coefficients, 0.
    normalized = (d[:, live]/scales[live]).numpy()
    # Normalize the target as well: small LoRA norms must not trigger an
    # absolute active-set stopping criterion before any variable enters.
    yscale = float(y.norm())
    weights, _ = nnls(normalized, y.numpy()/yscale, maxiter=10000)
    gradient = normalized.T @ (normalized @ weights-y.numpy()/yscale)
    projected_gradient = np.where(weights > 0, gradient, np.minimum(gradient, 0.))
    error = float(np.max(np.abs(projected_gradient), initial=0.))
    if not math.isfinite(error) or error > 1e-6:
        raise ValueError(f'NNLS KKT check failed: {error:g}')
    coefficients[live] = torch.from_numpy(weights)*yscale/scales[live]
    return d @ coefficients, coefficients, error


class RouteProjector:
    def __init__(self, arm, coordinates, donors, shapes):
        if arm not in ARMS:
            raise ValueError('Unknown route arm: '+arm)
        self.arm = arm.removesuffix('-rot')
        self.keys = list(shapes)
        self.shapes = dict(shapes)
        self.designs, self.bases, self.source_bases = {}, {}, {}
        self.donors = {k: v.detach().double().cpu() for k, v in donors.items()}
        self.last_coefficients, self.kkt_error = {}, 0.
        self.reconstruction_errors = []
        if self.arm in ('N', 'F'):
            self.dimension = sum(math.prod(s) for s in shapes.values()) if self.arm == 'F' else 0
            return
        for k in self.keys:
            d = self.donors[k]
            encoded = d @ coordinates.encode_maps[k]
            self.designs[k] = encoded.flatten(1).T.contiguous()
            if self.arm == 'S':
                self.bases[k] = column_basis(self.designs[k])
            if self.arm == 'C':
                self.source_bases[k] = column_basis(d.permute(1, 0, 2).reshape(d.shape[1], -1))
        if self.arm == 'P':
            self.design = torch.cat([self.designs[k] for k in self.keys], dim=0)
            self.dimension = column_basis(self.design).shape[1]
        elif self.arm == 'C':
            self.dimension = sum(q.shape[1]*self.shapes[k][1] for k, q in self.source_bases.items())
        else:
            self.dimension = sum(column_basis(d).shape[1] for d in self.designs.values())

    def project(self, values, radius=None):
        if radius is not None and (not math.isfinite(radius) or radius < 0):
            raise ValueError('Invalid route radius')
        finite(*values.values())
        self.last_coefficients, self.kkt_error = {}, 0.
        if self.arm == 'N':
            result = {k: torch.zeros_like(v) for k, v in values.items()}
        elif self.arm == 'F':
            result = {k: v.detach().clone() for k, v in values.items()}
        elif self.arm == 'P':
            target = torch.cat([values[k].detach().double().cpu().flatten() for k in self.keys])
            projected, weights, self.kkt_error = cone_projection(self.design, target)
            result, offset = {}, 0
            for k in self.keys:
                size = values[k].numel()
                result[k] = projected[offset:offset+size].reshape(values[k].shape).to(values[k])
                offset += size
            self.last_coefficients['all_modules'] = weights
        else:
            result = {}
            for k in self.keys:
                y = values[k].detach().double().cpu()
                if self.arm == 'M':
                    p, weights, error = cone_projection(self.designs[k], y.flatten())
                    self.last_coefficients[k] = weights
                    self.kkt_error = max(self.kkt_error, error)
                    p = p.reshape(y.shape)
                elif self.arm == 'S':
                    q = self.bases[k]
                    p = (q @ (q.T @ y.flatten())).reshape(y.shape)
                    self.last_coefficients[k] = torch.linalg.pinv(self.designs[k], rtol=RTOL) @ p.flatten()
                else:
                    q = self.source_bases[k]
                    p = q @ (q.T @ y)
                result[k] = p.to(values[k])
        length = norm(result.values())
        shrink = min(1., radius/length) if radius is not None and length > 0 else 1.
        result = {k: v*shrink for k, v in result.items()}
        self.last_coefficients = {k: v*shrink for k, v in self.last_coefficients.items()}
        finite(*result.values())
        if radius is not None and norm(result.values()) > radius*(1+2e-6)+1e-12:
            raise ValueError('Projected route update exceeds radius')
        return result

    def reconstruct_c(self, residual):
        """Return minimum-norm C and verify the actual source reconstruction."""
        result = {}
        for k in self.keys:
            donors = self.donors[k]
            m, _, rank = donors.shape
            d = donors.permute(1, 0, 2).reshape(donors.shape[1], -1)
            r = residual[k].detach().double().cpu()
            c = (torch.linalg.pinv(d, rtol=RTOL) @ (m*r)).reshape(m, rank, r.shape[1])
            actual = torch.bmm(donors, c).mean(0)
            # FP32 projection/decode rounds entries outside the exact FP64
            # source space. Test the residual norm, not relative error of an
            # individual near-zero coordinate in a large matrix.
            error, length = float((actual-r).norm()), float(r.norm())
            if error > 2e-6*length+1e-10:
                raise ValueError('C reconstruction failed for '+k)
            self.reconstruction_errors.append(dict(module=k, reconstruction_error=error,
                relative_reconstruction_error=error/length if length else None,
                reconstruction_rtol=2e-6, reconstruction_atol=1e-10))
            result[k] = c
        return result


def projected_step(values, gradients, projector, radius):
    finite(*gradients.values())
    length = norm(gradients.values())
    if length <= 1e-12 or radius == 0:
        return {k: v.detach().clone() for k, v in values.items()}, length
    candidate = {k: v.detach()-(radius/2)*gradients[k]/length for k, v in values.items()}
    return projector.project(candidate, radius), length
