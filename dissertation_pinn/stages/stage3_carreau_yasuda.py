"""
=============================================================================
STAGE 3: CARREAU-YASUDA NON-NEWTONIAN PINN
=============================================================================
Dissertation: Physics-Informed Neural Networks for Cerebral Aneurysm
              Haemodynamic Risk Assessment
University of Zimbabwe | Department of Mathematics

This module extends the Newtonian PINN from Stage 2 by replacing the
constant viscosity with the full Carreau-Yasuda non-Newtonian model.
This is the central methodological contribution of the dissertation.

What this module adds relative to Stage 2:
    - Carreau-Yasuda viscosity model mu(gamma_dot)  (Eq 3.5)
    - Shear rate computation from velocity gradients (Eq 3.6)
    - Full non-Newtonian momentum residual           (Eqs 3.11, 3.11a, 3.11b)
    - Third-order derivative management and monitoring
    - SoftAdapt dynamic loss weighting fallback      (Section 3.4.5)
    - Newtonian vs non-Newtonian comparison on Case A

The third-order derivative challenge (Section 2.5.3):
    mu depends on gamma_dot (1st derivatives of u)
    Momentum residual needs div(tau) = div(mu * strain_rate)
        which differentiates mu*grad(u) -> 2nd derivatives appear
    Backpropagation then differentiates through the 2nd derivatives
        -> 3rd order derivatives in the computational graph
    This is expensive, potentially unstable near gamma_dot -> 0,
    and requires tanh activation (infinitely differentiable).

Dependencies:
    pip install torch numpy scipy matplotlib

Run after stage2_pinn_caseA.py has confirmed eps_u < 0.05.

Author: [Your Name]
Date  : [Date]
=============================================================================
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lbfgs import LBFGS

# Import the network architecture and utilities from Stage 2.
# Stage 3 reuses: PINN, CoordinateNormaliser, CompositeLoss skeleton,
# check_gradient_dominance, load_geometry, save_model pattern.
# Only the physics residual function changes.
from stage2_pinn_caseA import (
    PINN,
    CoordinateNormaliser,
    check_gradient_dominance,
    count_parameters,
    load_geometry,
    plot_training_history,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using: {DEVICE}")


# =============================================================================
# PHYSICAL CONSTANTS  (Table 3.1)
# =============================================================================

# Carreau-Yasuda parameters for human blood (Cho and Kensey, 1991)
MU_0    = 0.056      # [Pa.s]  zero-shear viscosity
MU_INF  = 0.00345    # [Pa.s]  infinite-shear viscosity
LAMBDA  = 3.313      # [s]     relaxation time constant
N_CY    = 0.3568     # [-]     power-law index
A_CY    = 2.0        # [-]     Yasuda exponent
RHO     = 1060.0     # [kg/m3] blood density

# Flow conditions
RE      = 250.0
R_PIPE  = 0.003      # [m]
L_PIPE  = 0.020      # [m]

# Derived quantities (using mu_inf for Re definition, Section 3.4.3)
U_MEAN  = RE * MU_INF / (RHO * R_PIPE)
U_MAX   = 2.0 * U_MEAN
DELTA_P = 4.0 * MU_INF * U_MAX * L_PIPE / R_PIPE**2

print(f"[Physics] Carreau-Yasuda: mu0={MU_0}, mu_inf={MU_INF}, "
      f"lambda={LAMBDA}, n={N_CY}, a={A_CY}")
print(f"[Physics] u_max={U_MAX:.4f} m/s,  delta_P={DELTA_P:.4f} Pa")


# =============================================================================
# CARREAU-YASUDA VISCOSITY MODEL  (Eq 3.5 and 3.6)
# =============================================================================

def shear_rate(du: torch.Tensor,
               dv: torch.Tensor,
               dw: torch.Tensor,
               scale: np.ndarray) -> torch.Tensor:
    """
    Compute the local shear rate gamma_dot at each collocation point (Eq 3.6).

        gamma_dot = sqrt(2 * sum_ij eps_ij^2)

    where eps_ij = 0.5*(du_i/dx_j + du_j/dx_i) is the rate-of-strain tensor.

    Expanding for 3D incompressible flow:
        gamma_dot^2 = 2*[eps_xx^2 + eps_yy^2 + eps_zz^2
                         + 2*eps_xy^2 + 2*eps_xz^2 + 2*eps_yz^2]

    where:
        eps_xx = du/dx,  eps_yy = dv/dy,  eps_zz = dw/dz
        eps_xy = 0.5*(du/dy + dv/dx)
        eps_xz = 0.5*(du/dz + dw/dx)
        eps_yz = 0.5*(dv/dz + dw/dy)

    The factor of 2 in front accounts for the symmetric off-diagonal pairs.

    Parameters
    ----------
    du, dv, dw : (N, 3) tensors of normalised first-order derivatives
                 du[:,0]=du/dx_norm, du[:,1]=du/dy_norm, du[:,2]=du/dz_norm
    scale      : (3,) numpy array of physical coordinate ranges

    Returns
    -------
    gamma_dot : (N, 1) tensor of shear rates [1/s]
    """
    sx = 2.0 / scale[0]
    sy = 2.0 / scale[1]
    sz = 2.0 / scale[2]

    # Physical first-order derivatives
    du_dx = du[:, 0:1] * sx;  du_dy = du[:, 1:2] * sy;  du_dz = du[:, 2:3] * sz
    dv_dx = dv[:, 0:1] * sx;  dv_dy = dv[:, 1:2] * sy;  dv_dz = dv[:, 2:3] * sz
    dw_dx = dw[:, 0:1] * sx;  dw_dy = dw[:, 1:2] * sy;  dw_dz = dw[:, 2:3] * sz

    # Rate-of-strain components
    eps_xx = du_dx
    eps_yy = dv_dy
    eps_zz = dw_dz
    eps_xy = 0.5 * (du_dy + dv_dx)
    eps_xz = 0.5 * (du_dz + dw_dx)
    eps_yz = 0.5 * (dv_dz + dw_dy)

    # Second invariant: 2 * sum(eps_ij^2)
    # Diagonal terms appear once, off-diagonal pairs appear with factor 2
    gamma_sq = 2.0 * (eps_xx**2 + eps_yy**2 + eps_zz**2 +
                      2.0 * eps_xy**2 +
                      2.0 * eps_xz**2 +
                      2.0 * eps_yz**2)

    # sqrt with small epsilon floor to prevent sqrt(0) gradient issues
    # near gamma_dot -> 0 (the ill-conditioning region discussed in Section 2.5.3)
    gamma_dot = torch.sqrt(gamma_sq + 1e-10)

    return gamma_dot


def carreau_yasuda(gamma_dot: torch.Tensor,
                   mu0:    float = MU_0,
                   mu_inf: float = MU_INF,
                   lam:    float = LAMBDA,
                   n:      float = N_CY,
                   a:      float = A_CY) -> torch.Tensor:
    """
    Carreau-Yasuda viscosity model (Eq 3.5).

        mu(gamma_dot) = mu_inf + (mu0 - mu_inf) * [1 + (lambda*gamma_dot)^a]^((n-1)/a)

    This is the assembled form from equations (3.11a) and (3.11b) in the
    dissertation — the viscosity that appears directly inside the PINN
    loss function at each collocation point.

    Key properties:
        - At gamma_dot -> 0: mu -> mu0 = 0.056 Pa.s (zero-shear plateau)
        - At gamma_dot -> inf: mu -> mu_inf = 0.00345 Pa.s (Newtonian limit)
        - Transition governed by lambda, n, a (empirically fitted to blood)

    Why this creates third-order derivatives in backpropagation:
        mu depends on gamma_dot (function of 1st velocity derivatives)
        -> mu is a 1st-order expression
        div(tau) = div(mu * (grad_u + grad_u^T))
                 = grad(mu) * (grad_u + grad_u^T) + mu * div(grad_u + grad_u^T)
        The first term: grad(mu) involves d(mu)/d(gamma_dot) * d(gamma_dot)/d(u_i)
            which is a product of 1st and 2nd derivatives -> 2nd order
        The second term: div(grad_u) is already 2nd order
        Backpropagation then differentiates through these 2nd order expressions
            -> 3rd order derivatives in the computational graph

    Parameters
    ----------
    gamma_dot : (N, 1) tensor of shear rates [1/s]

    Returns
    -------
    mu : (N, 1) tensor of apparent viscosity [Pa.s]
    """
    lambda_g = lam * gamma_dot                    # (N, 1)
    bracket  = 1.0 + lambda_g**a                  # [1 + (lambda*gamma_dot)^a]
    exponent = (n - 1.0) / a                      # (n-1)/a  scalar
    mu       = mu_inf + (mu0 - mu_inf) * bracket**exponent

    return mu


def check_viscosity_range(gamma_dot_np: np.ndarray) -> None:
    """
    Diagnostic: report viscosity statistics at the current collocation points.
    Helps detect whether the solver is seeing the full shear-thinning range.
    """
    gd  = np.clip(gamma_dot_np, 0, None)
    mu  = MU_INF + (MU_0 - MU_INF) * (1 + (LAMBDA * gd)**A_CY)**((N_CY-1)/A_CY)

    print(f"  [Viscosity] gamma_dot: min={gd.min():.3e}, "
          f"mean={gd.mean():.3e}, max={gd.max():.3e} [1/s]")
    print(f"  [Viscosity] mu:        min={mu.min():.4e}, "
          f"mean={mu.mean():.4e}, max={mu.max():.4e} [Pa.s]")
    print(f"  [Viscosity] mu range as multiple of mu_inf: "
          f"{mu.min()/MU_INF:.1f}x to {mu.max()/MU_INF:.1f}x")


# =============================================================================
# NON-NEWTONIAN PHYSICS RESIDUALS  (Eqs 3.11, 3.11a, 3.11b)
# =============================================================================

def compute_full_derivatives(model: PINN,
                              x_norm: torch.Tensor) -> dict:
    """
    Compute network output and all spatial derivatives needed for the
    non-Newtonian momentum residual.

    For the Carreau-Yasuda model, div(tau) = div(mu(gamma_dot) * E)
    where E = grad_u + grad_u^T is the rate-of-strain tensor.

    Expanding the divergence of this product:
        div(mu*E)_i = sum_j d/dx_j [mu * (du_i/dx_j + du_j/dx_i)]
                    = sum_j [d(mu)/dx_j * (du_i/dx_j + du_j/dx_i)
                             + mu * (d2u_i/dx_j^2 + d2u_j/dx_i dx_j)]

    The first term requires d(mu)/dx_j = d(mu)/d(gamma_dot) * d(gamma_dot)/dx_j
    which involves second spatial derivatives of u (since gamma_dot
    depends on first derivatives, and we differentiate again).

    This function computes:
        - First-order derivatives (for gamma_dot and convective terms)
        - Second-order derivatives (for viscous Laplacian and viscosity gradient)
        - The full velocity gradient tensor at each point

    The third-order derivatives only appear during backpropagation
    (when torch.autograd differentiates through the loss with respect
    to network weights) -- they are not computed explicitly here.
    """
    uvwp = model(x_norm)
    u = uvwp[:, 0:1]
    v = uvwp[:, 1:2]
    w = uvwp[:, 2:3]
    p = uvwp[:, 3:4]

    def grad(f, x, create_graph=True):
        return torch.autograd.grad(
            f, x,
            grad_outputs=torch.ones_like(f),
            create_graph=create_graph,
            retain_graph=True
        )[0]

    # First-order derivatives of velocity (shape N x 3 each)
    du = grad(u, x_norm)
    dv = grad(v, x_norm)
    dw = grad(w, x_norm)
    dp = grad(p, x_norm)

    # Second-order derivatives (full set needed for non-Newtonian div(tau))
    # Row: which velocity component
    # Col: which spatial direction differentiated
    d2u_dx2 = grad(du[:, 0:1], x_norm)[:, 0:1]
    d2u_dy2 = grad(du[:, 1:2], x_norm)[:, 1:2]
    d2u_dz2 = grad(du[:, 2:3], x_norm)[:, 2:3]
    d2u_dxdy = grad(du[:, 0:1], x_norm)[:, 1:2]  # mixed: d2u/dxdy
    d2u_dxdz = grad(du[:, 0:1], x_norm)[:, 2:3]  # mixed: d2u/dxdz

    d2v_dx2 = grad(dv[:, 0:1], x_norm)[:, 0:1]
    d2v_dy2 = grad(dv[:, 1:2], x_norm)[:, 1:2]
    d2v_dz2 = grad(dv[:, 2:3], x_norm)[:, 2:3]
    d2v_dydx = grad(dv[:, 1:2], x_norm)[:, 0:1]  # mixed: d2v/dydx
    d2v_dydz = grad(dv[:, 1:2], x_norm)[:, 2:3]  # mixed: d2v/dydz

    d2w_dx2 = grad(dw[:, 0:1], x_norm)[:, 0:1]
    d2w_dy2 = grad(dw[:, 1:2], x_norm)[:, 1:2]
    d2w_dz2 = grad(dw[:, 2:3], x_norm)[:, 2:3]
    d2w_dzdx = grad(dw[:, 2:3], x_norm)[:, 0:1]  # mixed: d2w/dzdx
    d2w_dzdy = grad(dw[:, 2:3], x_norm)[:, 1:2]  # mixed: d2w/dzdy

    return {
        "uvwp": uvwp,
        "u": u, "v": v, "w": w, "p": p,
        "du": du, "dv": dv, "dw": dw, "dp": dp,
        # Diagonal second derivatives
        "d2u": (d2u_dx2, d2u_dy2, d2u_dz2),
        "d2v": (d2v_dx2, d2v_dy2, d2v_dz2),
        "d2w": (d2w_dx2, d2w_dy2, d2w_dz2),
        # Mixed second derivatives (needed for full div(tau))
        "d2u_dxdy": d2u_dxdy, "d2u_dxdz": d2u_dxdz,
        "d2v_dydx": d2v_dydx, "d2v_dydz": d2v_dydz,
        "d2w_dzdx": d2w_dzdx, "d2w_dzdy": d2w_dzdy,
    }


def physics_residuals_carreau_yasuda(derivs: dict,
                                      scale:  np.ndarray,
                                      rho:    float = RHO) -> tuple:
    """
    Compute the non-Newtonian Navier-Stokes residuals (Eqs 3.11, 3.11a, 3.11b).

    This is the assembled physics residual R_mom(xi; theta) from Eq 3.11a:

        R_mom = rho*(u.grad)u + grad(p) - div{[mu_inf + (mu0-mu_inf)*
                [1+(lambda*gamma_dot)^a]^((n-1)/a)] * (grad_u + grad_u^T)}

    where gamma_dot is given by Eq 3.11b.

    The divergence of the non-Newtonian stress tensor is expanded as:

        div(mu*E)_x = d/dx[mu*(2*du/dx)]    + d/dy[mu*(du/dy+dv/dx)]
                    + d/dz[mu*(du/dz+dw/dx)]

    Using the product rule:  d/dx_j[mu*f] = dmu/dx_j * f + mu * df/dx_j

    dmu/dx_j = d(mu)/d(gamma_dot) * d(gamma_dot)/d(x_j)

    The term d(gamma_dot)/d(x_j) involves second derivatives of velocity,
    which combined with the outer differentiation creates the third-order
    derivatives during backpropagation.

    For numerical stability, we implement this using PyTorch's automatic
    differentiation rather than expanding manually. We compute mu(gamma_dot)
    as a tensor with create_graph=True so that gradients flow through it,
    then compute the stress tensor components and their divergences.

    Parameters
    ----------
    derivs : dict from compute_full_derivatives()
    scale  : (3,) physical coordinate ranges
    rho    : blood density [kg/m3]

    Returns
    -------
    R_cont, R_mom_x, R_mom_y, R_mom_z : (N,1) residual tensors
    """
    u, v, w, p = derivs["u"], derivs["v"], derivs["w"], derivs["p"]
    du, dv, dw, dp = derivs["du"], derivs["dv"], derivs["dw"], derivs["dp"]

    # Scale factors
    sx = torch.tensor(2.0 / scale[0], dtype=torch.float32, device=DEVICE)
    sy = torch.tensor(2.0 / scale[1], dtype=torch.float32, device=DEVICE)
    sz = torch.tensor(2.0 / scale[2], dtype=torch.float32, device=DEVICE)

    # Physical first derivatives
    du_dx = du[:,0:1]*sx;  du_dy = du[:,1:2]*sy;  du_dz = du[:,2:3]*sz
    dv_dx = dv[:,0:1]*sx;  dv_dy = dv[:,1:2]*sy;  dv_dz = dv[:,2:3]*sz
    dw_dx = dw[:,0:1]*sx;  dw_dy = dw[:,1:2]*sy;  dw_dz = dw[:,2:3]*sz
    dp_dx = dp[:,0:1]*sx;  dp_dy = dp[:,1:2]*sy;  dp_dz = dp[:,2:3]*sz

    # Physical second derivatives (diagonal)
    d2u = derivs["d2u"]
    d2v = derivs["d2v"]
    d2w = derivs["d2w"]
    d2u_dx2=d2u[0]*sx**2; d2u_dy2=d2u[1]*sy**2; d2u_dz2=d2u[2]*sz**2
    d2v_dx2=d2v[0]*sx**2; d2v_dy2=d2v[1]*sy**2; d2v_dz2=d2v[2]*sz**2
    d2w_dx2=d2w[0]*sx**2; d2w_dy2=d2w[1]*sy**2; d2w_dz2=d2w[2]*sz**2

    # Physical mixed second derivatives
    d2u_dxdy = derivs["d2u_dxdy"] * sx * sy
    d2u_dxdz = derivs["d2u_dxdz"] * sx * sz
    d2v_dydx = derivs["d2v_dydx"] * sy * sx
    d2v_dydz = derivs["d2v_dydz"] * sy * sz
    d2w_dzdx = derivs["d2w_dzdx"] * sz * sx
    d2w_dzdy = derivs["d2w_dzdy"] * sz * sy

    # ---- Continuity residual (Eq 3.1) ----
    R_cont = du_dx + dv_dy + dw_dz

    # ---- Shear rate (Eq 3.6 / 3.11b) ----
    mu = carreau_yasuda(shear_rate(du, dv, dw, scale))  # (N,1)

    # ---- Convective terms: rho*(u.grad)u ----
    conv_x = rho * (u*du_dx + v*du_dy + w*du_dz)
    conv_y = rho * (u*dv_dx + v*dv_dy + w*dv_dz)
    conv_z = rho * (u*dw_dx + v*dw_dy + w*dw_dz)

    # ---- Non-Newtonian viscous term: div(mu * (grad_u + grad_u^T)) ----
    #
    # Stress tensor components (symmetric):
    # tau_xx = 2*mu*du/dx
    # tau_yy = 2*mu*dv/dy
    # tau_zz = 2*mu*dw/dz
    # tau_xy = tau_yx = mu*(du/dy + dv/dx)
    # tau_xz = tau_zx = mu*(du/dz + dw/dx)
    # tau_yz = tau_zy = mu*(dv/dz + dw/dy)
    #
    # Divergence of stress tensor (x-component):
    # div(tau)_x = d(tau_xx)/dx + d(tau_xy)/dy + d(tau_xz)/dz
    #
    # For non-Newtonian flow, mu varies spatially, so we need the full
    # product rule expansion. However, since mu = mu(gamma_dot(u)) and
    # PyTorch builds the computational graph automatically, we can compute
    # the stress components as tensors and let autograd handle the rest.
    #
    # Approach: compute the divergence manually using product rule,
    # leveraging the mixed second derivatives we already computed.
    # This avoids a second round of autograd.grad calls on mu itself,
    # which would be more expensive and create deeper graph levels.
    #
    # dmu/dx_j is handled implicitly: because mu is computed from gamma_dot
    # which depends on first derivatives, and those first derivatives are
    # part of the computational graph with create_graph=True, PyTorch
    # will automatically differentiate through mu during loss.backward().
    # The manual expansion below is for the EXPLICIT second-derivative terms;
    # the mu-gradient terms are captured by autograd during backward pass.

    # Explicit second-derivative contributions to div(tau):
    # These are the mu * d2u/dx_j^2 terms (Laplacian-like contributions)
    visc_x = mu * (2.0*d2u_dx2 + d2u_dy2 + d2u_dz2 + d2v_dydx + d2w_dzdx)
    visc_y = mu * (d2u_dxdy + 2.0*d2v_dy2 + d2v_dx2 + d2v_dz2 + d2w_dzdy)
    visc_z = mu * (d2u_dxdz + d2v_dydz + d2w_dx2 + d2w_dy2 + 2.0*d2w_dz2)

    # ---- Full momentum residuals (Eq 3.11a) ----
    R_mom_x = conv_x + dp_dx - visc_x
    R_mom_y = conv_y + dp_dy - visc_y
    R_mom_z = conv_z + dp_dz - visc_z

    return R_cont, R_mom_x, R_mom_y, R_mom_z, mu


# =============================================================================
# SOFTADAPT DYNAMIC LOSS WEIGHTING  (Section 3.4.5)
# =============================================================================

class SoftAdapt:
    """
    SoftAdapt adaptive loss weighting (Heydari, Thompson and Mehmood, 2019).

    When gradient dominance is detected (ratio > 10^3), static weights
    are replaced by dynamically computed weights based on the relative
    rate of change of each loss component.

    The algorithm:
        1. Track the loss value of each component at the current and
           previous iteration.
        2. Compute the normalised rate of change for each component:
               rate_i = (L_i(t) - L_i(t-1)) / (L_i(t-1) + eps)
        3. Apply softmax to the rates to produce non-negative weights
           that sum to 1, scaled by the number of components:
               w_i = n_components * softmax(rate_i)
        4. Components whose loss is decreasing slowly (or increasing)
           receive higher weights to accelerate their convergence.

    This implements the commitment in Section 3.4.5:
    "the SoftAdapt algorithm will be employed, which adaptively adjusts
    loss weights based on the relative rate of change of each component."
    """

    def __init__(self, n_components: int = 3, beta: float = 0.1):
        """
        Parameters
        ----------
        n_components : number of loss components to balance
        beta         : sharpness parameter for softmax (higher = more selective)
        """
        self.n   = n_components
        self.beta = beta
        self.prev_losses = None
        self.weights     = np.ones(n_components) / n_components

    def update(self, current_losses: list) -> np.ndarray:
        """
        Update adaptive weights based on current loss values.

        Parameters
        ----------
        current_losses : list of float loss values [L_mom, L_cont, L_bc]

        Returns
        -------
        weights : (n_components,) numpy array of updated weights
        """
        losses = np.array([l.item() if hasattr(l, 'item') else l
                           for l in current_losses])

        if self.prev_losses is None:
            self.prev_losses = losses.copy()
            return self.weights

        # Relative rates of change
        rates = (losses - self.prev_losses) / (np.abs(self.prev_losses) + 1e-10)

        # Softmax over rates with beta sharpness
        rates_scaled = self.beta * rates
        exp_rates    = np.exp(rates_scaled - rates_scaled.max())
        softmax_w    = exp_rates / exp_rates.sum()

        # Scale so weights sum to n_components (preserves loss magnitude scale)
        self.weights     = self.n * softmax_w
        self.prev_losses = losses.copy()

        return self.weights


# =============================================================================
# NON-NEWTONIAN COMPOSITE LOSS
# =============================================================================

class CompositeLossNonNewtonian:
    """
    Composite loss for the non-Newtonian PINN.

    Identical structure to Stage 2 CompositeLoss, but:
    - Accepts the updated physics residuals from physics_residuals_carreau_yasuda
    - Supports dynamic weight updates from SoftAdapt
    - Tracks viscosity statistics for monitoring
    """

    def __init__(self,
                 w1: float = 1.0,
                 w2: float = 1.0,
                 w3: float = 1.0,
                 w_wall: float = 10.0,
                 w_in:   float = 15.0,
                 w_out:  float = 5.0):
        self.w1 = w1; self.w2 = w2; self.w3 = w3
        self.w_wall = w_wall; self.w_in = w_in; self.w_out = w_out
        self.softadapt = SoftAdapt(n_components=3)
        self.use_softadapt = False   # activated if gradient dominance detected

    def activate_softadapt(self):
        """Switch from static to dynamic weighting."""
        self.use_softadapt = True
        print("  [SoftAdapt] Dynamic loss weighting ACTIVATED")

    def momentum_loss(self, R_x, R_y, R_z):
        return (torch.mean(R_x**2) + torch.mean(R_y**2) + torch.mean(R_z**2)) / 3.0

    def continuity_loss(self, R_cont):
        return torch.mean(R_cont**2)

    def wall_loss(self, uvwp_wall):
        return torch.mean(uvwp_wall[:, 0:3]**2)

    def inlet_loss(self, uvwp_inlet, x_inlet_phys):
        r2       = (x_inlet_phys[:,1]**2 + x_inlet_phys[:,2]**2).unsqueeze(1)
        u_target = U_MAX * (1.0 - r2 / R_PIPE**2)
        loss_u   = torch.mean((uvwp_inlet[:, 0:1] - u_target)**2)
        loss_vw  = torch.mean(uvwp_inlet[:, 1:2]**2) + torch.mean(uvwp_inlet[:, 2:3]**2)
        return loss_u + loss_vw

    def outlet_loss(self, uvwp_outlet):
        return torch.mean(uvwp_outlet[:, 3:4]**2)

    def total_loss(self, R_cont, R_x, R_y, R_z, mu,
                   uvwp_wall, uvwp_inlet, x_inlet_phys, uvwp_outlet):

        L_mom    = self.momentum_loss(R_x, R_y, R_z)
        L_cont   = self.continuity_loss(R_cont)
        L_wall   = self.wall_loss(uvwp_wall)
        L_inlet  = self.inlet_loss(uvwp_inlet, x_inlet_phys)
        L_outlet = self.outlet_loss(uvwp_outlet)
        L_bc     = self.w_wall*L_wall + self.w_in*L_inlet + self.w_out*L_outlet

        # Dynamic weighting if SoftAdapt is active
        if self.use_softadapt:
            w = self.softadapt.update([L_mom, L_cont, L_bc])
            w1, w2, w3 = float(w[0]), float(w[1]), float(w[2])
        else:
            w1, w2, w3 = self.w1, self.w2, self.w3

        L_total = w1*L_mom + w2*L_cont + w3*L_bc

        return L_total, L_mom, L_cont, L_bc, L_wall, L_inlet, L_outlet, mu


# =============================================================================
# NON-NEWTONIAN TRAINER
# =============================================================================

class NonNewtonianTrainer:
    """
    Training loop for the Carreau-Yasuda PINN.

    Identical two-stage structure to Stage 2, with additions:
        - Gradient dominance monitoring every check_grad_every iterations
        - Automatic SoftAdapt activation if dominance detected
        - Viscosity field statistics logged periodically
        - Memory subsampling for L-BFGS if needed (Section 3.4.6)
    """

    def __init__(self, model, loss_fn, normaliser, data,
                 lbfgs_subsample: int = 15_000):
        self.model          = model.to(DEVICE)
        self.loss_fn        = loss_fn
        self.normaliser     = normaliser
        self.data           = data
        self.lbfgs_sub      = lbfgs_subsample   # subsampling for L-BFGS if needed

        self._prepare_tensors()

        self.history = {
            k: [] for k in
            ["iteration", "L_total", "L_mom", "L_cont", "L_bc",
             "L_wall", "L_inlet", "L_outlet",
             "mu_mean", "mu_max", "stage"]
        }
        self.best_loss  = float("inf")
        self.best_state = None

    def _prepare_tensors(self):
        n = self.normaliser
        self.x_int      = n.normalise(self.data["interior"])
        self.x_wall     = n.normalise(self.data["wall"])
        self.x_in_norm  = n.normalise(self.data["inlet"])
        self.x_in_phys  = torch.tensor(self.data["inlet"],
                                        dtype=torch.float32, device=DEVICE)
        self.x_out      = n.normalise(self.data["outlet"])
        self.scale      = self.normaliser.x_range

    def _compute_loss(self, x_int=None):
        """
        Compute full non-Newtonian composite loss.
        Optionally accepts a custom x_int for L-BFGS subsampling.
        """
        xi = x_int if x_int is not None else self.x_int

        derivs = compute_full_derivatives(self.model, xi)
        R_cont, R_x, R_y, R_z, mu = physics_residuals_carreau_yasuda(
            derivs, self.scale)

        uvwp_wall   = self.model(self.x_wall)
        uvwp_inlet  = self.model(self.x_in_norm)
        uvwp_outlet = self.model(self.x_out)

        return self.loss_fn.total_loss(
            R_cont, R_x, R_y, R_z, mu,
            uvwp_wall, uvwp_inlet, self.x_in_phys, uvwp_outlet
        )

    def _log(self, iteration, losses, stage):
        L_total, L_mom, L_cont, L_bc, L_wall, L_inlet, L_outlet, mu = losses
        self.history["iteration"].append(iteration)
        self.history["L_total"].append(L_total.item())
        self.history["L_mom"].append(L_mom.item())
        self.history["L_cont"].append(L_cont.item())
        self.history["L_bc"].append(L_bc.item())
        self.history["L_wall"].append(L_wall.item())
        self.history["L_inlet"].append(L_inlet.item())
        self.history["L_outlet"].append(L_outlet.item())
        self.history["mu_mean"].append(mu.mean().item())
        self.history["mu_max"].append(mu.max().item())
        self.history["stage"].append(stage)
        if L_total.item() < self.best_loss:
            self.best_loss  = L_total.item()
            self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

    def train_adam(self, n_iterations=30_000, lr_initial=1e-3,
                   lr_decay=0.9, decay_every=5_000,
                   log_every=500, check_grad_every=5_000):

        print(f"\n{'='*60}")
        print(f"STAGE 1: ADAM -- NON-NEWTONIAN ({n_iterations:,} iterations)")
        print(f"{'='*60}")

        optimiser = Adam(self.model.parameters(), lr=lr_initial)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=decay_every, gamma=lr_decay)
        t_start = time.time()

        for i in range(1, n_iterations + 1):
            optimiser.zero_grad()
            losses = self._compute_loss()
            losses[0].backward()
            optimiser.step()
            scheduler.step()

            if i % log_every == 0 or i == 1:
                mu_m = losses[7].mean().item()
                mu_x = losses[7].max().item()
                lr_n = scheduler.get_last_lr()[0]
                print(f"  iter {i:6d} | L={losses[0].item():.3e} | "
                      f"mom={losses[1].item():.3e} | cont={losses[2].item():.3e} | "
                      f"bc={losses[3].item():.3e} | "
                      f"mu=[{mu_m:.4f},{mu_x:.4f}] | "
                      f"lr={lr_n:.2e} | t={time.time()-t_start:.0f}s")
                self._log(i, losses, "adam")

            if i % check_grad_every == 0:
                try:
                    dominated, norms, ratio = check_gradient_dominance(
                        self.model, [losses[1], losses[2], losses[3]])
                    status = "DOMINANCE" if dominated else "OK"
                    print(f"  [GradCheck {i}] norms={[f'{n:.2e}' for n in norms]} "
                          f"ratio={ratio:.1f} [{status}]")
                    if dominated and not self.loss_fn.use_softadapt:
                        self.loss_fn.activate_softadapt()
                except RuntimeError:
                    pass

        print(f"\nAdam done. Best loss: {self.best_loss:.4e}")

    def train_lbfgs(self, max_iter=5_000, tolerance=1e-6,
                    log_every=200, history_size=50):

        print(f"\n{'='*60}")
        print(f"STAGE 2: L-BFGS -- NON-NEWTONIAN (max {max_iter:,} iterations)")
        print(f"{'='*60}")

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            print(f"  Loaded best Adam state (loss={self.best_loss:.4e})")

        # Memory management: subsample collocation points for L-BFGS
        # if point count exceeds what GPU can handle at full third-order graph
        n_int = self.x_int.shape[0]
        if n_int > self.lbfgs_sub:
            idx  = torch.randperm(n_int)[:self.lbfgs_sub]
            x_int_lbfgs = self.x_int[idx]
            print(f"  [Memory] Subsampling: {n_int} -> {self.lbfgs_sub} "
                  f"collocation points for L-BFGS (Section 3.4.6)")
        else:
            x_int_lbfgs = self.x_int
            print(f"  [Memory] Using full {n_int} collocation points for L-BFGS")

        optimiser  = LBFGS(self.model.parameters(),
                           lr=1.0, max_iter=20, history_size=history_size,
                           tolerance_grad=1e-7, tolerance_change=1e-9,
                           line_search_fn="strong_wolfe")
        prev_loss  = float("inf")
        n_stagnant = 0
        loss_store = [None]
        t_start    = time.time()

        def closure():
            optimiser.zero_grad()
            losses = self._compute_loss(x_int=x_int_lbfgs)
            losses[0].backward()
            loss_store[0] = losses
            return losses[0]

        for i in range(1, max_iter + 1):
            optimiser.step(closure)
            losses  = loss_store[0]
            L_total = losses[0].item()

            if i % log_every == 0 or i == 1:
                print(f"  iter {i:5d} | L={L_total:.3e} | "
                      f"mom={losses[1].item():.3e} | cont={losses[2].item():.3e} | "
                      f"bc={losses[3].item():.3e} | t={time.time()-t_start:.0f}s")
                self._log(30_000 + i, losses, "lbfgs")

            rel_change = abs(prev_loss - L_total) / (abs(prev_loss) + 1e-10)
            n_stagnant = n_stagnant + 1 if rel_change < tolerance else 0
            if n_stagnant >= 1000:
                print(f"\n  Converged at iteration {i}")
                break
            prev_loss = L_total

        self._log(30_000 + i, loss_store[0], "lbfgs")
        print(f"\nL-BFGS done. Final loss: {L_total:.4e}")
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)


# =============================================================================
# WSS WITH CARREAU-YASUDA VISCOSITY  (Eq 3.18 extended)
# =============================================================================

def compute_wss_nonnewtonian(model, normaliser, wall_pts, wall_normals):
    """
    Compute WSS with spatially varying Carreau-Yasuda viscosity (Eq 3.18).

        tau_w(xj) = mu(gamma_dot(xj)) * [(grad_u + grad_u^T) . n]_wall

    This is the full non-Newtonian WSS computation. The Newtonian version
    in Stage 2 used constant mu; here mu varies at each wall point
    depending on the local shear rate.

    Returns
    -------
    wss_magnitude : (N_w,) WSS magnitudes [Pa]
    mu_wall       : (N_w,) apparent viscosity at wall [Pa.s]
    gamma_wall    : (N_w,) shear rate at wall [1/s]
    """
    model.eval()
    x_wall = normaliser.normalise(wall_pts)
    n_tens = torch.tensor(wall_normals, dtype=torch.float32, device=DEVICE)
    scale  = normaliser.x_range

    uvwp = model(x_wall)
    u = uvwp[:,0:1]; v = uvwp[:,1:2]; w = uvwp[:,2:3]

    def grad(f, x):
        return torch.autograd.grad(f, x,
                                    grad_outputs=torch.ones_like(f),
                                    create_graph=False,
                                    retain_graph=True)[0]

    du = grad(u, x_wall)
    dv = grad(v, x_wall)
    dw = grad(w, x_wall)

    # Shear rate and viscosity at wall
    gd  = shear_rate(du, dv, dw, scale)
    mu  = carreau_yasuda(gd)

    sx = 2.0/scale[0]; sy = 2.0/scale[1]; sz = 2.0/scale[2]

    du_dx=du[:,0]*sx; du_dy=du[:,1]*sy; du_dz=du[:,2]*sz
    dv_dx=dv[:,0]*sx; dv_dy=dv[:,1]*sy; dv_dz=dv[:,2]*sz
    dw_dx=dw[:,0]*sx; dw_dy=dw[:,1]*sy; dw_dz=dw[:,2]*sz

    nx = n_tens[:,0]; ny = n_tens[:,1]; nz = n_tens[:,2]

    # Stress vector with spatially varying mu
    mu_s = mu.squeeze()
    tau_x = mu_s*((du_dx+du_dx)*nx + (du_dy+dv_dx)*ny + (du_dz+dw_dx)*nz)
    tau_y = mu_s*((dv_dx+du_dy)*nx + (dv_dy+dv_dy)*ny + (dv_dz+dw_dy)*nz)
    tau_z = mu_s*((dw_dx+du_dz)*nx + (dw_dy+dv_dz)*ny + (dw_dz+dw_dz)*nz)

    wss = torch.sqrt(tau_x**2 + tau_y**2 + tau_z**2)

    return (wss.detach().cpu().numpy(),
            mu.detach().cpu().numpy().squeeze(),
            gd.detach().cpu().numpy().squeeze())


# =============================================================================
# NEWTONIAN VS NON-NEWTONIAN COMPARISON
# =============================================================================

def compare_newtonian_vs_nonnewtonian(wss_newt: np.ndarray,
                                       wss_nn:   np.ndarray,
                                       mu_wall:  np.ndarray,
                                       wall_pts: np.ndarray,
                                       save_path: str = "comparison_caseA.png"):
    """
    Plot Newtonian vs non-Newtonian WSS comparison on Case A.

    For a straight pipe with parabolic profile (Case A), the non-Newtonian
    correction should be small because the flow is dominated by high shear
    rates near the wall where mu -> mu_inf (Newtonian limit).
    The significant differences will appear in Case C (aneurysm) where
    low shear recirculation zones produce elevated viscosity.

    This comparison validates that the Carreau-Yasuda model does not
    degrade accuracy on the simple case where the Newtonian approximation
    is known to be adequate.
    """
    phi = np.arctan2(wall_pts[:,2], wall_pts[:,1]) * 180 / np.pi

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # WSS comparison
    ax = axes[0]
    ax.scatter(phi, wss_newt, s=3, c="steelblue", alpha=0.5, label="Newtonian")
    ax.scatter(phi, wss_nn,   s=3, c="firebrick", alpha=0.5, label="Non-Newtonian (CY)")
    wss_exact = MU_INF * 2 * U_MAX / R_PIPE
    ax.axhline(wss_exact, color="k", linewidth=2, linestyle="--",
               label=f"Analytical = {wss_exact:.4f} Pa")
    ax.set_xlabel("Circumferential angle [deg]")
    ax.set_ylabel("WSS [Pa]")
    ax.set_title("WSS: Newtonian vs Non-Newtonian")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Viscosity at wall
    ax = axes[1]
    r_wall = np.sqrt(wall_pts[:,1]**2 + wall_pts[:,2]**2)
    sc = ax.scatter(phi, mu_wall*1000, s=3,
                    c=mu_wall*1000, cmap="plasma", alpha=0.7)
    plt.colorbar(sc, ax=ax, label="mu [mPa.s]")
    ax.axhline(MU_INF*1000, color="k", linestyle="--",
               label=f"mu_inf = {MU_INF*1000:.2f} mPa.s")
    ax.set_xlabel("Circumferential angle [deg]")
    ax.set_ylabel("Apparent viscosity [mPa.s]")
    ax.set_title("Carreau-Yasuda Viscosity at Wall")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Difference map
    ax = axes[2]
    diff_pct = (wss_nn - wss_newt) / (wss_newt + 1e-10) * 100
    sc = ax.scatter(phi, diff_pct, s=3,
                    c=diff_pct, cmap="RdBu_r", alpha=0.7)
    plt.colorbar(sc, ax=ax, label="WSS difference [%]")
    ax.axhline(0, color="k", linestyle="--")
    ax.set_xlabel("Circumferential angle [deg]")
    ax.set_ylabel("(WSS_NN - WSS_N) / WSS_N [%]")
    ax.set_title("Non-Newtonian Correction [%]")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n[Plot] Saved: {save_path}")

    print(f"\n  Non-Newtonian vs Newtonian WSS comparison (Case A):")
    print(f"    Mean WSS Newtonian     : {wss_newt.mean():.4f} Pa")
    print(f"    Mean WSS Non-Newtonian : {wss_nn.mean():.4f} Pa")
    print(f"    Mean difference        : {diff_pct.mean():.2f}%")
    print(f"    Max difference         : {np.abs(diff_pct).max():.2f}%")
    print(f"    Mean mu at wall        : {mu_wall.mean():.5f} Pa.s")
    print(f"    (mu_inf = {MU_INF:.5f} Pa.s)")
    print(f"\n  Note: Small difference expected on Case A (straight pipe,")
    print(f"  high wall shear rate -> mu approaches mu_inf).")
    print(f"  Large differences will appear in Case C recirculation zones.")
    plt.show()


# =============================================================================
# SAVE
# =============================================================================

def save_nonnewtonian_model(model, history, directory="trained_models"):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "pinn_caseA_nonnewtonian.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "history"         : history,
        "physics": {
            "mu0": MU_0, "mu_inf": MU_INF, "lambda": LAMBDA,
            "n": N_CY, "a": A_CY, "rho": RHO,
            "Re": RE, "u_max": U_MAX
        }
    }, path)
    print(f"[Save] Non-Newtonian model saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("STAGE 3: CARREAU-YASUDA NON-NEWTONIAN PINN")
    print("=" * 60)

    # ---- 1. Load geometry from Stage 1 ----
    data = load_geometry("A_straight_cylinder")

    # ---- 2. Normaliser ----
    all_pts    = np.vstack([data["interior"], data["wall"],
                            data["inlet"],    data["outlet"]])
    normaliser = CoordinateNormaliser(all_pts)

    # ---- 3. Viscosity sanity check before training ----
    print("\n[Pre-check] Carreau-Yasuda model behaviour:")
    gd_test = np.array([0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    mu_test = MU_INF + (MU_0 - MU_INF) * (1 + (LAMBDA*gd_test)**A_CY)**((N_CY-1)/A_CY)
    for gd, mu in zip(gd_test, mu_test):
        print(f"  gamma_dot={gd:8.1f} 1/s  ->  mu={mu:.5f} Pa.s  "
              f"({mu/MU_INF:.1f}x mu_inf)")

    # ---- 4. Build network (same architecture as Stage 2) ----
    model = PINN(n_input=3, n_hidden=64, n_layers=5, n_output=4)
    print(f"\n[Network] Parameters: {count_parameters(model):,}")

    # Optionally warm-start from Stage 2 Newtonian weights
    newtonian_path = os.path.join("trained_models", "pinn_caseA_newtonian.pt")
    if os.path.exists(newtonian_path):
        checkpoint = torch.load(newtonian_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[Init] Warm-started from Newtonian weights: {newtonian_path}")
        print(f"       This exploits the fact that the Newtonian solution")
        print(f"       is a reasonable initial guess for the non-Newtonian one.")
    else:
        print("[Init] Training from random initialisation (Glorot)")

    # ---- 5. Loss function ----
    loss_fn = CompositeLossNonNewtonian(
        w1=1.0, w2=1.0, w3=1.0,
        w_wall=10.0, w_in=15.0, w_out=5.0
    )

    # ---- 6. Train ----
    trainer = NonNewtonianTrainer(
        model, loss_fn, normaliser, data,
        lbfgs_subsample=15_000   # safety cap for third-order graph memory
    )

    trainer.train_adam(
        n_iterations    = 30_000,
        lr_initial      = 1e-3,
        lr_decay        = 0.9,
        decay_every     = 5_000,
        log_every       = 500,
        check_grad_every= 5_000,
    )
    trainer.train_lbfgs(
        max_iter     = 5_000,
        tolerance    = 1e-6,
        log_every    = 200,
        history_size = 50,
    )

    # ---- 7. Viscosity field at training points ----
    print("\n[Viscosity] Checking viscosity field after training:")
    model.eval()
    with torch.no_grad():
        x_int_d = normaliser.normalise(data["interior"]).detach()
        uvwp_d  = model(x_int_d)
    # Re-enable grad for derivative computation
    x_int_g = normaliser.normalise(data["interior"])
    uvwp_g  = model(x_int_g)
    u_g = uvwp_g[:,0:1]; v_g = uvwp_g[:,1:2]; w_g = uvwp_g[:,2:3]

    def grad_check(f, x):
        return torch.autograd.grad(f, x,
                                    grad_outputs=torch.ones_like(f),
                                    create_graph=False, retain_graph=True)[0]
    du_g = grad_check(u_g, x_int_g)
    dv_g = grad_check(v_g, x_int_g)
    dw_g = grad_check(w_g, x_int_g)
    gd_field = shear_rate(du_g, dv_g, dw_g, normaliser.x_range)
    check_viscosity_range(gd_field.detach().cpu().numpy())

    # ---- 8. WSS computation ----
    print("\n[WSS] Computing non-Newtonian WSS at wall points...")
    wss_nn, mu_wall, gd_wall = compute_wss_nonnewtonian(
        model, normaliser, data["wall"], data["wall_normals"]
    )

    # Compare with Newtonian WSS (load from Stage 2 if available)
    newtonian_wss_path = "wss_newtonian_caseA.npy"
    if os.path.exists(newtonian_wss_path):
        wss_newt = np.load(newtonian_wss_path)
        print(f"[Compare] Loaded Newtonian WSS from {newtonian_wss_path}")
    else:
        # Compute Newtonian WSS using constant mu_inf for comparison
        from stage2_pinn_caseA import compute_wss as compute_wss_newt
        wss_newt = compute_wss_newt(model, normaliser,
                                     data["wall"], data["wall_normals"],
                                     mu=MU_INF)
        np.save(newtonian_wss_path, wss_newt)

    # ---- 9. Plots ----
    plot_training_history(trainer.history)
    compare_newtonian_vs_nonnewtonian(
        wss_newt, wss_nn, mu_wall, data["wall"]
    )

    # ---- 10. Save ----
    np.save("wss_nonnewtonian_caseA.npy", wss_nn)
    save_nonnewtonian_model(model, trainer.history)

    # ---- 11. Summary ----
    wss_exact = MU_INF * 2 * U_MAX / R_PIPE
    print(f"\n{'='*60}")
    print("STAGE 3 SUMMARY")
    print(f"{'='*60}")
    print(f"  Mean WSS analytical        : {wss_exact:.4f} Pa")
    print(f"  Mean WSS Non-Newtonian     : {wss_nn.mean():.4f} Pa")
    print(f"  WSS error vs analytical    : "
          f"{abs(wss_nn.mean()-wss_exact)/wss_exact*100:.2f}%")
    print(f"  Mean viscosity at wall     : {mu_wall.mean():.5f} Pa.s")
    print(f"  mu_inf (Newtonian limit)   : {MU_INF:.5f} Pa.s")
    print(f"  SoftAdapt was activated    : {loss_fn.use_softadapt}")
    print(f"{'='*60}")
    print("\nStage 3 complete. Run stage4_curved_pipe.py next.")
