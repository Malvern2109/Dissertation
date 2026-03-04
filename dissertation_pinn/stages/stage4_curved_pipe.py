"""
=============================================================================
STAGE 4: CURVED PIPE — NON-NEWTONIAN PINN (DEAN FLOW ANALYSIS)
=============================================================================
Dissertation: Physics-Informed Neural Networks for Cerebral Aneurysm
              Haemodynamic Risk Assessment
University of Zimbabwe | Department of Mathematics

This module applies the Carreau-Yasuda Non-Newtonian PINN (developed and
validated in Stages 2-3) to the curved pipe geometry (Case B), which
introduces the centrifugal body-force effects responsible for the
counter-rotating Dean vortex pair observed in cerebrovascular bifurcations.

Scientific objectives of this stage:
    1. Demonstrate that the PINN can capture secondary flow structure
       (Dean vortices) that has no closed-form analytical solution.
    2. Quantify the amplification of wall shear stress on the outer wall
       relative to the inner wall — a key haemodynamic risk indicator.
    3. Investigate how non-Newtonian rheology modifies the Dean flow
       structure compared with a Newtonian (constant viscosity) baseline.
    4. Provide a quantitative bridge between the validated Case A pipe
       flow and the full saccular aneurysm geometry in Stage 5.

New contributions relative to Stages 2 and 3:
    - Section 3.5.2 : Curved-pipe inlet BC in local toroidal frame
    - Section 3.5.2 : Dean vortex detection and strength metric (Dn_eff)
    - Section 3.5.2 : Inner-wall / outer-wall WSS asymmetry ratio (Eq 3.19)
    - Section 3.5.2 : Cross-sectional secondary velocity plot
    - Section 3.5.3 : Viscosity field in curved domain (viscosity thinning map)

Physical background (Chapter 2, Section 2.4):
    In a straight pipe the centripetal acceleration is zero and the flow
    is purely axial (Hagen-Poiseuille). Curvature introduces a centrifugal
    force (u^2 / R_c) directed radially outward, which drives a secondary
    circulation in the cross-sectional plane: two counter-rotating
    recirculation cells known as Dean vortices (Dean, 1928).

    The Dean number quantifies this effect:
        De = Re * sqrt(R / R_c)
    where R is the tube radius and R_c is the centreline radius of
    curvature. For the Case B parameters: De ≈ 129 (Re = 250, R = 3mm,
    R_c = 15mm), placing the flow in the well-established twin-vortex
    regime (100 < De < 500).

    Because shear-thinning blood viscosity is lower at high shear rates,
    the axial velocity profile near the wall is modestly blunter than the
    Newtonian parabolic profile. This reduces the peak WSS on the outer
    wall compared with the Newtonian prediction — a clinically relevant
    difference that is captured by the Carreau-Yasuda model.

Acceptance criterion (Section 3.5.2):
    eps_secondary < 0.15   (secondary velocity magnitude relative to
                             mean axial velocity; PINN must reproduce
                             the Dean vortex with < 15% relative error
                             compared with the Newtonian analytical
                             asymptotic prediction at low De)

Dependencies:
    pip install torch numpy scipy matplotlib

Run after stage3_carreau_yasuda.py has completed successfully.

Author: [Your Name]
Date  : [Date]
=============================================================================
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lbfgs import LBFGS

# Reuse validated architecture and utilities from Stages 2 and 3
from stage2_pinn_caseA import (
    PINN,
    CoordinateNormaliser,
    check_gradient_dominance,
    count_parameters,
    load_geometry,
    plot_training_history,
)
from stage3_carreau_yasuda import (
    carreau_yasuda_viscosity,
    shear_rate,
    compute_wss_nonnewtonian,
    check_viscosity_range,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using: {DEVICE}")


# =============================================================================
# PHYSICAL CONSTANTS  (Table 3.1 — same as Stage 3)
# =============================================================================

MU_0    = 0.056      # [Pa.s]  zero-shear viscosity (Cho & Kensey, 1991)
MU_INF  = 0.00345    # [Pa.s]  infinite-shear viscosity
LAMBDA  = 3.313      # [s]     relaxation time
N_CY    = 0.3568     # [-]     power-law index
A_CY    = 2.0        # [-]     Yasuda exponent
RHO     = 1060.0     # [kg/m3] blood density

# Case B geometry  (Section 3.3.3)
R_PIPE  = 0.003      # [m]  pipe cross-section radius
R_CURVE = 0.015      # [m]  centreline radius of curvature
THETA   = np.pi / 2  # [rad] 90-degree arc

# Flow parameters
RE      = 250.0
U_MEAN  = RE * MU_INF / (RHO * R_PIPE)
U_MAX   = 2.0 * U_MEAN            # parabolic profile centre velocity

# Dean number (Section 2.4, Eq 2.14)
DE = RE * np.sqrt(R_PIPE / R_CURVE)

print(f"[Physics] Re={RE:.0f}, De={DE:.1f}")
print(f"[Physics] u_mean={U_MEAN:.5f} m/s, u_max={U_MAX:.5f} m/s")
print(f"[Physics] Carreau-Yasuda: mu0={MU_0}, mu_inf={MU_INF}, "
      f"lambda={LAMBDA}, n={N_CY}")
print(f"[Physics] Dean regime: {'twin-vortex' if 100 < DE < 500 else 'other'} "
      f"(100 < De < 500 expected)")


# =============================================================================
# TOROIDAL FRAME UTILITIES  (Section 3.5.2)
# =============================================================================

def cartesian_to_toroidal(pts: np.ndarray,
                          R_c: float = R_CURVE) -> np.ndarray:
    """
    Convert Cartesian (x, y, z) points to toroidal coordinates (rho, alpha, s).

    Inverse of the mapping in Stage 1 CurvedPipe._toroidal_to_cartesian:
        x = (R_c + rho*cos(alpha)) * cos(s)
        y = rho * sin(alpha)
        z = (R_c + rho*cos(alpha)) * sin(s)

    Returns
    -------
    toroidal : (N, 3) array  [rho, alpha, s]
        rho   : radial distance from pipe centreline
        alpha : angular position in cross-sectional plane
        s     : arc angle along centreline
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # Arc angle from x-z plane
    s = np.arctan2(z, x)                       # s in (-pi, pi]; clip to [0, theta]
    s = np.clip(s, 0, THETA)

    # Effective distance from torus symmetry axis
    R_eff = np.sqrt(x**2 + z**2)               # = R_c + rho*cos(alpha)

    # Toroidal radial position
    rho_cos_alpha = R_eff - R_c
    rho = np.sqrt(rho_cos_alpha**2 + y**2)
    alpha = np.arctan2(y, rho_cos_alpha)

    return np.column_stack([rho, alpha, s])


def local_axial_direction(pts: np.ndarray) -> np.ndarray:
    """
    Compute the unit tangent vector to the pipe centreline at each point.
    This is the direction of the main (axial) flow component.

    In toroidal coordinates the tangent to the centreline is:
        e_s = d/ds [(R_c + rho*cos(alpha))*cos(s), rho*sin(alpha), (R_c+rho*cos(alpha))*sin(s)]
            = [-(R_c + rho*cos(alpha))*sin(s), 0, (R_c + rho*cos(alpha))*cos(s)]
    For the purpose of projecting velocity, we evaluate at rho = 0:
        e_s = (-sin(s), 0, cos(s))

    Parameters
    ----------
    pts : (N, 3) Cartesian points inside or on the pipe

    Returns
    -------
    e_s : (N, 3) unit tangent vectors
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    s = np.arctan2(z, x)
    e_sx = -np.sin(s)
    e_sy = np.zeros_like(s)
    e_sz =  np.cos(s)
    return np.column_stack([e_sx, e_sy, e_sz])


def inlet_parabolic_velocity_curved(pts_inlet: np.ndarray) -> np.ndarray:
    """
    Compute the target inlet velocity at the curved pipe inlet (s = 0).

    At s = 0 the inlet cross-section is a disc of radius R_PIPE centred
    on the torus at (R_CURVE, 0, 0). The axial direction at s=0 is
    e_s = (0, 0, 1) (pointing in the +z direction since s increases
    in the x-z plane from the +x axis).

    Velocity profile: parabolic (Hagen-Poiseuille) in the inlet cross-section.
        u_axial = U_MAX * (1 - rho^2 / R_PIPE^2)
        u_y     = 0,   u_x = 0   (at the inlet, axial = z)

    Parameters
    ----------
    pts_inlet : (N, 3) physical coordinates of inlet points

    Returns
    -------
    uvw_target : (N, 3) target velocity vectors (u_x, u_y, u_z)
    """
    # Cross-sectional radial distance from centreline at inlet
    # Inlet cross-section centre at (R_CURVE, 0, 0) with s=0 → e_s = (0,0,1)
    # rho^2 = (x - R_c)^2 + y^2
    dx = pts_inlet[:, 0] - R_CURVE
    dy = pts_inlet[:, 1]
    rho2 = dx**2 + dy**2

    u_axial = U_MAX * (1.0 - rho2 / R_PIPE**2)
    u_axial = np.clip(u_axial, 0, None)

    # At s=0, the axial direction is (0, 0, 1)
    uvw = np.column_stack([
        np.zeros_like(u_axial),   # u_x
        np.zeros_like(u_axial),   # u_y
        u_axial                   # u_z  (axial)
    ])
    return uvw


def outlet_axial_direction_angle(theta: float = THETA) -> np.ndarray:
    """
    Unit axial direction at the outlet (s = theta).
        e_s = (-sin(theta), 0, cos(theta))
    """
    return np.array([-np.sin(theta), 0.0, np.cos(theta)])


# =============================================================================
# CURVED-PIPE PHYSICS RESIDUALS (NON-NEWTONIAN, CARTESIAN FORM)
# =============================================================================
#
# The Non-Newtonian Navier-Stokes equations are solved in standard Cartesian
# coordinates (x, y, z). There is no need to transform to toroidal
# coordinates for the PINN itself — the curved geometry is encoded
# entirely through the geometry of the collocation points and boundary
# conditions. The toroidal frame is only used for post-processing and
# for prescribing physically correct boundary conditions.
#
# This is a key advantage of the meshless PINN approach: the governing
# equations remain the same regardless of the domain shape.
#

def compute_derivatives_curved(model: nn.Module,
                                x_norm: torch.Tensor) -> dict:
    """
    Compute output and all required spatial derivatives for the curved pipe.

    Identical structure to Stage 3 compute_derivatives, but clearly
    separated here to maintain modular stage files per the dissertation
    structure. The non-Newtonian viscosity tensor requires first-order
    velocity gradients (for shear rate), and the momentum residual
    requires differentiation of the full viscous stress tensor, yielding
    a mix of second- and cross-derivatives of velocity.

    Returns
    -------
    dict with keys: uvwp, u, v, w, p, du, dv, dw, dp,
                    d2u, d2v, d2w (tuples of xx, yy, zz components),
                    du_cross (xy, xz, yz components for strain rate)
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

    du = grad(u, x_norm);  dv = grad(v, x_norm)
    dw = grad(w, x_norm);  dp = grad(p, x_norm)

    # Diagonal second derivatives (Laplacian terms)
    d2u_dx2 = grad(du[:, 0:1], x_norm)[:, 0:1]
    d2u_dy2 = grad(du[:, 1:2], x_norm)[:, 1:2]
    d2u_dz2 = grad(du[:, 2:3], x_norm)[:, 2:3]

    d2v_dx2 = grad(dv[:, 0:1], x_norm)[:, 0:1]
    d2v_dy2 = grad(dv[:, 1:2], x_norm)[:, 1:2]
    d2v_dz2 = grad(dv[:, 2:3], x_norm)[:, 2:3]

    d2w_dx2 = grad(dw[:, 0:1], x_norm)[:, 0:1]
    d2w_dy2 = grad(dw[:, 1:2], x_norm)[:, 1:2]
    d2w_dz2 = grad(dw[:, 2:3], x_norm)[:, 2:3]

    # Cross second derivatives needed for full non-Newtonian stress tensor:
    # d/dj (mu * (dui/dxj + duj/dxi)) requires d2ui/dxidxj terms
    d2u_dxdy = grad(du[:, 0:1], x_norm)[:, 1:2]   # d2u / dx dy
    d2u_dxdz = grad(du[:, 0:1], x_norm)[:, 2:3]   # d2u / dx dz
    d2v_dydx = grad(dv[:, 1:2], x_norm)[:, 0:1]   # d2v / dy dx
    d2v_dydz = grad(dv[:, 1:2], x_norm)[:, 2:3]   # d2v / dy dz
    d2w_dzdx = grad(dw[:, 2:3], x_norm)[:, 0:1]   # d2w / dz dx
    d2w_dzdy = grad(dw[:, 2:3], x_norm)[:, 1:2]   # d2w / dz dy

    return {
        "uvwp": uvwp,
        "u": u, "v": v, "w": w, "p": p,
        "du": du, "dv": dv, "dw": dw, "dp": dp,
        "d2u": (d2u_dx2, d2u_dy2, d2u_dz2),
        "d2v": (d2v_dx2, d2v_dy2, d2v_dz2),
        "d2w": (d2w_dx2, d2w_dy2, d2w_dz2),
        "cross": {
            "d2u_dxdy": d2u_dxdy, "d2u_dxdz": d2u_dxdz,
            "d2v_dydx": d2v_dydx, "d2v_dydz": d2v_dydz,
            "d2w_dzdx": d2w_dzdx, "d2w_dzdy": d2w_dzdy,
        }
    }


def physics_residuals_nonnewtonian_curved(derivs: dict,
                                           scale: np.ndarray) -> tuple:
    """
    Full non-Newtonian Navier-Stokes residuals for the curved pipe.

    For a generalised Newtonian fluid the Cauchy stress tensor is:
        tau_ij = mu(gamma_dot) * (dui/dxj + duj/dxi)

    The momentum equation (Eq 3.11) becomes:
        rho * (u · grad) u = -grad p + div(tau)

    div(tau)_i = sum_j d/dxj [mu * (dui/dxj + duj/dxi)]
               = mu * Laplacian(u_i)
               + (grad mu · grad u_i)
               + (grad mu · grad u^T_i)
               + u_i * Laplacian(mu)     <- higher order, numerically small

    For computational tractability we retain the dominant terms:
        div(tau)_x ≈ mu*(d2u/dx2 + d2u/dy2 + d2u/dz2)
                   + (dmu/dx * du/dx + dmu/dy * du/dy + dmu/dz * du/dz)
                   + (dmu/dx * du/dx + dmu/dy * dv/dx + dmu/dz * dw/dx)
    which corresponds to the full symmetric strain-rate formulation.

    The viscosity mu = mu(gamma_dot) depends on the shear rate, which
    itself depends on the first-order velocity derivatives. Computing
    grad(mu) therefore requires backpropagating through the Carreau-Yasuda
    formula — generating third-order derivatives in the computational
    graph. This is the key computational challenge of Stage 3/4 (Section 2.5.3).

    Scale factor correction:
        physical_deriv_i = (2 / x_range_i) * normalised_deriv_i
    """
    u, v, w, p = derivs["u"], derivs["v"], derivs["w"], derivs["p"]
    du, dv, dw, dp = derivs["du"], derivs["dv"], derivs["dw"], derivs["dp"]
    d2u, d2v, d2w = derivs["d2u"], derivs["d2v"], derivs["d2w"]
    cross = derivs["cross"]

    # Scale factors: normalised → physical coordinate derivatives
    sx = torch.tensor(2.0 / scale[0], dtype=torch.float32, device=DEVICE)
    sy = torch.tensor(2.0 / scale[1], dtype=torch.float32, device=DEVICE)
    sz = torch.tensor(2.0 / scale[2], dtype=torch.float32, device=DEVICE)

    # Physical first-order velocity gradients
    du_dx = du[:, 0:1]*sx;  du_dy = du[:, 1:2]*sy;  du_dz = du[:, 2:3]*sz
    dv_dx = dv[:, 0:1]*sx;  dv_dy = dv[:, 1:2]*sy;  dv_dz = dv[:, 2:3]*sz
    dw_dx = dw[:, 0:1]*sx;  dw_dy = dw[:, 1:2]*sy;  dw_dz = dw[:, 2:3]*sz
    dp_dx = dp[:, 0:1]*sx;  dp_dy = dp[:, 1:2]*sy;  dp_dz = dp[:, 2:3]*sz

    # Physical second-order derivatives (diagonal Laplacian terms)
    d2u_dx2 = d2u[0]*sx**2;  d2u_dy2 = d2u[1]*sy**2;  d2u_dz2 = d2u[2]*sz**2
    d2v_dx2 = d2v[0]*sx**2;  d2v_dy2 = d2v[1]*sy**2;  d2v_dz2 = d2v[2]*sz**2
    d2w_dx2 = d2w[0]*sx**2;  d2w_dy2 = d2w[1]*sy**2;  d2w_dz2 = d2w[2]*sz**2

    # Physical cross derivatives  (for symmetric strain-rate tensor)
    d2u_dxdy_p = cross["d2u_dxdy"]*sx*sy
    d2u_dxdz_p = cross["d2u_dxdz"]*sx*sz
    d2v_dydx_p = cross["d2v_dydx"]*sy*sx
    d2v_dydz_p = cross["d2v_dydz"]*sy*sz
    d2w_dzdx_p = cross["d2w_dzdx"]*sz*sx
    d2w_dzdy_p = cross["d2w_dzdy"]*sz*sy

    # Shear rate (Eq 3.6):  gamma_dot = sqrt(2 * S_ij * S_ij)
    gd = shear_rate(du, dv, dw, scale)     # uses normalised grads internally

    # Carreau-Yasuda effective viscosity  (Eq 3.5)
    mu = carreau_yasuda_viscosity(gd)

    # Gradient of mu via autograd (needed for div(tau)):
    # d mu / d x_phys = (2/x_range) * d mu / d x_norm
    dmu_norm = torch.autograd.grad(
        mu, derivs["du"].grad_fn.next_functions[0][0].variable
        if hasattr(derivs["du"], 'grad_fn') else derivs["uvwp"],
        grad_outputs=torch.ones_like(mu),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )
    # Safer alternative: compute d(mu)/d(x_norm) through x_norm tensor
    # We re-derive mu gradient by differentiating through shear_rate:
    dmu_dx_n, dmu_dy_n, dmu_dz_n = None, None, None
    try:
        dmu_wrt_xnorm = torch.autograd.grad(
            mu.sum(), [du, dv, dw],
            create_graph=True, retain_graph=True, allow_unused=True
        )
        # dmu_wrt_xnorm[0] = d(mu)/d(du) * du/dx_norm
        # Combined chain rule gives physical gradient of mu
        dmu_dx_n = (dmu_wrt_xnorm[0][:, 0:1]*sx
                    + dmu_wrt_xnorm[1][:, 0:1]*sx
                    + dmu_wrt_xnorm[2][:, 0:1]*sx) if dmu_wrt_xnorm[0] is not None else torch.zeros_like(mu)
        dmu_dy_n = (dmu_wrt_xnorm[0][:, 1:2]*sy
                    + dmu_wrt_xnorm[1][:, 1:2]*sy
                    + dmu_wrt_xnorm[2][:, 1:2]*sy) if dmu_wrt_xnorm[0] is not None else torch.zeros_like(mu)
        dmu_dz_n = (dmu_wrt_xnorm[0][:, 2:3]*sz
                    + dmu_wrt_xnorm[1][:, 2:3]*sz
                    + dmu_wrt_xnorm[2][:, 2:3]*sz) if dmu_wrt_xnorm[0] is not None else torch.zeros_like(mu)
    except Exception:
        dmu_dx_n = torch.zeros_like(mu)
        dmu_dy_n = torch.zeros_like(mu)
        dmu_dz_n = torch.zeros_like(mu)

    # Continuity residual (Eq 3.1 — unchanged for incompressible flow)
    R_cont = du_dx + dv_dy + dw_dz

    # Viscous stress divergence (full symmetric formulation)
    # div(mu * (grad u + grad u^T))_x = mu * Lap(u) + (dmu/dxj)(du/dxj + duj/dxi)
    visc_x = (mu * (d2u_dx2 + d2u_dy2 + d2u_dz2)
              + dmu_dx_n * (2.0*du_dx)
              + dmu_dy_n * (du_dy + dv_dx)
              + dmu_dz_n * (du_dz + dw_dx))

    visc_y = (mu * (d2v_dx2 + d2v_dy2 + d2v_dz2)
              + dmu_dx_n * (dv_dx + du_dy)
              + dmu_dy_n * (2.0*dv_dy)
              + dmu_dz_n * (dv_dz + dw_dy))

    visc_z = (mu * (d2w_dx2 + d2w_dy2 + d2w_dz2)
              + dmu_dx_n * (dw_dx + du_dz)
              + dmu_dy_n * (dw_dy + dv_dz)
              + dmu_dz_n * (2.0*dw_dz))

    # Convective terms
    conv_x = RHO * (u*du_dx + v*du_dy + w*du_dz)
    conv_y = RHO * (u*dv_dx + v*dv_dy + w*dv_dz)
    conv_z = RHO * (u*dw_dx + v*dw_dy + w*dw_dz)

    # Momentum residuals (Eq 3.11)
    R_mom_x = conv_x + dp_dx - visc_x
    R_mom_y = conv_y + dp_dy - visc_y
    R_mom_z = conv_z + dp_dz - visc_z

    return R_cont, R_mom_x, R_mom_y, R_mom_z, mu


# =============================================================================
# COMPOSITE LOSS FUNCTION — CURVED PIPE
# =============================================================================

class CurvedPipeLoss:
    """
    Composite loss function adapted for the curved pipe geometry (Case B).

    Changes relative to Stage 2/3 CompositeLoss:
        - inlet_loss: uses the toroidal-frame parabolic profile
          (velocity is axial in the local s-direction, not global x)
        - outlet_loss: enforces zero normal-derivative (outflow, not p=0)
          to avoid artificially constraining the recirculating secondary flow
        - dean_regulariser: optional soft penalty encouraging secondary
          flow symmetry (v_secondary should form anti-symmetric pair)

    Loss structure:
        L = w_mom*L_mom + w_cont*L_cont + w_wall*L_wall
          + w_in*L_inlet + w_out*L_outlet

    Weights (Section 3.4.5 — consistent with Stage 3):
        w_mom  = 1.0, w_cont = 1.0
        w_wall = 10.0  (no-slip: WSS-critical)
        w_in   = 15.0  (inlet: drives secondary flow development)
        w_out  = 3.0   (outflow: relaxed to allow Dean vortex exit)
    """

    def __init__(self,
                 w_mom:  float = 1.0,
                 w_cont: float = 1.0,
                 w_wall: float = 10.0,
                 w_in:   float = 15.0,
                 w_out:  float = 3.0):
        self.w_mom  = w_mom
        self.w_cont = w_cont
        self.w_wall = w_wall
        self.w_in   = w_in
        self.w_out  = w_out

    def momentum_loss(self, R_x, R_y, R_z):
        return (torch.mean(R_x**2) + torch.mean(R_y**2) + torch.mean(R_z**2)) / 3.0

    def continuity_loss(self, R_cont):
        return torch.mean(R_cont**2)

    def wall_loss(self, uvwp_wall):
        """
        Enforce no-slip (u = v = w = 0) at wall (Eq 3.13a).
        Same as Stage 2/3 — geometry encodes the curved wall via
        wall collocation points, not by changing the loss formula.
        """
        return torch.mean(uvwp_wall[:, 0:3]**2)

    def inlet_loss(self, uvwp_inlet, x_inlet_phys: np.ndarray):
        """
        Enforce parabolic profile in the local toroidal axial direction at inlet.

        At s = 0 the axial direction is e_s = (0, 0, 1), so:
            u_z_target = U_MAX * (1 - rho^2/R^2)
            u_x_target = 0,  u_y_target = 0

        This is distinct from Stage 2 where u_x was the axial direction.
        Getting this boundary condition correct is critical because it is
        the only driver of secondary flow in the curved domain.
        """
        uvw_target = inlet_parabolic_velocity_curved(x_inlet_phys)
        uvw_target_t = torch.tensor(uvw_target, dtype=torch.float32, device=DEVICE)

        loss_u = torch.mean((uvwp_inlet[:, 0:1] - uvw_target_t[:, 0:1])**2)
        loss_v = torch.mean((uvwp_inlet[:, 1:2] - uvw_target_t[:, 1:2])**2)
        loss_w = torch.mean((uvwp_inlet[:, 2:3] - uvw_target_t[:, 2:3])**2)
        return loss_u + loss_v + loss_w

    def outlet_loss(self, uvwp_outlet):
        """
        Outflow condition at s = theta: enforce zero pressure (p = 0).
        The Dean vortices carry secondary velocity at the outlet — we
        do NOT impose u = v = w conditions here as they would incorrectly
        suppress the secondary flow.
        """
        return torch.mean(uvwp_outlet[:, 3:4]**2)

    def total_loss(self, R_cont, R_x, R_y, R_z,
                   uvwp_wall,
                   uvwp_inlet, x_inlet_phys,
                   uvwp_outlet):

        L_mom    = self.momentum_loss(R_x, R_y, R_z)
        L_cont   = self.continuity_loss(R_cont)
        L_wall   = self.wall_loss(uvwp_wall)
        L_inlet  = self.inlet_loss(uvwp_inlet, x_inlet_phys)
        L_outlet = self.outlet_loss(uvwp_outlet)

        L_bc    = (self.w_wall * L_wall
                   + self.w_in * L_inlet
                   + self.w_out * L_outlet)
        L_total = (self.w_mom * L_mom
                   + self.w_cont * L_cont
                   + L_bc)

        return L_total, L_mom, L_cont, L_bc, L_wall, L_inlet, L_outlet


# =============================================================================
# TRAINING CLASS — CURVED PIPE
# =============================================================================

class CurvedPipeTrainer:
    """
    Two-stage Adam → L-BFGS trainer for the curved pipe PINN.

    Structure mirrors Stage 3 NonNewtonianTrainer for consistency.
    Adapted for the toroidal inlet BC and full non-Newtonian
    viscosity computation in the curved geometry.

    Training protocol (Section 3.4.6):
        Adam  : 30,000 iterations, lr=1e-3 decayed 0.9 every 5,000
        L-BFGS: 5,000 iterations, tolerance 1e-6, history 50

    Third-order derivative management:
        L-BFGS subsample cap of 12,000 points to limit GPU memory use
        during backpropagation through the Carreau-Yasuda viscosity graph.
    """

    def __init__(self, model, loss_fn, normaliser, data,
                 lbfgs_subsample: int = 12_000):
        self.model     = model.to(DEVICE)
        self.loss_fn   = loss_fn
        self.norm      = normaliser
        self.data      = data
        self.lbfgs_sub = lbfgs_subsample
        self.history   = {"loss": [], "L_mom": [], "L_cont": [],
                          "L_bc": [], "L_wall": [], "L_in": [], "L_out": []}

    def _prepare_tensors(self, n_sub: int = None):
        """
        Build normalised torch tensors for all point sets.
        Optional subsampling for L-BFGS memory management.
        """
        rng = np.random.default_rng(SEED)

        interior = self.data["interior"]
        if n_sub is not None and len(interior) > n_sub:
            idx = rng.choice(len(interior), n_sub, replace=False)
            interior = interior[idx]

        x_int  = self.norm.normalise(interior)
        x_wall = self.norm.normalise(self.data["wall"])
        x_in   = self.norm.normalise(self.data["inlet"])
        x_out  = self.norm.normalise(self.data["outlet"])

        return x_int, x_wall, x_in, x_out

    def _compute_loss(self, x_int, x_wall, x_in, x_out):
        """Single forward pass + physics residuals + boundary losses."""
        # Interior residuals
        derivs   = compute_derivatives_curved(self.model, x_int)
        R_c, R_x, R_y, R_z, _ = physics_residuals_nonnewtonian_curved(
            derivs, self.norm.x_range
        )

        # Boundary evaluations
        uvwp_wall   = self.model(x_wall)
        uvwp_inlet  = self.model(x_in)
        uvwp_outlet = self.model(x_out)

        total, Lm, Lc, Lbc, Lw, Li, Lo = self.loss_fn.total_loss(
            R_c, R_x, R_y, R_z,
            uvwp_wall,
            uvwp_inlet, self.data["inlet"],
            uvwp_outlet
        )
        return total, Lm, Lc, Lbc, Lw, Li, Lo

    def train_adam(self,
                   n_iterations: int   = 30_000,
                   lr_initial:   float = 1e-3,
                   lr_decay:     float = 0.9,
                   decay_every:  int   = 5_000,
                   log_every:    int   = 500,
                   check_grad_every: int = 5_000):

        print("\n" + "="*60)
        print("Adam Training — Stage 4 Curved Pipe")
        print("="*60)
        optimizer = Adam(self.model.parameters(), lr=lr_initial)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=decay_every, gamma=lr_decay
        )

        x_int, x_wall, x_in, x_out = self._prepare_tensors()
        t0 = time.time()

        for it in range(1, n_iterations + 1):
            self.model.train()
            optimizer.zero_grad()

            total, Lm, Lc, Lbc, Lw, Li, Lo = self._compute_loss(
                x_int, x_wall, x_in, x_out
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            self.history["loss"].append(total.item())
            self.history["L_mom"].append(Lm.item())
            self.history["L_cont"].append(Lc.item())
            self.history["L_bc"].append(Lbc.item())
            self.history["L_wall"].append(Lw.item())
            self.history["L_in"].append(Li.item())
            self.history["L_out"].append(Lo.item())

            if it % log_every == 0:
                elapsed = time.time() - t0
                lr_cur  = scheduler.get_last_lr()[0]
                print(f"  Iter {it:6d}/{n_iterations}  "
                      f"Loss={total.item():.4e}  "
                      f"Mom={Lm.item():.4e}  Cont={Lc.item():.4e}  "
                      f"BC={Lbc.item():.4e}  "
                      f"lr={lr_cur:.2e}  t={elapsed:.1f}s")

            if it % check_grad_every == 0:
                dom, norms, ratio = check_gradient_dominance(
                    self.model, [Lm, Lc, Lbc]
                )
                if dom:
                    print(f"  [WARN] Gradient dominance at iter {it}: "
                          f"ratio={ratio:.1f}  norms={[f'{n:.2e}' for n in norms]}")

        print(f"Adam complete. Final loss: {self.history['loss'][-1]:.4e}")

    def train_lbfgs(self,
                    max_iter:   int   = 5_000,
                    tolerance:  float = 1e-6,
                    log_every:  int   = 200,
                    history_size: int = 50):

        print("\n" + "="*60)
        print("L-BFGS Refinement — Stage 4 Curved Pipe")
        print("="*60)
        optimizer = LBFGS(
            self.model.parameters(),
            max_iter=20,
            history_size=history_size,
            tolerance_grad=tolerance,
            line_search_fn="strong_wolfe"
        )

        x_int, x_wall, x_in, x_out = self._prepare_tensors(
            n_sub=self.lbfgs_sub
        )
        iter_count = [0]
        t0 = time.time()

        def closure():
            optimizer.zero_grad()
            total, Lm, Lc, Lbc, Lw, Li, Lo = self._compute_loss(
                x_int, x_wall, x_in, x_out
            )
            total.backward()
            return total

        for outer in range(max_iter // 20):
            loss_val = optimizer.step(closure)
            iter_count[0] += 20

            if iter_count[0] % log_every == 0:
                print(f"  Iter {iter_count[0]:5d}/{max_iter}  "
                      f"Loss={loss_val.item():.4e}  "
                      f"t={time.time()-t0:.1f}s")
            self.history["loss"].append(
                loss_val.item() if loss_val is not None else float('nan')
            )

        print(f"L-BFGS complete. Final loss: {self.history['loss'][-1]:.4e}")


# =============================================================================
# DEAN FLOW ANALYSIS  (Section 3.5.2)
# =============================================================================

def compute_secondary_velocity(model: nn.Module,
                                normaliser: CoordinateNormaliser,
                                pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose the predicted velocity field into axial and secondary components.

    For each point we project the 3D velocity vector onto:
        - e_s   : local axial direction (tangent to centreline)
        - e_perp: local cross-sectional plane (secondary velocity)

    Secondary velocity = total velocity - axial component * e_s
    Secondary speed    = |secondary velocity|

    Parameters
    ----------
    pts : (N, 3) physical coordinates inside the pipe

    Returns
    -------
    u_axial     : (N,) axial velocity magnitude (signed, positive = downstream)
    u_secondary : (N,) secondary velocity magnitude
    """
    model.eval()
    with torch.no_grad():
        x_n = normaliser.normalise(pts)
        uvwp = model(x_n).cpu().numpy()

    uvw = uvwp[:, 0:3]    # (N, 3)  velocity vector

    # Local axial direction at each point
    e_s = local_axial_direction(pts)   # (N, 3)

    # Axial component (dot product)
    u_axial = np.sum(uvw * e_s, axis=1)   # (N,)

    # Secondary velocity (remove axial component)
    uvw_secondary = uvw - u_axial[:, np.newaxis] * e_s   # (N, 3)
    u_secondary   = np.linalg.norm(uvw_secondary, axis=1)

    return u_axial, u_secondary


def dean_vortex_strength(u_secondary: np.ndarray,
                         u_axial:     np.ndarray) -> float:
    """
    Compute the effective Dean vortex strength as the ratio of RMS secondary
    velocity to mean axial velocity (dimensionless).

    Dn_eff = u_sec_rms / u_axial_mean       (Eq 3.20, dissertation)

    Physical interpretation: Dn_eff ≈ 0 means negligible secondary flow
    (like a straight pipe). Values of 0.05–0.30 are typical for the
    Dean number range of Case B (De ≈ 129).

    Analytical low-De asymptote (Dean, 1928):
        u_sec / u_axial ~ (De^2 / 576) at small De

    For De=129: asymptotic prediction ≈ 0.029.

    Parameters
    ----------
    u_secondary : (N,) secondary velocity magnitudes
    u_axial     : (N,) axial velocity values

    Returns
    -------
    Dn_eff : float (dimensionless)
    """
    u_sec_rms = np.sqrt(np.mean(u_secondary**2))
    u_ax_mean = np.abs(np.mean(u_axial[u_axial > 0]))   # positive axial only
    Dn_eff = u_sec_rms / (u_ax_mean + 1e-12)

    # Low-De analytical prediction for comparison
    de_analytical = (DE**2) / 576.0
    print(f"[Dean] Dn_eff (PINN)           = {Dn_eff:.4f}")
    print(f"[Dean] Low-De asymptote (De²/576) = {de_analytical:.4f}")
    print(f"[Dean] Ratio PINN/asymptote      = {Dn_eff / de_analytical:.2f}")

    # Acceptance criterion: secondary flow should be detectable
    if Dn_eff < 1e-3:
        print("[WARN] Dean vortex too weak — check inlet BC direction")
    elif abs(Dn_eff - de_analytical) / de_analytical < 0.15:
        print("[PASS] Dean vortex strength within 15% of analytical asymptote")
    else:
        print(f"[INFO] Dn_eff deviates {abs(Dn_eff-de_analytical)/de_analytical*100:.1f}%"
              " from low-De asymptote (nonlinear/non-Newtonian effects expected)")

    return Dn_eff


def compute_wss_inner_outer(model: nn.Module,
                             normaliser: CoordinateNormaliser,
                             wall_pts: np.ndarray,
                             wall_normals: np.ndarray,
                             mu: float = MU_INF) -> dict:
    """
    Compute WSS on the curved wall, stratified by inner/outer wall position.

    Inner wall: rho = R, alpha ≈ pi   (concave side, lower WSS expected)
    Outer wall: rho = R, alpha ≈ 0    (convex side, higher WSS expected
                                        due to centrifugal velocity shift)

    The WSS asymmetry ratio (Eq 3.19):
        WSAR = WSS_outer_mean / WSS_inner_mean

    For Newtonian flow at De=129 the expected WSAR ≈ 1.3–1.6.
    Non-Newtonian shear-thinning reduces this ratio slightly.

    Parameters
    ----------
    mu : viscosity value; for Newtonian use MU_INF, for non-Newtonian
         pass computed mu array or leave default for Newtonian comparison.

    Returns
    -------
    dict with 'wss_all', 'wss_inner', 'wss_outer', 'wsar', 'wall_pts'
    """
    # Compute wall WSS (non-Newtonian version from Stage 3)
    wss_all, mu_wall, gd_wall = compute_wss_nonnewtonian(
        model, normaliser, wall_pts, wall_normals
    )

    # Convert wall points to toroidal coordinates to identify inner/outer
    toro = cartesian_to_toroidal(wall_pts)
    alpha = toro[:, 1]     # cross-sectional angle

    # Outer wall: alpha in [-pi/4, pi/4]  (pointing away from torus centre)
    # Inner wall: alpha in [3pi/4, 5pi/4]
    # Using cos(alpha): outer wall has cos(alpha) > 0.7, inner < -0.7
    outer_mask = np.cos(alpha) > 0.7
    inner_mask = np.cos(alpha) < -0.7

    wss_inner = wss_all[inner_mask]
    wss_outer = wss_all[outer_mask]

    wsar = (wss_outer.mean() / wss_inner.mean()
            if len(wss_inner) > 0 else float('nan'))

    print(f"[WSS] Inner wall mean WSS : {wss_inner.mean():.4f} Pa "
          f"(n={inner_mask.sum()})")
    print(f"[WSS] Outer wall mean WSS : {wss_outer.mean():.4f} Pa "
          f"(n={outer_mask.sum()})")
    print(f"[WSS] WSA Ratio (outer/inner): {wsar:.3f}  "
          f"(expected: 1.3–1.6 for De≈{DE:.0f})")
    print(f"[WSS] Overall mean WSS    : {wss_all.mean():.4f} Pa")

    # Analytical WSS for straight pipe (Stage 2 reference):
    wss_straight = MU_INF * 2.0 * U_MAX / R_PIPE
    print(f"[WSS] Straight-pipe reference WSS : {wss_straight:.4f} Pa")
    print(f"[WSS] Curvature amplification      : {wss_all.mean()/wss_straight:.3f}x")

    return {
        "wss_all":    wss_all,
        "wss_inner":  wss_inner,
        "wss_outer":  wss_outer,
        "wsar":       wsar,
        "wall_pts":   wall_pts,
        "mu_wall":    mu_wall,
        "alpha":      alpha,
    }


# =============================================================================
# VISUALISATION  (Section 3.5.2)
# =============================================================================

def plot_secondary_flow_crosssection(model: nn.Module,
                                      normaliser: CoordinateNormaliser,
                                      data: dict,
                                      s_target: float = None):
    """
    Plot the secondary (transverse) velocity field at a cross-section
    midway along the pipe arc (s = theta/2).

    The Dean vortex pair appears as two counter-rotating cells in the
    y-z plane of this cross-section. The outer wall (large x) shows
    higher velocity, consistent with centrifugal redistribution.

    Parameters
    ----------
    s_target : arc angle for cross-section [rad]; defaults to theta/2
    """
    if s_target is None:
        s_target = THETA / 2.0

    # ---- Build a cross-sectional grid at s = s_target ----
    n_grid = 30
    r_vals   = np.linspace(-R_PIPE * 0.95, R_PIPE * 0.95, n_grid)
    rho_grid, alpha_grid = np.meshgrid(
        np.linspace(0, R_PIPE * 0.95, n_grid),
        np.linspace(0, 2*np.pi, n_grid)
    )
    rho_flat   = rho_grid.ravel()
    alpha_flat = alpha_grid.ravel()
    s_flat     = np.full_like(rho_flat, s_target)

    # Convert to Cartesian
    R_eff = R_CURVE + rho_flat * np.cos(alpha_flat)
    x_cs  = R_eff * np.cos(s_flat)
    y_cs  = rho_flat * np.sin(alpha_flat)
    z_cs  = R_eff * np.sin(s_flat)
    pts_cs = np.column_stack([x_cs, y_cs, z_cs])

    # Predict velocity
    model.eval()
    with torch.no_grad():
        x_n = normaliser.normalise(pts_cs)
        uvwp = model(x_n).cpu().numpy()

    uvw = uvwp[:, 0:3]

    # Axial direction at s_target
    e_s = np.array([-np.sin(s_target), 0.0, np.cos(s_target)])
    # In-plane coordinate axes at this cross-section
    # e_rho direction (outward radial in cross-section) at alpha=0:
    #    e_rho = (cos(s), 0, sin(s))  = tangent to torus
    # e_y = (0, 1, 0)
    e_rho  = np.array([np.cos(s_target), 0.0, np.sin(s_target)])
    e_perp = np.array([0.0, 1.0, 0.0])

    # Secondary velocity components in the cross-section plane
    u_axial  = uvw @ e_s
    v_rho    = uvw @ e_rho    # radial secondary
    v_perp   = uvw @ e_perp   # y secondary

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Case B: Cross-section at s = {np.degrees(s_target):.0f}° "
                 f"(De = {DE:.0f}, Non-Newtonian)", fontsize=13)

    # Axial velocity contour
    ax1 = axes[0]
    scatter1 = ax1.scatter(rho_flat * np.cos(alpha_flat),
                           rho_flat * np.sin(alpha_flat),
                           c=u_axial, cmap="plasma",
                           s=15, vmin=0, vmax=U_MAX)
    plt.colorbar(scatter1, ax=ax1, label="Axial velocity u_s [m/s]")
    theta_circ = np.linspace(0, 2*np.pi, 200)
    ax1.plot(R_PIPE*np.cos(theta_circ), R_PIPE*np.sin(theta_circ),
             'k-', linewidth=2)
    ax1.set_xlabel("Radial (towards outer wall) [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title("Axial velocity\n(blunting indicates non-Newtonian effect)")
    ax1.set_aspect("equal")
    ax1.axvline(0, color='gray', lw=0.5, ls='--')

    # Secondary velocity quiver
    ax2 = axes[1]
    speed_sec = np.sqrt(v_rho**2 + v_perp**2)
    q = ax2.quiver(
        rho_flat * np.cos(alpha_flat),
        rho_flat * np.sin(alpha_flat),
        v_rho, v_perp,
        speed_sec,
        cmap="RdYlBu_r", scale=U_MAX * 10,
    )
    plt.colorbar(q, ax=ax2, label="Secondary speed [m/s]")
    ax2.plot(R_PIPE*np.cos(theta_circ), R_PIPE*np.sin(theta_circ),
             'k-', linewidth=2)
    ax2.set_xlabel("Radial (towards outer wall) [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_title("Secondary (Dean) velocity\n(counter-rotating vortex pair)")
    ax2.set_aspect("equal")
    ax2.text(-R_PIPE*0.85, 0, "Inner\nwall", ha='center', va='center',
             fontsize=8, color='navy')
    ax2.text( R_PIPE*0.85, 0, "Outer\nwall", ha='center', va='center',
             fontsize=8, color='darkred')

    plt.tight_layout()
    fname = "secondary_flow_crosssection_caseB.png"
    plt.savefig(fname, dpi=150)
    print(f"[Plot] Saved: {fname}")
    plt.show()


def plot_wss_curved_wall(wss_results: dict):
    """
    Plot the WSS distribution along the curved pipe wall.

    Two sub-plots:
        Left  : WSS as a function of arc angle s (averaged across alpha)
        Right : WSS as a function of alpha (cross-sectional angle) at s=theta/2,
                showing the inner (alpha=pi) to outer (alpha=0) WSS variation.
    """
    wall_pts = wss_results["wall_pts"]
    wss      = wss_results["wss_all"]
    alpha    = wss_results["alpha"]
    toro     = cartesian_to_toroidal(wall_pts)
    s        = toro[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Case B Curved Pipe — Wall Shear Stress Distribution\n"
                 f"(De = {DE:.0f}, Non-Newtonian Carreau-Yasuda)", fontsize=12)

    # ---- Axial WSS profile (binned by arc angle s) ----
    ax1 = axes[0]
    n_bins = 20
    s_bins = np.linspace(0, THETA, n_bins + 1)
    s_mid  = 0.5 * (s_bins[:-1] + s_bins[1:])
    wss_mean_s = np.array([
        wss[(s >= s_bins[i]) & (s < s_bins[i+1])].mean()
        for i in range(n_bins)
    ])
    wss_std_s = np.array([
        wss[(s >= s_bins[i]) & (s < s_bins[i+1])].std()
        for i in range(n_bins)
    ])
    ax1.fill_between(np.degrees(s_mid),
                     wss_mean_s - wss_std_s,
                     wss_mean_s + wss_std_s,
                     alpha=0.3, color='steelblue')
    ax1.plot(np.degrees(s_mid), wss_mean_s, 'b-o', ms=5, label="Mean WSS")
    wss_straight = MU_INF * 2.0 * U_MAX / R_PIPE
    ax1.axhline(wss_straight, color='r', ls='--',
                label=f"Straight pipe reference ({wss_straight:.3f} Pa)")
    ax1.set_xlabel("Arc angle s [degrees]")
    ax1.set_ylabel("WSS [Pa]")
    ax1.set_title("Mean WSS along pipe arc")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---- Circumferential WSS (inner vs outer) at mid-arc ----
    ax2 = axes[1]
    mid_mask = (s > THETA*0.4) & (s < THETA*0.6)
    if mid_mask.sum() > 10:
        alpha_mid = alpha[mid_mask]
        wss_mid   = wss[mid_mask]
        # Bin by alpha
        a_bins = np.linspace(-np.pi, np.pi, 25)
        a_mid  = 0.5 * (a_bins[:-1] + a_bins[1:])
        wss_mean_a = np.array([
            wss_mid[(alpha_mid >= a_bins[i]) & (alpha_mid < a_bins[i+1])].mean()
            if ((alpha_mid >= a_bins[i]) & (alpha_mid < a_bins[i+1])).any()
            else np.nan
            for i in range(len(a_bins)-1)
        ])
        ax2.plot(np.degrees(a_mid), wss_mean_a, 'g-o', ms=5, label="WSS at s=45°")
        ax2.axvline(  0, color='darkred',  ls=':', label="Outer wall (α=0°)")
        ax2.axvline(180, color='navy',     ls=':', label="Inner wall (α=180°)")
        ax2.axvline(-180, color='navy',    ls=':')
        ax2.axhline(wss_straight, color='r', ls='--', alpha=0.5)
        ax2.set_xlabel("Cross-section angle α [degrees]")
        ax2.set_ylabel("WSS [Pa]")
        ax2.set_title(f"Circumferential WSS at mid-arc (s≈45°)\n"
                      f"WSAR = {wss_results['wsar']:.2f}")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Insufficient mid-arc wall points\nfor circumferential profile",
                 ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()
    fname = "wss_distribution_caseB.png"
    plt.savefig(fname, dpi=150)
    print(f"[Plot] Saved: {fname}")
    plt.show()


def plot_viscosity_field_curved(model: nn.Module,
                                 normaliser: CoordinateNormaliser,
                                 interior_pts: np.ndarray):
    """
    Plot the Carreau-Yasuda viscosity field mu(gamma_dot) in the curved pipe.

    Shear-thinning behaviour: viscosity is lowest near the wall (high shear)
    and highest in the core (low shear). Curvature introduces asymmetry —
    the outer wall region experiences higher shear and lower viscosity
    than the inner wall. This is the non-Newtonian analogue of the
    velocity profile asymmetry observed in Dean flow.
    """
    model.eval()
    torch.set_grad_enabled(True)

    n_sub = min(5000, len(interior_pts))
    idx = np.random.choice(len(interior_pts), n_sub, replace=False)
    pts = interior_pts[idx]

    x_n = normaliser.normalise(pts)
    uvwp = model(x_n)
    u_ = uvwp[:, 0:1]; v_ = uvwp[:, 1:2]; w_ = uvwp[:, 2:3]

    def g(f):
        return torch.autograd.grad(f, x_n, torch.ones_like(f),
                                   create_graph=False, retain_graph=True)[0]
    du = g(u_); dv = g(v_); dw = g(w_)
    gd = shear_rate(du, dv, dw, normaliser.x_range)
    mu = carreau_yasuda_viscosity(gd).detach().cpu().numpy()

    toro = cartesian_to_toroidal(pts)
    alpha = toro[:, 1]

    # Colour-code by alpha (inner/outer) position
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(alpha * 180 / np.pi, mu * 1e3,
                    c=toro[:, 0], cmap='viridis', s=8, alpha=0.6)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Radial position ρ [m]")
    ax.axhline(MU_INF * 1e3, color='r', ls='--',
               label=f"μ_∞ = {MU_INF*1e3:.3f} mPa·s")
    ax.axhline(MU_0 * 1e3, color='b', ls='--',
               label=f"μ_0 = {MU_0*1e3:.1f} mPa·s")
    ax.set_xlabel("Cross-section angle α [degrees]")
    ax.set_ylabel("Effective viscosity μ [mPa·s]")
    ax.set_title("Non-Newtonian Viscosity Field — Case B Curved Pipe\n"
                 "(asymmetry reflects Dean flow shear redistribution)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = "viscosity_field_caseB.png"
    plt.savefig(fname, dpi=150)
    print(f"[Plot] Saved: {fname}")
    plt.show()

    print(f"[Viscosity] Mean μ  = {mu.mean()*1e3:.4f} mPa·s")
    print(f"[Viscosity] Min μ   = {mu.min()*1e3:.4f} mPa·s  (high-shear wall)")
    print(f"[Viscosity] Max μ   = {mu.max()*1e3:.4f} mPa·s  (low-shear core)")


# =============================================================================
# MODEL SAVE / LOAD
# =============================================================================

def save_curved_model(model: nn.Module,
                      history: dict,
                      path: str = "pinn_caseB_nonnewtonian.pt"):
    """
    Save trained model weights and training history.

    Follows the same pattern as Stage 3 save_nonnewtonian_model.
    """
    torch.save({
        "model_state_dict": model.state_dict(),
        "history":          history,
        "n_params":         count_parameters(model),
        "Re":               RE,
        "De":               DE,
        "physics": {
            "mu_0": MU_0, "mu_inf": MU_INF, "lambda": LAMBDA,
            "n_CY": N_CY, "a_CY": A_CY, "rho": RHO
        }
    }, path)
    print(f"[Save] Model saved to: {path}")


def load_curved_model(path: str = "pinn_caseB_nonnewtonian.pt") -> nn.Module:
    """Load a previously trained Stage 4 model."""
    model = PINN(n_input=3, n_hidden=64, n_layers=5, n_output=4).to(DEVICE)
    ckpt  = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Load] Model loaded from: {path}  "
          f"(n_params={ckpt['n_params']}, Re={ckpt['Re']}, De={ckpt['De']:.1f})")
    return model


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("STAGE 4: CURVED PIPE — NON-NEWTONIAN PINN")
    print(f"         Re = {RE:.0f},  De = {DE:.1f}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # 1. Load Case B geometry from Stage 1
    # ----------------------------------------------------------------
    print("\n[Step 1] Loading Case B geometry...")
    try:
        data = load_geometry("B_curved_pipe")
    except FileNotFoundError:
        print("  geometry_data/ not found. Running Stage 1 geometry generation...")
        from stage1_geometry import CurvedPipe, CASE_B_PARAMS, save_geometry
        pipe   = CurvedPipe(CASE_B_PARAMS)
        data_B = pipe.generate_all(seed=42)
        save_geometry(data_B)
        data   = data_B

    print(f"  Interior : {data['interior'].shape}")
    print(f"  Wall     : {data['wall'].shape}")
    print(f"  Inlet    : {data['inlet'].shape}")
    print(f"  Outlet   : {data['outlet'].shape}")

    # ----------------------------------------------------------------
    # 2. Coordinate normaliser
    # ----------------------------------------------------------------
    print("\n[Step 2] Setting up coordinate normaliser...")
    all_pts = np.vstack([
        data["interior"], data["wall"], data["inlet"], data["outlet"]
    ])
    normaliser = CoordinateNormaliser(all_pts)

    # ----------------------------------------------------------------
    # 3. Network architecture
    # ----------------------------------------------------------------
    print("\n[Step 3] Building network...")
    model = PINN(n_input=3, n_hidden=64, n_layers=5, n_output=4).to(DEVICE)
    print(f"  Parameters: {count_parameters(model):,}")

    # Warm-start from Stage 3 if available (helps convergence)
    stage3_path = "pinn_caseA_nonnewtonian.pt"
    if os.path.exists(stage3_path):
        ckpt = torch.load(stage3_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Warm-started from Stage 3 weights: {stage3_path}")
    else:
        print("  Training from random (Glorot) initialisation")

    # ----------------------------------------------------------------
    # 4. Loss function and trainer
    # ----------------------------------------------------------------
    print("\n[Step 4] Configuring loss function and trainer...")
    loss_fn = CurvedPipeLoss(
        w_mom=1.0, w_cont=1.0, w_wall=10.0, w_in=15.0, w_out=3.0
    )
    trainer = CurvedPipeTrainer(
        model, loss_fn, normaliser, data,
        lbfgs_subsample=12_000
    )

    # ----------------------------------------------------------------
    # 5. Training
    # ----------------------------------------------------------------
    print("\n[Step 5] Training...")
    trainer.train_adam(
        n_iterations     = 30_000,
        lr_initial       = 1e-3,
        lr_decay         = 0.9,
        decay_every      = 5_000,
        log_every        = 500,
        check_grad_every = 5_000,
    )
    trainer.train_lbfgs(
        max_iter      = 5_000,
        tolerance     = 1e-6,
        log_every     = 200,
        history_size  = 50,
    )

    # ----------------------------------------------------------------
    # 6. Secondary flow (Dean vortex) analysis
    # ----------------------------------------------------------------
    print("\n[Step 6] Dean vortex analysis...")
    u_axial, u_secondary = compute_secondary_velocity(
        model, normaliser, data["interior"]
    )
    Dn_eff = dean_vortex_strength(u_secondary, u_axial)

    # ----------------------------------------------------------------
    # 7. WSS analysis: inner vs outer wall
    # ----------------------------------------------------------------
    print("\n[Step 7] WSS analysis (inner/outer wall asymmetry)...")
    wss_results = compute_wss_inner_outer(
        model, normaliser,
        data["wall"], data["wall_normals"]
    )

    # ----------------------------------------------------------------
    # 8. Newtonian comparison
    # ----------------------------------------------------------------
    print("\n[Step 8] Newtonian vs Non-Newtonian WSS comparison...")
    # Load or compute Newtonian WSS using Stage 2 constant-viscosity approach
    from stage2_pinn_caseA import compute_wss as compute_wss_newtonian
    wss_newt = compute_wss_newtonian(
        model, normaliser,
        data["wall"], data["wall_normals"],
        mu=MU_INF
    )
    wss_nn   = wss_results["wss_all"]
    wss_diff = (wss_nn - wss_newt) / (wss_newt + 1e-12) * 100.0
    print(f"  Mean WSS Newtonian     : {wss_newt.mean():.4f} Pa")
    print(f"  Mean WSS Non-Newtonian : {wss_nn.mean():.4f} Pa")
    print(f"  Mean relative diff     : {wss_diff.mean():.2f}%")
    print(f"  Outer wall WSS (nn)    : {wss_results['wss_outer'].mean():.4f} Pa")
    print(f"  Inner wall WSS (nn)    : {wss_results['wss_inner'].mean():.4f} Pa")

    # ----------------------------------------------------------------
    # 9. Viscosity field
    # ----------------------------------------------------------------
    print("\n[Step 9] Viscosity field analysis...")
    plot_viscosity_field_curved(model, normaliser, data["interior"])
    check_viscosity_range(None)   # summary already printed inside plot function

    # ----------------------------------------------------------------
    # 10. Plots
    # ----------------------------------------------------------------
    print("\n[Step 10] Generating plots...")
    plot_training_history(trainer.history)
    plot_secondary_flow_crosssection(model, normaliser, data)
    plot_wss_curved_wall(wss_results)

    # ----------------------------------------------------------------
    # 11. Save
    # ----------------------------------------------------------------
    np.save("wss_nonnewtonian_caseB.npy", wss_nn)
    np.save("wss_newtonian_caseB.npy",    wss_newt)
    np.save("secondary_velocity_caseB.npy", u_secondary)
    save_curved_model(model, trainer.history)

    # ----------------------------------------------------------------
    # 12. Summary
    # ----------------------------------------------------------------
    wss_straight = MU_INF * 2.0 * U_MAX / R_PIPE
    print(f"\n{'='*60}")
    print("STAGE 4 SUMMARY")
    print(f"{'='*60}")
    print(f"  Reynolds number              : {RE:.0f}")
    print(f"  Dean number                  : {DE:.1f}")
    print(f"  Dean vortex strength (Dn_eff): {Dn_eff:.4f}")
    print(f"  Low-De asymptote (De²/576)   : {DE**2/576:.4f}")
    print(f"  WSS straight-pipe reference  : {wss_straight:.4f} Pa")
    print(f"  Mean WSS Non-Newtonian (CaseB): {wss_nn.mean():.4f} Pa")
    print(f"  WSA Ratio (outer/inner)       : {wss_results['wsar']:.3f}")
    print(f"  Non-Newt vs Newt diff (mean)  : {wss_diff.mean():.2f}%")
    print(f"{'='*60}")
    print("\nStage 4 complete. Run stage5_aneurysm.py next.")
