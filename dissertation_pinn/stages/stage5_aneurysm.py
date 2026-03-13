"""
=============================================================================
STAGE 5: SACCULAR ANEURYSM — NON-NEWTONIAN PINN (PRIMARY CLINICAL CASE)
=============================================================================
Dissertation: Physics-Informed Neural Networks for Cerebral Aneurysm
              Haemodynamic Risk Assessment
University of Zimbabwe | Department of Mathematics

This is the central contribution of the dissertation.  Having validated
the PINN architecture on Hagen-Poiseuille flow (Stage 2), confirmed the
non-Newtonian Carreau-Yasuda implementation (Stage 3), and demonstrated
secondary-flow resolution in a curved pipe (Stage 4), the fully validated
solver is here applied to Case C — the saccular (sidewall) aneurysm
geometry that represents the primary clinical domain.

Scientific objectives:
    1. Resolve the full 3D non-Newtonian velocity and pressure fields inside
       a simplified cerebral aneurysm at physiologically relevant Reynolds
       numbers (Re = 100, 250, 400).
    2. Compute the Wall Shear Stress (WSS) distribution over the entire
       aneurysm wall, distinguishing:
           - Parent artery wall (reference)
           - Aneurysm neck region (flow transition zone)
           - Aneurysm dome (target for rupture-risk assessment)
    3. Identify haemodynamic risk markers:
           - Low-WSS zones  (< 0.4 Pa)  — associated with aneurysm growth
           - High-WSS zones (> 2.5 Pa)  — associated with wall remodelling
           - Impingement zone (pressure maximum on dome)
           - WSS Gradient (WSSG) — spatial rate of change of WSS
    4. Quantify the effect of non-Newtonian rheology relative to the
       Newtonian (constant viscosity) assumption at each Re.
    5. Produce the haemodynamic risk map required for Chapter 4 results.

New contributions relative to Stages 2-4:
    - Section 3.5.3 : Aneurysm-specific loss weights (neck enhancement)
    - Section 3.5.3 : Multi-Re training and solution family
    - Section 4.2   : WSS distribution on sac vs parent artery
    - Section 4.3   : WSSG computation via double automatic differentiation
    - Section 4.4   : Low-WSS and high-WSS area fractions (risk indices)
    - Section 4.5   : Pressure coefficient on dome (impingement)
    - Section 4.6   : Non-Newtonian correction factor kappa_NN (Eq 4.1)

Physical background (Chapter 2, Sections 2.3 and 2.4):
    Cerebral aneurysms are focal dilatations of cerebral arteries that
    arise preferentially at bifurcations and sharp bends (Section 2.2).
    Haemodynamic stress is central to both aneurysm initiation and
    rupture. Two competing hypotheses exist (Section 2.3.1):
        a) LOW WSS hypothesis: sustained low WSS induces endothelial
           dysfunction → inflammatory remodelling → wall weakening
           (Meng et al., 2014; Cebral et al., 2011).
        b) HIGH WSS hypothesis: concentrated impingement jet produces
           high WSS and pressure that mechanically damages the wall
           (Shojima et al., 2004).
    Both are characterised by the haemodynamic quantities computed here.

    The key advantage of blood's shear-thinning behaviour in this context
    (Section 2.5.2) is that the Carreau-Yasuda model predicts higher
    effective viscosity in the low-shear recirculation zone inside the
    aneurysm sac, which in turn reduces the WSS there compared with the
    Newtonian prediction — making Newtonian models non-conservative for
    low-WSS risk assessment.

Acceptance criteria (Section 3.5.3):
    eps_u_artery < 0.05    (5% L2 error in parent artery region)
    WSS_dome < WSS_artery  (qualitative: dome WSS lower than parent artery)
    |WSS_nn - WSS_Newt| / WSS_Newt > 0.01  (measurable non-Newtonian effect)

Dependencies:
    pip install torch numpy scipy matplotlib

Run after stage4_curved_pipe.py has completed.

Author: [Your Name]
Date  : [Date]
=============================================================================
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.gridspec import GridSpec
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lbfgs import LBFGS

# Reuse all validated components from earlier stages
from stage2_pinn_caseA import (
    PINN,
    CoordinateNormaliser,
    check_gradient_dominance,
    count_parameters,
    load_geometry,
    plot_training_history,
    compute_wss as compute_wss_newtonian,
)
from stage3_carreau_yasuda import (
    carreau_yasuda,
    shear_rate,
    compute_wss_nonnewtonian,
    check_viscosity_range,
    compute_full_derivatives,
    physics_residuals_carreau_yasuda,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using: {DEVICE}")


# =============================================================================
# HARD ANSATZ WRAPPER  (Section 3.5.4 — Exact Inlet Boundary Enforcement)
# =============================================================================

class HardAnsatzPINN(nn.Module):
    """
    Wraps the base PINN to enforce the parabolic inlet BC exactly
    by construction (Hard Ansatz).

    Motivation (Section 3.5.4):
        In the soft-enforcement approach the inlet condition is added as a
        weighted penalty to the loss (w_in * L_inlet).  For complex geometries
        the loss landscape has sharp narrow basins (Krishnapriyan et al. 2021),
        and L-BFGS can overshoot these basins, landing at a point where
        the inlet penalty is not fully satisfied.  The result is a non-zero
        eps_u error even when Adam found a near-perfect solution.

        The Hard Ansatz removes the problem at the architecture level.  The
        network output is algebraically blended with the analytical inlet
        profile:

            u_out = (1 - D) * u_parabolic(y,z)  +  D * u_nn(x,y,z)
            v_out =                               D * v_nn(x,y,z)
            w_out =                               D * w_nn(x,y,z)
            p_out =                                   p_nn(x,y,z)

        where  D(x) = sigmoid(alpha * (x_norm - x_inlet_norm))
        is a smooth mask:  D ≈ 0 at inlet (x = 0),  D ≈ 1 everywhere else.

        At x = 0:  D = 0  →  u_out = u_parabolic  (exact, regardless of u_nn)
        Away from inlet:  D = 1  →  u_out = u_nn   (network has full freedom)

        Consequences:
            - eps_u at the inlet is zero by construction.
            - The inlet loss term (w_in * L_inlet) is removed from the
              total loss — it is identically zero and no longer needed.
            - All other loss terms (momentum, continuity, wall, neck,
              outlet, pressure reference) remain unchanged.

    Implementation note:
        Physical y, z coordinates are recovered from normalised inputs via
        pure PyTorch tensor operations so the computation graph is preserved
        for automatic differentiation.

    Parameters
    ----------
    base_model  : base PINN (stage2 architecture)
    u_max       : peak inlet velocity [m/s]  — Re-dependent
    normaliser  : CoordinateNormaliser from stage2 (provides scale/mean)
    alpha       : sigmoid sharpness (default 20.0 — transition over ~10% of domain)
    """

    def __init__(self, base_model: nn.Module, u_max: float,
                 normaliser, alpha: float = 20.0):
        super().__init__()
        self.base  = base_model
        self.u_max = u_max
        self.alpha = alpha

        # Normaliser uses: x_norm = 2*(x - x_min)/x_range - 1
        # Inverse:         x_phys = (x_norm + 1) * x_range/2 + x_min
        self.register_buffer('y_range_half',
            torch.tensor(float(normaliser.x_range[1]) / 2.0, dtype=torch.float32))
        self.register_buffer('y_min',
            torch.tensor(float(normaliser.x_min[1]),          dtype=torch.float32))
        self.register_buffer('z_range_half',
            torch.tensor(float(normaliser.x_range[2]) / 2.0, dtype=torch.float32))
        self.register_buffer('z_min',
            torch.tensor(float(normaliser.x_min[2]),          dtype=torch.float32))

        # Normalised x-coordinate of the inlet plane (physical x = 0)
        # x_norm = 2*(0 - x_min[0])/x_range[0] - 1
        x_in_norm = 2.0 * (0.0 - float(normaliser.x_min[0])) / float(normaliser.x_range[0]) - 1.0
        self.register_buffer('x_inlet_norm',
            torch.tensor(x_in_norm, dtype=torch.float32))

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Hard-Ansatz forward pass.

        Parameters
        ----------
        x_norm : (N, 3) normalised coordinates

        Returns
        -------
        uvwp : (N, 4)  velocity (u,v,w) + pressure (p)
               Inlet BC satisfied exactly for the velocity field.
        """
        # Raw network output — free to do whatever it likes
        uvwp_nn = self.base(x_norm)

        # ── Distance mask ──────────────────────────────────────────────
        # D ≈ 0 at inlet (x_norm ≈ x_inlet_norm), D ≈ 1 elsewhere
        x_coord = x_norm[:, 0:1]
        D = torch.sigmoid(self.alpha * (x_coord - self.x_inlet_norm))

        # ── Recover physical y, z  (pure torch — stays in autograd graph) ─
        # Inverse normalisation: x_phys = (x_norm + 1) * x_range/2 + x_min
        y_phys = (x_norm[:, 1:2] + 1.0) * self.y_range_half + self.y_min
        z_phys = (x_norm[:, 2:3] + 1.0) * self.z_range_half + self.z_min

        # ── Analytical inlet profile ───────────────────────────────────
        r2 = y_phys ** 2 + z_phys ** 2
        u_inlet = self.u_max * (1.0 - r2 / R_A ** 2)
        u_inlet = torch.clamp(u_inlet, min=0.0)   # outside r=R_A → zero

        # ── Blend: inlet zone uses analytical; bulk uses network ───────
        u_out = (1.0 - D) * u_inlet + D * uvwp_nn[:, 0:1]  # axial velocity
        v_out =                        D * uvwp_nn[:, 1:2]  # transverse: 0 at inlet
        w_out =                        D * uvwp_nn[:, 2:3]  # transverse: 0 at inlet
        p_out =                            uvwp_nn[:, 3:4]  # pressure unconstrained

        return torch.cat([u_out, v_out, w_out, p_out], dim=1)


# =============================================================================
# PHYSICAL CONSTANTS  (Table 3.1)
# =============================================================================

MU_0    = 0.056      # [Pa.s]  zero-shear viscosity
MU_INF  = 0.00345    # [Pa.s]  infinite-shear viscosity
LAMBDA  = 3.313      # [s]     relaxation time
N_CY    = 0.3568     # [-]     power-law index
A_CY    = 2.0        # [-]     Yasuda exponent
RHO     = 1060.0     # [kg/m3] blood density

# Case C geometry  (Section 3.3.3, Table 3.2)
R_A     = 0.002      # [m]  parent artery radius
L_A     = 0.025      # [m]  parent artery length
R_S     = 0.004      # [m]  aneurysm sac radius
NECK_R  = 0.0015     # [m]  aneurysm neck radius

# Sac centre (Section 3.3.3 — tangent attachment)
SAC_CENTRE = np.array([L_A / 2.0, R_A + R_S - NECK_R, 0.0])

# Reynolds numbers to study (Section 3.5.3 — parametric Re sweep)
# Re = rho * u_mean * D / mu_inf  (using mu_inf for consistent Re definition)
RE_LIST = [100, 250, 400]

# Clinical WSS thresholds (Section 2.3.1, Table 2.1)
WSS_LOW_THRESH  = 0.4   # [Pa]  below → endothelial dysfunction risk
WSS_HIGH_THRESH = 2.5   # [Pa]  above → wall remodelling risk

print(f"[Geometry] R_artery={R_A*1e3:.1f}mm, R_sac={R_S*1e3:.1f}mm, "
      f"neck_r={NECK_R*1e3:.1f}mm")
print(f"[Geometry] Sac centre: {SAC_CENTRE}")
print(f"[Clinical] Low-WSS threshold  : {WSS_LOW_THRESH} Pa")
print(f"[Clinical] High-WSS threshold : {WSS_HIGH_THRESH} Pa")


# =============================================================================
# FLOW PARAMETERS  (Re-dependent, Section 3.4.3)
# =============================================================================

def flow_params(Re: float) -> dict:
    """
    Compute flow quantities for a given Reynolds number.

    Re = rho * u_mean * (2*R_A) / mu_inf
    → u_mean = Re * mu_inf / (rho * 2*R_A)

    Parabolic inlet:  u_max = 2 * u_mean
    Pressure scale:   delta_P = 4 * mu_inf * u_max * L_A / R_A^2

    Parameters
    ----------
    Re : Reynolds number (defined using mu_inf for comparability across
         Newtonian/non-Newtonian cases)

    Returns
    -------
    dict with u_mean, u_max, delta_P, wss_analytical_artery
    """
    u_mean  = Re * MU_INF / (RHO * 2.0 * R_A)
    u_max   = 2.0 * u_mean
    delta_P = 4.0 * MU_INF * u_max * L_A / R_A**2
    wss_art = MU_INF * 2.0 * u_max / R_A      # straight-pipe WSS reference

    return {"Re": Re, "u_mean": u_mean, "u_max": u_max,
            "delta_P": delta_P, "wss_artery": wss_art}


# Print parameters for primary case
_fp = flow_params(250)
print(f"\n[Physics] Re=250: u_mean={_fp['u_mean']:.5f} m/s, "
      f"u_max={_fp['u_max']:.5f} m/s")
print(f"[Physics] delta_P={_fp['delta_P']:.4f} Pa, "
      f"WSS_artery_ref={_fp['wss_artery']:.4f} Pa")


# =============================================================================
# ANEURYSM GEOMETRY UTILITIES
# =============================================================================

def classify_wall_points(wall_pts: np.ndarray) -> dict:
    """
    Classify wall points into three anatomical regions:
        'artery'  : parent artery lateral wall (cylinder surface)
        'neck'    : transition zone around the aneurysm neck opening
        'dome'    : aneurysm sac wall (spherical surface)

    Classification logic:
        - A point is on the sac wall if its distance from SAC_CENTRE ≈ R_S
        - Of those, neck points are within 1.5 * NECK_R of the neck centre
        - Remaining sac points are dome
        - All other wall points are parent artery

    This classification is used to:
        1. Compute region-specific WSS statistics
        2. Apply enhanced loss weights in the neck region
        3. Generate the clinical risk map (Section 4.4)

    Parameters
    ----------
    wall_pts : (N_w, 3) wall collocation points

    Returns
    -------
    dict with boolean masks 'artery', 'neck', 'dome', and 'sac'
    """
    # Distance from sac centre → identifies sac wall points
    d_sac = np.linalg.norm(wall_pts - SAC_CENTRE, axis=1)
    on_sac = d_sac < R_S * 1.05    # small tolerance

    # Neck region: near the neck opening circle
    # Neck circle is at (L_A/2, R_A, 0) in the plane connecting artery and sac
    neck_centre = np.array([L_A / 2.0, R_A, 0.0])
    d_neck = np.linalg.norm(wall_pts - neck_centre, axis=1)
    near_neck = d_neck < 1.5 * NECK_R

    # Classification
    neck_mask   = on_sac & near_neck
    dome_mask   = on_sac & ~near_neck
    artery_mask = ~on_sac

    print(f"[Classify] Artery wall : {artery_mask.sum():5d} points "
          f"({artery_mask.mean()*100:.1f}%)")
    print(f"[Classify] Neck region : {neck_mask.sum():5d} points "
          f"({neck_mask.mean()*100:.1f}%)")
    print(f"[Classify] Dome wall   : {dome_mask.sum():5d} points "
          f"({dome_mask.mean()*100:.1f}%)")

    return {
        "artery": artery_mask,
        "neck":   neck_mask,
        "dome":   dome_mask,
        "sac":    on_sac,
    }


def classify_interior_points(interior_pts: np.ndarray) -> dict:
    """
    Classify interior collocation points as inside the parent artery
    or inside the aneurysm sac. Used for region-specific physics analysis.

    Parameters
    ----------
    interior_pts : (N_c, 3)

    Returns
    -------
    dict with 'artery_interior' and 'sac_interior' boolean masks
    """
    r_yz = np.sqrt(interior_pts[:, 1]**2 + interior_pts[:, 2]**2)
    in_artery = ((interior_pts[:, 0] >= 0) &
                 (interior_pts[:, 0] <= L_A) &
                 (r_yz <= R_A))

    d_sac = np.linalg.norm(interior_pts - SAC_CENTRE, axis=1)
    in_sac = d_sac <= R_S

    return {"artery_interior": in_artery, "sac_interior": in_sac}


# =============================================================================
# COMPOSITE LOSS FUNCTION — ANEURYSM (SECTION 3.5.3)
# =============================================================================

class AneurysmLoss:
    """
    Composite loss function for the saccular aneurysm PINN.

    Relative to Stage 3 (Case A), the aneurysm loss has two important
    modifications:

    1. NECK ENHANCEMENT WEIGHT (w_neck):
       The neck region connects the parent artery to the aneurysm sac and
       exhibits the steepest velocity gradients in the domain. A dedicated
       neck sub-loss (MSE of velocity at neck wall points) is included with
       an elevated weight to ensure the no-slip condition is well enforced
       at this geometrically critical location.

           L_neck = MSE(u, v, w at neck wall points) → weight w_neck = 20.0

    2. PRESSURE REFERENCE AT DOME:
       The aneurysm sac interior is a slow recirculation zone. Without
       an explicit pressure reference inside the sac, the network may
       converge to a pressure field that is offset from the physical
       solution. We add a soft penalty enforcing that the mean pressure
       inside the sac matches the artery pressure at the neck cross-section.

           L_p_ref = (mean(p_sac) - p_neck_target)^2   → weight w_pref = 2.0

    Full loss:
        L = w_mom*L_mom + w_cont*L_cont
          + w_wall*L_wall + w_neck*L_neck
          + w_in*L_inlet  + w_out*L_outlet
          + w_pref*L_p_ref

    Weights (Section 3.4.5 / Table 3.3):
        w_mom  = 1.0,  w_cont = 1.0
        w_wall = 10.0  (no-slip: entire artery + sac wall)
        w_neck = 20.0  (neck: geometrically critical, WSS-sensitive)
        w_in   = 0.0   (REMOVED — inlet BC enforced exactly by Hard Ansatz)
        w_out  = 5.0   (outlet zero-pressure)
        w_pref = 2.0   (sac pressure reference, soft constraint)
    """

    def __init__(self,
                 w_mom:  float = 1.0,
                 w_cont: float = 1.0,
                 w_wall: float = 10.0,
                 w_neck: float = 20.0,
                 w_in:   float = 0.0,    # ZERO — inlet enforced by Hard Ansatz architecture
                 w_out:  float = 5.0,
                 w_pref: float = 2.0,
                 u_max:  float = 0.5):    # for P_SCALE momentum normalisation
        self.w_mom  = w_mom
        self.w_cont = w_cont
        self.w_wall = w_wall
        self.w_neck = w_neck
        self.w_in   = w_in
        self.w_out  = w_out
        self.w_pref = w_pref
        # P_SCALE: normalise momentum residuals from O(1e7) → O(1)
        # so they compete fairly with BC losses (which are O(1e-1 to 1e-3))
        self.P_SCALE = RHO * u_max**2 + 1e-10

    def momentum_loss(self, R_x, R_y, R_z):
        # Divide by P_SCALE to bring momentum residuals to O(1)
        return (torch.mean((R_x / self.P_SCALE)**2)
                + torch.mean((R_y / self.P_SCALE)**2)
                + torch.mean((R_z / self.P_SCALE)**2)) / 3.0

    def continuity_loss(self, R_cont):
        return torch.mean(R_cont**2)

    def wall_loss(self, uvwp_wall):
        """No-slip on the full wall (artery + sac)."""
        return torch.mean(uvwp_wall[:, 0:3]**2)

    def neck_loss(self, uvwp_neck):
        """
        Enhanced no-slip at the neck wall points.
        The neck is where the flow transitions from confined artery flow
        to the recirculating sac flow — velocity gradients are steepest
        here, and this is where WSS risk indicators change most sharply.
        """
        return torch.mean(uvwp_neck[:, 0:3]**2)

    def inlet_loss(self, uvwp_inlet, x_inlet_phys: np.ndarray, u_max: float):
        """
        Parabolic inlet along the x-axis (Case C inlet is at x=0, disc
        of radius R_A). Axial direction is global +x (unlike Case B).
        The transverse components v, w are zero.

        u_x_target = u_max * (1 - (y^2 + z^2) / R_A^2)
        """
        y = x_inlet_phys[:, 1]
        z = x_inlet_phys[:, 2]
        r2 = torch.tensor(y**2 + z**2, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        u_target = u_max * (1.0 - r2 / R_A**2)
        u_target = torch.clamp(u_target, min=0.0)

        loss_u  = torch.mean((uvwp_inlet[:, 0:1] - u_target)**2)
        loss_vw = torch.mean(uvwp_inlet[:, 1:2]**2) + torch.mean(uvwp_inlet[:, 2:3]**2)
        return loss_u + loss_vw

    def outlet_loss(self, uvwp_outlet):
        """Zero gauge pressure at the artery outlet (x = L_A)."""
        return torch.mean(uvwp_outlet[:, 3:4]**2)

    def pressure_ref_loss(self, uvwp_sac_interior):
        """
        Soft pressure reference inside the aneurysm sac (Section 3.5.3).
        Penalise if mean sac pressure drifts far from zero gauge.
        Without this, the solution can converge to a physically correct
        velocity field but with a pressure offset inside the closed sac.
        """
        if uvwp_sac_interior is None or uvwp_sac_interior.shape[0] == 0:
            return torch.tensor(0.0, device=DEVICE)
        p_sac_mean = uvwp_sac_interior[:, 3:4].mean()
        return p_sac_mean**2

    def total_loss(self, R_cont, R_x, R_y, R_z,
                   uvwp_wall, uvwp_neck,
                   uvwp_inlet, x_inlet_phys, u_max,
                   uvwp_outlet,
                   uvwp_sac_int=None):

        L_mom    = self.momentum_loss(R_x, R_y, R_z)
        L_cont   = self.continuity_loss(R_cont)
        L_wall   = self.wall_loss(uvwp_wall)
        L_neck   = self.neck_loss(uvwp_neck)
        L_inlet  = self.inlet_loss(uvwp_inlet, x_inlet_phys, u_max)
        L_outlet = self.outlet_loss(uvwp_outlet)
        L_pref   = self.pressure_ref_loss(uvwp_sac_int)

        L_bc = (self.w_wall * L_wall
                + self.w_neck * L_neck
                + self.w_in   * L_inlet
                + self.w_out  * L_outlet
                + self.w_pref * L_pref)

        L_total = self.w_mom * L_mom + self.w_cont * L_cont + L_bc

        return (L_total, L_mom, L_cont, L_bc,
                L_wall, L_neck, L_inlet, L_outlet, L_pref)


# =============================================================================
# TRAINING CLASS — ANEURYSM
# =============================================================================

class AneurysmTrainer:
    """
    Two-stage Adam → L-BFGS trainer for the saccular aneurysm PINN.

    The aneurysm geometry has three key differences from Case A/B that
    require careful handling during training:

    1. Increased domain complexity: the combined artery + sac geometry
       has a non-convex boundary, which can produce local loss minima
       if the training points near the neck are not adequately sampled.
       The enhanced near-neck sampling from Stage 1 (30% of interior
       points near the neck) addresses this at the geometry level.

    2. Larger parameter count (25,000 interior + 5,000 wall points)
       requires more aggressive L-BFGS subsampling to avoid OOM.

    3. Multi-Re training: for the parametric Re sweep (Re=100, 250, 400),
       each Re requires a separate training run starting from the
       Re=250 solution as a warm-start (avoids from-scratch training).

    Training protocol (Section 3.4.6):
        Adam  : 40,000 iterations (more than Case A/B due to complexity)
        L-BFGS: 8,000 iterations
        lbfgs_subsample: 10,000 points
    """

    def __init__(self, model, loss_fn, normaliser, data,
                 fp: dict,
                 neck_mask: np.ndarray,
                 sac_int_mask: np.ndarray,
                 lbfgs_subsample: int = 10_000):
        self.model       = model.to(DEVICE)
        self.loss_fn     = loss_fn
        self.norm        = normaliser
        self.data        = data
        self.fp          = fp               # flow params dict (Re-dependent)
        self.neck_mask   = neck_mask        # mask into wall points
        self.sac_int_mask= sac_int_mask     # mask into interior points
        self.lbfgs_sub   = lbfgs_subsample
        self.history     = {
            "loss": [], "L_mom": [], "L_cont": [], "L_bc": [],
            "L_wall": [], "L_neck": [], "L_in": [],
            "L_out": [], "L_pref": []
        }
        # Best-state tracking for Adam → L-BFGS handoff
        self.best_loss  = float("inf")
        self.best_state = None

    def _prepare_tensors(self, n_sub: int = None):
        """Build normalised tensors; optionally subsample interior."""
        rng = np.random.default_rng(SEED)
        interior = self.data["interior"]
        if n_sub is not None and len(interior) > n_sub:
            idx = rng.choice(len(interior), n_sub, replace=False)
            interior = interior[idx]
            # Recompute sac_int_mask for subsampled interior
            d_sac = np.linalg.norm(interior - SAC_CENTRE, axis=1)
            sac_sub = d_sac <= R_S
        else:
            sac_sub = self.sac_int_mask[:len(interior)]

        x_int  = self.norm.normalise(interior)
        x_wall = self.norm.normalise(self.data["wall"])
        x_neck = self.norm.normalise(self.data["wall"][self.neck_mask])
        x_in   = self.norm.normalise(self.data["inlet"])
        x_out  = self.norm.normalise(self.data["outlet"])

        # Sac interior points (for pressure reference loss)
        sac_pts  = interior[sac_sub]
        x_sac_int = self.norm.normalise(sac_pts) if len(sac_pts) > 0 else None

        return x_int, x_wall, x_neck, x_in, x_out, x_sac_int

    def _compute_loss(self, x_int, x_wall, x_neck,
                      x_in, x_out, x_sac_int):
        """Single forward/physics pass."""
        # Physics residuals at interior points
        derivs = compute_full_derivatives(self.model, x_int)
        R_cont, R_x, R_y, R_z, _mu = physics_residuals_carreau_yasuda(
            derivs, self.norm.x_range
        )

        # Boundary evaluations
        uvwp_wall   = self.model(x_wall)
        uvwp_neck   = self.model(x_neck)
        uvwp_inlet  = self.model(x_in)
        uvwp_outlet = self.model(x_out)
        uvwp_sac    = self.model(x_sac_int) if x_sac_int is not None else None

        out = self.loss_fn.total_loss(
            R_cont, R_x, R_y, R_z,
            uvwp_wall, uvwp_neck,
            uvwp_inlet, self.data["inlet"], self.fp["u_max"],
            uvwp_outlet,
            uvwp_sac
        )
        return out

    def train_inlet_warmup(self, n_iterations: int = None, lr: float = 1e-3):
        """
        Phase 0: BC-only warmup before main Adam training.
        Physics weights are near-zero so the network first learns the
        inlet parabolic profile and no-slip before physics turns on.
        This prevents the physics gradient from crushing the inlet
        gradient from iteration 1.
        """
        Re = self.fp["Re"]
        if n_iterations is None:
            n_iterations = 5000 if Re == 250 else 3000

        print(f"\n{'='*60}")
        print(f"Inlet Warmup — Stage 5 Aneurysm  (Re = {Re})")
        print(f"  {n_iterations} iterations, physics nearly off (w=0.001)")
        print(f"{'='*60}")

        # Save nominal weights, set physics near-zero
        nom_mom  = self.loss_fn.w_mom
        nom_cont = self.loss_fn.w_cont
        nom_wall = self.loss_fn.w_wall
        nom_neck = self.loss_fn.w_neck
        nom_in   = self.loss_fn.w_in

        self.loss_fn.w_mom  = 0.001
        self.loss_fn.w_cont = 0.001
        self.loss_fn.w_wall = 80.0
        self.loss_fn.w_neck = 100.0
        # w_in stays 0.0 — inlet enforced by Hard Ansatz, not by loss weight
        self.loss_fn.w_in   = 0.0

        optimizer = Adam(self.model.parameters(), lr=lr)
        x_int, x_wall, x_neck, x_in, x_out, x_sac_int = self._prepare_tensors()
        t0 = time.time()

        for it in range(1, n_iterations + 1):
            self.model.train()
            optimizer.zero_grad()
            (total, Lm, Lc, Lbc,
             Lw, Lnk, Li, Lo, Lp) = self._compute_loss(
                x_int, x_wall, x_neck, x_in, x_out, x_sac_int
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            # Track best warmup state
            if total.item() < self.best_loss:
                self.best_loss  = total.item()
                self.best_state = {k: v.cpu().clone()
                                   for k, v in self.model.state_dict().items()}

            if it % 500 == 0:
                print(f"  Warmup {it:5d}/{n_iterations}  "
                      f"Loss={total.item():.4e}  "
                      f"Inlet={Li.item():.4e}  "
                      f"Wall={Lw.item():.4e}  "
                      f"t={time.time()-t0:.1f}s")

        # Restore nominal weights
        self.loss_fn.w_mom  = nom_mom
        self.loss_fn.w_cont = nom_cont
        self.loss_fn.w_wall = nom_wall
        self.loss_fn.w_neck = nom_neck
        self.loss_fn.w_in   = nom_in
        print(f"Warmup complete (Re={Re}). Nominal weights restored.")

    def train_adam(self,
                   n_iterations:  int   = 40_000,
                   lr_initial:    float = 1e-3,
                   lr_decay:      float = 0.9,
                   decay_every:   int   = 5_000,
                   log_every:     int   = 500,
                   check_grad_every: int = 5_000):

        Re = self.fp["Re"]
        print(f"\n{'='*60}")
        print(f"Adam Training — Stage 5 Aneurysm  (Re = {Re})")
        print(f"{'='*60}")

        # Reset best-state so warmup checkpoints (physics off, artificially
        # low loss) never contaminate the L-BFGS starting point
        self.best_loss  = float("inf")
        self.best_state = None

        # Store nominal weights for curriculum ramp
        base_w_mom  = self.loss_fn.w_mom
        base_w_cont = self.loss_fn.w_cont

        optimizer = Adam(self.model.parameters(), lr=lr_initial)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=decay_every, gamma=lr_decay
        )
        x_int, x_wall, x_neck, x_in, x_out, x_sac_int = self._prepare_tensors()
        t0 = time.time()

        for it in range(1, n_iterations + 1):
            self.model.train()
            optimizer.zero_grad()

            # Curriculum ramp: physics weight 0.001 → 1.0 over first 10,000 iters
            ramp = min(1.0, 0.001 + 0.999 * (it / 10_000))
            self.loss_fn.w_mom  = base_w_mom  * ramp
            self.loss_fn.w_cont = base_w_cont * ramp

            (total, Lm, Lc, Lbc,
             Lw, Lnk, Li, Lo, Lp) = self._compute_loss(
                x_int, x_wall, x_neck, x_in, x_out, x_sac_int
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Track best fully-physics-weighted state for L-BFGS
            if total.item() < self.best_loss:
                self.best_loss  = total.item()
                self.best_state = {k: v.cpu().clone()
                                   for k, v in self.model.state_dict().items()}

            for key, val in zip(
                ["loss","L_mom","L_cont","L_bc","L_wall","L_neck","L_in","L_out","L_pref"],
                [total, Lm, Lc, Lbc, Lw, Lnk, Li, Lo, Lp]
            ):
                self.history[key].append(val.item())

            if it % log_every == 0:
                print(f"  Iter {it:6d}/{n_iterations}  "
                      f"Loss={total.item():.4e}  "
                      f"Mom={Lm.item():.4e}  Cont={Lc.item():.4e}  "
                      f"Neck={Lnk.item():.4e}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}  "
                      f"t={time.time()-t0:.1f}s")

            if it % check_grad_every == 0:
                try:
                    dom, norms, ratio = check_gradient_dominance(
                        self.model, [Lm, Lc, Lbc]
                    )
                    if dom:
                        print(f"  [WARN] Gradient dominance at iter {it}: "
                              f"ratio={ratio:.1f}")
                except RuntimeError:
                    pass

        print(f"Adam complete (Re={Re}). "
              f"Final loss: {self.history['loss'][-1]:.4e}")

    def train_lbfgs(self,
                    max_iter:    int   = 8_000,
                    tolerance:   float = 1e-6,
                    log_every:   int   = 200,
                    history_size: int  = 50):

        Re = self.fp["Re"]
        print(f"\n{'='*60}")
        print(f"L-BFGS Refinement — Stage 5 Aneurysm  (Re = {Re})")
        print(f"{'='*60}")

        # Load best Adam state — must be the physics-weighted checkpoint,
        # NOT the warmup checkpoint (best_loss is reset at Adam start)
        if self.best_state is not None:
            self.model.load_state_dict(
                {k: v.to(DEVICE) for k, v in self.best_state.items()}
            )
            print(f"  Loaded best Adam state (loss={self.best_loss:.4e})")

        optimizer = LBFGS(
            self.model.parameters(),
            max_iter=20,
            history_size=history_size,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe"
        )
        x_int, x_wall, x_neck, x_in, x_out, x_sac_int = self._prepare_tensors(
            n_sub=self.lbfgs_sub
        )
        iter_count = [0]
        t0 = time.time()

        def closure():
            optimizer.zero_grad()
            (total, *_) = self._compute_loss(
                x_int, x_wall, x_neck, x_in, x_out, x_sac_int
            )
            total.backward()
            return total

        for outer in range(max_iter // 20):
            loss_val = optimizer.step(closure)
            iter_count[0] += 20
            if iter_count[0] % log_every == 0:
                print(f"  Iter {iter_count[0]:5d}/{max_iter}  "
                      f"Loss={loss_val.item():.4e}  t={time.time()-t0:.1f}s")
            self.history["loss"].append(
                loss_val.item() if loss_val is not None else float('nan')
            )

        print(f"L-BFGS complete (Re={Re}). "
              f"Final loss: {self.history['loss'][-1]:.4e}")


# =============================================================================
# WSS ANALYSIS — ANEURYSM  (Section 4.2 – 4.4)
# =============================================================================

def compute_aneurysm_wss(model: nn.Module,
                          normaliser: CoordinateNormaliser,
                          wall_pts:     np.ndarray,
                          wall_normals: np.ndarray,
                          region_masks: dict) -> dict:
    """
    Compute the full non-Newtonian WSS distribution across the aneurysm
    wall and compute region-specific statistics.

    Region statistics (Section 4.2):
        - Mean WSS per region (artery, neck, dome)
        - Max WSS per region
        - WSS normalised by parent artery reference (dimensionless)

    Risk metrics (Section 4.4):
        - Low-WSS area fraction:  fraction of sac wall with WSS < 0.4 Pa
        - High-WSS area fraction: fraction of sac wall with WSS > 2.5 Pa
        - WSS concentration index: max(WSS_dome) / mean(WSS_dome)
        - Non-Newtonian correction factor kappa_NN (Eq 4.1):
              kappa_NN = WSS_nn / WSS_Newt  (per-point ratio)

    Returns
    -------
    dict with wss arrays, region stats, risk metrics, and mu_wall
    """
    # Non-Newtonian WSS
    wss_nn, mu_wall, gd_wall = compute_wss_nonnewtonian(
        model, normaliser, wall_pts, wall_normals
    )
    # Newtonian WSS (constant mu_inf)
    wss_newt = compute_wss_newtonian(
        model, normaliser, wall_pts, wall_normals, mu=MU_INF
    )

    # Non-Newtonian correction factor kappa_NN (Eq 4.1)
    kappa_nn = wss_nn / (wss_newt + 1e-10)

    # Region statistics
    regions = ["artery", "neck", "dome"]
    stats = {}
    for reg in regions:
        mask = region_masks[reg]
        if mask.sum() == 0:
            stats[reg] = {"mean": np.nan, "max": np.nan,
                          "min": np.nan, "std": np.nan}
            continue
        w = wss_nn[mask]
        stats[reg] = {
            "mean": w.mean(), "max": w.max(),
            "min": w.min(),   "std": w.std(),
            "n":   mask.sum()
        }
        print(f"[WSS Region] {reg:8s}: "
              f"mean={w.mean():.4f}  max={w.max():.4f}  "
              f"min={w.min():.4f}  std={w.std():.4f}  Pa")

    # Risk metrics on the sac (neck + dome)
    sac_mask = region_masks["sac"]
    wss_sac  = wss_nn[sac_mask]

    low_wss_frac  = (wss_sac < WSS_LOW_THRESH).mean()
    high_wss_frac = (wss_sac > WSS_HIGH_THRESH).mean()
    wss_conc_idx  = (wss_sac.max() / (wss_sac.mean() + 1e-10))

    print(f"\n[Risk Metrics]")
    print(f"  Low-WSS area fraction  (< {WSS_LOW_THRESH} Pa) : "
          f"{low_wss_frac*100:.1f}%")
    print(f"  High-WSS area fraction (> {WSS_HIGH_THRESH} Pa): "
          f"{high_wss_frac*100:.1f}%")
    print(f"  WSS concentration index                        : "
          f"{wss_conc_idx:.2f}")
    print(f"  Mean kappa_NN (sac)    : {kappa_nn[sac_mask].mean():.3f}")
    print(f"  Min kappa_NN (sac)     : {kappa_nn[sac_mask].min():.3f}  "
          f"(most non-Newtonian point)")

    # WSS normalised by parent artery mean
    wss_art_mean = wss_nn[region_masks["artery"]].mean()
    wss_norm     = wss_nn / (wss_art_mean + 1e-10)
    print(f"  Artery mean WSS        : {wss_art_mean:.4f} Pa  (normalisation ref)")
    print(f"  Dome mean WSS / Artery : {stats['dome']['mean'] / wss_art_mean:.3f}")

    return {
        "wss_nn":     wss_nn,
        "wss_newt":   wss_newt,
        "kappa_nn":   kappa_nn,
        "mu_wall":    mu_wall,
        "gd_wall":    gd_wall,
        "wss_norm":   wss_norm,
        "stats":      stats,
        "low_wss_frac":  low_wss_frac,
        "high_wss_frac": high_wss_frac,
        "wss_conc_idx":  wss_conc_idx,
    }


# =============================================================================
# WSS SPATIAL GRADIENT (WSSG)  (Section 4.3)
# =============================================================================

def compute_wssg(wss: np.ndarray,
                 wall_pts: np.ndarray,
                 k_neighbours: int = 8) -> np.ndarray:
    """
    Compute the Wall Shear Stress Gradient (WSSG) at each wall point.

    WSSG = |grad_wall(WSS)|  [Pa/m]

    Method: for each wall point, find k nearest neighbours on the wall
    and use a weighted least-squares fit to estimate the spatial gradient
    in the local tangent plane (two-dimensional surface gradient).

    WSSG is a secondary risk indicator: high WSSG at the neck suggests
    regions where the boundary layer changes rapidly — associated with
    endothelial cell deformation and inflammatory signalling.

    Parameters
    ----------
    wss      : (N_w,) WSS values at wall points
    wall_pts : (N_w, 3) wall point coordinates
    k_neighbours : number of nearest neighbours for gradient estimation

    Returns
    -------
    wssg : (N_w,) WSSG magnitudes [Pa/m]
    """
    tree = cKDTree(wall_pts)
    wssg = np.zeros(len(wall_pts))

    for i, pt in enumerate(wall_pts):
        # Find k nearest neighbours (excluding self)
        dists, idx = tree.query(pt, k=k_neighbours + 1)
        idx   = idx[1:]    # exclude self
        dists = dists[1:]

        if dists.min() < 1e-12:
            continue

        # Local displacement vectors
        dx = wall_pts[idx] - pt          # (k, 3)
        dw = wss[idx] - wss[i]           # (k,)

        # Inverse-distance-weighted least squares: dx^T * dx * grad ≈ dx^T * dw
        W    = np.diag(1.0 / (dists + 1e-12))
        A    = dx.T @ W @ dx             # (3, 3)
        b    = dx.T @ W @ dw             # (3,)

        try:
            grad_wss = np.linalg.lstsq(A, b, rcond=None)[0]
            wssg[i]  = np.linalg.norm(grad_wss)
        except np.linalg.LinAlgError:
            wssg[i] = 0.0

    print(f"[WSSG] Mean WSSG : {wssg.mean():.2f} Pa/m")
    print(f"[WSSG] Max WSSG  : {wssg.max():.2f} Pa/m  "
          f"(at point idx {wssg.argmax()})")
    return wssg


# =============================================================================
# PRESSURE AND FLOW FIELD ANALYSIS  (Section 4.5)
# =============================================================================

def analyse_pressure_field(model: nn.Module,
                            normaliser: CoordinateNormaliser,
                            interior_pts: np.ndarray,
                            region_masks_int: dict,
                            fp: dict) -> dict:
    """
    Extract pressure field and compute the pressure coefficient on the dome.

    Key quantities (Section 4.5):
        - Pressure drop along the parent artery: delta_P_artery
        - Mean pressure inside the sac: p_sac_mean
        - Pressure coefficient at the impingement point:
              Cp = (p_max_dome - p_ref) / (0.5 * rho * u_mean^2)
          where p_ref is the outlet pressure (= 0 gauge).

    The impingement point is identified as the wall point where the
    total pressure (static + dynamic) is maximum on the sac wall —
    this corresponds to the stagnation point of the impingement jet.

    Parameters
    ----------
    fp : flow params dict from flow_params(Re)

    Returns
    -------
    dict with pressure statistics and Cp
    """
    model.eval()
    with torch.no_grad():
        x_n  = normaliser.normalise(interior_pts)
        uvwp = model(x_n).cpu().numpy()

    p     = uvwp[:, 3]
    u_vel = uvwp[:, 0]

    p_artery = p[region_masks_int["artery_interior"]]
    p_sac    = p[region_masks_int["sac_interior"]]

    # Pressure drop along artery
    x_coords = interior_pts[:, 0]
    x_art    = x_coords[region_masks_int["artery_interior"]]
    p_art    = p_artery

    # Fit linear pressure gradient
    if len(x_art) > 10:
        coeffs = np.polyfit(x_art, p_art, 1)
        dpdx_pinn = coeffs[0]            # [Pa/m]
        dpdx_theo = -(fp["delta_P"] / L_A)
        print(f"[Pressure] PINN dp/dx : {dpdx_pinn:.3f} Pa/m")
        print(f"[Pressure] Theory dp/dx: {dpdx_theo:.3f} Pa/m")
        print(f"[Pressure] Relative error: "
              f"{abs(dpdx_pinn-dpdx_theo)/abs(dpdx_theo)*100:.2f}%")
    else:
        dpdx_pinn = np.nan

    # Pressure inside the sac
    p_sac_mean = p_sac.mean() if len(p_sac) > 0 else np.nan
    p_sac_max  = p_sac.max()  if len(p_sac) > 0 else np.nan

    # Pressure coefficient at impingement
    q_ref = 0.5 * RHO * fp["u_mean"]**2
    Cp    = (p_sac_max - 0.0) / (q_ref + 1e-12)   # p_ref = 0 (outlet)

    print(f"[Pressure] Mean p in sac  : {p_sac_mean:.4f} Pa")
    print(f"[Pressure] Max  p in sac  : {p_sac_max:.4f} Pa")
    print(f"[Pressure] Cp (impingement): {Cp:.3f}  "
          f"(q_ref = {q_ref:.4f} Pa)")

    return {
        "p_field":    p,
        "p_artery":   p_artery,
        "p_sac":      p_sac,
        "p_sac_mean": p_sac_mean,
        "Cp":         Cp,
        "dpdx_pinn":  dpdx_pinn,
    }


def analyse_velocity_field(model: nn.Module,
                            normaliser: CoordinateNormaliser,
                            interior_pts: np.ndarray,
                            region_masks_int: dict,
                            fp: dict) -> dict:
    """
    Compute velocity statistics and identify the recirculation zone.

    Key quantities:
        - Mean axial velocity in parent artery (validates inlet BC)
        - Velocity magnitude in sac (characterises recirculation strength)
        - Recirculation index: fraction of sac interior where the axial
          velocity component is negative (flow reversal)
        - Impingement jet speed: max velocity near the dome

    Parameters
    ----------
    fp : flow_params dict

    Returns
    -------
    dict with velocity statistics
    """
    model.eval()
    with torch.no_grad():
        x_n  = normaliser.normalise(interior_pts)
        uvwp = model(x_n).cpu().numpy()

    u, v, w = uvwp[:, 0], uvwp[:, 1], uvwp[:, 2]
    speed   = np.sqrt(u**2 + v**2 + w**2)

    art_mask = region_masks_int["artery_interior"]
    sac_mask = region_masks_int["sac_interior"]

    u_artery = u[art_mask]
    u_sac    = u[sac_mask]
    spd_sac  = speed[sac_mask]

    # Recirculation index: fraction of sac where axial flow reverses
    recirculation_index = (u_sac < 0).mean()

    # Velocity ratio: mean sac speed relative to mean artery speed
    spd_artery = speed[art_mask]
    vel_ratio  = spd_sac.mean() / (spd_artery.mean() + 1e-12)

    print(f"[Velocity] Mean axial velocity (artery) : {u_artery.mean():.5f} m/s")
    print(f"[Velocity] Target u_mean                : {fp['u_mean']:.5f} m/s")
    print(f"[Velocity] Artery u_mean relative error : "
          f"{abs(u_artery.mean()-fp['u_mean'])/fp['u_mean']*100:.2f}%")
    print(f"[Velocity] Mean speed in sac            : {spd_sac.mean():.5f} m/s")
    print(f"[Velocity] Velocity ratio (sac/artery)  : {vel_ratio:.3f}")
    print(f"[Velocity] Recirculation index          : "
          f"{recirculation_index*100:.1f}%  "
          f"(fraction of sac with reverse axial flow)")

    return {
        "speed":               speed,
        "u_artery_mean":       u_artery.mean(),
        "spd_sac_mean":        spd_sac.mean(),
        "vel_ratio":           vel_ratio,
        "recirculation_index": recirculation_index,
    }


# =============================================================================
# MULTI-Re PARAMETRIC STUDY  (Section 3.5.3)
# =============================================================================

def multi_re_study(data: dict,
                   normaliser: CoordinateNormaliser,
                   neck_mask: np.ndarray,
                   sac_int_mask: np.ndarray,
                   region_masks: dict,
                   region_masks_int: dict,
                   re_list: list = RE_LIST,
                   base_model_path: str = "pinn_caseC_Re250.pt") -> dict:
    """
    Train and analyse the PINN at multiple Reynolds numbers.

    Protocol:
        1. Train fully at Re=250 (primary case, warm-start from Stage 4)
        2. Warm-start Re=100 and Re=400 models from the Re=250 solution
           (the velocity field at different Re are physically close enough
            that warm-starting reduces training to ~40% of the from-scratch cost)

    For each Re, compute:
        - wss_nn        : Non-Newtonian WSS on full wall
        - wss_newt      : Newtonian WSS
        - kappa_nn      : Non-Newtonian correction factor
        - low_wss_frac  : Low-WSS area fraction on sac
        - Cp            : Dome pressure coefficient
        - recirculation : Recirculation index

    Returns
    -------
    results : dict keyed by Re, each containing all computed quantities
    """
    results = {}

    for Re in re_list:
        print(f"\n{'#'*60}")
        print(f"# Re = {Re}")
        print(f"{'#'*60}")

        fp = flow_params(Re)

        # Build model — Hard Ansatz wraps the base PINN
        base_m = PINN(n_input=3, n_hidden=64, n_layers=5, n_output=4).to(DEVICE)
        model  = HardAnsatzPINN(base_m, fp["u_max"], normaliser).to(DEVICE)

        # Warm-start — load into base (state dicts saved without 'base.' prefix)
        if Re != 250 and os.path.exists(base_model_path):
            ckpt = torch.load(base_model_path, map_location=DEVICE, weights_only=False)
            model.base.load_state_dict(ckpt["model_state_dict"])
            print(f"  Warm-started from Re=250 model: {base_model_path}")
        elif os.path.exists(f"pinn_caseC_Re{Re}.pt"):
            ckpt = torch.load(f"pinn_caseC_Re{Re}.pt", map_location=DEVICE,
                              weights_only=False)
            model.base.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded existing model for Re={Re}")
        else:
            # Warm-start from Stage 4 curved pipe if available
            for stage4_path in ["trained_models/pinn_caseB_nonnewtonian.pt",
                                "pinn_caseB_nonnewtonian.pt"]:
                if os.path.exists(stage4_path):
                    ckpt = torch.load(stage4_path, map_location=DEVICE,
                                      weights_only=False)
                    model.base.load_state_dict(ckpt["model_state_dict"])
                    print(f"  Warm-started from Stage 4: {stage4_path}")
                    break

        # Train
        loss_fn = AneurysmLoss(u_max=fp["u_max"])
        trainer = AneurysmTrainer(
            model, loss_fn, normaliser, data, fp,
            neck_mask, sac_int_mask, lbfgs_subsample=10_000
        )
        n_adam = 40_000 if Re == 250 else 30_000
        trainer.train_inlet_warmup()
        trainer.train_adam(n_iterations=n_adam)
        trainer.train_lbfgs(max_iter=8_000)

        # Analysis
        wss_data = compute_aneurysm_wss(
            model, normaliser,
            data["wall"], data["wall_normals"], region_masks
        )
        wssg = compute_wssg(wss_data["wss_nn"], data["wall"])
        p_data = analyse_pressure_field(
            model, normaliser, data["interior"], region_masks_int, fp
        )
        v_data = analyse_velocity_field(
            model, normaliser, data["interior"], region_masks_int, fp
        )

        results[Re] = {
            "model":    model,
            "fp":       fp,
            "history":  trainer.history,
            "wss":      wss_data,
            "wssg":     wssg,
            "pressure": p_data,
            "velocity": v_data,
        }

        # Save model and arrays
        save_aneurysm_model(model, trainer.history, Re)
        np.save(f"wss_nn_caseC_Re{Re}.npy",   wss_data["wss_nn"])
        np.save(f"wss_newt_caseC_Re{Re}.npy",  wss_data["wss_newt"])
        np.save(f"kappa_nn_caseC_Re{Re}.npy",  wss_data["kappa_nn"])
        np.save(f"wssg_caseC_Re{Re}.npy",      wssg)

    return results


# =============================================================================
# VISUALISATION  (Section 4.2 – 4.6)
# =============================================================================

def plot_wss_risk_map(wss_nn: np.ndarray,
                     wall_pts: np.ndarray,
                     region_masks: dict,
                     Re: float,
                     title_suffix: str = ""):
    """
    Plot the WSS haemodynamic risk map on the aneurysm wall.

    Three-panel figure:
        Panel 1 : WSS magnitude colour map (full wall, x-y projection)
        Panel 2 : WSS on sac only, with risk thresholds highlighted
        Panel 3 : Cross-section through the sac mid-plane (x = L_A/2)
                  showing WSS vs position on the sac wall

    The risk map is the primary clinical output of the dissertation —
    it shows which regions of the aneurysm wall are at haemodynamic risk.
    """
    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(f"Haemodynamic WSS Risk Map — Case C Aneurysm  "
                 f"(Re={Re:.0f}{title_suffix})", fontsize=13)
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    # Colour map bounds
    wss_vmin = 0.0
    wss_vmax = max(wss_nn.max(), WSS_HIGH_THRESH * 1.2)

    # ---- Panel 1: Full wall WSS (x-y projection) ----
    ax1 = fig.add_subplot(gs[0])
    sc1 = ax1.scatter(wall_pts[:, 0] * 1e3, wall_pts[:, 1] * 1e3,
                      c=wss_nn, cmap="RdYlGn_r",
                      vmin=wss_vmin, vmax=wss_vmax, s=4, alpha=0.7)
    plt.colorbar(sc1, ax=ax1, label="WSS [Pa]", fraction=0.046)
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    ax1.set_title("Full Wall WSS\n(artery + sac)")
    ax1.set_aspect("equal")

    # Mark sac region outline
    sac_mask = region_masks["sac"]
    ax1.scatter(wall_pts[sac_mask, 0] * 1e3, wall_pts[sac_mask, 1] * 1e3,
                c='none', edgecolors='black', s=6, linewidths=0.4, alpha=0.5)

    # ---- Panel 2: Sac WSS with risk thresholds ----
    ax2 = fig.add_subplot(gs[1])
    wss_sac = wss_nn[sac_mask]
    pts_sac = wall_pts[sac_mask]

    # Colour by risk category
    risk_colours = np.where(wss_sac < WSS_LOW_THRESH,  0,   # low: red
                   np.where(wss_sac > WSS_HIGH_THRESH, 2,   # high: blue
                                                        1))  # normal: green
    cmap_risk = plt.cm.get_cmap("RdYlGn", 3)
    sc2 = ax2.scatter(pts_sac[:, 0] * 1e3, pts_sac[:, 1] * 1e3,
                      c=risk_colours, cmap=cmap_risk,
                      vmin=-0.5, vmax=2.5, s=8)
    cb2 = plt.colorbar(sc2, ax=ax2, ticks=[0, 1, 2], fraction=0.046)
    cb2.ax.set_yticklabels([f"Low\n(<{WSS_LOW_THRESH}Pa)",
                            "Normal", f"High\n(>{WSS_HIGH_THRESH}Pa)"])
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("y [mm]")
    ax2.set_title("Sac Risk Map\n(Low/Normal/High WSS)")
    ax2.set_aspect("equal")

    low_pct  = (wss_sac < WSS_LOW_THRESH).mean() * 100
    high_pct = (wss_sac > WSS_HIGH_THRESH).mean() * 100
    ax2.text(0.05, 0.95, f"Low:  {low_pct:.1f}%\nHigh: {high_pct:.1f}%",
             transform=ax2.transAxes, va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # ---- Panel 3: WSS along sac circumference (mid-plane cross-section) ----
    ax3 = fig.add_subplot(gs[2])
    # Project sac wall points onto angle from sac centre in y-z plane
    sac_dx = pts_sac[:, 0] - SAC_CENTRE[0]
    sac_dy = pts_sac[:, 1] - SAC_CENTRE[1]
    sac_dz = pts_sac[:, 2] - SAC_CENTRE[2]
    sac_angle = np.degrees(np.arctan2(sac_dy, sac_dx))

    # Bin WSS by angle
    angle_bins = np.linspace(-180, 180, 37)
    angle_mids = 0.5 * (angle_bins[:-1] + angle_bins[1:])
    wss_binned = np.full(len(angle_mids), np.nan)
    for i in range(len(angle_mids)):
        mask_bin = ((sac_angle >= angle_bins[i]) &
                    (sac_angle < angle_bins[i+1]))
        if mask_bin.sum() > 0:
            wss_binned[i] = wss_sac[mask_bin].mean()

    valid = ~np.isnan(wss_binned)
    ax3.bar(angle_mids[valid], wss_binned[valid], width=10,
            color=plt.cm.RdYlGn_r(
                (wss_binned[valid] - wss_vmin) / (wss_vmax - wss_vmin + 1e-10)
            ), edgecolor='gray', linewidth=0.3)
    ax3.axhline(WSS_LOW_THRESH,  color='red',  ls='--', lw=1.5,
                label=f"Low threshold ({WSS_LOW_THRESH} Pa)")
    ax3.axhline(WSS_HIGH_THRESH, color='blue', ls='--', lw=1.5,
                label=f"High threshold ({WSS_HIGH_THRESH} Pa)")
    ax3.set_xlabel("Angle from neck (°)")
    ax3.set_ylabel("Mean WSS [Pa]")
    ax3.set_title("WSS vs Sac Angle\n(0° = neck, 180° = dome apex)")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-180, 180)

    plt.tight_layout()
    fname = f"wss_risk_map_caseC_Re{Re:.0f}.png"
    plt.savefig(fname, dpi=150)
    print(f"[Plot] Saved: {fname}")
    plt.show()


def plot_multi_re_comparison(results: dict):
    """
    Compare WSS, low-WSS fraction, and kappa_NN across all Reynolds numbers.

    Four-panel summary figure for Section 4.6:
        Panel 1 : Mean WSS on sac vs Re  (Newtonian and Non-Newtonian)
        Panel 2 : Low-WSS area fraction vs Re
        Panel 3 : Mean kappa_NN vs Re  (quantifies non-Newtonian effect)
        Panel 4 : Recirculation index vs Re
    """
    re_vals = sorted(results.keys())

    wss_mean_nn   = [results[r]["wss"]["stats"]["dome"]["mean"] for r in re_vals]
    wss_mean_newt = [results[r]["wss"]["wss_newt"][
                         # compute mean over sac mask in post
                         np.ones(len(results[r]["wss"]["wss_newt"]), dtype=bool)
                     ].mean() for r in re_vals]
    low_frac      = [results[r]["wss"]["low_wss_frac"]   * 100 for r in re_vals]
    kappa_mean    = [results[r]["wss"]["kappa_nn"].mean() for r in re_vals]
    recirc        = [results[r]["velocity"]["recirculation_index"] * 100
                     for r in re_vals]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Multi-Re Parametric Study — Case C Saccular Aneurysm",
                 fontsize=13)

    ax1, ax2, ax3, ax4 = axes.flatten()

    # Panel 1: Mean dome WSS vs Re
    ax1.plot(re_vals, wss_mean_nn,   'bo-', ms=8, label="Non-Newtonian (CY)")
    ax1.plot(re_vals, wss_mean_newt, 'rs--', ms=8, label="Newtonian (μ=μ∞)")
    ax1.axhline(WSS_LOW_THRESH,  color='orange', ls=':', lw=1.5,
                label=f"Low risk ({WSS_LOW_THRESH} Pa)")
    ax1.set_xlabel("Reynolds Number Re")
    ax1.set_ylabel("Mean Dome WSS [Pa]")
    ax1.set_title("Dome Mean WSS vs Re")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Low-WSS fraction vs Re
    ax2.plot(re_vals, low_frac, 'r^-', ms=8)
    ax2.set_xlabel("Reynolds Number Re")
    ax2.set_ylabel("Low-WSS Area Fraction [%]")
    ax2.set_title(f"Low-WSS Fraction (< {WSS_LOW_THRESH} Pa) vs Re")
    ax2.grid(True, alpha=0.3)
    for r, f in zip(re_vals, low_frac):
        ax2.annotate(f"{f:.1f}%", (r, f), textcoords="offset points",
                     xytext=(5, 5), fontsize=9)

    # Panel 3: kappa_NN vs Re
    ax3.plot(re_vals, kappa_mean, 'gD-', ms=8)
    ax3.axhline(1.0, color='gray', ls='--', lw=1, label="Newtonian limit")
    ax3.set_xlabel("Reynolds Number Re")
    ax3.set_ylabel("Mean κ_NN = WSS_nn / WSS_Newt")
    ax3.set_title("Non-Newtonian Correction Factor κ_NN vs Re")
    ax3.set_ylim(0.5, 1.2)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Recirculation index vs Re
    ax4.plot(re_vals, recirc, 'ms-', ms=8)
    ax4.set_xlabel("Reynolds Number Re")
    ax4.set_ylabel("Recirculation Index [%]")
    ax4.set_title("Sac Recirculation Index vs Re\n"
                  "(fraction with reverse axial flow)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = "multi_re_comparison_caseC.png"
    plt.savefig(fname, dpi=150)
    print(f"[Plot] Saved: {fname}")
    plt.show()


def plot_velocity_sac_midplane(model: nn.Module,
                                normaliser: CoordinateNormaliser,
                                interior_pts: np.ndarray,
                                Re: float):
    """
    Plot the velocity field in the sac mid-plane (z = 0 cross-section).

    Shows the intra-aneurysmal flow pattern including:
        - Impingement jet from the neck
        - Recirculation vortex inside the sac
        - Dome stagnation zone (low velocity, low WSS)

    This is a key dissertation figure linking the velocity field
    to the haemodynamic risk markers computed from WSS.
    """
    # Select points near the z=0 mid-plane
    z_tol   = R_A * 0.5
    midplane = np.abs(interior_pts[:, 2]) < z_tol

    # Build a regular 2D grid in x-y space
    n_grid = 40
    x_vals = np.linspace(0, L_A, n_grid)
    y_vals = np.linspace(-R_S - R_A, R_A + 2*R_S, n_grid)
    xg, yg = np.meshgrid(x_vals, y_vals)
    zg     = np.zeros_like(xg)
    grid_pts = np.column_stack([xg.ravel(), yg.ravel(), zg.ravel()])

    # Evaluate velocity on grid
    model.eval()
    batch_size = 500
    uvwp_list  = []
    with torch.no_grad():
        for i in range(0, len(grid_pts), batch_size):
            pts_b = grid_pts[i:i+batch_size]
            x_n   = normaliser.normalise(pts_b)
            uvwp_list.append(model(x_n).cpu().numpy())
    uvwp_grid = np.vstack(uvwp_list)

    u_g = uvwp_grid[:, 0].reshape(n_grid, n_grid)
    v_g = uvwp_grid[:, 1].reshape(n_grid, n_grid)
    p_g = uvwp_grid[:, 3].reshape(n_grid, n_grid)
    speed_g = np.sqrt(u_g**2 + v_g**2).reshape(n_grid, n_grid)

    # Mask out points outside the domain
    from stage1_geometry import SaccularAneurysm, CASE_C_PARAMS
    aneurysm_geom = SaccularAneurysm(CASE_C_PARAMS)
    in_domain = aneurysm_geom._in_domain(grid_pts).reshape(n_grid, n_grid)
    speed_g[~in_domain] = np.nan
    p_g[~in_domain]     = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Case C Aneurysm — Mid-plane Flow Field  (Re={Re:.0f})",
                 fontsize=13)

    # Speed contour + streamlines (left panel)
    ax1 = axes[0]
    cf1 = ax1.contourf(xg * 1e3, yg * 1e3, speed_g,
                       levels=20, cmap='plasma')
    plt.colorbar(cf1, ax=ax1, label="|u| [m/s]")
    # Streamlines for in-domain region
    try:
        ax1.streamplot(xg[0] * 1e3, yg[:, 0] * 1e3,
                       np.where(in_domain, u_g, 0),
                       np.where(in_domain, v_g, 0),
                       color='white', linewidth=0.7, density=1.5,
                       arrowsize=0.8)
    except Exception:
        pass
    ax1.set_xlabel("x [mm]"); ax1.set_ylabel("y [mm]")
    ax1.set_title("Velocity Magnitude + Streamlines")
    # Draw domain outline
    theta_c = np.linspace(0, 2*np.pi, 200)
    # Sac outline
    ax1.plot((SAC_CENTRE[0] + R_S*np.cos(theta_c)) * 1e3,
             (SAC_CENTRE[1] + R_S*np.sin(theta_c)) * 1e3,
             'w--', lw=1.2, alpha=0.7)
    ax1.set_aspect("equal")

    # Pressure field (right panel)
    ax2 = axes[1]
    cf2 = ax2.contourf(xg * 1e3, yg * 1e3, p_g,
                       levels=20, cmap='coolwarm')
    plt.colorbar(cf2, ax=ax2, label="p [Pa]")
    ax2.set_xlabel("x [mm]"); ax2.set_ylabel("y [mm]")
    ax2.set_title("Pressure Field")
    ax2.plot((SAC_CENTRE[0] + R_S*np.cos(theta_c)) * 1e3,
             (SAC_CENTRE[1] + R_S*np.sin(theta_c)) * 1e3,
             'k--', lw=1.2, alpha=0.7)
    ax2.set_aspect("equal")

    plt.tight_layout()
    fname = f"velocity_midplane_caseC_Re{Re:.0f}.png"
    plt.savefig(fname, dpi=150)
    print(f"[Plot] Saved: {fname}")
    plt.show()


def plot_wssg_distribution(wssg: np.ndarray,
                           wall_pts: np.ndarray,
                           region_masks: dict,
                           Re: float):
    """
    Plot the WSSG distribution, highlighting the neck region
    where gradients are clinically significant.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Wall Shear Stress Gradient (WSSG) — Re={Re:.0f}", fontsize=12)

    # WSSG spatial map
    ax1 = axes[0]
    sc = ax1.scatter(wall_pts[:, 0] * 1e3, wall_pts[:, 1] * 1e3,
                     c=wssg, cmap='hot_r',
                     vmin=0, vmax=np.percentile(wssg, 95), s=5)
    plt.colorbar(sc, ax=ax1, label="WSSG [Pa/m]")
    neck_mask = region_masks["neck"]
    ax1.scatter(wall_pts[neck_mask, 0] * 1e3, wall_pts[neck_mask, 1] * 1e3,
                c='cyan', s=8, alpha=0.6, label="Neck region")
    ax1.set_xlabel("x [mm]"); ax1.set_ylabel("y [mm]")
    ax1.set_title("WSSG Spatial Distribution")
    ax1.set_aspect("equal")
    ax1.legend(fontsize=8)

    # WSSG histogram by region
    ax2 = axes[1]
    for reg, colour in [("artery", "steelblue"), ("neck", "orange"),
                        ("dome", "firebrick")]:
        mask = region_masks[reg]
        if mask.sum() > 5:
            ax2.hist(wssg[mask], bins=30, alpha=0.6,
                     color=colour, label=reg.capitalize(),
                     density=True)
    ax2.set_xlabel("WSSG [Pa/m]")
    ax2.set_ylabel("Probability density")
    ax2.set_title("WSSG Distribution by Region")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"wssg_caseC_Re{Re:.0f}.png"
    plt.savefig(fname, dpi=150)
    print(f"[Plot] Saved: {fname}")
    plt.show()


# =============================================================================
# MODEL SAVE / LOAD
# =============================================================================

def save_aneurysm_model(model: nn.Module,
                        history: dict,
                        Re: float,
                        directory: str = "trained_models"):
    """Save trained aneurysm model with metadata.

    If the model is a HardAnsatzPINN wrapper, the base PINN state dict
    is saved (without the 'base.' prefix) so it remains compatible with
    load_aneurysm_model() and Stage 6.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"pinn_caseC_Re{Re:.0f}.pt")

    # Extract base state dict for compatibility with plain-PINN loaders
    if hasattr(model, 'base'):
        state_dict = model.base.state_dict()
        n_params   = count_parameters(model.base)
    else:
        state_dict = model.state_dict()
        n_params   = count_parameters(model)

    torch.save({
        "model_state_dict": state_dict,
        "history":          history,
        "n_params":         n_params,
        "Re":               Re,
        "geometry": {
            "R_A": R_A, "L_A": L_A, "R_S": R_S, "neck_r": NECK_R
        },
        "physics": {
            "mu_0": MU_0, "mu_inf": MU_INF, "lambda": LAMBDA,
            "n_CY": N_CY, "a_CY": A_CY, "rho": RHO
        }
    }, path)
    # Also save a copy in the working directory for easy access from Stage 6
    torch.save(torch.load(path, weights_only=False), f"pinn_caseC_Re{Re:.0f}.pt")
    print(f"[Save] Model saved to: {path}")


def load_aneurysm_model(Re: float,
                        directory: str = "trained_models") -> nn.Module:
    """Load a trained aneurysm model."""
    path = os.path.join(directory, f"pinn_caseC_Re{Re:.0f}.pt")
    if not os.path.exists(path):
        path = f"pinn_caseC_Re{Re:.0f}.pt"
    model = PINN(n_input=3, n_hidden=64, n_layers=5, n_output=4).to(DEVICE)
    ckpt  = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Load] Model loaded: {path}  (Re={ckpt['Re']}, "
          f"n_params={ckpt['n_params']})")
    return model


# =============================================================================
# ACCEPTANCE CRITERION CHECK  (Section 3.5.3)
# =============================================================================

def check_acceptance_criteria(vel_data: dict,
                               wss_data: dict,
                               fp: dict) -> bool:
    """
    Evaluate the three pre-specified acceptance criteria for Stage 5.

    Criteria (Section 3.5.3):
        AC1: eps_u_artery < 0.05
             Relative L2 error on mean axial velocity in parent artery
        AC2: WSS_dome < WSS_artery
             Mean dome WSS must be lower than mean artery WSS (qualitative)
        AC3: |WSS_nn - WSS_Newt| / WSS_Newt > 0.01
             Measurable non-Newtonian effect (at least 1% difference)

    Returns
    -------
    all_passed : bool
    """
    print("\n" + "="*50)
    print("ACCEPTANCE CRITERIA CHECK")
    print("="*50)

    all_passed = True

    # AC1: artery velocity error
    eps_u = abs(vel_data["u_artery_mean"] - fp["u_mean"]) / fp["u_mean"]
    ac1 = eps_u < 0.05
    print(f"  AC1: eps_u_artery = {eps_u*100:.2f}%  "
          f"(threshold: < 5%)  → {'PASS' if ac1 else 'FAIL'}")
    all_passed &= ac1

    # AC2: dome WSS < artery WSS
    wss_dome   = wss_data["stats"]["dome"]["mean"]
    wss_artery = wss_data["stats"]["artery"]["mean"]
    ac2 = wss_dome < wss_artery
    print(f"  AC2: WSS_dome={wss_dome:.4f} Pa < "
          f"WSS_artery={wss_artery:.4f} Pa  → {'PASS' if ac2 else 'FAIL'}")
    all_passed &= ac2

    # AC3: non-Newtonian effect > 1%
    kappa_mean = wss_data["kappa_nn"].mean()
    nn_effect  = abs(1.0 - kappa_mean)
    ac3 = nn_effect > 0.01
    print(f"  AC3: |κ_NN - 1| = {nn_effect*100:.2f}%  "
          f"(threshold: > 1%)  → {'PASS' if ac3 else 'FAIL'}")
    all_passed &= ac3

    print(f"\n  Overall: {'ALL CRITERIA PASSED' if all_passed else 'SOME CRITERIA FAILED'}")
    print("="*50)
    return all_passed


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("STAGE 5: SACCULAR ANEURYSM — NON-NEWTONIAN PINN")
    print("         Primary Clinical Case")
    print("=" * 60)

    # ----------------------------------------------------------------
    # 1. Load Case C geometry
    # ----------------------------------------------------------------
    print("\n[Step 1] Loading Case C geometry...")
    try:
        data = load_geometry("C_saccular_aneurysm")
    except FileNotFoundError:
        print("  geometry_data/ not found. Regenerating Case C geometry...")
        from stage1_geometry import SaccularAneurysm, CASE_C_PARAMS, save_geometry
        aneurysm_geom = SaccularAneurysm(CASE_C_PARAMS)
        data_C = aneurysm_geom.generate_all(seed=42)
        save_geometry(data_C)
        data = data_C

    print(f"  Interior : {data['interior'].shape}")
    print(f"  Wall     : {data['wall'].shape}")
    print(f"  Inlet    : {data['inlet'].shape}")
    print(f"  Outlet   : {data['outlet'].shape}")

    # ----------------------------------------------------------------
    # 2. Classify wall and interior points by anatomical region
    # ----------------------------------------------------------------
    print("\n[Step 2] Classifying wall and interior points by region...")
    region_masks     = classify_wall_points(data["wall"])
    region_masks_int = classify_interior_points(data["interior"])

    neck_mask    = region_masks["neck"]
    sac_int_mask = region_masks_int["sac_interior"]

    # ----------------------------------------------------------------
    # 3. Coordinate normaliser
    # ----------------------------------------------------------------
    print("\n[Step 3] Setting up coordinate normaliser...")
    all_pts    = np.vstack([data["interior"], data["wall"],
                            data["inlet"],    data["outlet"]])
    normaliser = CoordinateNormaliser(all_pts)

    # ----------------------------------------------------------------
    # 4. Primary case: Re = 250
    # ----------------------------------------------------------------
    print("\n[Step 4] Primary training run: Re = 250")
    fp_250 = flow_params(250)
    print(f"  u_max={fp_250['u_max']:.5f} m/s, "
          f"delta_P={fp_250['delta_P']:.4f} Pa")

    model_250 = HardAnsatzPINN(
        PINN(n_input=3, n_hidden=64, n_layers=5, n_output=4),
        u_max=fp_250["u_max"],
        normaliser=normaliser
    ).to(DEVICE)
    print(f"  Parameters: {count_parameters(model_250.base):,}  "
          f"[Hard Ansatz: inlet BC enforced by architecture]")

    # Warm-start from Stage 4 if available
    # Load into model.base — state dicts are stored without 'base.' prefix
    for warm_path in ["trained_models/pinn_caseB_nonnewtonian.pt",
                      "trained_models/pinn_caseA_nonnewtonian.pt",
                      "pinn_caseB_nonnewtonian.pt",
                      "pinn_caseA_nonnewtonian.pt"]:
        if os.path.exists(warm_path):
            ckpt = torch.load(warm_path, map_location=DEVICE, weights_only=False)
            model_250.base.load_state_dict(ckpt["model_state_dict"])
            print(f"  Warm-started from: {warm_path}")
            break
    else:
        print("  Training from random (Glorot) initialisation")

    loss_fn_250 = AneurysmLoss(u_max=fp_250["u_max"])
    trainer_250 = AneurysmTrainer(
        model_250, loss_fn_250, normaliser, data,
        fp_250, neck_mask, sac_int_mask,
        lbfgs_subsample=10_000
    )
    trainer_250.train_inlet_warmup()
    trainer_250.train_adam(n_iterations=40_000)
    trainer_250.train_lbfgs(max_iter=8_000)

    # ----------------------------------------------------------------
    # 5. Primary analysis: Re = 250
    # ----------------------------------------------------------------
    print("\n[Step 5] WSS analysis — Re = 250")
    wss_data_250 = compute_aneurysm_wss(
        model_250, normaliser,
        data["wall"], data["wall_normals"], region_masks
    )

    print("\n[Step 5b] WSSG computation — Re = 250")
    wssg_250 = compute_wssg(wss_data_250["wss_nn"], data["wall"])

    print("\n[Step 5c] Pressure field analysis — Re = 250")
    p_data_250 = analyse_pressure_field(
        model_250, normaliser, data["interior"], region_masks_int, fp_250
    )

    print("\n[Step 5d] Velocity field analysis — Re = 250")
    v_data_250 = analyse_velocity_field(
        model_250, normaliser, data["interior"], region_masks_int, fp_250
    )

    # ----------------------------------------------------------------
    # 6. Acceptance criteria
    # ----------------------------------------------------------------
    print("\n[Step 6] Checking acceptance criteria...")
    passed = check_acceptance_criteria(v_data_250, wss_data_250, fp_250)

    # ----------------------------------------------------------------
    # 7. Plots — Re = 250
    # ----------------------------------------------------------------
    print("\n[Step 7] Generating plots — Re = 250")
    pass  # plot_training_history skipped — Stage 5 history keys differ from Stage 2 format
    plot_wss_risk_map(wss_data_250["wss_nn"], data["wall"],
                      region_masks, Re=250)
    plot_wssg_distribution(wssg_250, data["wall"], region_masks, Re=250)
    plot_velocity_sac_midplane(model_250, normaliser, data["interior"], Re=250)

    # ----------------------------------------------------------------
    # 8. Save Re=250 results
    # ----------------------------------------------------------------
    save_aneurysm_model(model_250, trainer_250.history, Re=250)
    np.save("wss_nn_caseC_Re250.npy",   wss_data_250["wss_nn"])
    np.save("wss_newt_caseC_Re250.npy", wss_data_250["wss_newt"])
    np.save("kappa_nn_caseC_Re250.npy", wss_data_250["kappa_nn"])
    np.save("wssg_caseC_Re250.npy",     wssg_250)

    # ----------------------------------------------------------------
    # 9. Multi-Re parametric study (Re = 100, 400)
    # ----------------------------------------------------------------
    print("\n[Step 8] Multi-Re parametric study...")
    multi_re_results = {250: {
        "model":    model_250,
        "fp":       fp_250,
        "history":  trainer_250.history,
        "wss":      wss_data_250,
        "wssg":     wssg_250,
        "pressure": p_data_250,
        "velocity": v_data_250,
    }}

    for Re_other in [100, 400]:
        fp_other  = flow_params(Re_other)
        model_oth = HardAnsatzPINN(
            PINN(n_input=3, n_hidden=64, n_layers=5, n_output=4),
            u_max=fp_other["u_max"],
            normaliser=normaliser
        ).to(DEVICE)
        # Warm-start from Re=250 solution (base state dict, no 'base.' prefix)
        ckpt_250  = torch.load("pinn_caseC_Re250.pt", map_location=DEVICE,
                               weights_only=False)
        model_oth.base.load_state_dict(ckpt_250["model_state_dict"])
        print(f"\n  Training Re={Re_other} (warm-started from Re=250)...")

        loss_oth    = AneurysmLoss(u_max=fp_other["u_max"])
        trainer_oth = AneurysmTrainer(
            model_oth, loss_oth, normaliser, data,
            fp_other, neck_mask, sac_int_mask, lbfgs_subsample=10_000
        )
        trainer_oth.train_inlet_warmup()
        trainer_oth.train_adam(n_iterations=30_000)
        trainer_oth.train_lbfgs(max_iter=5_000)

        wss_oth  = compute_aneurysm_wss(model_oth, normaliser,
                                         data["wall"], data["wall_normals"],
                                         region_masks)
        wssg_oth = compute_wssg(wss_oth["wss_nn"], data["wall"])
        p_oth    = analyse_pressure_field(model_oth, normaliser,
                                          data["interior"], region_masks_int,
                                          fp_other)
        v_oth    = analyse_velocity_field(model_oth, normaliser,
                                          data["interior"], region_masks_int,
                                          fp_other)

        plot_wss_risk_map(wss_oth["wss_nn"], data["wall"],
                          region_masks, Re=Re_other)

        save_aneurysm_model(model_oth, trainer_oth.history, Re=Re_other)
        np.save(f"wss_nn_caseC_Re{Re_other}.npy",   wss_oth["wss_nn"])
        np.save(f"wss_newt_caseC_Re{Re_other}.npy",  wss_oth["wss_newt"])
        np.save(f"kappa_nn_caseC_Re{Re_other}.npy",  wss_oth["kappa_nn"])

        multi_re_results[Re_other] = {
            "model":    model_oth, "fp":      fp_other,
            "history":  trainer_oth.history, "wss": wss_oth,
            "wssg":     wssg_oth, "pressure": p_oth, "velocity": v_oth,
        }

    # ----------------------------------------------------------------
    # 10. Multi-Re comparison plots
    # ----------------------------------------------------------------
    print("\n[Step 9] Multi-Re comparison plots...")
    plot_multi_re_comparison(multi_re_results)

    # ----------------------------------------------------------------
    # 11. Summary
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STAGE 5 SUMMARY")
    print(f"{'='*60}")
    for Re in RE_LIST:
        r = multi_re_results[Re]
        ws = r["wss"]
        vs = r["velocity"]
        print(f"\n  Re = {Re}:")
        print(f"    Dome mean WSS (nn)    : {ws['stats']['dome']['mean']:.4f} Pa")
        print(f"    Artery mean WSS (nn)  : {ws['stats']['artery']['mean']:.4f} Pa")
        print(f"    Low-WSS fraction      : {ws['low_wss_frac']*100:.1f}%")
        print(f"    Mean κ_NN             : {ws['kappa_nn'].mean():.3f}")
        print(f"    Recirculation index   : {vs['recirculation_index']*100:.1f}%")
        print(f"    Dome Cp               : {r['pressure']['Cp']:.3f}")

    print(f"\n  Acceptance criteria (Re=250): "
          f"{'PASSED' if passed else 'FAILED'}")
    print(f"{'='*60}")
    print("\nStage 5 complete. Run stage6_risk_assessment.py next.")
