"""
=============================================================================
STAGE 2: NEWTONIAN PINN — CASE A VALIDATION (HAGEN-POISEUILLE)
=============================================================================
Dissertation: Physics-Informed Neural Networks for Cerebral Aneurysm
              Haemodynamic Risk Assessment
University of Zimbabwe | Department of Mathematics

This module implements the core PINN solver for the straight cylinder
geometry (Case A) using a Newtonian (constant viscosity) fluid model.
The purpose of this stage is to validate the network architecture,
loss function formulation, and training procedure against the exact
Hagen-Poiseuille analytical solution before introducing the additional
complexity of the Carreau-Yasuda non-Newtonian model in Stage 3.

What this module implements:
    - Section 3.4.4 : Network architecture (5 hidden layers, 64 neurons, tanh)
    - Section 3.4.5 : Composite loss function, equations (3.10) to (3.13c)
    - Section 3.4.6 : Two-stage Adam to L-BFGS training with gradient monitoring
    - Section 3.5.1 : Hagen-Poiseuille validation, relative L2 error (Eq 3.16)
    - Section 3.5.3 : WSS computation via automatic differentiation (Eq 3.18)

Acceptance criterion (pre-specified, Section 3.5.1):
    eps_u < 0.05   (5% relative L2 error on the velocity field)

Dependencies:
    pip install torch numpy scipy matplotlib

Run after stage1_geometry.py has been executed and geometry_data/ exists.

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

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using: {DEVICE}")


# =============================================================================
# PHYSICAL CONSTANTS  (Chapter 3, Table 3.1 and Section 3.4.3)
# =============================================================================

# For the Newtonian validation case, viscosity is fixed at mu_inf.
# This ensures direct comparability with the Hagen-Poiseuille analytical
# solution, which assumes constant viscosity (Section 3.5.1).
MU_NEWTONIAN = 0.00345      # [Pa.s]  infinite-shear viscosity
RHO          = 1060.0       # [kg/m3] blood density
RE           = 250.0        # [-]     Reynolds number

# Geometry (Case A)
R_PIPE = 0.003              # [m]  pipe radius
L_PIPE = 0.020              # [m]  pipe length

# Derived flow quantities
U_MEAN  = RE * MU_NEWTONIAN / (RHO * R_PIPE)
U_MAX   = 2.0 * U_MEAN
DELTA_P = 4.0 * MU_NEWTONIAN * U_MAX * L_PIPE / R_PIPE**2

print(f"[Physics] Re={RE}, u_mean={U_MEAN:.4f} m/s, u_max={U_MAX:.4f} m/s")
print(f"[Physics] delta_P={DELTA_P:.4f} Pa")
print(f"[Physics] WSS_analytical = {MU_NEWTONIAN * 2 * U_MAX / R_PIPE:.4f} Pa")


# =============================================================================
# SECTION 3.4.4 - NETWORK ARCHITECTURE
# =============================================================================

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for 3D incompressible flow.

    Architecture (Section 3.4.4):
        Input  : (x, y, z) spatial coordinates         -- 3 neurons
        Hidden : 5 fully connected layers, 64 neurons each, tanh activation
        Output : (u, v, w, p) velocity components + pressure -- 4 neurons

    Why tanh and not ReLU:
        The physics residuals require second-order spatial derivatives of
        the velocity field. ReLU has zero second derivatives almost
        everywhere, making it unsuitable for PDE-constrained training.
        tanh is infinitely differentiable, supporting all required
        derivative orders via PyTorch automatic differentiation.

    Why Glorot (Xavier) initialisation:
        tanh saturates at large input magnitudes (gradient approaches 0).
        Glorot initialisation keeps initial weights in the linear region
        of tanh, preventing saturation before training starts.
        W ~ Uniform(-sqrt(6/(n_in+n_out)), +sqrt(6/(n_in+n_out))).
    """

    def __init__(self,
                 n_input:  int = 3,
                 n_hidden: int = 64,
                 n_layers: int = 5,
                 n_output: int = 4):
        super().__init__()

        layers = []
        layers.append(nn.Linear(n_input, n_hidden))
        layers.append(nn.Tanh())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(n_hidden, n_output))
        self.network = nn.Sequential(*layers)
        self._initialise_weights()

    def _initialise_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: maps spatial coordinates to flow variables.

        Parameters
        ----------
        x : (N, 3) tensor of (x, y, z) coordinates with requires_grad=True

        Returns
        -------
        out : (N, 4) tensor of (u, v, w, p)
        """
        return self.network(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# COORDINATE NORMALISATION
# =============================================================================

class CoordinateNormaliser:
    """
    Normalise input coordinates to [-1, 1] before feeding to the network.

    Without normalisation, coordinates in metres (e.g. x in [0, 0.02])
    produce very small weight gradients, causing slow convergence.
    The linear mapping applied is:
        x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    """

    def __init__(self, all_points: np.ndarray):
        self.x_min   = all_points.min(axis=0)
        self.x_max   = all_points.max(axis=0)
        self.x_range = self.x_max - self.x_min

        print(f"[Normaliser] x: [{self.x_min[0]:.4f}, {self.x_max[0]:.4f}]")
        print(f"[Normaliser] y: [{self.x_min[1]:.4f}, {self.x_max[1]:.4f}]")
        print(f"[Normaliser] z: [{self.x_min[2]:.4f}, {self.x_max[2]:.4f}]")

    def normalise(self, pts: np.ndarray) -> torch.Tensor:
        """Convert numpy array to normalised torch tensor on DEVICE."""
        pts_norm = 2.0 * (pts - self.x_min) / self.x_range - 1.0
        return torch.tensor(pts_norm, dtype=torch.float32,
                            requires_grad=True, device=DEVICE)


# =============================================================================
# SECTION 3.4.5 - PHYSICS RESIDUALS (NEWTONIAN)
# =============================================================================

def compute_derivatives(model: PINN, x_norm: torch.Tensor) -> dict:
    """
    Compute network output and all required spatial derivatives
    via automatic differentiation.

    For the Newtonian case we need:
        First-order  : du/dx, du/dy, du/dz etc.  (continuity)
        Second-order : d2u/dx2, d2u/dy2, d2u/dz2 (viscous Laplacian)

    All derivatives are in normalised coordinates.
    Physical derivatives = (2/x_range) * normalised derivatives.
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

    # First-order: shape (N, 3)
    du = grad(u, x_norm)
    dv = grad(v, x_norm)
    dw = grad(w, x_norm)
    dp = grad(p, x_norm)

    # Second-order (Laplacian terms)
    d2u_dx2 = grad(du[:, 0:1], x_norm)[:, 0:1]
    d2u_dy2 = grad(du[:, 1:2], x_norm)[:, 1:2]
    d2u_dz2 = grad(du[:, 2:3], x_norm)[:, 2:3]

    d2v_dx2 = grad(dv[:, 0:1], x_norm)[:, 0:1]
    d2v_dy2 = grad(dv[:, 1:2], x_norm)[:, 1:2]
    d2v_dz2 = grad(dv[:, 2:3], x_norm)[:, 2:3]

    d2w_dx2 = grad(dw[:, 0:1], x_norm)[:, 0:1]
    d2w_dy2 = grad(dw[:, 1:2], x_norm)[:, 1:2]
    d2w_dz2 = grad(dw[:, 2:3], x_norm)[:, 2:3]

    return {
        "uvwp": uvwp,
        "u": u, "v": v, "w": w, "p": p,
        "du": du, "dv": dv, "dw": dw, "dp": dp,
        "d2u": (d2u_dx2, d2u_dy2, d2u_dz2),
        "d2v": (d2v_dx2, d2v_dy2, d2v_dz2),
        "d2w": (d2w_dx2, d2w_dy2, d2w_dz2),
    }


def physics_residuals_newtonian(derivs: dict,
                                 scale: np.ndarray,
                                 mu: float = MU_NEWTONIAN,
                                 rho: float = RHO) -> tuple:
    """
    Compute the Newtonian Navier-Stokes residuals at collocation points.

    Continuity (Eq 3.1):
        du/dx + dv/dy + dw/dz = 0

    Momentum x (Eq 3.4 with constant mu):
        rho*(u*du/dx + v*du/dy + w*du/dz) = -dp/dx + mu*(d2u/dx2 + d2u/dy2 + d2u/dz2)

    Similarly for y and z momentum components.

    Scale factor correction: physical_deriv = (2/x_range) * normalised_deriv
    """
    u, v, w, p = derivs["u"], derivs["v"], derivs["w"], derivs["p"]
    du, dv, dw, dp = derivs["du"], derivs["dv"], derivs["dw"], derivs["dp"]
    d2u, d2v, d2w = derivs["d2u"], derivs["d2v"], derivs["d2w"]

    sx = torch.tensor(2.0 / scale[0], dtype=torch.float32, device=DEVICE)
    sy = torch.tensor(2.0 / scale[1], dtype=torch.float32, device=DEVICE)
    sz = torch.tensor(2.0 / scale[2], dtype=torch.float32, device=DEVICE)

    # Physical first-order derivatives
    du_dx = du[:, 0:1]*sx;  du_dy = du[:, 1:2]*sy;  du_dz = du[:, 2:3]*sz
    dv_dx = dv[:, 0:1]*sx;  dv_dy = dv[:, 1:2]*sy;  dv_dz = dv[:, 2:3]*sz
    dw_dx = dw[:, 0:1]*sx;  dw_dy = dw[:, 1:2]*sy;  dw_dz = dw[:, 2:3]*sz
    dp_dx = dp[:, 0:1]*sx;  dp_dy = dp[:, 1:2]*sy;  dp_dz = dp[:, 2:3]*sz

    # Physical second-order derivatives
    d2u_dx2 = d2u[0]*sx**2;  d2u_dy2 = d2u[1]*sy**2;  d2u_dz2 = d2u[2]*sz**2
    d2v_dx2 = d2v[0]*sx**2;  d2v_dy2 = d2v[1]*sy**2;  d2v_dz2 = d2v[2]*sz**2
    d2w_dx2 = d2w[0]*sx**2;  d2w_dy2 = d2w[1]*sy**2;  d2w_dz2 = d2w[2]*sz**2

    # Continuity residual (Eq 3.1)
    R_cont = du_dx + dv_dy + dw_dz

    # Momentum residuals (Eq 3.4 with constant mu)
    conv_x = rho * (u*du_dx + v*du_dy + w*du_dz)
    conv_y = rho * (u*dv_dx + v*dv_dy + w*dv_dz)
    conv_z = rho * (u*dw_dx + v*dw_dy + w*dw_dz)

    visc_x = mu * (d2u_dx2 + d2u_dy2 + d2u_dz2)
    visc_y = mu * (d2v_dx2 + d2v_dy2 + d2v_dz2)
    visc_z = mu * (d2w_dx2 + d2w_dy2 + d2w_dz2)

    R_mom_x = conv_x + dp_dx - visc_x
    R_mom_y = conv_y + dp_dy - visc_y
    R_mom_z = conv_z + dp_dz - visc_z

    return R_cont, R_mom_x, R_mom_y, R_mom_z


# =============================================================================
# SECTION 3.4.5 - COMPOSITE LOSS FUNCTION  (Eqs 3.10 to 3.13c)
# =============================================================================

class CompositeLoss:
    """
    Full composite loss function from Section 3.4.5.

    L(theta) = w1*L_mom + w2*L_cont + w3*L_bc              (Eq 3.10)

    L_bc = w_wall*L_wall + w_in*L_inlet + w_out*L_outlet    (Eq 3.13)

    L_wall   = MSE no-slip at wall        (Eq 3.13a)
    L_inlet  = MSE parabolic inlet        (Eq 3.13b)
    L_outlet = MSE zero-pressure outlet   (Eq 3.13c)

    Weights (Section 3.4.5):
        w1 = w2 = w3 = 1.0
        w_wall  = 10.0   (no-slip: WSS-critical, large boundary)
        w_in    = 15.0   (inlet: most complex BC, WSS-sensitive)
        w_out   = 5.0    (outlet: simple scalar constraint)
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

    def momentum_loss(self, R_x, R_y, R_z):
        # Normalise by (rho*u_max^2) to bring from O(1e7) down to O(1).
        # Without this, momentum gradient is millions of times stronger
        # than BC gradient and the inlet profile never gets learned.
        P_SCALE = RHO * U_MAX**2 + 1e-10
        return (torch.mean((R_x / P_SCALE)**2) +
                torch.mean((R_y / P_SCALE)**2) +
                torch.mean((R_z / P_SCALE)**2)) / 3.0

    def continuity_loss(self, R_cont):
        return torch.mean(R_cont**2)

    def wall_loss(self, uvwp_wall):
        """Eq 3.13a: enforce u = v = w = 0 at wall."""
        return torch.mean(uvwp_wall[:, 0:3]**2)

    def inlet_loss(self, uvwp_inlet, x_inlet_phys):
        """Eq 3.13b: enforce parabolic inlet profile."""
        r2       = (x_inlet_phys[:, 1]**2 + x_inlet_phys[:, 2]**2).unsqueeze(1)
        u_target = U_MAX * (1.0 - r2 / R_PIPE**2)
        loss_u   = torch.mean((uvwp_inlet[:, 0:1] - u_target)**2)
        loss_vw  = torch.mean(uvwp_inlet[:, 1:2]**2) + torch.mean(uvwp_inlet[:, 2:3]**2)
        return loss_u + loss_vw

    def outlet_loss(self, uvwp_outlet):
        """Eq 3.13c: enforce p = 0 at outlet."""
        return torch.mean(uvwp_outlet[:, 3:4]**2)

    def total_loss(self, R_cont, R_x, R_y, R_z,
                   uvwp_wall, uvwp_inlet, x_inlet_phys, uvwp_outlet):
        L_mom    = self.momentum_loss(R_x, R_y, R_z)
        L_cont   = self.continuity_loss(R_cont)
        L_wall   = self.wall_loss(uvwp_wall)
        L_inlet  = self.inlet_loss(uvwp_inlet, x_inlet_phys)
        L_outlet = self.outlet_loss(uvwp_outlet)
        L_bc     = self.w_wall*L_wall + self.w_in*L_inlet + self.w_out*L_outlet
        L_total  = self.w1*L_mom + self.w2*L_cont + self.w3*L_bc
        return L_total, L_mom, L_cont, L_bc, L_wall, L_inlet, L_outlet


# =============================================================================
# GRADIENT DOMINANCE MONITOR  (Section 3.4.5)
# =============================================================================

def check_gradient_dominance(model, loss_components, threshold=1e3):
    """
    Check whether gradient dominance is occurring across loss components.

    Computes the L2 norm of gradients from each loss component w.r.t.
    all network parameters. Flags dominance if max/min ratio > threshold.

    This implements the monitoring commitment in Section 3.4.5.
    """
    norms = []
    for loss in loss_components:
        grads = torch.autograd.grad(
            loss, model.parameters(),
            retain_graph=True, create_graph=False, allow_unused=True
        )
        grad_flat = torch.cat([g.flatten() for g in grads if g is not None])
        norms.append(grad_flat.norm().item())

    ratio = max(norms) / (min(norms) + 1e-10)
    return ratio > threshold, norms, ratio


# =============================================================================
# SECTION 3.4.6 - TRAINING PROCEDURE
# =============================================================================

class PINNTrainer:
    """
    Manages the two-stage Adam to L-BFGS training (Section 3.4.6).

    Stage 1 - Adam (30,000 iterations):
        Learning rate 1e-3, decayed by 0.9 every 5,000 iterations.
        Handles the rough initial loss landscape efficiently.

    Stage 2 - L-BFGS (up to 5,000 iterations):
        Full-batch evaluation via closure function.
        Convergence: relative change in loss < 1e-6 for 1,000 iterations.
    """

    def __init__(self, model, loss_fn, normaliser, data):
        self.model      = model.to(DEVICE)
        self.loss_fn    = loss_fn
        self.normaliser = normaliser
        self.data       = data
        self._prepare_tensors()

        self.history = {
            k: [] for k in
            ["iteration", "L_total", "L_mom", "L_cont", "L_bc",
             "L_wall", "L_inlet", "L_outlet", "stage"]
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

    def _compute_loss(self):
        derivs = compute_derivatives(self.model, self.x_int)
        R_cont, R_x, R_y, R_z = physics_residuals_newtonian(derivs, self.scale)
        uvwp_wall   = self.model(self.x_wall)
        uvwp_inlet  = self.model(self.x_in_norm)
        uvwp_outlet = self.model(self.x_out)
        return self.loss_fn.total_loss(
            R_cont, R_x, R_y, R_z,
            uvwp_wall, uvwp_inlet, self.x_in_phys, uvwp_outlet
        )

    def _log(self, iteration, losses, stage):
        L_total, L_mom, L_cont, L_bc, L_wall, L_inlet, L_outlet = losses
        self.history["iteration"].append(iteration)
        self.history["L_total"].append(L_total.item())
        self.history["L_mom"].append(L_mom.item())
        self.history["L_cont"].append(L_cont.item())
        self.history["L_bc"].append(L_bc.item())
        self.history["L_wall"].append(L_wall.item())
        self.history["L_inlet"].append(L_inlet.item())
        self.history["L_outlet"].append(L_outlet.item())
        self.history["stage"].append(stage)
        if L_total.item() < self.best_loss:
            self.best_loss  = L_total.item()
            self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

    def train_adam(self, n_iterations=30_000, lr_initial=1e-3,
                   lr_decay=0.9, decay_every=5_000,
                   log_every=500, check_grad_every=5_000):
        print(f"\n{'='*60}")
        print(f"STAGE 1: ADAM ({n_iterations:,} iterations)")
        print(f"{'='*60}")

        optimiser = Adam(self.model.parameters(), lr=lr_initial)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=decay_every, gamma=lr_decay)
        t_start = time.time()

        for i in range(1, n_iterations + 1):
            # Curriculum: ramp physics weight 0.001->1.0 over first 10k iters
            # Prevents large momentum residual from destroying BC warmup
            ramp = min(1.0, 0.001 + 0.999 * (i / 10_000))
            self.loss_fn.w1 = ramp
            self.loss_fn.w2 = ramp

            optimiser.zero_grad()
            losses = self._compute_loss()
            losses[0].backward()
            # Gradient clipping prevents loss explosions (seen at iter ~8000/18000/26000)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimiser.step()
            scheduler.step()
            # Save best state every iteration (not just at log checkpoints)
            if losses[0].item() < self.best_loss:
                self.best_loss  = losses[0].item()
                self.best_state = {k: v.clone()
                                   for k, v in self.model.state_dict().items()}

            if i % log_every == 0 or i == 1:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  iter {i:6d} | L={losses[0].item():.3e} | "
                      f"mom={losses[1].item():.3e} | cont={losses[2].item():.3e} | "
                      f"bc={losses[3].item():.3e} | lr={lr_now:.2e} | "
                      f"t={time.time()-t_start:.0f}s")
                self._log(i, losses, "adam")

            if i % check_grad_every == 0:
                try:
                    dominated, norms, ratio = check_gradient_dominance(
                        self.model, [losses[1], losses[2], losses[3]])
                    status = "DOMINANCE DETECTED" if dominated else "OK"
                    print(f"  [GradCheck {i}] norms={[f'{n:.2e}' for n in norms]} "
                          f"ratio={ratio:.1f} [{status}] ramp={ramp:.3f}")
                except RuntimeError:
                    print(f"  [GradCheck {i}] skipped (graph freed) ramp={ramp:.3f}")

        print(f"\nAdam done. Best loss: {self.best_loss:.4e}")

    def train_lbfgs(self, max_iter=5_000, tolerance=1e-6,
                    log_every=200, history_size=50):
        print(f"\n{'='*60}")
        print(f"STAGE 2: L-BFGS (max {max_iter:,} iterations)")
        print(f"{'='*60}")

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            print(f"  Loaded best Adam state (in-memory, loss={self.best_loss:.4e})")

        optimiser = LBFGS(self.model.parameters(),
                          lr=1.0, max_iter=20, history_size=history_size,
                          tolerance_grad=1e-7, tolerance_change=1e-9,
                          line_search_fn="strong_wolfe")

        prev_loss  = float("inf")
        n_stagnant = 0
        loss_store = [None]
        t_start    = time.time()

        def closure():
            optimiser.zero_grad()
            losses = self._compute_loss()
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
# SECTION 3.5.1 - VALIDATION AGAINST HAGEN-POISEUILLE
# =============================================================================

def validate_hagen_poiseuille(model, normaliser, data, n_test=5_000, seed=99):
    """
    Validate trained PINN against exact Hagen-Poiseuille solution.

    Metric (Eq 3.16):
        eps_u = ||u_PINN - u_exact||_2 / ||u_exact||_2

    Acceptance criterion: eps_u < 0.05

    Evaluated on 5,000 fresh test points not used during training.
    """
    print(f"\n{'='*60}")
    print("VALIDATION: HAGEN-POISEUILLE BENCHMARK")
    print(f"{'='*60}")
    model.eval()

    rng     = np.random.default_rng(seed)
    r_test  = R_PIPE * np.sqrt(rng.uniform(0, 1, n_test))
    phi     = rng.uniform(0, 2*np.pi, n_test)
    x_test  = rng.uniform(0, L_PIPE, n_test)
    y_test  = r_test * np.cos(phi)
    z_test  = r_test * np.sin(phi)
    pts     = np.column_stack([x_test, y_test, z_test])

    # Analytical solution (Eq 3.14 and 3.15)
    r2      = y_test**2 + z_test**2
    u_exact = np.clip(U_MAX * (1.0 - r2/R_PIPE**2), 0, None)
    p_exact = DELTA_P * (1.0 - x_test/L_PIPE)

    with torch.no_grad():
        x_norm = normaliser.normalise(pts).detach()
        pred   = model(x_norm).cpu().numpy()

    u_pred = pred[:, 0]
    p_pred = pred[:, 3]

    def rel_l2(a, b):
        return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-10)

    eps_u = rel_l2(u_pred, u_exact)
    eps_p = rel_l2(p_pred, p_exact)
    passed = eps_u < 0.05

    # Peak centreline velocity
    r_arr  = np.sqrt(y_test**2 + z_test**2)
    cl_mask = r_arr < 0.1 * R_PIPE
    u_max_pred = u_pred[cl_mask].max() if cl_mask.sum() > 0 else u_pred.max()

    print(f"\n  eps_u (velocity L2 error) = {eps_u:.4f}  "
          f"({'PASS' if passed else 'FAIL'}, criterion < 0.05)")
    print(f"  eps_p (pressure L2 error) = {eps_p:.4f}")
    print(f"  Peak centreline u: analytical={U_MAX:.5f}  PINN={u_max_pred:.5f} m/s")
    print(f"  Peak velocity error: {abs(u_max_pred-U_MAX)/U_MAX*100:.3f}%")

    if passed:
        print(f"\n  ACCEPTANCE CRITERION MET -- proceed to Stage 3")
    else:
        print(f"\n  ACCEPTANCE CRITERION NOT MET")
        print(f"  Suggested actions:")
        print(f"    - Increase Adam iterations to 50,000")
        print(f"    - Increase n_hidden to 128")
        print(f"    - Increase w_in (inlet BC weight)")

    return {"eps_u": eps_u, "eps_p": eps_p,
            "u_max_pred": u_max_pred, "u_max_exact": U_MAX,
            "passed": passed,
            "test_pts": pts, "u_exact": u_exact, "u_pred": u_pred,
            "p_exact": p_exact, "p_pred": p_pred}


# =============================================================================
# SECTION 3.5.3 - WSS VIA AUTOMATIC DIFFERENTIATION (Eq 3.18)
# =============================================================================

def compute_wss(model, normaliser, wall_pts, wall_normals, mu=MU_NEWTONIAN):
    """
    Compute WSS magnitude at all wall points (Eq 3.18).

    tau_w(xj) = mu * [(grad_u + grad_u^T) . n]_wall

    For Newtonian case mu is constant.
    Stage 3 will extend this to spatially varying Carreau-Yasuda mu(gamma_dot).
    """
    model.eval()
    x_wall = normaliser.normalise(wall_pts)
    n_tens = torch.tensor(wall_normals, dtype=torch.float32, device=DEVICE)
    scale  = normaliser.x_range

    uvwp = model(x_wall)
    u = uvwp[:, 0:1]; v = uvwp[:, 1:2]; w = uvwp[:, 2:3]

    def grad(f, x):
        return torch.autograd.grad(f, x,
                                    grad_outputs=torch.ones_like(f),
                                    create_graph=False,
                                    retain_graph=True)[0]

    sx = 2.0/scale[0]; sy = 2.0/scale[1]; sz = 2.0/scale[2]

    du = grad(u, x_wall)
    dv = grad(v, x_wall)
    dw = grad(w, x_wall)

    du_dx=du[:,0]*sx; du_dy=du[:,1]*sy; du_dz=du[:,2]*sz
    dv_dx=dv[:,0]*sx; dv_dy=dv[:,1]*sy; dv_dz=dv[:,2]*sz
    dw_dx=dw[:,0]*sx; dw_dy=dw[:,1]*sy; dw_dz=dw[:,2]*sz

    nx = n_tens[:,0]; ny = n_tens[:,1]; nz = n_tens[:,2]

    # Stress vector: tau . n = mu*(grad_u + grad_u^T) . n
    tau_x = mu*((du_dx+du_dx)*nx + (du_dy+dv_dx)*ny + (du_dz+dw_dx)*nz)
    tau_y = mu*((dv_dx+du_dy)*nx + (dv_dy+dv_dy)*ny + (dv_dz+dw_dy)*nz)
    tau_z = mu*((dw_dx+du_dz)*nx + (dw_dy+dv_dz)*ny + (dw_dz+dw_dz)*nz)

    wss = torch.sqrt(tau_x**2 + tau_y**2 + tau_z**2)
    return wss.detach().cpu().numpy()


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_training_history(history, save_path="training_history_caseA.png"):
    """Figure 4.1: Convergence plot across both training stages."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    iters  = np.array(history["iteration"])
    stages = np.array(history["stage"])

    adam_m  = stages == "adam"
    lbfgs_m = stages == "lbfgs"

    ax = axes[0]
    if adam_m.any():
        ax.semilogy(iters[adam_m],  np.array(history["L_total"])[adam_m],
                    "b-", label="Total Loss (Adam)", linewidth=1.5)
    if lbfgs_m.any():
        ax.semilogy(iters[lbfgs_m], np.array(history["L_total"])[lbfgs_m],
                    "r-", label="Total Loss (L-BFGS)", linewidth=1.5)
    if adam_m.any() and lbfgs_m.any():
        ax.axvline(iters[lbfgs_m][0], color="k", linestyle="--",
                   alpha=0.5, label="Adam to L-BFGS switch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Case A Training Convergence")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    colours = {"L_mom":"steelblue","L_cont":"darkorange",
               "L_wall":"green","L_inlet":"purple","L_outlet":"brown"}
    for key, col in colours.items():
        ax.semilogy(iters, np.array(history[key]),
                    label=key, linewidth=1.0, color=col)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Loss (log scale)")
    ax.set_title("Loss Component Breakdown")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Plot] Saved: {save_path}")
    plt.show()


def plot_velocity_profile(results, save_path="velocity_profile_caseA.png"):
    """Figure 4.2: PINN vs analytical axial velocity profile."""
    pts    = results["test_pts"]
    u_pred = results["u_pred"]
    u_ex   = results["u_exact"]

    mid = (pts[:,0] > 0.4*L_PIPE) & (pts[:,0] < 0.6*L_PIPE)
    r_mid = np.sqrt(pts[mid,1]**2 + pts[mid,2]**2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    r_line = np.linspace(0, R_PIPE, 200)
    ax.plot(r_line*1000, U_MAX*(1-r_line**2/R_PIPE**2),
            "k-", linewidth=2, label="Analytical")
    ax.scatter(r_mid*1000, u_pred[mid], s=8, c="steelblue", alpha=0.6,
               label=f"PINN (eps_u={results['eps_u']:.3f})")
    ax.set_xlabel("r [mm]"); ax.set_ylabel("u [m/s]")
    ax.set_title("Axial Velocity at z = L/2")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    r_all = np.sqrt(pts[:,1]**2 + pts[:,2]**2)
    err   = np.abs(u_pred - u_ex)
    sc = ax.scatter(r_all*1000, err, s=3, c=err, cmap="hot_r", alpha=0.5)
    plt.colorbar(sc, ax=ax, label="|u_PINN - u_exact| [m/s]")
    ax.set_xlabel("r [mm]"); ax.set_ylabel("Absolute error [m/s]")
    ax.set_title("Pointwise Velocity Error")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Plot] Saved: {save_path}")
    plt.show()


def plot_wss(wss_pinn, wall_pts, save_path="wss_caseA.png"):
    """Figure 4.3: WSS circumferential distribution vs analytical value."""
    wss_exact = MU_NEWTONIAN * 2 * U_MAX / R_PIPE
    phi = np.arctan2(wall_pts[:,2], wall_pts[:,1]) * 180 / np.pi
    cov = wss_pinn.std() / wss_pinn.mean() * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(phi, wss_pinn, s=4, c="steelblue", alpha=0.5,
               label=f"PINN WSS (mean={wss_pinn.mean():.4f} Pa)")
    ax.axhline(wss_exact, color="red", linewidth=2, linestyle="--",
               label=f"Analytical = {wss_exact:.4f} Pa")
    ax.set_xlabel("Circumferential angle [deg]")
    ax.set_ylabel("WSS [Pa]")
    ax.set_title(f"Wall Shear Stress -- Case A (CoV = {cov:.2f}%)")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Plot] Saved: {save_path}")
    print(f"  Mean WSS PINN      : {wss_pinn.mean():.4f} Pa")
    print(f"  Mean WSS analytical: {wss_exact:.4f} Pa")
    print(f"  WSS relative error : {abs(wss_pinn.mean()-wss_exact)/wss_exact*100:.2f}%")
    print(f"  Circumferential CoV: {cov:.2f}%")
    plt.show()


# =============================================================================
# SAVE / LOAD
# =============================================================================

def save_model(model, history, results, directory="trained_models"):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "pinn_caseA_newtonian.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "validation": {k: v for k, v in results.items()
                       if isinstance(v, (float, bool, int))},
        "physics": {"mu": MU_NEWTONIAN, "rho": RHO,
                    "Re": RE, "u_max": U_MAX, "delta_P": DELTA_P}
    }, path)
    print(f"[Save] Model saved: {path}")


def load_geometry(case_name="A_straight_cylinder", directory="geometry_data"):
    case_dir = os.path.join(directory, case_name)
    data = {"case": case_name}
    for key in ["interior", "wall", "wall_normals", "inlet", "outlet"]:
        data[key] = np.load(os.path.join(case_dir, f"{key}.npy"))
    print(f"[Load] Geometry loaded: {case_dir}")
    return data


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("STAGE 2: NEWTONIAN PINN -- CASE A")
    print("=" * 60)

    # 1. Load geometry from Stage 1
    data = load_geometry("A_straight_cylinder")

    # 2. Build coordinate normaliser
    all_pts    = np.vstack([data["interior"], data["wall"],
                            data["inlet"],    data["outlet"]])
    normaliser = CoordinateNormaliser(all_pts)

    # 3. Build network
    model  = PINN(n_input=3, n_hidden=64, n_layers=5, n_output=4)
    n_p    = count_parameters(model)
    print(f"\n[Network] 3 -> [64x5] -> 4  |  Parameters: {n_p:,}")

    # 4. Loss function
    # With normalised momentum (O(1)), w_in=200 gives BC 200x more weight
    # than a single momentum point — sufficient to enforce parabolic profile
    loss_fn = CompositeLoss(w1=1.0, w2=1.0, w3=1.0,
                             w_wall=50.0, w_in=200.0, w_out=20.0)

    # ── Phase 0: BC-only warmup (3,000 iterations) ───────────────────────
    # Train BCs alone before adding physics. Without this, the large
    # momentum residual at iter 1 (O(1e7)) drowns out the inlet BC signal
    # and the network never learns the parabolic velocity profile.
    print("\n[Warmup] Phase 0: BC-only (3,000 iters)...")
    import torch.optim as _optim
    _wo = _optim.Adam(model.parameters(), lr=1e-3)
    x_in_w   = normaliser.normalise(data["inlet"]).to(DEVICE)
    x_wall_w = normaliser.normalise(data["wall"]).to(DEVICE)
    x_out_w  = normaliser.normalise(data["outlet"]).to(DEVICE)
    for _wi in range(1, 3001):
        _wo.zero_grad()
        uvwp_in   = model(x_in_w)
        uvwp_wall = model(x_wall_w)
        uvwp_out  = model(x_out_w)
        _y = data["inlet"][:, 1]; _z = data["inlet"][:, 2]
        _r2 = torch.tensor(_y**2 + _z**2, dtype=torch.float32,
                            device=x_in_w.device).unsqueeze(1)
        _u_tgt = U_MAX * (1.0 - _r2 / R_PIPE**2)
        _u_tgt = torch.clamp(_u_tgt, min=0.0)
        _L = (200.0 * (torch.mean((uvwp_in[:, 0:1] - _u_tgt)**2) +
                       torch.mean(uvwp_in[:, 1:2]**2) +
                       torch.mean(uvwp_in[:, 2:3]**2)) +
               50.0 * torch.mean(uvwp_wall[:, 0:3]**2) +
               20.0 * torch.mean(uvwp_out[:, 3:4]**2))
        _L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        _wo.step()
        if _wi % 1000 == 0:
            print(f"  warmup {_wi}/3000 | bc={_L.item():.4e}")
    print("[Warmup] Phase 0 complete. Starting full physics training...\n")

    # 5. Train — store warmup result as initial best state
    trainer = PINNTrainer(model, loss_fn, normaliser, data)
    trainer.best_state = {k: v.clone() for k, v in model.state_dict().items()}
    trainer.best_loss  = float("inf")
    trainer.train_adam(n_iterations=50_000, lr_initial=5e-4,
                       lr_decay=0.95, decay_every=5_000,
                       log_every=500, check_grad_every=5_000)
    trainer.train_lbfgs(max_iter=5_000, tolerance=1e-6,
                        log_every=200, history_size=50)

    # 6. Validate
    results = validate_hagen_poiseuille(model, normaliser, data)

    # 7. Compute WSS
    print("\n[WSS] Computing via automatic differentiation...")
    wss = compute_wss(model, normaliser,
                      data["wall"], data["wall_normals"])

    # 8. Plots
    plot_training_history(trainer.history)
    plot_velocity_profile(results)
    plot_wss(wss, data["wall"])

    # 9. Save
    save_model(model, trainer.history, results)

    # 10. Summary (Table 4.1)
    wss_exact = MU_NEWTONIAN * 2 * U_MAX / R_PIPE
    print(f"\n{'='*60}")
    print("CASE A VALIDATION SUMMARY  (Table 4.1)")
    print(f"{'='*60}")
    print(f"  u_max   analytical : {U_MAX:.6f} m/s")
    print(f"  u_max   PINN       : {results['u_max_pred']:.6f} m/s")
    print(f"  eps_u              : {results['eps_u']:.4f}  ({results['eps_u']*100:.2f}%)")
    print(f"  WSS     analytical : {wss_exact:.4f} Pa")
    print(f"  WSS     PINN       : {wss.mean():.4f} Pa")
    print(f"  WSS error          : {abs(wss.mean()-wss_exact)/wss_exact*100:.2f}%")
    print(f"  Criterion (< 5%)   : {'PASSED' if results['passed'] else 'FAILED'}")
    print(f"{'='*60}")

    if results["passed"]:
        print("\nStage 2 complete. Run stage3_carreau_yasuda.py next.")
    else:
        print("\nCriterion not met. Review architecture or training settings.")
