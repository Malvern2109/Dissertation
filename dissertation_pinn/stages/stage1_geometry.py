"""
=============================================================================
STAGE 1: GEOMETRY GENERATION AND POINT CLOUD SAMPLING
=============================================================================
Dissertation: Physics-Informed Neural Networks for Cerebral Aneurysm
              Haemodynamic Risk Assessment
University of Zimbabwe | Department of Mathematics

This module generates the three geometric domains described in Chapter 3
and produces the collocation point clouds used for PINN training.

Three cases are implemented:
    Case A - Straight cylinder       (primary Hagen-Poiseuille validation)
    Case B - Curved pipe / toroid    (Dean flow, intermediate complexity)
    Case C - Saccular aneurysm       (primary clinical domain)

For each case the module produces:
    - interior_points : (N_c, 3)  collocation points inside the domain
    - wall_points     : (N_w, 3)  points on the vessel wall
    - inlet_points    : (N_i, 3)  points on the inlet cross-section
    - outlet_points   : (N_o, 3)  points on the outlet cross-section
    - wall_normals    : (N_w, 3)  outward unit normals at wall points
                                  (needed for WSS computation in Stage 6)

Dependencies:
    pip install numpy scipy matplotlib
    pip install gmsh          # geometry kernel
    pip install pyvista       # 3D visualisation (optional but recommended)

Author: [Your Name]
Date  : [Date]
=============================================================================
"""

import numpy as np
from scipy.stats import qmc          # Latin Hypercube Sampling
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Optional imports - gracefully degrade if not installed
try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False
    print("[WARNING] gmsh not found. Using parametric geometry fallback.")

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("[WARNING] pyvista not found. Using matplotlib for visualisation.")


# =============================================================================
# GEOMETRIC PARAMETERS (from Chapter 3, Section 3.3.3)
# =============================================================================

CASE_A_PARAMS = {
    "R"      : 0.003,    # pipe radius [m]  -- 3mm, medium cerebral artery
    "L"      : 0.020,    # pipe length [m]
    "N_int"  : 10_000,   # interior collocation points
    "N_wall" : 2_000,    # wall boundary points
    "N_io"   : 500,      # inlet / outlet points each
}

CASE_B_PARAMS = {
    "R"      : 0.003,    # cross-section radius [m]
    "R_c"    : 0.015,    # centreline radius of curvature [m]
    "theta"  : np.pi/2,  # subtended angle [rad]  -- 90 degrees
    "N_int"  : 15_000,
    "N_wall" : 3_000,
    "N_io"   : 500,
}

CASE_C_PARAMS = {
    "R_a"    : 0.002,    # parent artery radius [m]
    "L_a"    : 0.025,    # parent artery length [m]
    "R_s"    : 0.004,    # aneurysm sac radius [m]
    "neck_r" : 0.0015,   # aneurysm neck radius [m]  (half neck diameter)
    # Aneurysm centre offset from artery centreline
    # Sac centre is at y = R_a + R_s - neck_r (tangent attachment)
    "N_int"  : 25_000,
    "N_wall" : 5_000,
    "N_io"   : 500,
    # Enhanced sampling fractions near critical regions
    "frac_neck_enhanced" : 0.30,   # 30% of interior points near neck/dome
}


# =============================================================================
# LATIN HYPERCUBE SAMPLER  (Section 3.3.2)
# =============================================================================

def lhs_sample(n_samples: int, n_dims: int, seed: int = 42) -> np.ndarray:
    """
    Generate a Latin Hypercube Sample in [0,1]^n_dims.

    LHS ensures uniform coverage of the domain volume: every row of the
    [0,1]^n unit hypercube is sampled exactly once per subdivision.
    This produces better spatial coverage than pure random sampling for
    the same number of points, which is important for resolving steep
    velocity gradients near the aneurysm wall.

    Parameters
    ----------
    n_samples : int   -- number of points to generate
    n_dims    : int   -- spatial dimensionality (3 for our 3D domains)
    seed      : int   -- random seed for reproducibility

    Returns
    -------
    samples : (n_samples, n_dims) array in [0, 1]^n_dims
    """
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    return sampler.random(n=n_samples)


# =============================================================================
# CASE A: STRAIGHT CYLINDER
# =============================================================================

class StraightCylinder:
    """
    Generates point clouds for a right circular cylinder.

    Coordinate system:
        x-axis : axial direction (flow direction)
        y, z   : transverse directions
        Cylinder centreline lies along the x-axis.
        Inlet at x = 0, outlet at x = L.

    The exact Hagen-Poiseuille analytical solution on this geometry is:
        u_x(r) = u_max * (1 - r^2 / R^2)
        u_y = u_z = 0
        p(x)  = p_in - (delta_P / L) * x
    where r = sqrt(y^2 + z^2).
    This is the validation benchmark for Stage 2.
    """

    def __init__(self, params: dict):
        self.R = params["R"]
        self.L = params["L"]
        self.N_int  = params["N_int"]
        self.N_wall = params["N_wall"]
        self.N_io   = params["N_io"]

    # ------------------------------------------------------------------
    # Interior collocation points
    # ------------------------------------------------------------------
    def sample_interior(self, seed: int = 42) -> np.ndarray:
        """
        Sample N_int points uniformly inside the cylinder volume.

        Strategy: sample in cylindrical coordinates (r, phi, x) with
        r drawn from sqrt(U[0,1]) * R to ensure uniform area coverage
        (naive uniform r would oversample the axis relative to the wall).
        Then convert to Cartesian.
        """
        rng = np.random.default_rng(seed)

        # r: uniform in area => r = R * sqrt(uniform)
        r   = self.R * np.sqrt(rng.uniform(0, 1, self.N_int))
        phi = rng.uniform(0, 2 * np.pi, self.N_int)
        x   = rng.uniform(0, self.L, self.N_int)

        y = r * np.cos(phi)
        z = r * np.sin(phi)

        return np.column_stack([x, y, z])

    # ------------------------------------------------------------------
    # Wall boundary points + outward normals
    # ------------------------------------------------------------------
    def sample_wall(self, seed: int = 43) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample N_wall points on the lateral (curved) wall surface.

        The outward unit normal at a wall point (x, y, z) on a cylinder
        of radius R centred on the x-axis is:
            n = (0, y/R, z/R)
        because the wall is at r = R and the normal is radially outward.
        """
        rng = np.random.default_rng(seed)

        phi = rng.uniform(0, 2 * np.pi, self.N_wall)
        x   = rng.uniform(0, self.L, self.N_wall)

        y = self.R * np.cos(phi)
        z = self.R * np.sin(phi)

        points  = np.column_stack([x, y, z])
        normals = np.column_stack([
            np.zeros(self.N_wall),   # no normal component in axial direction
            np.cos(phi),             # y-component of radial unit vector
            np.sin(phi)              # z-component of radial unit vector
        ])

        return points, normals

    # ------------------------------------------------------------------
    # Inlet and outlet cross-section points
    # ------------------------------------------------------------------
    def sample_inlet(self, seed: int = 44) -> np.ndarray:
        """
        Sample N_io points on the inlet disc (x = 0).
        Used to enforce the parabolic inlet velocity profile (Eq 3.8).
        """
        rng = np.random.default_rng(seed)
        r   = self.R * np.sqrt(rng.uniform(0, 1, self.N_io))
        phi = rng.uniform(0, 2 * np.pi, self.N_io)

        x = np.zeros(self.N_io)
        y = r * np.cos(phi)
        z = r * np.sin(phi)

        return np.column_stack([x, y, z])

    def sample_outlet(self, seed: int = 45) -> np.ndarray:
        """
        Sample N_io points on the outlet disc (x = L).
        Used to enforce the zero-gauge pressure outlet condition (Eq 3.9).
        """
        pts = self.sample_inlet(seed=seed)
        pts[:, 0] = self.L   # shift x from 0 to L
        return pts

    # ------------------------------------------------------------------
    # Convenience: generate all point clouds at once
    # ------------------------------------------------------------------
    def generate_all(self, seed: int = 42) -> dict:
        interior          = self.sample_interior(seed=seed)
        wall, wall_n      = self.sample_wall(seed=seed + 1)
        inlet             = self.sample_inlet(seed=seed + 2)
        outlet            = self.sample_outlet(seed=seed + 3)

        data = {
            "interior"     : interior,
            "wall"         : wall,
            "wall_normals" : wall_n,
            "inlet"        : inlet,
            "outlet"       : outlet,
            "params"       : CASE_A_PARAMS,
            "case"         : "A_straight_cylinder",
        }

        print(f"[Case A] Interior  : {interior.shape}")
        print(f"[Case A] Wall      : {wall.shape}")
        print(f"[Case A] Inlet     : {inlet.shape}")
        print(f"[Case A] Outlet    : {outlet.shape}")
        return data

    # ------------------------------------------------------------------
    # Analytical solution (for validation in Stage 2)
    # ------------------------------------------------------------------
    def hagen_poiseuille(self,
                         points: np.ndarray,
                         u_max: float,
                         delta_P: float) -> dict:
        """
        Compute the exact Hagen-Poiseuille analytical solution at given points.

        u_x(x,y,z) = u_max * (1 - (y^2 + z^2) / R^2)
        u_y = u_z  = 0
        p(x,y,z)   = p_in - (delta_P / L) * x
                   = delta_P * (1 - x/L)   [choosing p_in = delta_P]

        The mean velocity is u_mean = u_max / 2  for parabolic profile.
        The wall shear stress is:
            tau_w = mu * du/dr|_{r=R} = mu * 2*u_max / R  (magnitude)

        Parameters
        ----------
        points  : (N, 3) array of evaluation points
        u_max   : centreline velocity [m/s]
        delta_P : inlet-outlet pressure difference [Pa]

        Returns
        -------
        dict with keys 'u', 'v', 'w', 'p' each of shape (N,)
        """
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r2 = y**2 + z**2

        u_x = u_max * (1.0 - r2 / self.R**2)
        u_x = np.clip(u_x, 0, None)   # enforce u >= 0 (inside cylinder only)

        return {
            "u" : u_x,
            "v" : np.zeros_like(u_x),
            "w" : np.zeros_like(u_x),
            "p" : delta_P * (1.0 - x / self.L),
        }


# =============================================================================
# CASE B: CURVED PIPE (TOROIDAL SEGMENT)
# =============================================================================

class CurvedPipe:
    """
    Generates point clouds for a toroidal pipe segment.

    The centreline of the pipe follows a circular arc of radius R_c
    in the x-z plane, centred at the origin, sweeping from theta=0
    to theta=theta_max.

    Coordinate system for a point in the pipe:
        Let s = arc angle along the centreline (0 to theta_max)
        Let (rho, alpha) = polar coordinates in the local cross-sectional
                           plane perpendicular to the centreline.

    Cartesian conversion:
        x = (R_c + rho*cos(alpha)) * cos(s)
        y = rho * sin(alpha)
        z = (R_c + rho*cos(alpha)) * sin(s)

    The Dean number De = Re * sqrt(R / R_c) characterises the strength
    of the centrifugal secondary flow (Dean vortices).
    """

    def __init__(self, params: dict):
        self.R       = params["R"]
        self.R_c     = params["R_c"]
        self.theta   = params["theta"]
        self.N_int   = params["N_int"]
        self.N_wall  = params["N_wall"]
        self.N_io    = params["N_io"]

    def _toroidal_to_cartesian(self,
                                rho: np.ndarray,
                                alpha: np.ndarray,
                                s: np.ndarray) -> np.ndarray:
        """
        Convert toroidal coordinates (rho, alpha, s) to Cartesian (x, y, z).

        rho   : radial distance from pipe centreline (0 to R)
        alpha : angle in cross-sectional plane (0 to 2pi)
        s     : arc angle along centreline (0 to theta_max)
        """
        # Distance from torus symmetry axis = R_c + rho*cos(alpha)
        R_eff = self.R_c + rho * np.cos(alpha)

        x = R_eff * np.cos(s)
        y = rho   * np.sin(alpha)
        z = R_eff * np.sin(s)

        return np.column_stack([x, y, z])

    def sample_interior(self, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)

        # rho sampled as sqrt(uniform) * R for uniform area distribution
        rho   = self.R * np.sqrt(rng.uniform(0, 1, self.N_int))
        alpha = rng.uniform(0, 2 * np.pi, self.N_int)
        s     = rng.uniform(0, self.theta, self.N_int)

        return self._toroidal_to_cartesian(rho, alpha, s)

    def sample_wall(self, seed: int = 43) -> tuple[np.ndarray, np.ndarray]:
        """
        Wall points lie at rho = R.
        The outward normal in toroidal coordinates at rho = R is
        simply the radial direction in the cross-sectional plane:
            n_local = (cos(alpha), sin(alpha)) in the (rho, alpha) plane.
        Converting to Cartesian requires the local frame vectors.
        """
        rng = np.random.default_rng(seed)

        alpha = rng.uniform(0, 2 * np.pi, self.N_wall)
        s     = rng.uniform(0, self.theta, self.N_wall)
        rho   = np.full(self.N_wall, self.R)

        points = self._toroidal_to_cartesian(rho, alpha, s)

        # Outward normal in Cartesian:
        # The cross-section is perpendicular to the tangent of the centreline.
        # Local radial direction (rho-hat) in Cartesian:
        #   e_rho = cos(alpha) * e_R  +  sin(alpha) * e_y
        # where e_R = (-sin(s), 0, cos(s))  [outward torus radial in x-z plane]
        # Actually the local frame:
        #   e_s    = (-sin(s), 0, cos(s))   tangent to centreline
        #   e_y    = (0, 1, 0)              out-of-plane
        #   e_rho  = cos(alpha)*(cos(s),0,sin(s)) + sin(alpha)*(0,1,0)
        #   This is the normal pointing from centreline towards pipe wall

        n_x = np.cos(alpha) * np.cos(s)
        n_y = np.sin(alpha) * np.ones_like(s)
        n_z = np.cos(alpha) * np.sin(s)

        normals = np.column_stack([n_x, n_y, n_z])

        return points, normals

    def sample_inlet(self, seed: int = 44) -> np.ndarray:
        """Inlet cross-section at s = 0."""
        rng  = np.random.default_rng(seed)
        rho  = self.R * np.sqrt(rng.uniform(0, 1, self.N_io))
        alpha = rng.uniform(0, 2 * np.pi, self.N_io)
        s     = np.zeros(self.N_io)
        return self._toroidal_to_cartesian(rho, alpha, s)

    def sample_outlet(self, seed: int = 45) -> np.ndarray:
        """Outlet cross-section at s = theta_max."""
        pts = self.sample_inlet(seed=seed)
        # Re-generate at s = theta_max
        rng  = np.random.default_rng(seed)
        rho  = self.R * np.sqrt(rng.uniform(0, 1, self.N_io))
        alpha = rng.uniform(0, 2 * np.pi, self.N_io)
        s     = np.full(self.N_io, self.theta)
        return self._toroidal_to_cartesian(rho, alpha, s)

    def generate_all(self, seed: int = 42) -> dict:
        interior          = self.sample_interior(seed=seed)
        wall, wall_n      = self.sample_wall(seed=seed + 1)
        inlet             = self.sample_inlet(seed=seed + 2)
        outlet            = self.sample_outlet(seed=seed + 3)

        data = {
            "interior"     : interior,
            "wall"         : wall,
            "wall_normals" : wall_n,
            "inlet"        : inlet,
            "outlet"       : outlet,
            "params"       : CASE_B_PARAMS,
            "case"         : "B_curved_pipe",
        }

        print(f"[Case B] Interior  : {interior.shape}")
        print(f"[Case B] Wall      : {wall.shape}")
        print(f"[Case B] Inlet     : {inlet.shape}")
        print(f"[Case B] Outlet    : {outlet.shape}")
        return data


# =============================================================================
# CASE C: SACCULAR ANEURYSM
# =============================================================================

class SaccularAneurysm:
    """
    Generates point clouds for a simplified sidewall saccular aneurysm.

    Geometry (following Khademi et al., 2024 conventions):
        - Straight parent artery: radius R_a, length L_a, along x-axis
        - Spherical aneurysm sac: radius R_s, attached laterally (in +y direction)
        - The sac centre is at (L_a/2, R_a + R_s - neck_r, 0)
          so the neck circle of radius neck_r is tangent to the artery wall.

    Two-region interior sampling strategy:
        - 70% of interior points: uniform random across entire bounding box
          (with rejection to stay inside domain)
        - 30% of interior points: concentrated near the aneurysm neck and
          dome, where velocity gradients are steepest (Section 3.3.3)

    This enhanced sampling near critical regions is essential for accurate
    WSS computation via automatic differentiation in Stage 6.
    """

    def __init__(self, params: dict):
        self.R_a    = params["R_a"]
        self.L_a    = params["L_a"]
        self.R_s    = params["R_s"]
        self.neck_r = params["neck_r"]
        self.N_int  = params["N_int"]
        self.N_wall = params["N_wall"]
        self.N_io   = params["N_io"]
        self.frac_enhanced = params["frac_neck_enhanced"]

        # Aneurysm sac centre (attached to +y side of artery)
        self.sac_centre = np.array([
            self.L_a / 2.0,                         # midpoint along artery
            self.R_a + self.R_s - self.neck_r,       # offset in y
            0.0
        ])

    # ------------------------------------------------------------------
    # Membership tests
    # ------------------------------------------------------------------
    def _in_artery(self, pts: np.ndarray) -> np.ndarray:
        """
        Boolean mask: True if point is inside the straight artery cylinder.
        Artery: x in [0, L_a], r_yz = sqrt(y^2 + z^2) <= R_a
        """
        r_yz = np.sqrt(pts[:, 1]**2 + pts[:, 2]**2)
        return (pts[:, 0] >= 0) & (pts[:, 0] <= self.L_a) & (r_yz <= self.R_a)

    def _in_sac(self, pts: np.ndarray) -> np.ndarray:
        """
        Boolean mask: True if point is inside the spherical aneurysm sac.
        Sac: distance from sac_centre <= R_s
        """
        diff = pts - self.sac_centre
        dist = np.sqrt(np.sum(diff**2, axis=1))
        return dist <= self.R_s

    def _in_domain(self, pts: np.ndarray) -> np.ndarray:
        """Point is in domain if it is in artery OR sac."""
        return self._in_artery(pts) | self._in_sac(pts)

    # ------------------------------------------------------------------
    # Interior collocation points with enhanced near-neck sampling
    # ------------------------------------------------------------------
    def sample_interior(self, seed: int = 42) -> np.ndarray:
        """
        Sample N_int interior points with enhanced density near the neck/dome.

        Two-phase sampling:
        Phase 1 (70%): Rejection sampling from bounding box, uniform random.
                       Accepts all points inside the combined domain.
        Phase 2 (30%): Concentrated near the neck region.
                       Samples from a sphere of radius 1.5*neck_r centred
                       at the neck, ensuring sufficient resolution there.
        """
        rng = np.random.default_rng(seed)

        N_uniform  = int(self.N_int * (1 - self.frac_enhanced))
        N_enhanced = self.N_int - N_uniform

        # -- Phase 1: uniform interior sampling --
        uniform_pts = []
        # Bounding box for the combined domain
        x_min, x_max = 0, self.L_a
        y_min = -max(self.R_a, self.sac_centre[1] + self.R_s)
        y_max =  self.sac_centre[1] + self.R_s
        z_min, z_max = -self.R_s, self.R_s

        while len(uniform_pts) < N_uniform:
            # Oversample, then reject points outside domain
            n_try = (N_uniform - len(uniform_pts)) * 4
            candidates = rng.uniform(
                [x_min, y_min, z_min],
                [x_max, y_max, z_max],
                size=(n_try, 3)
            )
            inside = self._in_domain(candidates)
            uniform_pts.extend(candidates[inside].tolist())

        uniform_pts = np.array(uniform_pts[:N_uniform])

        # -- Phase 2: enhanced near-neck sampling --
        # Neck is a circle of radius neck_r in the x=L_a/2 plane
        # at y = R_a (the artery wall), z = 0.
        # Sample from a sphere of 2*neck_r around the neck centre.
        neck_centre = np.array([self.L_a / 2, self.R_a, 0.0])
        enhanced_pts = []

        while len(enhanced_pts) < N_enhanced:
            n_try = (N_enhanced - len(enhanced_pts)) * 6
            # Random points in a sphere of radius 2*R_s centred near neck
            radius = 2.0 * self.R_s * rng.uniform(0, 1, n_try)**(1/3)
            phi    = rng.uniform(0, 2*np.pi, n_try)
            costh  = rng.uniform(-1, 1, n_try)
            sinth  = np.sqrt(1 - costh**2)

            dx = radius * sinth * np.cos(phi)
            dy = radius * sinth * np.sin(phi)
            dz = radius * costh

            candidates = neck_centre + np.column_stack([dx, dy, dz])
            inside = self._in_domain(candidates)
            enhanced_pts.extend(candidates[inside].tolist())

        enhanced_pts = np.array(enhanced_pts[:N_enhanced])

        all_interior = np.vstack([uniform_pts, enhanced_pts])
        # Shuffle so the two phases are interleaved (better mini-batch diversity)
        rng.shuffle(all_interior)

        return all_interior

    # ------------------------------------------------------------------
    # Wall boundary points
    # ------------------------------------------------------------------
    def sample_wall(self, seed: int = 43) -> tuple[np.ndarray, np.ndarray]:
        """
        Wall consists of two surfaces:
            1. Artery lateral wall (cylinder, excluding neck opening)
            2. Aneurysm sac wall  (sphere, excluding neck opening)

        The neck opening is a circular hole of radius neck_r at x = L_a/2,
        y = R_a, z = 0. Points near the opening are excluded from both
        surfaces to avoid double-counting.

        Wall normals:
            Artery : radially outward from artery axis = (0, y/R_a, z/R_a)
                     (ignoring axial component, valid for lateral wall)
            Sac    : radially outward from sac centre
        """
        rng = np.random.default_rng(seed)

        N_artery = self.N_wall // 2
        N_sac    = self.N_wall - N_artery

        # ---- Artery wall ----
        art_pts, art_normals = [], []
        while len(art_pts) < N_artery:
            n_try = (N_artery - len(art_pts)) * 3
            phi   = rng.uniform(0, 2*np.pi, n_try)
            x     = rng.uniform(0, self.L_a, n_try)

            y = self.R_a * np.cos(phi)
            z = self.R_a * np.sin(phi)

            pts_try = np.column_stack([x, y, z])

            # Exclude the neck opening region
            dist_to_neck = np.sqrt(
                (x - self.L_a/2)**2 +
                (y - self.R_a)**2 +
                z**2
            )
            valid = dist_to_neck > self.neck_r * 0.8   # small clearance buffer

            for i in range(n_try):
                if valid[i] and len(art_pts) < N_artery:
                    art_pts.append(pts_try[i])
                    art_normals.append([0.0, np.cos(phi[i]), np.sin(phi[i])])

        art_pts     = np.array(art_pts[:N_artery])
        art_normals = np.array(art_normals[:N_artery])

        # ---- Sac wall ----
        sac_pts, sac_normals = [], []
        while len(sac_pts) < N_sac:
            n_try = N_sac * 4
            # Sample uniformly on sphere surface
            phi_s  = rng.uniform(0, 2*np.pi, n_try)
            costh  = rng.uniform(-1, 1, n_try)
            sinth  = np.sqrt(1 - costh**2)

            x_s = self.sac_centre[0] + self.R_s * sinth * np.cos(phi_s)
            y_s = self.sac_centre[1] + self.R_s * sinth * np.sin(phi_s)
            z_s = self.sac_centre[2] + self.R_s * costh

            pts_try = np.column_stack([x_s, y_s, z_s])

            # Exclude points inside the artery (these are in the neck region)
            not_in_artery = ~self._in_artery(pts_try)

            for i in range(n_try):
                if not_in_artery[i] and len(sac_pts) < N_sac:
                    sac_pts.append(pts_try[i])
                    # Outward normal: from sac centre towards the surface point
                    n = pts_try[i] - self.sac_centre
                    sac_normals.append(n / np.linalg.norm(n))

        sac_pts     = np.array(sac_pts[:N_sac])
        sac_normals = np.array(sac_normals[:N_sac])

        all_wall    = np.vstack([art_pts,     sac_pts])
        all_normals = np.vstack([art_normals, sac_normals])

        return all_wall, all_normals

    def sample_inlet(self, seed: int = 44) -> np.ndarray:
        """Inlet disc at x = 0, radius R_a."""
        rng = np.random.default_rng(seed)
        r   = self.R_a * np.sqrt(rng.uniform(0, 1, self.N_io))
        phi = rng.uniform(0, 2*np.pi, self.N_io)
        x   = np.zeros(self.N_io)
        y   = r * np.cos(phi)
        z   = r * np.sin(phi)
        return np.column_stack([x, y, z])

    def sample_outlet(self, seed: int = 45) -> np.ndarray:
        """Outlet disc at x = L_a, radius R_a."""
        pts = self.sample_inlet(seed=seed)
        pts[:, 0] = self.L_a
        return pts

    def generate_all(self, seed: int = 42) -> dict:
        interior          = self.sample_interior(seed=seed)
        wall, wall_n      = self.sample_wall(seed=seed + 1)
        inlet             = self.sample_inlet(seed=seed + 2)
        outlet            = self.sample_outlet(seed=seed + 3)

        data = {
            "interior"     : interior,
            "wall"         : wall,
            "wall_normals" : wall_n,
            "inlet"        : inlet,
            "outlet"       : outlet,
            "params"       : CASE_C_PARAMS,
            "case"         : "C_saccular_aneurysm",
            "sac_centre"   : self.sac_centre,
        }

        print(f"[Case C] Interior  : {interior.shape}")
        print(f"[Case C] Wall      : {wall.shape}")
        print(f"[Case C] Inlet     : {inlet.shape}")
        print(f"[Case C] Outlet    : {outlet.shape}")
        return data


# =============================================================================
# VISUALISATION  (matplotlib fallback, pyvista preferred)
# =============================================================================

def visualise_case(data: dict, max_pts: int = 3000):
    """
    Plot the point cloud geometry using matplotlib (3D scatter).
    Shows interior, wall, inlet and outlet points in different colours.

    max_pts : subsample for plotting speed (does not affect saved data)
    """
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    def subsample(arr, n):
        if len(arr) > n:
            idx = np.random.choice(len(arr), n, replace=False)
            return arr[idx]
        return arr

    interior = subsample(data["interior"], max_pts)
    wall      = subsample(data["wall"],     max_pts // 2)
    inlet     = data["inlet"]
    outlet    = data["outlet"]

    ax.scatter(*interior.T, s=0.5, c="steelblue",  alpha=0.3, label="Interior")
    ax.scatter(*wall.T,     s=1.0, c="firebrick",  alpha=0.6, label="Wall")
    ax.scatter(*inlet.T,    s=3.0, c="limegreen",  alpha=0.9, label="Inlet")
    ax.scatter(*outlet.T,   s=3.0, c="darkorange", alpha=0.9, label="Outlet")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"Point Cloud: {data['case']}")
    ax.legend(loc="upper left", markerscale=5)

    plt.tight_layout()
    fname = f"geometry_{data['case']}.png"
    plt.savefig(fname, dpi=150)
    print(f"[Visualisation] Saved: {fname}")
    plt.show()


# =============================================================================
# SAVE / LOAD UTILITIES
# =============================================================================

def save_geometry(data: dict, directory: str = "geometry_data"):
    """
    Save all point cloud arrays as compressed numpy files.
    Each case saves to its own subdirectory for clean organisation.

    Files:
        interior.npy, wall.npy, wall_normals.npy, inlet.npy, outlet.npy
    """
    case_dir = os.path.join(directory, data["case"])
    os.makedirs(case_dir, exist_ok=True)

    for key in ["interior", "wall", "wall_normals", "inlet", "outlet"]:
        np.save(os.path.join(case_dir, f"{key}.npy"), data[key])

    # Save params as a text file for reference
    with open(os.path.join(case_dir, "params.txt"), "w") as f:
        for k, v in data["params"].items():
            f.write(f"{k} = {v}\n")
        if "sac_centre" in data:
            f.write(f"sac_centre = {data['sac_centre']}\n")

    print(f"[Save] Geometry saved to: {case_dir}/")


def load_geometry(case_name: str, directory: str = "geometry_data") -> dict:
    """
    Load previously saved point cloud arrays.

    Parameters
    ----------
    case_name : one of 'A_straight_cylinder', 'B_curved_pipe',
                        'C_saccular_aneurysm'
    """
    case_dir = os.path.join(directory, case_name)
    data = {"case": case_name}

    for key in ["interior", "wall", "wall_normals", "inlet", "outlet"]:
        fpath = os.path.join(case_dir, f"{key}.npy")
        data[key] = np.load(fpath)

    print(f"[Load] Geometry loaded from: {case_dir}/")
    return data


# =============================================================================
# VALIDATION CHECKS
# =============================================================================

def validate_geometry(data: dict):
    """
    Run basic sanity checks on the generated point clouds.

    Checks:
        1. No NaN or Inf values in any array
        2. Wall normals are unit vectors (|n| = 1)
        3. For Case A: all interior points satisfy r < R
        4. For Case C: all interior points are inside the domain
    """
    print(f"\n[Validation] Checking geometry: {data['case']}")
    passed = True

    for key in ["interior", "wall", "wall_normals", "inlet", "outlet"]:
        arr = data[key]
        if not np.all(np.isfinite(arr)):
            print(f"  FAIL: {key} contains NaN or Inf")
            passed = False
        else:
            print(f"  OK  : {key} -- shape {arr.shape}, range "
                  f"[{arr.min():.4f}, {arr.max():.4f}]")

    # Check wall normals are unit vectors
    norms = np.linalg.norm(data["wall_normals"], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-6):
        print(f"  FAIL: wall_normals are not unit vectors "
              f"(mean |n| = {norms.mean():.6f})")
        passed = False
    else:
        print(f"  OK  : wall_normals are unit vectors "
              f"(mean |n| = {norms.mean():.6f})")

    # Case A specific: all interior r < R
    if data["case"] == "A_straight_cylinder":
        R   = data["params"]["R"]
        pts = data["interior"]
        r   = np.sqrt(pts[:, 1]**2 + pts[:, 2]**2)
        if np.any(r >= R):
            pct = 100 * np.mean(r >= R)
            print(f"  FAIL: {pct:.2f}% of interior points outside cylinder")
            passed = False
        else:
            print(f"  OK  : all interior points inside cylinder "
                  f"(max r = {r.max():.6f}, R = {R})")

    if passed:
        print("[Validation] ALL CHECKS PASSED\n")
    else:
        print("[Validation] SOME CHECKS FAILED -- review geometry\n")

    return passed


# =============================================================================
# MAIN: GENERATE ALL THREE CASES
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("STAGE 1: GEOMETRY GENERATION")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Case A: Straight Cylinder
    # ----------------------------------------------------------------
    print("\n--- Case A: Straight Cylinder ---")
    cyl   = StraightCylinder(CASE_A_PARAMS)
    data_A = cyl.generate_all(seed=42)
    validate_geometry(data_A)
    save_geometry(data_A)
    visualise_case(data_A)

    # Quick check: does the analytical solution look correct?
    # Use Re = 250 at mu_inf = 0.00345 Pa.s, rho = 1060 kg/m3
    MU_INF = 0.00345
    RHO    = 1060.0
    R_A    = CASE_A_PARAMS["R"]
    # u_mean = Re * mu / (rho * R)
    u_mean = 250 * MU_INF / (RHO * R_A)
    u_max  = 2 * u_mean                  # parabolic profile: u_max = 2*u_mean
    delta_P = 4 * MU_INF * u_max * CASE_A_PARAMS["L"] / R_A**2

    analytical = cyl.hagen_poiseuille(data_A["interior"], u_max, delta_P)
    print(f"  Analytical check: u_max = {u_max:.5f} m/s")
    print(f"  Peak u on interior: {analytical['u'].max():.5f} m/s")
    print(f"  Expected delta_P   = {delta_P:.4f} Pa")
    print(f"  WSS_analytical     = {MU_INF * 2 * u_max / R_A:.4f} Pa")

    # ----------------------------------------------------------------
    # Case B: Curved Pipe
    # ----------------------------------------------------------------
    print("\n--- Case B: Curved Pipe ---")
    pipe   = CurvedPipe(CASE_B_PARAMS)
    data_B = pipe.generate_all(seed=42)
    validate_geometry(data_B)
    save_geometry(data_B)
    visualise_case(data_B)

    # Report Dean number at Re=250
    R_p  = CASE_B_PARAMS["R"]
    R_c  = CASE_B_PARAMS["R_c"]
    Re   = 250
    De   = Re * np.sqrt(R_p / R_c)
    print(f"  Dean number at Re=250: De = {De:.1f}  (target: 100-400)")

    # ----------------------------------------------------------------
    # Case C: Saccular Aneurysm
    # ----------------------------------------------------------------
    print("\n--- Case C: Saccular Aneurysm ---")
    aneurysm = SaccularAneurysm(CASE_C_PARAMS)
    data_C   = aneurysm.generate_all(seed=42)
    validate_geometry(data_C)
    save_geometry(data_C)
    visualise_case(data_C)

    print("\n" + "=" * 60)
    print("STAGE 1 COMPLETE")
    print("All geometry data saved to ./geometry_data/")
    print("Next step: run stage2_pinn_caseA.py")
    print("=" * 60)
