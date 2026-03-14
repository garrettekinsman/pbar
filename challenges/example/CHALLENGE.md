# Lattice Cube Compression Challenge

*Date: 2026-03-13*
*Version: 1.0*

---

## Executive Summary

Design a lattice structure within a fixed cubic envelope that maximizes **specific compressive strength** (ultimate compressive load divided by mass). The structure must be manufacturable by FDM 3D printing in PETG without supports. This challenge benchmarks PBAR (population-based annealed research) against greedy optimization for structural topology problems.

The problem is deliberately constrained to be:
- **Computationally tractable** — single FEA evaluation in under 60 seconds
- **Physically testable** — print and crush on a bench-top load frame
- **LLM-editable** — parameterized DSL that a language model can meaningfully modify
- **[PI] enough** — known to have local optima that greedy selection gets trapped in

---

## 1. Geometry Constraints

### 1.1 Envelope

| Parameter | Value | Notes |
|-----------|-------|-------|
| Outer dimensions | 50 × 50 × 50 mm | Cube envelope, axis-aligned |
| Top plate | 50 × 50 × 1.5 mm | Solid PETG, load distribution surface |
| Bottom plate | 50 × 50 × 1.5 mm | Solid PETG, fixed boundary surface |
| Lattice zone | 50 × 50 × 47 mm | Interior volume between plates |

The top and bottom plates are **mandatory and fixed** — they are not part of the optimization. They exist to provide uniform load introduction and boundary conditions, exactly as a physical test fixture would require. All lattice geometry must be fully bonded to both plates (no gaps).

### 1.2 Nozzle & Feature Size Constraints

**Fixed nozzle:** 0.4 mm diameter (standard)

| Parameter | Min | Max | Rationale |
|-----------|-----|-----|-----------|
| Nozzle diameter | 0.4 mm | 0.4 mm | Fixed — standard nozzle, no changes allowed |
| Strut diameter | 1.2 mm | 8.0 mm | Min = 3× nozzle width. Below this, extrusion is unreliable. |
| Wall thickness | 0.8 mm | 8.0 mm | Min = 2× nozzle width. Thin walls delaminate under load. |
| Node fillet radius | 0.4 mm | — | Minimum at strut junctions to avoid stress singularities in FEA. |
| Minimum span (unsupported) | — | 15 mm | Horizontal unsupported bridges. Beyond 15 mm, sag is excessive. |
| Line width | 0.4 mm | 0.48 mm | Constrained by nozzle (100-120% of nozzle diameter) |

### 1.3 Printability Constraints (FDM)

| Constraint | Value | Rationale |
|------------|-------|-----------|
| Maximum overhang angle | 50° from vertical | Conservative for PETG. 45° is textbook; 50° accounts for cooling. |
| Minimum layer height | 0.2 mm | Standard for 0.4 mm nozzle |
| Maximum layer height | 0.28 mm | Limited by nozzle diameter |
| Print orientation | Z-up (as designed) | Top/bottom plates print as first/last layers |
| Support material | **None permitted** | The lattice must be self-supporting. This is a hard constraint. |
| Minimum angle from horizontal for downward-facing surfaces | 40° | Below this, surface quality and bonding degrade catastrophically |

### 1.4 Connectivity Requirements

- **No floating islands.** Every element of the lattice must be reachable from every other element via a continuous material path.
- **No trapped volumes.** The lattice must be fully drainable (no enclosed internal cavities). This is both a printability requirement (no trapped support material) and a mass verification requirement (enclosed voids cannot be weighed).
- **Plate bonding.** Every strut or wall must terminate at either the top plate, the bottom plate, or another strut/wall. Free-floating endpoints are forbidden.
- **Minimum connectivity.** Each interior node must connect to at least 3 struts (prevents kinematic mechanisms under load).

---

## 2. Material Specification: PLA

All properties are **conservative lower-bound values** for generic PLA filament, printed with standard FDM parameters (0.4 mm nozzle, 210°C hotend, 60°C bed, 0.2 mm layer height).

### 2.1 Bulk Material Properties

| Property | Symbol | Value | Units | Source/Notes |
|----------|--------|-------|-------|--------------|
| Density | ρ | 1,240 | kg/m³ | Solid filament density |
| Tensile yield strength | σ_y | 50 | MPa | Conservative; datasheets range 50–65 |
| Compressive yield strength | σ_cy | 60 | MPa | PLA is stronger in compression |
| Elastic modulus (tension) | E | 2,500 | MPa | Conservative; datasheets range 2,500–3,500 |
| Elastic modulus (compression) | E_c | 2,500 | MPa | Assumed equal to tensile for simplicity |
| Poisson's ratio | ν | 0.36 | — | Typical for PLA |
| Elongation at break | ε_b | 3 | % | PLA is brittle; FDM parts even more so |

### 2.2 FDM Derating Factors

Printed PLA is **not** isotropic. Layer adhesion is the weak link. Apply these derating factors to bulk properties:

| Direction | Factor | Effective σ_cy | Effective E_c | Rationale |
|-----------|--------|----------------|---------------|-----------|
| In-plane (XY) | 1.0 | 60 MPa | 2,500 MPa | Full material strength within layers |
| Inter-layer (Z) | 0.55 | 33 MPa | 1,375 MPa | Layer adhesion; PLA bonds worse than PETG |
| 45° to layers | 0.75 | 45 MPa | 1,875 MPa | Interpolated; shear + normal component |

**Implementation in FEA:** Use **transversely isotropic** material model with Z as the weak axis. If the FEA solver does not support anisotropy, use the **Z-direction (worst-case) values everywhere** — this is conservative and avoids unconservative predictions for struts aligned with the build direction.

### 2.3 Effective Print Density

Due to infill and extrusion tolerances, printed lattice density differs from solid material density:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Effective wall density | 1,180 kg/m³ | ~95% of solid, accounts for inter-bead voids |
| Solid plate density | 1,220 kg/m³ | ~98.5% of solid (100% infill plates) |

Use **ρ_eff = 1,180 kg/m³** for lattice mass calculations and **ρ_plate = 1,220 kg/m³** for the top/bottom plates.

---

## 3. Load Cases

### 3.1 Primary: Uniaxial Compression (Z-direction)

This is the only load case used for scoring.

| Parameter | Value |
|-----------|-------|
| Load type | Distributed pressure on top plate (uniform) |
| Load direction | -Z (downward) |
| Load magnitude | **Variable** — we find the failure load |
| Bottom BC | Fixed (all DOF constrained on bottom plate outer face) |
| Top BC | Pressure applied to top plate outer face; lateral DOF free |

**Boundary condition detail:** The bottom plate is fully fixed (encastre — u_x = u_y = u_z = 0, no rotation). This models a rigid, high-friction test platen. The top plate receives a uniform pressure with no lateral constraint — this models a test platen that can slide laterally but not tilt (spherical seat in physical testing).

### 3.2 Failure Criteria

The structure "fails" at the **lowest** of:

1. **Yielding:** Any element reaches the compressive yield stress (σ_cy or derated value per §2.2). Check von Mises stress against yield.
2. **Buckling:** Linear buckling eigenvalue analysis. The critical buckling load is the first positive eigenvalue times the applied reference load.
3. **Excessive deflection:** Top plate deflects more than 2.5 mm (5% of cube height). This is a serviceability limit.

The **predicted failure load** F_fail is:

```
F_fail = min(F_yield, F_buckling, F_deflection_limit)
```

Where:
- F_yield = load at which max von Mises stress first reaches σ_cy (per §2.2 derating)
- F_buckling = first linear buckling eigenvalue × reference load
- F_deflection_limit = load at which max top-plate deflection reaches 2.5 mm

### 3.3 Safety Factor

Apply a safety factor of **1.5** to the predicted failure load for the "usable" load:

```
F_usable = F_fail / 1.5
```

This is not used in scoring directly — it exists for correlation with physical testing. A structure that scores well on F_fail but has a buckling mode just above the yield load is fragile. The scoring formula (§5) addresses this through the multi-mode penalty.

### 3.4 Secondary Loads (Not Scored, But Reported)

For information and future use, also compute:

- **Off-axis compression:** 10° tilt from Z-axis (models misalignment in physical test)
- **Shear:** 500 N lateral force on top plate while under 50% of F_fail compression

These are reported but do not affect the primary score. They provide data for future challenge revisions.

---

## 4. Objective Function

### 4.1 What We Optimize

**Maximize specific compressive strength:**

```
S = F_fail / m_lattice
```

Where:
- F_fail = predicted failure load [N] per §3.2
- m_lattice = mass of lattice structure only [kg] (excluding top and bottom plates)

Units: N/kg (equivalent to m/s², i.e., specific force in acceleration units).

**Why this metric:**
- Minimizing mass at fixed strength is ill-posed (degenerate solutions at zero mass)
- Maximizing strength at fixed mass requires fixing volume fraction a priori, which constrains the search space unnecessarily
- Specific strength (F/m) naturally balances both: adding mass only helps if it adds proportionally more strength

### 4.2 Volume Fraction Bounds

To prevent degenerate solutions:

| Parameter | Min | Max | Rationale |
|-----------|-----|-----|-----------|
| Volume fraction (lattice zone) | 5% | 40% | Below 5%: unprintable. Above 40%: it's not a lattice, it's a block. |

Volume fraction = V_material / V_envelope, where V_envelope = 50 × 50 × 47 mm³ = 117,500 mm³.

| | Volume (mm³) | Mass (g) |
|---|---|---|
| 5% lattice | 5,875 | 7.1 |
| 40% lattice | 47,000 | 56.9 |
| Top + bottom plates | 7,500 | 9.4 |
| **Total (5% lattice)** | 13,375 | **16.5** |
| **Total (40% lattice)** | 54,500 | **66.3** |

---

## 5. Scoring Formula

### 5.1 Primary Score (Lower Is Better)

For PBAR compatibility, the score must be **lower is better** (same convention as val_bpb).

```python
def compute_score(F_fail_N, mass_lattice_kg, volume_fraction,
                  min_feature_mm, max_overhang_deg, is_connected, 
                  has_trapped_voids, F_buckling_N, F_yield_N):
    """
    Compute PBAR-compatible score. Lower is better.
    
    Returns:
        score: float, lower is better. Range [0, +inf).
               Feasible designs score roughly 0.001–0.1.
               Infeasible designs score 1000+.
    """
    
    # === FEASIBILITY CHECKS (hard constraints) ===
    penalty = 0.0
    
    if not is_connected:
        penalty += 1000.0  # Floating islands
    
    if has_trapped_voids:
        penalty += 500.0   # Non-drainable cavities
    
    if volume_fraction < 0.05:
        penalty += 1000.0  # Below minimum VF
    elif volume_fraction > 0.40:
        penalty += 1000.0  # Above maximum VF
    
    if min_feature_mm < 1.2:
        # Graduated penalty for slightly-too-thin features
        penalty += 100.0 * (1.2 - min_feature_mm) / 1.2
    
    if max_overhang_deg < 40.0:
        # Overhang from horizontal (complement of angle from vertical)
        # Lower angle from horizontal = worse overhang
        penalty += 100.0 * (40.0 - max_overhang_deg) / 40.0
    
    if F_fail_N <= 0:
        return 10000.0 + penalty  # Structure has no load capacity
    
    if mass_lattice_kg <= 0:
        return 10000.0 + penalty  # Degenerate (no material)
    
    # === PRIMARY METRIC ===
    # Specific strength: F/m [N/kg] — higher is better
    specific_strength = F_fail_N / mass_lattice_kg
    
    # Invert to get "lower is better" score
    # Reference specific strength: 50,000 N/kg (order of magnitude for PETG lattice)
    # This normalization keeps scores in a reasonable range
    score = 50000.0 / specific_strength
    
    # === ROBUSTNESS BONUS ===
    # Reward structures where buckling and yielding fail at similar loads
    # (indicates well-balanced design — no single weak failure mode)
    if F_buckling_N > 0 and F_yield_N > 0:
        mode_ratio = min(F_buckling_N, F_yield_N) / max(F_buckling_N, F_yield_N)
        # mode_ratio = 1.0 means both modes fail together (ideal)
        # mode_ratio << 1.0 means one mode dominates (fragile)
        robustness_bonus = 0.1 * (1.0 - mode_ratio)  # 0 to 0.1 penalty
        score += robustness_bonus
    
    # === TOTAL ===
    score += penalty
    
    return score
```

### 5.2 Score Interpretation

| Score Range | Interpretation |
|-------------|---------------|
| < 0.5 | Excellent — specific strength > 100 kN/kg |
| 0.5 – 1.0 | Good — competitive with known lattice topologies |
| 1.0 – 2.0 | Mediocre — room for significant improvement |
| 2.0 – 10.0 | Poor — far from optimal |
| > 1000 | Infeasible — constraint violation |

### 5.3 Tie-Breaking

If two designs have equal score (within 0.001), prefer:
1. Lower volume fraction (lighter is better when equally strong)
2. Higher buckling-to-yield ratio (more robust)
3. Simpler topology (fewer nodes/struts — Occam's razor for structures)

---

## 6. Known Baselines

These are established lattice topologies from the literature, with expected performance ranges for PETG at ~15–20% volume fraction. They serve as calibration targets and "par" for the optimization.

### 6.1 Baseline Topologies

| Topology | Type | Volume Fraction | Expected F_fail (kN) | Expected S (kN/kg) | Strengths | Weaknesses |
|----------|------|-----------------|----------------------|---------------------|-----------|------------|
| **Octet truss** | Strut-based | 15% | 5–8 | 40–55 | High stiffness, stretch-dominated | Hard to print (45° overhangs), stress concentrations at nodes |
| **BCC (body-centered cubic)** | Strut-based | 15% | 3–5 | 25–40 | Easy to print (all struts >45°), good isotropy | Bending-dominated, lower specific strength |
| **Gyroid (TPMS)** | Surface-based | 15% | 6–10 | 45–65 | Excellent printability (no overhangs), smooth stress distribution | Harder to parameterize, computationally expensive to mesh |
| **Diamond** | Strut-based | 15% | 4–7 | 35–50 | Good balance of stiffness and printability | Moderate performance overall |
| **Schwarz-P (TPMS)** | Surface-based | 15% | 5–8 | 40–60 | Very printable, large open channels | Slightly lower than gyroid |
| **Simple cubic** | Strut-based | 15% | 2–4 | 15–30 | Trivial to parameterize | Bending-dominated, poor efficiency |
| **Graded gyroid** | Surface-based | Variable (5–25%) | 8–12 | 55–75 | Material where it matters | Complex parameterization |

### 6.2 Baseline Targets

For the optimization benchmark:

| Tier | Target Score | Target S (kN/kg) | Meaning |
|------|-------------|-------------------|---------|
| **Par** | 1.0 | ~50 | Matches simple BCC/diamond lattice |
| **Good** | 0.7 | ~70 | Matches optimized TPMS (gyroid) |
| **Excellent** | 0.5 | ~100 | Exceeds published results for PETG lattices |
| **Breakthrough** | < 0.4 | > 125 | Novel topology — warrants physical testing |

### 6.3 Literature References

- Gibson & Ashby, *Cellular Solids* (1997) — foundational lattice mechanics
- Deshpande, Ashby, & Fleck, "Effective properties of the octet-truss lattice material" (2001) — stretch vs. bending dominated
- Al-Ketan & Abu Al-Rus, "Multifunctional TPMS lattices" (2019) — gyroid/Schwarz-P FEA data
- Maconachie et al., "SLM lattice structures: Properties, applications" (2019) — extensive experimental data
- Zhang et al., "FDM-printed PETG lattice compression testing" (2023) — directly relevant experimental data

**Note:** Most literature uses SLM (metal) or SLA (resin) lattices. FDM PETG data is sparser. The baselines above are estimated by derating metal lattice data using the PETG material properties from §2. Physical testing will calibrate these estimates.

---

## 7. DSL Representation

### 7.1 Requirements

The lattice parameterization must be:
1. **LLM-editable** — a language model must be able to read, understand, and meaningfully modify it
2. **Deterministic** — same DSL → same mesh, always
3. **Constraint-expressible** — easy to verify feature sizes, overhangs, connectivity
4. **Compact** — fits in a single file under 500 lines

### 7.2 Chosen Representation: Hybrid Node-Strut + TPMS

Support two lattice families in a single DSL, plus free-form combinations:

```python
# lattice_cube.dsl — Example lattice definition

LATTICE_CONFIG = {
    "envelope": [50.0, 50.0, 50.0],  # mm
    "plate_thickness": 1.5,            # mm, top and bottom
    "representation": "strut",         # "strut" | "tpms" | "hybrid"
}

# === STRUT-BASED LATTICE ===
# Define by unit cell type and parameters

STRUT_LATTICE = {
    "unit_cell": "bcc",              # "bcc" | "octet" | "diamond" | "cubic" | "custom"
    "cell_size": [10.0, 10.0, 10.0], # mm — tiling period in X, Y, Z
    "strut_diameter": 2.0,           # mm — uniform diameter
    "node_fillet": 0.5,              # mm — fillet at junctions
    
    # Grading: vary strut diameter spatially
    "grading": {
        "enabled": False,
        "function": "linear_z",       # "uniform" | "linear_z" | "radial" | "custom"
        "min_diameter": 1.5,          # mm at thinnest
        "max_diameter": 3.0,          # mm at thickest
    },
    
    # Custom unit cell (only if unit_cell == "custom")
    "custom_nodes": [
        # [x, y, z] in normalized coordinates [0, 1]
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0],
    ],
    "custom_edges": [
        # [node_i, node_j]
        [0, 1],
        [1, 2],
    ],
}

# === TPMS-BASED LATTICE ===
# Define by implicit surface type and parameters

TPMS_LATTICE = {
    "surface_type": "gyroid",        # "gyroid" | "schwarz_p" | "schwarz_d" | "lidinoid"
    "cell_size": [10.0, 10.0, 10.0], # mm — period in X, Y, Z
    "wall_thickness": 1.5,           # mm — shell thickness
    "level_set": 0.0,                # Offset from zero-level surface
    
    # Grading
    "grading": {
        "enabled": False,
        "function": "linear_z",
        "min_thickness": 1.0,
        "max_thickness": 2.5,
    },
}

# === HYBRID ===
# Combine strut skeleton with TPMS infill in different zones

HYBRID_LATTICE = {
    "zones": [
        {
            "type": "strut",
            "region": {"z_min": 0.0, "z_max": 15.0},
            "params": {"unit_cell": "octet", "strut_diameter": 2.5},
        },
        {
            "type": "tpms",
            "region": {"z_min": 15.0, "z_max": 35.0},
            "params": {"surface_type": "gyroid", "wall_thickness": 1.2},
        },
        {
            "type": "strut",
            "region": {"z_min": 35.0, "z_max": 47.0},
            "params": {"unit_cell": "octet", "strut_diameter": 2.5},
        },
    ],
    "zone_transition_blend": 2.0,  # mm — blend region between zones
}
```

### 7.3 What the LLM Can Edit

| Parameter | Range | Type | LLM Difficulty |
|-----------|-------|------|----------------|
| `unit_cell` type | Enum | Discrete choice | Easy |
| `cell_size` | [5, 25] mm | Continuous (3D) | Easy |
| `strut_diameter` | [1.2, 8.0] mm | Continuous | Easy |
| `wall_thickness` | [0.8, 8.0] mm | Continuous | Easy |
| `grading.function` | Enum | Discrete choice | Easy |
| `grading` min/max | Constrained | Continuous | Medium |
| `custom_nodes` | [0,1]³ | Continuous (3D × N) | Hard |
| `custom_edges` | Connectivity | Discrete graph | Hard |
| `zones` (hybrid) | Multi-region | Structured | Medium |
| `level_set` offset | [-1, 1] | Continuous | Easy |

**Observation:** The DSL is structured so that simple optimizations (tuning diameters, cell sizes, choosing between BCC/octet/gyroid) are easy for an LLM, while more creative optimizations (custom unit cells, hybrid zones, graded designs) require deeper reasoning. This naturally tests whether PBAR's exploration enables discovering the harder, more creative solutions that greedy would miss.

### 7.4 DSL Validation Rules

Before evaluation, the DSL is checked for:

1. All feature sizes within bounds (§1.2)
2. All continuous parameters are finite, positive where required
3. Custom node coordinates in [0, 1]³
4. Custom edges reference valid node indices
5. No duplicate edges
6. Zone regions fully tile the lattice zone (no gaps, no overlaps beyond blend)
7. File parses without errors

Invalid DSLs score **10000.0** (same as §5.1 infeasible penalty).

---

## 8. Evaluation Pipeline

### 8.1 Pipeline Overview

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────┐
│  DSL File   │────▶│  Mesh Gen    │────▶│  Validation  │────▶│    FEA    │
│  (*.dsl)    │     │  (trimesh)   │     │  (checks)    │     │ (CalculiX)│
└─────────────┘     └──────────────┘     └──────────────┘     └───────────┘
                                                                     │
     ┌───────────────────────────────────────────────────────────────┘
     ▼
┌──────────────┐     ┌──────────────┐     ┌───────────┐
│  Post-Proc   │────▶│  Score Calc  │────▶│  Result   │
│  (results)   │     │  (§5.1)      │     │  (float)  │
└──────────────┘     └──────────────┘     └───────────┘
```

### 8.2 Stage Details

#### Stage 1: DSL → Mesh

**Tool:** Custom Python script using `trimesh` and `numpy`

```python
def dsl_to_mesh(dsl_path: str) -> trimesh.Trimesh:
    """Convert DSL definition to watertight triangle mesh."""
    config = load_dsl(dsl_path)
    
    if config["representation"] == "strut":
        mesh = generate_strut_lattice(config)
    elif config["representation"] == "tpms":
        mesh = generate_tpms_lattice(config)
    elif config["representation"] == "hybrid":
        mesh = generate_hybrid_lattice(config)
    
    # Add top and bottom plates
    mesh = add_plates(mesh, config)
    
    # Ensure watertight
    assert mesh.is_watertight, "Mesh must be watertight for FEA"
    
    return mesh
```

**Strut generation:** Sweep circles along edges, union with filleted nodes. Use trimesh boolean operations or Blender boolean modifier for robustness.

**TPMS generation:** Evaluate implicit function on a voxel grid (resolution: 0.5 mm), extract isosurface via marching cubes, offset to create shell of specified thickness.

**Target mesh size:** 50,000–200,000 tetrahedra after volumetric meshing (sufficient for convergence, feasible for desktop FEA).

#### Stage 2: Mesh Validation

Before FEA, verify:

| Check | Method | Fail Action |
|-------|--------|-------------|
| Watertight | `mesh.is_watertight` | Score = 10000 |
| Connected | `trimesh.graph.connected_components` | Score = 10000 if >1 body |
| No trapped voids | Flood-fill from exterior | Score += 500 per void |
| Feature size check | Distance field analysis | Score += graduated penalty |
| Overhang check | Face normal analysis | Score += graduated penalty |
| Volume fraction | `mesh.volume / V_envelope` | Score = 10000 if out of [5%, 40%] |
| Mass calculation | `volume × ρ_eff` | Record for scoring |

#### Stage 3: FEA — Linear Static + Buckling

**Tool:** CalculiX (open-source, available on macOS via Homebrew)

**Why CalculiX:**
- Free and open-source (no licensing constraints)
- Abaqus-compatible input format (well-documented)
- Handles linear static and linear buckling in one run
- Fast enough for desktop: 100k-element model in 30–60 seconds
- Battle-tested in aerospace and automotive

**Alternative fallback:** FEniCS (Python-native, easier integration but slower for large models and no built-in buckling)

**Meshing:** `tetgen` or `gmsh` for volumetric tetrahedral mesh from STL surface.

```bash
# Generate volume mesh
gmsh -3 -optimize lattice.stl -o lattice.inp -format inp

# Run CalculiX (linear static + buckling)
ccx lattice_static   # Static analysis: stress, displacement
ccx lattice_buckle    # Linear buckling: eigenvalues
```

**CalculiX input template (static):**

```
*HEADING
Lattice Cube Compression - Static Analysis
*NODE, INPUT=nodes.inp
*ELEMENT, TYPE=C3D10, INPUT=elements.inp
*MATERIAL, NAME=PETG
*ELASTIC
1800.0, 0.38
*SOLID SECTION, ELSET=ALL, MATERIAL=PETG
*STEP
*STATIC
*BOUNDARY
BOTTOM_NODES, 1, 3, 0.0
*DLOAD
TOP_FACE, P, 1.0
*NODE FILE
U
*EL FILE
S
*END STEP
```

**CalculiX input template (buckling):**

```
*STEP
*BUCKLE
5
*BOUNDARY
BOTTOM_NODES, 1, 3, 0.0
*DLOAD
TOP_FACE, P, 1.0
*NODE FILE
U
*END STEP
```

#### Stage 4: Post-Processing

Extract from FEA results:

| Quantity | Source | Used For |
|----------|--------|----------|
| Max von Mises stress | Static analysis | F_yield calculation |
| Max displacement (top plate) | Static analysis | F_deflection_limit |
| First buckling eigenvalue | Buckling analysis | F_buckling |
| Reference load | Input | Scale factor for all loads |

```python
def postprocess_fea(static_results, buckle_results, reference_load_N):
    """Extract failure loads from FEA results."""
    
    # Yield: scale reference load by (yield_stress / max_stress)
    max_vm_stress = static_results.max_von_mises  # at reference load
    F_yield = reference_load_N * (SIGMA_CY_DERATED / max_vm_stress)
    
    # Buckling: first eigenvalue × reference load
    lambda_1 = buckle_results.eigenvalues[0]
    F_buckling = lambda_1 * reference_load_N
    
    # Deflection: scale reference load by (limit / max_deflection)
    max_deflection = static_results.max_displacement_z  # at reference load
    if max_deflection > 0:
        F_deflection = reference_load_N * (2.5 / max_deflection)  # 2.5 mm limit
    else:
        F_deflection = float('inf')
    
    F_fail = min(F_yield, F_buckling, F_deflection)
    
    return {
        'F_fail': F_fail,
        'F_yield': F_yield,
        'F_buckling': F_buckling,
        'F_deflection': F_deflection,
        'failure_mode': 'yield' if F_fail == F_yield 
                       else 'buckling' if F_fail == F_buckling 
                       else 'deflection',
    }
```

#### Stage 5: Score Calculation

Apply the formula from §5.1.

### 8.3 Timing Budget

| Stage | Estimated Time | Notes |
|-------|---------------|-------|
| DSL parse + validate | < 1 s | Pure Python |
| Mesh generation (strut) | 5–15 s | Boolean ops are slow; cache when possible |
| Mesh generation (TPMS) | 10–30 s | Marching cubes on 100³ grid |
| Mesh validation | 1–3 s | Graph + geometry checks |
| Volume meshing (gmsh) | 5–15 s | ~100k tets |
| FEA static (CalculiX) | 15–45 s | Depends on mesh density |
| FEA buckling (CalculiX) | 10–30 s | 5 eigenvalues |
| Post-processing | < 1 s | Parsing result files |
| Score computation | < 1 s | Arithmetic |
| **Total per evaluation** | **45–120 s** | **Target: < 90 s mean** |

For PBAR with 4 branches × 3 experiments/generation × 50 generations = 600 evaluations:
- **Sequential:** 600 × 90s = 15 hours
- **Parallel (4 branches):** ~4 hours

This is comparable to the autoresearch-mlx training loop timing, making it a fair benchmark.

### 8.4 Tool Installation

```bash
# CalculiX
brew install calculix

# Gmsh
brew install gmsh

# Python dependencies (add to pyproject.toml)
# trimesh, numpy, scipy, meshio
pip install trimesh numpy scipy meshio gmsh
```

---

## 9. Success Criteria

### 9.1 PBAR vs. Greedy Comparison

Both optimization methods get **the same total evaluation budget:** 600 evaluations.

- **Greedy:** Sequential, always keep the best, revert otherwise (classic autoresearch loop)
- **PBAR:** 4 branches, annealed selection, periodic pruning

| Criterion | Threshold | Meaning |
|-----------|-----------|---------|
| **PBAR wins on final score** | PBAR score < Greedy score | PBAR found a better design |
| **Statistically significant** | p < 0.05 over 5 independent runs | Not just luck |
| **Meaningful improvement** | ≥ 10% better specific strength | Engineering significance, not just statistical |
| **Diversity** | PBAR produces ≥ 3 distinct topologies in top-10 | Evidence of actual exploration |

### 9.2 Minimum Viable Result

The challenge is considered **successfully defined** if:

1. At least one baseline topology (§6) evaluates to a score between 0.5 and 2.0 (validating that the scoring formula produces reasonable numbers)
2. The greedy optimizer can improve upon a random starting point within 50 evaluations
3. The evaluation pipeline runs end-to-end in under 120 seconds per evaluation on an M4 Max

### 9.3 Physical Validation Criterion

The ultimate test: **print the winning design and crush it.**

| Metric | Threshold |
|--------|-----------|
| Predicted vs. actual F_fail | Within 30% | 
| Correct failure mode prediction | Yield vs. buckling matches |
| Printable without modifications | Prints successfully on first attempt |

A 30% error between FEA prediction and physical test is acceptable for FDM PETG lattices (anisotropy, geometric imperfections, and material variability all contribute). If the FEA consistently **overpredicts** by more than 30%, the material derating factors (§2.2) need to be made more conservative.

### 9.4 What "PBAR Won" Means

If after 600 evaluations:
- PBAR best score < Greedy best score by ≥ 10%: **PBAR wins — population + annealing helps**
- PBAR best score ≈ Greedy best score (within 10%): **Draw — the problem may not have enough local minima to benefit from exploration**
- PBAR best score > Greedy best score by ≥ 10%: **PBAR loses — the overhead of population management isn't worth it for this problem**

In all cases, we learn something valuable about when population-based methods outperform greedy ones.

---

## 10. Failure Modes to Watch For

*From experience with pressure vessels — and lattice structures have similar gotchas:*

### 10.1 Engineering Failure Modes

| Failure Mode | How It Manifests | Mitigation |
|--------------|-----------------|------------|
| **Euler buckling of slender struts** | Long thin struts snap before yielding | Slenderness check: L/r < 50 (recommended) |
| **Layer delamination** | Z-direction tensile stress peels layers apart | Derated Z-properties (§2.2) |
| **Node stress concentration** | FEA shows singularity at sharp junctions | Mandatory fillet radius (§1.2) |
| **Print warping** | Large flat surfaces curl during printing | Keep plate bridges < 15 mm between lattice anchors |
| **Stringing between struts** | Thin wisps of material between features | Min strut spacing ≥ 3 mm |
| **Thermal stress during printing** | Residual stress from uneven cooling | Not modeled — accept as source of prediction error |

### 10.2 Optimization Failure Modes

| Failure Mode | How It Manifests | Mitigation |
|--------------|-----------------|------------|
| **Degenerate geometry** | Optimizer finds a solid block (max VF) | Volume fraction upper bound (40%) |
| **Feature size gaming** | Optimizer makes extremely thin features to minimize mass | Feature size lower bounds + printability check |
| **FEA mesh dependence** | Results change with mesh density | Fix mesh density in pipeline; spot-check convergence |
| **Symmetry exploitation** | Optimizer makes highly symmetric structures that are fragile to asymmetric loads | Secondary load cases (§3.4) provide data |
| **Overfitting to FEA** | Designs that score well in simulation but fail in reality | Physical testing (§9.3) is the ground truth |

---

## Appendix A: Quick Reference Card

```
╔══════════════════════════════════════════════════════════════╗
║               LATTICE CUBE COMPRESSION CHALLENGE            ║
╠══════════════════════════════════════════════════════════════╣
║  Envelope:     50 × 50 × 50 mm cube                        ║
║  Plates:       1.5 mm top + bottom (fixed, not optimized)   ║
║  Material:     PETG, FDM printed, no supports               ║
║  Load:         Uniaxial compression, -Z                     ║
║                                                              ║
║  OBJECTIVE:    Maximize F_fail / m_lattice (N/kg)           ║
║  SCORE:        50000 / specific_strength  (lower = better)  ║
║                                                              ║
║  CONSTRAINTS:                                                ║
║    Strut diameter:    1.2 – 8.0 mm                          ║
║    Wall thickness:    0.8 – 8.0 mm                          ║
║    Volume fraction:   5% – 40%                              ║
║    Max overhang:      50° from vertical                     ║
║    No supports, no floating islands, no trapped voids       ║
║                                                              ║
║  BASELINES:                                                  ║
║    Par (BCC/diamond):    score ≈ 1.0  (~50 kN/kg)          ║
║    Good (gyroid):        score ≈ 0.7  (~70 kN/kg)          ║
║    Excellent:            score ≈ 0.5  (~100 kN/kg)         ║
║                                                              ║
║  TOOLS:  trimesh → gmsh → CalculiX → score                 ║
║  TIME:   < 90 seconds per evaluation                        ║
║  BUDGET: 600 evaluations (PBAR and Greedy, equal)           ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Appendix B: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-13 | Wernher von Braun | Initial specification |

---

*"A lattice structure is a compromise between material and void. The art is knowing where the material earns its keep — and where it is merely dead weight. In rocketry, dead weight is the enemy. In optimization, dead weight is an opportunity."*

*— Wernher von Braun, 2030*
