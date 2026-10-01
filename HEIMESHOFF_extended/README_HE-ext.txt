README — HEIMESHOFF_extended Method (Extended Gamma Method)

1. OVERVIEW ---------------------------------------------------------------------------------------
The Heimeshoff_extended method generalises the gamma-method to more than three lamellae.
It extends the analytical formulation used in the original Heimeshoff model (Eurocode 5) by introducing a matrix-based computation of the gamma-factors, allowing the simulation of multi-layered composite timber beams with arbitrary numbers of layers and connectors.

This version maintains the same physical principle - the partial composite action between lamellae - but replaces the explicit analytical gamma-formula with a numerical system solution that computes the reduction coefficients through a stiffness equilibrium model.

Number of lamellae: unlimited (n >= 2). HE_ext_Parameters() raises a
ValueError if n < 2 or if the connector data does not have exactly n - 1
rows. Unlike HEIMESHOFF, which is restricted to n = 2 or 3, there is no
upper limit here; the regularisation described in section 2 is what makes
that hold for symmetric layups with an odd number of layers as well.

Loading modes supported: 
 - SINE: Sinusoidal
 - UL: Uniform load
 - 3Pts: Point Load - 3-points bending test
 - 4Pts: 2 Point Loads - 4-points bending test

The reduction coefficients gamma are computed from a sinusoidal load
assumption and then reused as-is for every other loading mode: EIeff does
not change with x, so M(x), T(x) and w(x) are the ordinary isostatic
diagrams for each loading case, evaluated with that single EIeff.

Validation:
 - For n = 2 and n = 3 this method must reduce to the HEIMESHOFF gamma-method.
   Checked on symmetric and asymmetric layups: EIeff agrees to 1.5e-16 and
   the individual Ni to 2.1e-16, even though the two implementations use
   entirely different, mutually incompatible parameterisations of gamma and
   a (HE follows EN 1995-1-1 Annex B; HE_ext solves the matrix system below,
   where gamma_i can legitimately come out negative). The agreement of Ni
   despite that is the strongest available check on the matrix formulation.
 - Sectional equilibrium M(x) = sum_i Mi(x) + sum_i Ni(x)*(zG - zi) and the
   axial condition sum_i Ni(x) = 0 both hold to machine precision (< 1e-12)
   for every loading mode, for n = 2, 3, 4 and 5, symmetric or not.
 Re-running these checks is the recommended sanity test after any change to
 HE_ext_Rigidity.


2. THEORETICAL BACKGROUND ---------------------------------------------------------------------------------------
The original gamma-method defines the reduction coefficient for each lamella i as:
gamma_i = 1 / (1 + pi^2 * Ei * Ai * si / (ci * L_beam^2))

In the extended version, this relation is replaced by a coupled system of equations that accounts for the mutual influence between layers.
The effective composite stiffness is determined by solving the following matrix problem:
V * Gamma = S
	where:
	Gamma = [gamma_1, gamma_2, ..., gamma_n]^T is the vector of reduction coefficients
	V is the stiffness matrix including connector and axial stiffness terms,
	S is a vector representing the deformation compatibility between adjacent lamellae.

Note that a_i here is computed as z0_i - zGC, the distance from the
centroid of lamella i to the section's PLAIN elastic centroid (weighted by
EA only). This differs from EN 1995-1-1 / HEIMESHOFF, where a_i is measured
from the gamma-weighted neutral axis and therefore depends on the gamma_i
themselves. In this method a is fixed geometry, known before the system is
solved.

Once solved, the effective bending stiffness is computed as:
EIeff = sum(i=1,n) (Ei * Ii) + sum(i=1,n) (gamma_i * Ei * Ai * ai^2)

The effective stiffness EIeff is then used in the classical (isostatic)
beam equations to compute:
- Bending moment M(x)
- Shear force T(x)
- Deflection w(x)

LAMELLAE SITTING ON THE NEUTRAL AXIS - REGULARISATION
Row i of V is proportional to a_i, and so is column i. If a lamella's
centroid coincides exactly with the section's elastic centroid (a_i = 0 -
always the case for the central lamella of a symmetric layup with an odd
number of layers, including the reference 5-layer data set shipped with
this method), then:
 - column i is identically zero, so gamma_i appears in no equation at all
   and is structurally indeterminate;
 - row i is nevertheless a non-trivial equation coupling gamma_(i-1) and
   gamma_(i+1).
The system is therefore OVERdetermined, not merely singular. Row i is
consistent only if S_i = 0, which requires the two joint stiffnesses
adjacent to lamella i to be equal. That holds automatically for a symmetric
connector layout, but nothing enforces it in general.

An earlier version of this implementation solved the system with the
Moore-Penrose pseudo-inverse, which silently returns a least-squares answer
in that situation. For a symmetric connector layout it happened to be the
exact one; for an asymmetric one it was not, and sum_i Ni - which must be
exactly zero at every section - came out at about 30 % of a typical Ni,
with no error and no warning.

The fix is a limiting process. Instead of leaving a_i at exactly 0, it is
offset by a small eps, which makes V regular and allows a direct solve. As
eps -> 0 every physical quantity converges:
 - gamma_i itself diverges like 1/eps, which is expected and harmless;
 - the PRODUCT z_i = gamma_i * a_i - the quantity that actually enters Ni
   and the shear stresses - tends to a finite, non-zero limit;
 - gamma_i * EA_i * a_i^2 -> 0, so EIeff, and therefore w, M and T, are
   unaffected (the residual perturbation scales as eps^2).
Verified over six decades of eps: z_i, Ni, EIeff and the sectional
equilibrium are all stable from about eps = 1e-5 downwards, and
sum_i Ni = 0 holds to machine precision throughout.

eps is taken as a fraction of the lamella's OWN height (eps_frac * h_i,
with eps_frac = 1e-6 by default), not of the overall section depth: h_i is
the relevant local geometric scale, and a global one would give a thin
lamella in a deep section a disproportionate offset and a worse condition
number. eps_frac = 1e-6 keeps cond(V) near 1e8 - eight decades of headroom
in double precision - while holding the EIeff perturbation near 1e-13 and
the sectional-equilibrium error near 1e-9, far below the 1e-6 threshold of
the consistency check. Lowering eps_frac further buys accuracy that is
already irrelevant at the cost of conditioning.

One consequence is worth stating plainly, because it is counter-intuitive:
a lamella whose centroid lies on the elastic centroid CAN carry a non-zero
axial force. For the reference 5-layer data set (symmetric connectors) N3
is 0, as symmetry requires, and the regularised and pseudo-inverse results
coincide. With asymmetric connector stiffness N3 is not 0, and only the
regularised solution satisfies sum_i Ni = 0.


3. NUMERICAL IMPLEMENTATION ---------------------------------------------------------------------------------------
The script is organised into five main steps:

### Step 0 — Compute parameters
- Import lamellae and connector data (see section 4, INPUT DATA).
- Compute:
	- Section properties (E, G, h, b, A, I),
	- Axial and bending stiffnesses (EA, EI),
	- Elastic centroid zGC and the distances a = z0 - zGC,
	- Effective reduction coefficients (Gamma),
	- Effective stiffness (EIeff),
	- Deformation coefficients (z = Gamma * a).

HE_ext_Rigidity(n, c, s, a, EA, EI, L_beam, h=None, eps_frac=1e-6) builds
and solves the matrix system:
- V: coupling matrix (connector + axial stiffness)
- S: equilibrium vector
- Gamma: vector of reduction coefficients
It regularises any lamella with a_i = 0 as described in section 2, then
solves with numpy.linalg.solve. The pseudo-inverse is deliberately NOT used
any more: V is regular after regularisation, and a direct solve turns a
genuine degeneracy back into a visible LinAlgError instead of a
plausible-looking wrong answer.

The function returns FOUR values: Gamma, z, EIeff and a. The returned a is
the vector actually used to build and solve the system, and may differ from
the input by the eps offset. Every downstream quantity - Ni, z, EIeff, the
stresses, the connector forces - must be computed with it:
	Gamma, z, EIeff, a = HE_ext_Rigidity(n, c, s, a, EA, EI, L_beam, h=h)
Reverting to the raw geometric a for post-processing multiplies a diverging
gamma_i by an exact zero, silently drops that lamella's axial force and
breaks sum_i Ni = 0. The consistency check in Step 4 catches this.

### Step 1 — Load and internal forces
The distributed or point loads are defined using closed-form analytical
functions (M(x), T(x), w(x)) depending on the selected loading mode, using
the single EIeff from Step 0.

Per-lamella internal forces:
	Mi(x) = M(x) * EIi / EIeff
	Ni(x) = -M(x) * gamma_i * EAi * ai / EIeff
Mi is stored with the same sign convention as the GIRHAMMAR,
GIRHAMMAR_generalised and HEIMESHOFF methods (positive Mi for a positive
global M), and the stress formula in Step 2 is sigma = -z*Mi/Ii, so the
all_*.csv files can be compared column by column across the four methods.

### Step 2 — Lamellae stresses
For each lamella: sigmaN = Ni/Ai, sigmaM = -z*Mi/Ii, combined into
sigmaTOT (HE_ext_SigmaSection / HE_ext_SigmaMaxi).

Shear stresses (HE_ext_TauSection / HE_ext_TauMaxi / HE_ext_Ts) are built
lamella by lamella, working outward from the reference lamella toward the
bottom (idx_inf) and the top (idx_sup) of the section and accumulating the
running interface shear force Ts at each joint crossed. Checked: tau is
continuous across every interface (to about 1e-17) and vanishes at both
free edges of the section, for n = 3, 4 and 5, symmetric or not.

### Step 3 — Shear and connector forces
Ts(x) at each joint is integrated over each connector's tributary length
(HE_ext_ConnectorForces) to obtain individual connector forces; the signed
value of largest magnitude is reported per joint (Ts changes sign along the
beam, so the largest positive value alone would miss the critical connector
on the other half of the span).

Each joint gets its own connector positions, so joints with different
spacing (data_CON.csv holds one s value per joint) coexist in the output,
shorter ones padded with NaN. Checked: the sum of the connector forces of a
joint equals the integral of its Ts over the span, which validates the
tributary-length split including at the two beam ends.

### Step 4 — Consistency check
Sectional equilibrium is verified at a few sections along the span:
	M(x) = sum_i Mi(x) + sum_i Ni(x) * (zG - zi)
with zG the elastic centroid (sum_i Ni = 0 at every section, so the result
does not depend on the reference level), together with sum_i Ni = 0 itself.
Signs follow the sigma = -M*z/I convention.

Note what each half of this check covers: sum_i Ni = 0 is the one that
detects a broken gamma solution (it is what caught the pseudo-inverse
problem described in section 2), whereas the equilibrium identity alone
stays satisfied even then, because it depends only on EIeff and the Steiner
term absorbs the discrepancy. Neither exercises the stress distribution
within a lamella; the tau-continuity check of Step 2 covers that instead.


4. INPUT DATA ---------------------------------------------------------------------------------------
Lamellae and connector properties are read from two tab-separated text
files, `data_LAM.csv` and `data_CON.csv`, read by
HE_ext_FUNCTIONS.read_data(). This replaces the earlier Excel-based (.xlsx)
input; openpyxl is no longer a runtime dependency of the analysis scripts.
File format, column names, validation behaviour and the conversion utility
(xlsx_to_csv.py) are identical to the GIRHAMMAR method - see
GIRHAMMAR/README_GI.txt, section 4 - except that data_LAM.csv may have any
number of rows n >= 2, and data_CON.csv must then have exactly n - 1 rows.

Example data_LAM.csv (n = 5):
	h	b	E	G
	30	120	11000	100000000
	30	120	11000	100000000
	30	120	11000	100000000
	30	120	11000	100000000
	30	120	11000	100000000

Example data_CON.csv (n - 1 = 4 joints):
	c	s
	50	1
	50	1
	50	1
	50	1


5. REFERENCES ---------------------------------------------------------------------------------------
- B. Heimeshoff, 'Zur Berechnung von Biegetragern aus nachgiebig miteinander verbundenen Querschnittsteilen im Ingenieurholzbau', Holz als Roh-und Werkstoff, vol. 45, no. 6, pp. 237-241, June 1987, doi: 10.1007/BF02616416.
- European Committee for Standardization, EN 1995-1-1:2005 Eurocode 5: Design of timber structures - Part 1-1: General - Common rules and rules for buildings, Annex B (informative), Jan. 2005.
- L. Resch, 'Developpement d'elements de construction en bois de pays lamelles assembles par tourillons thermo-soudes', PHD Thesis, Nancy 1, 2009. Accessed: Nov. 21, 2023. [Online]. Available: https://www.theses.fr/2009NAN10142
- H. Kreuzinger and H. J. Blass, 'Calculation models for prefabricated wood-based loadbearing stressed skin panels for use in roofs', European Organisation for Technical Approvals (EOTA) TR 019. Accessed: Oct. 15, 2025. [Online]. Available: https://www.eota.eu/sites/default/files/uploads/Technical%20reports/tr019total.pdf
- T. Bogensperger, G. Silly, and G. Schickhofer, 'Comparison of Methods of Approximate Verification Procedures for Cross Laminated Timber', holz.bau forschungs gmbh, Institute for Timber Engineering and Wood Technology, Graz University of Technology, MMSM 2.2.3 sfem_mat, Sep. 2012.