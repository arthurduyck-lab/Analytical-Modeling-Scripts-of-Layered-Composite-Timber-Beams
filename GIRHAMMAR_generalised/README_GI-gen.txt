README — GIRHAMMAR_generalised Method

1. OVERVIEW ---------------------------------------------------------------------------------------
This method provides a generalized solution to the coupled bending-slip problem in multi-layer composite beams with partial interaction.
It extends Girhammar's theory to n lamellae connected by continuous or discrete shear interfaces, each with its own stiffness.
A modal approach based on Fourier series decomposition is used to solve the problem.

Number of lamellae: unlimited (n >= 2). GI_gen_Parameters() raises a
ValueError if n < 2 or if the connector data does not have exactly n - 1
rows; it does not otherwise restrict n.

Loading modes supported: 
 - SINE: Sinusoidal - modal resolution (single mode, closed-form)
 - FS - 3Pts: Point Load defined from a Fourier series SFq(x) (l = 10mm) - modal resolution
 - FS - UL: Uniform load defined from a Fourier series (l = L_beam) - modal resolution
 - FS - 4Pts: 2 Point Loads defined from a Fourier series SFq(x) (l = 10mm) - modal resolution

Point loads are approximated by a linear force applied to a segment of the beam. This segment length is l.

Validation: for n = 2, this method must reduce to the exact two-layer
Girhammar solution (GIRHAMMAR/README_GI.txt). Checked on the SINE case
(single mode, so no truncation error applies): GI_gen agrees with the
closed-form Girhammar deflection to 1.3e-16 relative error, i.e. machine
precision. For reference, GIRHAMMAR's own BVP solver (GI_SCRIPT_SINE.py)
agrees with the same closed-form value only to 3.0e-6, limited by the BVP
solver tolerance rather than by GI_gen. Re-running this n = 2 cross-check
is the recommended sanity test after any change to GI_gen_MODAL.


2. THEORETICAL BACKGROUND ---------------------------------------------------------------------------------------
The generalized Girhammar approach models the composite beam as a multi-layer system connected by shear interfaces.  
It captures both bending and slip effects without simplifying assumptions on the connection stiffness.

For each lamella i, the equilibrium equations are:
a) Compatibility condition: 
	Sum(i=1,n) EAi * ui' = 0

b) Axial equilibrium: 
	-EAi * ui'' - Ki_ip1 * (uip1 - ui + w' * vi_ip1) + Kim1_i * (ui - uim1 + w' * vim1_i) = 0 	for all i != s
		with s = floor(n/2) + 1

c) Global bending equilibrium:
	q = -w'''' * Sum(i=1,n) EIi + Sum(i=1,n) (vi * (Ki_ip1 * (uip1' - ui' + w'' * vi_ip1) + Kim1_i * (ui' - uim1' + w'' * vim1_i)))

Ki_ip1 and Kim1_i are the shear stiffnesses of the joints above and below
lamella i (0 at a free edge, i.e. Ki_ip1 = 0 for i = n and Kim1_i = 0 for
i = 1); vi_ip1 and vim1_i are the corresponding lever arms between lamella
centroids. Equation (b) is skipped for the reference lamella s, whose axial
displacement is fixed by the compatibility condition (a) instead — this is
what makes the assembled system square and well-posed (see GI_gen_MODAL
below).


3. NUMERICAL IMPLEMENTATION ---------------------------------------------------------------------------------------
Unlike GIRHAMMAR, this method needs only ONE numerical approach: the modal
resolution below is used for every loading mode, including SINE (for a
single sine load, the modal series has exactly one term, so no truncation
error or convergence question arises for that case).

### Step 0 — Compute parameters
- Import lamellae and connector data (see section 5, INPUT DATA).
- Compute:
	- Section properties (E, G, h, b, A, I),
	- Axial and bending stiffnesses (EA, EI),
	- Joint stiffnesses (Kim1_i, Ki_ip1) and lever arms (v, vim1_i, vi_ip1).

### Step 1 — Solving w(x) & ui(x): modal approach
Modal decomposition based on orthogonal sine modes:
q(x) = A_t * sin(k_t * x)
	with A_t = 2/pi * q_m * C_t/t
	     C_t = Fourier coefficients adapted to the loading mode
	     k_t = t * pi/L
w(x) = w0_t * sin(k_t * x)
ui(x) = u0i_t * cos(k_t * x)

Substituting into the governing system (a, b, c above) and matching each
mode gives, for every mode t, a square linear system of size (n + 1):
	A_sys(t) * X(t) = B_sys(t)
	where X(t) = [u0i_1, ..., u0i_n, w0_t]^T

The function GI_gen_MODAL assembles A_sys(t) and B_sys(t) row by row (one
row per lamella other than s for the axial equations, plus one row for the
compatibility condition and one for the global bending equation) and
solves the system with numpy.linalg.pinv. Using the pseudo-inverse rather
than a direct solve is deliberate here: it returns the least-squares
solution even if the assembled system were singular for a pathological
combination of stiffnesses, instead of raising a LinAlgError. It gives
the same result as an exact solve whenever the system is well-conditioned
(checked: relative difference 4.0e-16 on the reference n = 2 case).
No warning is issued if the system IS close to singular, though — if
results look suspicious for an unusual stiffness distribution, checking
numpy.linalg.cond(A_sys) is a reasonable first step.

Reconstruct global fields by summing over modes:
	w(x)  = Sum(t=1,N) w0_t  * sin(k_t * x)
	ui(x) = Sum(t=1,N) u0i_t * cos(k_t * x)
	where N = number of modes (N_modes in the scripts)

Convergence in N_modes: checked on the FS-3Pts reference case (point load
approximated by a 10 mm patch) for w, w''' (needed for M', hence T and Ts)
and ui'' (needed for Ts, see Step 3). All three are converged to better
than 1e-6 relative by N_modes = 500, and do not change further up to
N_modes = 10000. N_modes = 1000 is used throughout as a safety margin;
raising it further has no benefit and only costs runtime.

IMPORTANT DIFFERENCE WITH GIRHAMMAR (n = 2 exact method): unlike the exact
method, no polynomial smoothing of the derivatives is needed here, and none
is applied. In the exact Girhammar method w'''' is obtained from a 6th-order
ODE and its 5th derivative must be extracted from a truncated Fourier
series that does NOT converge (see GIRHAMMAR/README_GI.txt, section 3.B).
Here, by contrast, M(x) and T(x) are the ordinary isostatic bending moment
and shear force, computed directly in closed form for the loading case
(M_expr / T_expr in each script) rather than differentiated from the modal
series; the modal series is only used for w'' (bending) and ui' / ui''
(axial terms), i.e. derivatives of order <= 3 of the primary unknowns. The
corresponding modal coefficients (w0_t and u0i_t multiplied by k_t up to
the 3rd power) converge normally with t, unlike the 5th-order terms in the
exact method. This is why GI_gen_piecewise_derivatives-style smoothing is
neither present nor required in this method's scripts.

### Step 2 — Internal forces and deformations
	M(x), T(x): closed-form isostatic diagrams for the loading case
		(exact for SINE; for the point-load modes, computed for the
		idealised point load, not for the l = 10 mm patch actually
		applied to the beam — see the consistency check below)
	Mi = -EIi * w''			(bending moment carried by lamella i)
	Ni = EAi * ui'				(axial force in lamella i)
	dMi = -EIi * w'''			(= Mi', used for shear in Step 3)
	dNi = EAi * ui''			(= Ni', used for shear in Step 3)

### Step 3 — Lamellae stresses
For each lamella, sigmaN = Ni/Ai and sigmaM = -z*Mi/Ii are combined into
sigmaTOT (GI_gen_SigmaSection / GI_gen_SigmaMaxi).

Shear stresses (GI_gen_TauSection / GI_gen_TauMaxi / GI_gen_Ts) are built
by integrating the local shear flow lamella by lamella, working outward
from the reference lamella s (index idx_s = floor(n/2)) toward the bottom
(idx_inf) and the top (idx_sup) of the section, and accumulating the
running total of dNi (sum_dN_inf / sum_dN_sup) as the interface shear force
Ts at each joint crossed. Checked: tau is continuous across every
interface (to 4e-17) and vanishes at both free edges of the section, and
Ts/b matches the shear stress on either side of the corresponding joint.

### Step 4 — Shear and connector forces
Ts(x) at each joint is integrated over each connector's tributary length
(GI_gen_ConnectorForces) to obtain individual connector forces, and the
signed value of largest magnitude is reported per joint (Ts changes sign
along the beam, so the largest positive value alone would miss the
critical connector on the other half of the span).

### Step 5 (Step 6 for SINE) — Consistency check
Sectional equilibrium is verified away from any loaded zone:
	M(x) = Sum_i Mi(x) + Sum_i Ni(x) * (zG - zi)
with zG the elastic centroid of the section. (Since Sum_i Ni = 0 at every
section, the result does not actually depend on the reference level zG is
measured from; using the centroid simply keeps both terms of comparable
magnitude.) The sign of the axial term follows the sigma = -M*z/I
convention used in GI_gen_SigmaSection. Sum_i Ni(x) = 0 is checked too.

The check is evaluated away from the loaded zone. Under the load itself,
M(x) is the idealised point-load diagram (M_expr) while the beam is
actually loaded by the l = 10 mm patch used to build the Fourier series, so
the two differ there by qm*(l/2)**2/2 — a modelling difference, not a
numerical error (checked: exactly reproduces the 1.9e-3 relative gap seen
at midspan for the reference FS-3Pts case, versus 1.6e-7 a few points away
from the load).


4. INPUT DATA ---------------------------------------------------------------------------------------
Lamellae and connector properties are read from two tab-separated text
files, `data_LAM.csv` and `data_CON.csv`, read by GI_gen_FUNCTIONS.read_data().
This replaces the earlier Excel-based (.xlsx) input; openpyxl is no longer
a runtime dependency of the analysis scripts. The file format, column
names, validation behaviour and conversion utility (xlsx_to_csv.py) are
identical to the GIRHAMMAR (n = 2) method — see GIRHAMMAR/README_GI.txt,
section 4 — except that data_LAM.csv may have any number of rows n >= 2,
and data_CON.csv must then have exactly n - 1 rows.

Example data_LAM.csv (n = 3):
	h	b	E	G
	30	120	11000	100000000
	30	120	11000	100000000
	30	120	11000	100000000

Example data_CON.csv (n - 1 = 2 joints):
	c	s
	50	1
	50	1


5. REFERENCES ---------------------------------------------------------------------------------------
- U. A. Girhammar, 'A simplified analysis method for composite beams with interlayer slip', International Journal of Mechanical Sciences, vol. 51, no. 7, pp. 515-530, July 2009, doi: 10.1016/j.ijmecsci.2009.05.003.
- U. A. Girhammar and V. K. A. Gopu, 'Composite Beam-Columns with Interlayer Slip-Exact Analysis', J. Struct. Eng., vol. 119, no. 4, pp. 1265-1282, Apr. 1993, doi: 10.1061/(ASCE)0733-9445(1993)119:4(1265).
- L. Resch, 'Developpement d'elements de construction en bois de pays lamelles assembles par tourillons thermo-soudes', PHD Thesis, Nancy 1, 2009. Accessed: Nov. 21, 2023. [Online]. Available: https://www.theses.fr/2009NAN10142