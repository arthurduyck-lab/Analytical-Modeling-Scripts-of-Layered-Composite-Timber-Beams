README — GIRHAMMAR Method

1. OVERVIEW ---------------------------------------------------------------------------------------
This method offers a rigorous solution to the sixth-order differential equation governing the bending behaviour of a composite beam with partial slip between layers.
Two approaches are proposed in the scripts: 
- Direct numerical solution using a boundary value problem solver (BVP),
- Modal approach based on Fourier series decomposition.


Number of lamellae: n = 2 (strictly). GI_Parameters() raises a ValueError if the
input data defines a different number of layers — the exact closed-form
coefficients used here (EAp, zcg_full, EI_full, alpha, beta, gamma) are only
valid for two layers. Use GIRHAMMAR_generalised for n > 2.

Loading modes supported: 
 - SINE: Sinusoidal - numerical resolution (BVP)
 - UL: Uniform load - numerical resolution (BVP)
 - FS - 3Pts: Point Load defined from a Fourier series SFq(x) (l = 10mm) - modal resolution
 - FS - UL: Uniform load defined from a Fourier series (l = L_beam) - modal resolution
 - FS - 4Pts: 2 Point Loads defined from a Fourier series SFq(x) (l = 10mm) - modal resolution

Point loads are approximated by a linear force applied to a segment of the beam. This segment length is l.

Validation: the BVP and modal solvers are independent implementations of the
same problem. For a uniform load, GI_SCRIPT_UL.py (BVP) and GI_SCRIPT_FS-UL.py
(modal) agree on the maximum deflection to 1.2e-8 relative error. This
cross-check is the recommended sanity test after any change to either solver.


2. THEORETICAL BACKGROUND ---------------------------------------------------------------------------------------
The exact Girhammar approach models the composite beam as a multi-layer system connected by shear interfaces.  
It captures both bending and slip effects without simplifying assumptions on the connection stiffness.

The general governing equation for the deflection w(x) is:
w'''''' - alpha**2 * w'''' = q''/EI0 - alpha**2 * q/EI_full
	where:  
	- EI0: bending stiffness of the unconnected layers,  
	- EI_full: bending stiffness for full composite action,  
	- alpha: parameter related to the connection stiffness,  
	- q(x): distributed load.

Boundary conditions (for simply supported beams):
w(0) = w(L) = 0,
w''(0) = w''(L) = 0,
w''''(0) = w''''(L) = q/EI_zero.


3. NUMERICAL IMPLEMENTATION ---------------------------------------------------------------------------------------
###### A) NUMERICAL RESOLUTION (BVP)
Used by GI_SCRIPT_SINE.py and GI_SCRIPT_UL.py.

### Step 0 — Compute parameters
- Import lamellae and connector data (see section 5, INPUT DATA).
- Compute:
	- Section properties (E, G, h, b),
	- Axial and bending stiffnesses (EA, EI, EI0, EI_full),
	- Parameters related to the connection stiffness (alpha, beta, gamma).

### Step 1 — Solving w(x): numerical approach
Solve Girhammar's differential equation numerically as a boundary value problem (BVP) with 6 unknowns, using `scipy.integrate.solve_bvp`.
y0=w, y1=w', y2=w'', y3=w''', y4=w'''', y5=w'''''
	with y5' = alpha^2 * y4 + q''/EI0 - alpha^2 * q/EI_full

Here w'''' is obtained directly from the BVP solution, so it carries the full
shape of q(x), discontinuities included. The q-term below is therefore
required for M and T to be correct.

### Step 2 — Load, internal forces and deformation
M = EI_full / alpha**2 * w'''' - EI_full * w'' - EI_full/(alpha**2 * EI0) * q
T = EI_full / alpha**2 * w''''' - EI_full * w''' - EI_full/(alpha**2 * EI0) * q'
Ts = 1/r_tot * (M' + EI0 * w''')
	where Ts is the shear flow at the interface

Internal forces for lamella 1 (bottom layer):
N1 = 1/r_tot * [F * (r_tot - zcg_full) + M + EI0 * w'']
M1 = -EI[0] * w''
T1 = M1' + Ts * r1

Internal forces for lamella 2 (top layer):
N2 = 1/r_tot * [F * zcg_full - M - EI0 * w'']
M2 = -EI[1] * w''
T2 = M2' + Ts * r2

(zcg_full = EA[1]/EA0 * r_tot is the share of layer 2 in the composite
centroid, so N2 is the one weighted by zcg_full and N1 by its complement
r_tot - zcg_full. This was swapped in an earlier version of this README;
GI_SCRIPT_SINE.py / GI_SCRIPT_UL.py implement the version above. Since F = 0
in every script provided, N1 + N2 = 0 either way and the check below cannot
catch the sign of the F-term on its own — confirm it independently if F != 0
is ever used.)

Verified identities (checked numerically on GI_SCRIPT_SINE.py):
	N1 + N2 = 0
	M1 + M2 + N1 * r_tot = M   (exact to solver tolerance)
	T1 + T2 = T
	tau is continuous across the interface and vanishes at both free edges

### Step 3 — Lamellaes stresses
For each lamella, bending and axial contributions are calculated to find normal stresses. 
For each lamella, shear stresses are computed.

Note on GI_TAU_global: the z-axis of lamella 2 is flipped internally
(GI_TAU always places the interface at z_local = +r), so its shear profile
is mirrored before being expressed in global coordinates. This is
intentional — verified by continuity of tau at the interface — not an
inconsistency with GI_SIGMA_global, which does not need this flip.

### Step 4 — Shear and connector forces
The shear flow between lamellae is evaluated and integrated over each
connector's tributary length to obtain individual connector forces.

### Step 5 - Consistency results
The check compares the resultant reaction (from T at both supports) against
the exact applied load, and requires a relative error below 1e-3.


###### B) MODAL RESOLUTION
Used by GI_SCRIPT_FS-UL.py, GI_SCRIPT_FS-3Pts.py and GI_SCRIPT_FS-4Pts.py.
Semi-analytical approach based on sinusoidal modal decomposition.
Boundary conditions are automatically satisfied.

### Step 0 — Compute parameters
Same as step 0 of the BVP branch.

### Step 1 — Solving w(x): modal approach
Modal decomposition based on orthogonal sine modes:
q(x) = 2/pi * q_m * Sum[C_t/t * sin(k_t * x)]
	with k_t = t * pi/L    &    A_t = 2/pi * q_m * C_t/t
	     C_t = Fourier coefficients adapted to loading mode

q(x) = Sum [ A_t * sin(k_t * x) ]
w(x) = Sum [ B_t * sin(k_t * x) ]

Substituting w = Sum[B_t sin(k_t x)] and q = Sum[A_t sin(k_t x)] into the
governing equation and matching each mode gives:
	B_t * k_t**4 * (k_t**2 + alpha**2) = A_t * (k_t**2/EI0 + alpha**2/EI_full)
	B_t = A_t * S_t / (k_t**4 * (k_t**2 + alpha**2))
	with S_t = k_t**2/EI0 + alpha**2/EI_full

Projection onto each mode using orthogonality on [0, L]:
w_t(x) = B_t * sin(k_t * x)
w(x) = Sum[B_t * sin(k_t * x)]

IMPORTANT — this series does not converge for the higher derivatives: the
raw series for w''''' grows with the number of modes N instead of settling
(e.g. T(0) computed directly from the truncated series swings between
-6 300 and +58 000 as N goes from 500 to 20 000, for an exact value of 1000).
This is expected: for a point or patch load, q(x) is discontinuous, so its
Fourier series does not converge pointwise, and neither do the derivatives
obtained by differentiating it term by term. Increasing N does not fix
this and should not be used to try to do so. Step 2 (smoothing) is what
makes the higher derivatives usable, not the number of retained modes.

### Step 2 — Derivatives of piecewise w(x) and smoothing
To avoid problems related to singularities/non-convergence in the higher
derivatives, w(x) is split at every load discontinuity and, on each piece,
a polynomial of order `poly_order` is least-squares fitted to w(x) over the
sub-interval reduced by `margin` at the discontinuity side(s) (margin = l,
the load-patch length, so that the fit sees the smooth part of w(x) only).
The polynomial is then differentiated analytically up to 5 times.

Two implementation points are required for this to work in practice, both
present in GI_FUNCTIONS.GI_piecewise_derivatives:

- The fit must be done on a domain normalised to [-1, 1]
  (numpy.polynomial.Polynomial.fit), not on the raw x in [0, L]. The raw
  Vandermonde matrix on x in [0, L] has a condition number of order 1e31
  for L ~ 1000-2000 mm at poly_order = 10, which silently corrupts the fit
  (numpy issues a RankWarning) as soon as a higher order is requested.
  On the normalised domain the fit stays well conditioned up to
  poly_order ~ 20.

- poly_order must be at least ~12-16 for the 5th derivative to be accurate
  near the interval boundaries: each differentiation removes one order of
  useful shape information from the fitted polynomial, so a 5th derivative
  needs several orders of margin above what would look sufficient for w(x)
  itself. poly_order = 10 (as originally set) gives a consistency-check
  error of 1 to 2 %; poly_order = 14-16 (combined with the normalised fit
  above) brings it below 1e-6 for every loading mode tested here.
  Orders above ~20 degrade again (condition number grows past 1e8 even on
  the normalised domain) — poly_order is not a "the higher the better"
  parameter.

If the beam length changes significantly, or 1/alpha (the characteristic
length of the connection, ~300 mm for the reference data set) changes
significantly relative to L, re-run the consistency check (Step 6) before
trusting the result: the reasoning above only holds while the load-patch
length l stays small compared to 1/alpha.

### Step 3 — Load, internal forces and deformation
Because w'''' and w''''' here come from the smoothed polynomial fit, they
represent the REGULAR part of the true derivative only: the fit cannot
reproduce a jump, so the singular part (the load itself, where present) is
implicitly removed by the smoothing. Whether the q-term must be added back
in M and T therefore depends on whether q(x) is itself smooth enough for
the polynomial fit to reproduce it — checked numerically as follows.

  FS-UL (uniform load, smooth q over the whole span):
	M = EI_full/alpha**2 * w'''' - EI_full * w'' - EI_full/(alpha**2*EI0) * q
	T = EI_full/alpha**2 * w''''' - EI_full * w''' - EI_full/(alpha**2*EI0) * q'
	Ts = 1/r_tot * (T + EI0 * w''')
  The polynomial CAN represent the constant q = qm, so the fit keeps it in
  w'''' and the q-term must be subtracted explicitly, exactly as in the BVP
  branch. Omitting it here gives ~160 % error on M.

  FS-3Pts / FS-4Pts (point loads applied over a short patch, l = 10 mm):
	M = EI_full/alpha**2 * w'''' - EI_full * w''
	T = EI_full/alpha**2 * w''''' - EI_full * w'''
	Ts = 1/r_tot * (T + EI0 * w''')
  q(x) here is a narrow patch that the polynomial fit does not reproduce
  (it is excluded from the fitted region by `margin`), so the fit already
  returns the regular part only; adding the q-term a second time
  double-counts it and introduces a large spurious peak at the load
  (checked numerically: -7.9e7 instead of ~1.3e6 for the reference case).

In both cases Ts is written with T rather than M' for convenience; the two
are equivalent (T = M') to within the smoothing/differentiation error,
verified numerically here to 4.4e-7 relative, away from the load and the
beam ends.

Internal forces for lamella 1 and lamella 2: same formulas as Step 2 of the
BVP branch (section A above), using the M, T, Ts computed as above.

### Step 4 — Lamellaes stresses
Same as Step 3 of the BVP branch.

### Step 5 — Shear and connector forces
Same as Step 4 of the BVP branch.

### Step 6 - Consistency results
The reaction computed from T at the supports is compared against the EXACT
resultant of the applied load (F for FS-3Pts, 2*P for FS-4Pts, qm*L_beam for
FS-UL) — not against the numerical integral of the reconstructed q(x).
For FS-UL in particular, integrating the truncated Fourier series of a
uniform load underestimates the true resultant by about 1e-3 (Gibbs
phenomenon at the supports, where the reconstructed q(x) overshoots then
undershoots instead of jumping cleanly to qm): using that integral as the
reference makes the check fail on the reference value itself, even when T
is already accurate to 1e-5. Always compare against the exact, analytically
known resultant of the load case being run.


4. INPUT DATA ---------------------------------------------------------------------------------------
Lamellae and connector properties are read from two tab-separated text
files, `data_LAM.csv` and `data_CON.csv`, read by GI_FUNCTIONS.read_data().
This replaces the earlier Excel-based (.xlsx) input; openpyxl is no longer
a runtime dependency of the analysis scripts.

File format (both files):
	- First line: column headers, exactly as below.
	- One data row per lamella / per joint.
	- Columns separated by a tabulation character (sep='\t' by default,
	  configurable via the `sep` argument of read_data()).
	- Blank lines and lines starting with '#' are ignored.
	- Decimal mark: '.' by default; pass decimal=',' to read_data() if the
	  file was exported with a French/European locale.

data_LAM.csv columns: h [mm], b [mm], E [N/mm2], G [N/mm2] — one row per
lamella, ordered from the bottom layer (row 1) to the top layer (row 2).

data_CON.csv columns: c [N/mm], s [mm] — exactly one row (n - 1 = 1 joint
for n = 2 lamellae).

Example data_LAM.csv:
	h	b	E	G
	30	120	11000	690
	30	120	11000	690

Example data_CON.csv:
	c	s
	50	1

read_data() validates the file and raises an explicit error rather than
silently returning wrong numbers, e.g.:
	ValueError: Missing or non-numeric value(s) in 'data_LAM.csv' at data
	row(s) [2]. Check the separator (sep='\t') and the decimal mark
	(decimal='.').
GI_Parameters() additionally raises a ValueError if the number of lamellae
in data_LAM.csv is not exactly 2, or if the number of rows in data_CON.csv
does not match n - 1.

Converting existing .xlsx files: use xlsx_to_csv.py (provided alongside the
scripts) to convert data_LAM.xlsx / data_CON.xlsx into the tab-separated
format above:
	python xlsx_to_csv.py                # converts every .xlsx in the folder
	python xlsx_to_csv.py --overwrite    # replace existing .csv files
This utility is the only remaining place that depends on openpyxl.


5. REFERENCES ---------------------------------------------------------------------------------------
- U. A. Girhammar, 'A simplified analysis method for composite beams with interlayer slip', International Journal of Mechanical Sciences, vol. 51, no. 7, pp. 515-530, July 2009, doi: 10.1016/j.ijmecsci.2009.05.003.
- U. A. Girhammar and V. K. A. Gopu, 'Composite Beam-Columns with Interlayer Slip-Exact Analysis', J. Struct. Eng., vol. 119, no. 4, pp. 1265-1282, Apr. 1993, doi: 10.1061/(ASCE)0733-9445(1993)119:4(1265).
- L. Resch, 'Developpement d'elements de construction en bois de pays lamelles assembles par tourillons thermo-soudes', PHD Thesis, Nancy 1, 2009. Accessed: Nov. 21, 2023. [Online]. Available: https://www.theses.fr/2009NAN10142