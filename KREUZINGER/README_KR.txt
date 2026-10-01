README — KREUZINGER Method (Shear Analogy Method)

1. OVERVIEW ---------------------------------------------------------------------------------------
The Kreuzinger method models composite timber beams.
It replaces the composite beam by two equivalent fictitious beams (A and B) coupled through a continuous shear connection.

The method numerically solves a system of differential equations representing equilibrium and compatibility conditions between bending and shear deformations in the connected layers.

This implementation follows the formulation presented by H. Kreuzinger.

Number of lamellae: unlimited (n >= 2). KR_Rigidity() and KR_Parameters()
raise a ValueError if n < 2 or if the connector data does not have exactly
n - 1 rows.

Loading modes supported: 
 - SINE: Sinusoidal
 - UL: Uniform load
 - FS - 3Pts: Point Load defined from a Fourier series SFq(x) (l = 10mm)
 - FS - UL: Uniform load defined from a Fourier series (l = L_beam)
 - FS - 4Pts: 2 Point Loads defined from a Fourier series SFq(x) (l = 10mm)

Point loads are approximated by a linear force applied to a segment of the beam. This segment length is l.

Note on G: this is the ONLY method of the repository that uses the shear
modulus G. It enters GSef, the effective shear stiffness (section 2). The
GIRHAMMAR, GIRHAMMAR_generalised, HEIMESHOFF and HEIMESHOFF_extended methods
read G from data_LAM.csv but never use it - their formulations contain no
shear deformation of the layers themselves. Consequently, running KREUZINGER
with a realistic G (e.g. 650 MPa for spruce) while comparing it against the
other four adds a physical term on one side only, and the difference observed
then mixes two causes: the methods themselves, and that extra term. Setting G
to a very large value (1e8 in the reference data set) neutralises the layers'
own shear flexibility and puts all five methods on the same assumptions,
which is what the reference data set does deliberately. Use a realistic G
only when running KREUZINGER on its own. Checked: convergence, both
consistency checks and the physical bounds hold over the whole range
(G = 50 to 1e8); with G = 650 instead of 1e8 the deflection increases by
about 1.2 %, which is the layers' shear deformation being added.

Validation: convergence and both consistency checks were swept over
connector stiffness c/s from 1e-4 to 1e6 (the dimensionless stiffness
L*sqrt(alpha) from 0.01 to 2197), n = 2 to 25 lamellae, 40 lamellae of 5 mm,
beam lengths from 200 mm to 20 m, moduli contrasted by a factor 20,
contrasted thicknesses, per-joint connector stiffnesses differing by a factor
1000, and G from 50 to 1e8. All converge, with Step 4 and Step 10 between
1e-12 and 1e-6 and the deflection always inside the no-composite /
full-composite bounds. The three BVPs are robust; the one genuine failure
mode found is the Fourier sampling issue described in section 3, Step 1.
kr_campaign.py and kr_fs.py (shipped alongside) replay this campaign; they
are test harnesses, not part of the method, and can be deleted.


2. THEORETICAL BACKGROUND ---------------------------------------------------------------------------------------
The method is based on the analogy of two beams:
- Beam A carries a part of the bending load proportional to the bending stiffness of lamellae (EIa).
- Beam B accounts for the global "Steiner" effect influenced by the shear stiffness of the connections (EIb).

A continuous elastic connection between the beams transfers shear through an equivalent shear stiffness (GSef).  
The coupling is governed by a differential equation for the internal load distribution (p_k(x)):
p_k''(x) - alpha * p_k(x) = -GSef/EIa * p(x)

	where:
	p(x): applied load distribution,
	alpha = GSef * (1/EIa + 1/EIb)          [mm^-2]
	EIa = sum(i=1,n) (Ei * Ii)
	EIb = sum(i=1,n) (Ei * Ai * zi^2)
	zi = z0i - zG
	z0i = distance between z=0 and the center of lamella i
	zG = sum(i=1,n) (Ei * Ai * z0i) / (sum(i=1,n) (Ei * Ai))

GSef combines the shear flexibility of the layers themselves with that of
the connectors, over the lever arm a between the centroids of the outermost
lamellae:
	1/GSef = a^-2 * [ h1/(2*G1*b1)
	                  + sum(i=2,n-1) hi/(Gi*bi)
	                  + hn/(2*Gn*bn)
	                  + sum(j=1,n-1) sj/cj ]
	a = sum(hi) - h1/2 - hn/2
The last term is the connector contribution; the others are the layers'.
With G set very large, only the connector term survives (see the note on G
in section 1).

1/sqrt(alpha) is the characteristic length over which the load redistributes
between the two beams, and L*sqrt(alpha) is the dimensionless number that
governs how stiff the first BVP is. It is about 5 for the reference data
set.

Each equivalent beam then satisfies a fourth-order bending differential equation and an additional shear deformation term for beam B:
w_a'''' = p_a / EIa
w_b,M'''' = p_b / EIb
w_b,T'''' = (1/GSef) * p_b''
w_b = w_b,M + w_b,T

SIGN CONVENTION - IMPORTANT
The scripts integrate these with a leading minus, w'''' = -p/EI, which sets
the convention used throughout (w < 0 downwards, M = EI*w'' > 0 in sagging).
The shear branch only involves two integrations instead of four, so applying
the SAME leading minus to it makes the shear deflection come out upwards
while the bending one goes down. The shear ODE is therefore integrated as
+p_b''/GSef, not -p_b''/GSef.

An earlier version of this implementation had the minus on both branches and
compensated for it afterwards with
	w_b = -( |w_b,M| + |w_b,T| )
Taking absolute values forced the two contributions to add downwards whatever
their actual signs. On the reference case that gave numerically identical
results, which is why it went unnoticed: w_b,M was -0.076 mm and w_b,T was
+0.757 mm, their honest sum (+0.681) bore no relation to w_a = -0.833, and
the abs() hid this by producing -0.833 anyway. It also made the Step 4 check
(w_a = w_b) pass unconditionally, since both sides were being forced to the
same sign. The failure is only visible for a load that changes sign along the
span: for p(x) = sin(2*pi*x/L), abs() dragged the upward half downwards and
the Step 4 discrepancy jumped from 1.3e-6 to 2.1e-1. With the sign corrected
and no abs(), the two branches agree on their own
(w_a - (w_b,M + w_b,T) = 2.5e-6) for every loading mode.


3. NUMERICAL IMPLEMENTATION ---------------------------------------------------------------------------------------
The script is organised into ten steps.

### Step 0 — Compute parameters
- Import lamellae and connector data (see section 4, INPUT DATA).
- Compute:
	- Section properties (E, G, h, b),
	- Axial and bending stiffnesses (EA, EI),
	- Connection stiffness (GSef),
	- Coupling coefficient (alpha).

Functions: KR_Parameters(), KR_Rigidity().

### Step 1 — Solve for internal load redistribution (p_k(x))
Equation:
p_k'' - alpha*p_k = -(GSef/EIa)*p(x)
Boundary conditions: p_k(0)=0, p_k(L)=0

The boundary value problem (BVP) is solved numerically using `scipy.integrate.solve_bvp`.  

The resulting fields define:
p_a(x) = p(x) - p_k(x) (load on beam A),
p_b(x) = p_k(x) (load on beam B).

FOURIER SAMPLING - the one failure mode of this method
For the FS loading modes, p(x) is a truncated Fourier series of N_fs terms
(FS_q / FS_2q), whose shortest wavelength is 2*L/N_fs. If the grid step dx
exceeds L/N_fs, that series is ALIASED: the solvers then receive a load that
is not the one intended and converge perfectly well on the wrong problem.
solve_bvp reports success either way, so nothing signals the error. Measured
on the 3Pts reference case (L = 2600 mm, l = 10 mm, N_fs = 10000), varying
only n_points:

    n_points   dx      L/N_fs   dx <= L/N_fs   w_max
      1001    2.600    0.260        no        -2.6345   <- wrong sign
      2501    1.040    0.260        no         3.1875
      5001    0.520    0.260        no         6.4973
     10001    0.260    0.260        yes        7.1879   <- converged

Two conditions must hold, and both are now derived from the geometry in each
FS script rather than hard-coded:
	N_fs     >= 2*L/l          (the series must resolve the loaded patch)
	n_points >= N_fs + 1       (the grid must resolve the series)
	    N_fs = max(1000, ceil(2*L_beam/l))
	    n_points = max(10001, 10*N_fs + 1)
The factor 10 leaves an order of magnitude of margin on the Nyquist
condition. The previous fixed values (N_fs = 10000, n_points = 10001) sat
exactly at that limit with no margin at all, so any change of L_beam or l
silently broke the result - which is the likely explanation for runs that
looked like they "did not converge". The first condition matters too: on an
8 m beam with l = 10 mm, N_fs = 1000 left the Step 10 check at 3.2e-04, while
N_fs = 1600 = 2*L/l brought it to 1.7e-10.

KR_check_FS_sampling() re-checks both conditions at run time and raises an
explicit ValueError rather than letting an aliased run proceed. Each FS
script prints the sizing it selected.

Note that raising N_fs is not free: it forces a proportionally finer grid.
N_fs only has to resolve the loaded patch; beyond 2*L/l it buys nothing.

### Step 2 — Solve for deflection of beam A
Equation:
w_a'''' = p_a / EIa
Boundary conditions: w(0)=0, w(L)=0, w''(0)=0, w''(L)=0 (simply supported beam).  

Outputs:
- Deflection w_A(x),
- Bending moment M_A = EIa * w_A'',
- Shear force T_A = dM_A/dx, computed with KR_dMdx (see below).

KR_dMdx instead of np.gradient: np.gradient is second-order accurate in the
interior but drops to a one-sided FIRST-order difference at the two end
points. That matters because the Step 10 reaction check reads T exactly
there. For the uniform-load case it returned T(0) = 699.30 N instead of
700.00 N - a 1.0e-3 relative error that failed the check while the solution
itself was fine, every interior point being accurate to ~1e-12. KR_dMdx
keeps np.gradient in the interior and replaces the two end points by the
derivative of a local cubic fit, which restores full accuracy (Step 10 for
the uniform-load case went from 1.00e-03 to 1.32e-11).

### Step 3 — Solve for beam B
Two contributions are considered:

I. Bending part: w_b,M'''' = p_b/EIb
II. Shear part: w_b,T'''' = (1/GSef) * p_b''
   (integrated with a + sign relative to the bending branch - see the sign
   convention in section 2)

Both are solved using `solve_bvp`, and the total deflection is the plain sum:
w_b = w_b,M + w_b,T

Outputs:
- Deflection w_B(x),
- Bending moment M_B = EIb * w_b,M'',
- Shear force T_B = dM_B/dx (KR_dMdx).

### Step 4 — Consistency check (w_A = w_B)
The two equivalent beams must deflect identically. The check is RELATIVE,
normalised by max|w_a|: an absolute threshold of 1e-3 mm means 0.1 % of a
1 mm deflection but only 0.002 % of a 50 mm one, and it was failing the
FS-UL case at 5.35e-03 mm on a 5.53 mm deflection - i.e. 0.097 %, perfectly
acceptable.

This check is also what the abs() described in section 2 was neutralising:
forcing both branches to the same sign made it pass unconditionally. It is
only meaningful now that the superposition is a plain sum.

### Other Steps
- Step 5 - Compute and export efforts, loads and deformations into `.csv` files
- Step 6 - Stresses computation for beam A
- Step 7 - Stresses computation for beam B
- Step 8 - Total stresses (sum of A and B)
- Step 9 - Shear forces in connectors. Ts at each joint is integrated over
  each connector's tributary length (KR_ConnectorForces); the signed value of
  largest magnitude is reported per joint, since Ts changes sign along the
  beam and a plain maximum would miss the critical connector on the other
  half. Each joint gets its own connector positions, so joints with different
  spacing (data_CON.csv holds one s value per joint) coexist in the output,
  shorter ones padded with NaN.
- Step 10 - Consistency check of results: the reaction computed from T at the
  two supports is compared with the total applied load.


4. INPUT DATA ---------------------------------------------------------------------------------------
Lamellae and connector properties are read from two tab-separated text
files, `data_LAM.csv` and `data_CON.csv`, read by KR_FUNCTIONS.read_data().
This replaces the earlier Excel-based (.xlsx) input; openpyxl is no longer a
runtime dependency of the analysis scripts. File format, column names,
validation behaviour and the conversion utility (xlsx_to_csv.py) are
identical to the GIRHAMMAR method - see GIRHAMMAR/README_GI.txt, section 4 -
except that data_LAM.csv may have any number of rows n >= 2, and
data_CON.csv must then have exactly n - 1 rows.

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

The G column is set to 1e8 on purpose in this reference data set; see the
note on G in section 1 before replacing it with a realistic value.


5. REFERENCES ---------------------------------------------------------------------------------------
- H. Kreuzinger, 'Platten, Scheiben und Schalen', in Bauen mit Holz, vol. 101, 1999.
- L. Resch, 'Developpement d'elements de construction en bois de pays lamelles assembles par tourillons thermo-soudes', PHD Thesis, Nancy 1, 2009. Accessed: Nov. 21, 2023. [Online]. Available: https://www.theses.fr/2009NAN10142
- H. Kreuzinger and H. J. Blass, 'Calculation models for prefabricated wood-based loadbearing stressed skin panels for use in roofs', European Organisation for Technical Approvals (EOTA) TR 019. Accessed: Oct. 15, 2025. [Online]. Available: https://www.eota.eu/sites/default/files/uploads/Technical%20reports/tr019total.pdf
- T. Bogensperger, G. Silly, and G. Schickhofer, 'Comparison of Methods of Approximate Verification Procedures for Cross Laminated Timber', holz.bau forschungs gmbh, Institute for Timber Engineering and Wood Technology, Graz University of Technology, MMSM 2.2.3 sfem_mat, Sep. 2012.