README — HEIMESHOFF Method (Gamma Method)

1. OVERVIEW ---------------------------------------------------------------------------------------
The Heimeshoff method models composite timber beams using the gamma-method (EN 1995-1-1, Annex B).
It provides an analytical approach for beams composed of several timber lamellae connected by continuous shear connectors.
The global stiffness of the beam is derived from the effective bending stiffness EIeff, computed using reduction coefficients gamma that account for partial composite action.

Number of lamellae: 2 or 3 only. HE_Parameters() raises a ValueError for any
other value of n. This is a limitation of this implementation, not of the
gamma-method itself (EC5 Annex B gives the same reduction-coefficient
formula for an arbitrary number of layers); HEIMESHOFF_extended lifts this
restriction with a matrix formulation.

Loading modes supported: 
 - SINE: Sinusoidal
 - UL: Uniform load
 - 3Pts: Point Load - 3-points bending test
 - 4Pts: 2 Point Loads - 4-points bending test

The reduction coefficients gamma are computed from a sinusoidal load
assumption (EC5 Annex B is derived that way) and then reused as-is for
every other loading mode: EIeff does not change with x, so M(x), T(x) and
w(x) are the ordinary isostatic diagrams for each loading case, evaluated
with that single EIeff.

Validation: sectional equilibrium M(x) = sum_i Mi(x) + sum_i Ni(x)*(zG - zi)
holds to machine precision (< 2e-16 relative) for every loading mode, for
n = 2 and n = 3, symmetric or not. See section 3 for how this check is
performed and what it does and does not verify.


2. THEORETICAL BACKGROUND ---------------------------------------------------------------------------------------
The gamma-method approximates the composite action between lamellae through the factor gamma_i, defined for each layer i as:
gamma_i = 1 / (1 + pi^2 * Ei * Ai * si / (ci * L_beam^2))

The reference lamella (lamella 2, whichever position that is: the upper
one for n = 2, the central one for n = 3) is rigidly assumed not to slip
relative to itself, so EC5 fixes gamma_2 = 1 rather than computing it from
the formula above. HE_Parameters() applies this explicitly; do not remove
it under the impression that it is a special case of the general formula
(it is not: the formula above would give a value close to but different
from 1 if evaluated for lamella 2 as well).

Each lamella contributes to the total bending stiffness according to its gamma-factor:

EIeff = sum(i=1,n) (Ei * Ii) + sum(i=1,n) (gamma_i * Ei * Ai * ai^2)
	where ai is the vertical distance between the centroid of lamella i and the neutral axis of the composite section.

ai itself depends on the gamma_i (EC5 Annex B, eq. B.6): the position of
the neutral axis is not fixed geometry, it is the composite section's
elastic centroid once each layer's axial stiffness has been reduced by its
own gamma_i. For n = 3 (lamella 2 central):
	a2 = [gamma_3*E3*A3*(h2+h3) - gamma_1*E1*A1*(h1+h2)] / [2 * sum(gamma_i*Ei*Ai)]
	a1 = (h1+h2)/2 + a2
	a3 = (h2+h3)/2 - a2
For n = 2 this is the same formula with the third layer removed
(gamma_3*E3*A3 = 0):
	a2 = [0 - gamma_1*E1*A1*(h1+h2)] / [2 * (gamma_1*E1*A1 + gamma_2*E2*A2)]
	a1 = (h1+h2)/2 + a2
a2 is kept negative here (it is the signed offset of the reference lamella
from the neutral axis), so that the "+ for lamellae below the reference,
- for lamellae above" sign rule applied when computing Ni (Step 1 below)
gives the correct sign directly; a1 and a3 are the corresponding positive
EC5 magnitudes. This differs from EC5's own sign convention for a2 by a
sign flip, kept consistent internally in HE_Parameters and documented
there. An earlier version of this implementation used the n = 3 formula's
sign convention for n = 3 but an independently-derived, oppositely-signed
expression for n = 2; the two silently disagreed and every Ni for n = 2
came out with the wrong sign (the bottom lamella came out in compression
under a sagging moment). Since a only enters EIeff squared, w was
unaffected and the error was invisible on deflection; only Ni, the normal
stresses, the shear stresses and the connector forces were wrong.
The two formulas are now written as one, parameterised by n, to remove the
possibility of the two branches drifting apart again.

The effective stiffness EIeff is then used in the classical (isostatic)
beam equations to compute:
- Bending moment M(x)
- Shear force T(x)
- Deflection w(x)


3. NUMERICAL IMPLEMENTATION ---------------------------------------------------------------------------------------
The script is organised into five main steps:

### Step 0 — Compute parameters
- Import lamellae and connector data (see section 4, INPUT DATA).
- Compute:
	- Section properties (E, G, h, b, A, I),
	- Axial and bending stiffnesses (EA, EI),
	- Reduction coefficients (Gamma), neutral-axis offsets (a) and their
	  gamma-weighted counterpart (z = Gamma * a, used directly in the shear
	  stress formulas of Step 2),
	- Effective stiffness (EIeff).

Functions: HE_Parameters().

The reference lamella (gamma = 1) is identified in each script as
idx_lamref = argmax(Gamma) rather than hard-coded, so that the
classification into idx_inf / idx_cent / idx_sup below stays consistent
with whatever HE_Parameters actually computed, whether n = 2 or n = 3.

### Step 1 — Load and internal forces
The distributed or point loads are defined using closed-form analytical
functions (M(x), T(x), w(x)) depending on the selected loading mode, using
the single EIeff from Step 0.

Per-lamella internal forces:
	Mi(x) = M(x) * EIi / EIeff
	Ni(x) = +/- M(x) * gamma_i * EAi * ai / EIeff
		(+ for lamellae below the reference, - for lamellae above;
		 the sign of ai already accounts for the reference lamella itself,
		 whose Ni is 0 since ai = 0 there for n = 2, or is handled by the
		 same formula with its own signed ai for n = 3)

Mi is stored here with the SAME sign convention as the GIRHAMMAR and
GIRHAMMAR_generalised methods (positive Mi for a positive global M), so
that the all_*.csv files can be compared column by column between the
three methods. The stress formula in Step 2 uses sigma = -z*Mi/Ii,
matching that convention (an earlier version used the opposite pair of
conventions - negative Mi with sigma = +z*Mi/Ii - which gave the same
stresses but a Mi column with the opposite sign from the other two
methods).

### Step 2 — Lamellae stresses
For each lamella: sigmaN = Ni/Ai, sigmaM = -z*Mi/Ii, combined into
sigmaTOT (HE_SigmaSection / HE_SigmaMaxi).

Shear stresses (HE_TauSection / HE_TauMaxi) are built lamella by lamella,
working outward from the reference lamella toward the bottom (idx_inf) and
the top (idx_sup) of the section; HE_tauwjk gives the shear flow carried
at the joint between the reference lamella and its neighbour. Checked: tau
is continuous across every interface (to about 1e-17) and vanishes at
both free edges of the section, for both n = 2 and n = 3, symmetric or
not.

### Step 3 — Shear and connector forces
The shear flow between lamellae (Ts) is evaluated at each joint and
integrated over each connector's tributary length (HE_ConnectorForces) to
obtain individual connector forces; the signed value of largest magnitude
is reported per joint.

### Step 4 — Consistency check
Sectional equilibrium is verified at a few sections along the span:
	M(x) = sum_i Mi(x) + sum_i Ni(x) * (zG - zi)
with zG the elastic centroid of the section (sum_i Ni = 0 at every
section, so the result does not actually depend on the reference level zG
is measured from). This check exercises Step 0 (Gamma, a, EIeff) and Step
1 (Mi, Ni) together but says nothing about Step 2/3 (the stress and shear
distributions within each lamella); the tau-continuity check mentioned
above covers those instead.


4. INPUT DATA ---------------------------------------------------------------------------------------
Lamellae and connector properties are read from two tab-separated text
files, `data_LAM.csv` and `data_CON.csv`, read by HE_FUNCTIONS.read_data().
This replaces the earlier Excel-based (.xlsx) input; openpyxl is no longer
a runtime dependency of the analysis scripts. File format, column names,
validation behaviour and conversion utility (xlsx_to_csv.py) are identical
to the GIRHAMMAR method - see GIRHAMMAR/README_GI.txt, section 4 - except
that data_LAM.csv must have exactly 2 or 3 rows here.

Example data_LAM.csv (n = 3):
	h	b	E	G
	30	120	8500	650
	30	120	8500	650
	30	120	8500	650

Example data_CON.csv (n - 1 = 2 joints):
	c	s
	1000	50
	1000	50


5. REFERENCES ---------------------------------------------------------------------------------------
- B. Heimeshoff, 'Zur Berechnung von Biegetragern aus nachgiebig miteinander verbundenen Querschnittsteilen im Ingenieurholzbau', Holz als Roh-und Werkstoff, vol. 45, no. 6, pp. 237-241, June 1987, doi: 10.1007/BF02616416.
- European Committee for Standardization, EN 1995-1-1:2005 Eurocode 5: Design of timber structures - Part 1-1: General - Common rules and rules for buildings, Annex B (informative), Jan. 2005.
- L. Resch, 'Developpement d'elements de construction en bois de pays lamelles assembles par tourillons thermo-soudes', PHD Thesis, Nancy 1, 2009. Accessed: Nov. 21, 2023. [Online]. Available: https://www.theses.fr/2009NAN10142