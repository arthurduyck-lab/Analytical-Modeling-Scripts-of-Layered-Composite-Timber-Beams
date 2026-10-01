Analytical Modeling Scripts of Layered Composite Timber Beams

User Guide for scripts use
Authors: 
    Duyck Arthur (University of Mons, Department of Structural Mechanics - Timber Research Group) - Contact Person - Arthur.DUYCK@umons.ac.be
    Descamps Thierry (University of Mons, Department of Structural Mechanics - Timber Research Group) - Supervisor
    Van Parys Thierry (University of Mons, Department of Structural Mechanics - Timber Research Group) - Supervisor

1. OBJECTIVES ---------------------------------------------------------------------------------------
Several analytical methods exist to model the mechanical behavior of layered composite timber beams.
The proposed scripts compute internal forces, deformations, and connector forces through numerical resolution of analytical models derived from the literature for a simply supported beam.

Scope of application: comparison of five analytical approaches (GIRHAMMAR,
GIRHAMMAR_generalised, HEIMESHOFF, HEIMESHOFF_extended, KREUZINGER).
Language: Python (>= 3.9).

The point of the repository is the COMPARISON between methods. Everything
that follows - the common sign conventions, the shared consistency checks,
the note on G - exists so that the *_all_*.csv files produced by the five
methods can be read side by side without further reinterpretation.


2. REPOSITORY STRUCTURE ---------------------------------------------------------------------------------------
The repository is organized into several folders, each corresponding to a specific analytical modeling method:

| Folder                  | Method                          | n           | Description                                                                 |
| ----------------------- | ------------------------------- | ----------- | --------------------------------------------------------------------------- |
| `GIRHAMMAR`             | Girhammar theory (two layers)   | n = 2 only  | Exact solution of the 6th-order ODE, by BVP and by modal decomposition.     |
| `GIRHAMMAR_generalised` | Generalized Girhammar extension | n >= 2      | Modal solution of the coupled bending-slip system for n layers.             |
| `HEIMESHOFF`            | Gamma method (EC5)              | n = 2 or 3  | EN 1995-1-1 Annex B reduction coefficients, closed form.                    |
| `HEIMESHOFF_extended`   | Generalized gamma method        | n >= 2      | Same principle, gamma obtained from a matrix system instead of a formula.   |
| `KREUZINGER`            | Shear analogy method            | n >= 2      | Two equivalent beams (A and B) coupled by a continuous shear connection.    |

The n limits are enforced at run time: each method raises an explicit
ValueError if the lamellae data does not match, rather than silently
computing with the wrong number of layers.

Each folder also holds a method-specific README (README_GI.txt,
README_GI-gen.txt, README_HE.txt, README_HE-ext.txt, README_KR.txt) which
documents that method's formulation, sign conventions and known limitations.
Read the method README before modifying anything in its folder: several of
the implementation choices look wrong at first sight and are documented
precisely because a reviewer would otherwise "fix" them and break the
results.

At the repository root:

| File                           | Role                                                              |
| ------------------------------ | ----------------------------------------------------------------- |
| `README.txt`                   | This file.                                                        |
| `LICENSE`                      | BSD 3-Clause licence text (see section 9).                        |
| `CITATION.cff`                 | Machine-readable citation metadata, with the Zenodo DOI.          |
| `requirements.txt`             | Python dependencies (see section 3).                              |
| `xlsx_to_csv.py`               | One-off converter from the legacy .xlsx input files to .csv.       |
| `connector_forces_portable.py` | Reference version of the connector-force routine, shared by all.   |


3. SOFTWARE DEPENDENCIES ---------------------------------------------------------------------------------------
Python >= 3.9. All dependencies are listed in requirements.txt at the
repository root:

    pip install -r requirements.txt

The analysis scripts require:

numpy      - https://pypi.org/project/numpy/        (>= 1.20; 1.x and 2.x both supported)
scipy      - https://pypi.org/project/scipy/        (>= 1.5)
pandas     - https://pypi.org/project/pandas/       (>= 1.0)
matplotlib - https://pypi.org/project/matplotlib/   (>= 3.3)

openpyxl is NO LONGER a runtime dependency. It is needed only to run
xlsx_to_csv.py once, if converting legacy Excel input files, and is
therefore commented out in requirements.txt:

openpyxl   - https://pypi.org/project/openpyxl/     (optional, conversion only)

The lower bounds are deliberately loose: only long-established APIs are
used. requirements.txt also records the exact versions the released code was
developed and verified against (Python 3.12.3, numpy 2.4.4, scipy 1.17.1,
pandas 3.0.2, matplotlib 3.10.8), for anyone who needs to reproduce that
environment rather than the loosest compatible one.

Note on NumPy 2.0: np.trapz was removed and replaced by np.trapezoid. Each
FUNCTIONS module defines an alias resolving to whichever exists, so both
major versions work. (Before this was fixed the code could not run on ANY
version of NumPy: the function files called np.trapz while the scripts
called np.trapezoid.)


4. STRUCTURE AND EXECUTION ---------------------------------------------------------------------------------------
Each method folder contains:

| File                                | Role                                                                  |
| ----------------------------------- | --------------------------------------------------------------------- |
| `main.py`                           | Prints a reminder of how to use the folder. Not a launcher.           |
| `METHOD_SCRIPT_LOADINGMODE.py`      | The script to run, one per loading mode.                              |
| `METHOD_FUNCTIONS.py`               | Shared functions (parameters, stresses, connector forces).            |
| `data_LAM.csv`, `data_CON.csv`      | Input data (see section 5).                                           |

To run a calculation, execute the script matching the loading mode from
inside its method folder, e.g.:

    cd GIRHAMMAR
    python GI_SCRIPT_UL.py

The input files are read from the current working directory, so the script
must be launched from its own folder. Output folders (RES_all,
RES_interface, RES_stresses) are created automatically.

Loading modes available per method:

| Mode      | GIRHAMMAR | GI_generalised | HEIMESHOFF | HE_extended | KREUZINGER |
| --------- | --------- | -------------- | ---------- | ----------- | ---------- |
| SINE      | yes (BVP) | yes            | yes        | yes         | yes (BVP)  |
| UL        | yes (BVP) | -              | yes        | yes         | yes (BVP)  |
| 3Pts      | -         | -              | yes        | yes         | -          |
| 4Pts      | -         | -              | yes        | yes         | -          |
| FS - UL   | yes       | yes            | -          | -           | yes        |
| FS - 3Pts | yes       | yes            | -          | -           | yes        |
| FS - 4Pts | yes       | yes            | -          | -           | yes        |

Loading mode names:
- SINE      = sinusoidal loading (angular frequency = pi/L)
- UL        = uniform loading, applied as a closed-form load
- 3Pts      = point load at midspan, closed-form isostatic diagrams
- 4Pts      = two point loads, closed-form isostatic diagrams
- FS - ...  = the same load cases built from a truncated Fourier series, the
              point loads being smeared over a short patch of length l

3Pts/4Pts and FS-3Pts/FS-4Pts describe the SAME physical load case but are
not interchangeable numerically: the first pair uses the exact isostatic
M(x), T(x) and w(x) for an idealised point load, the second approximates the
point load by a patch of length l (10 mm by default) and reconstructs it
from a Fourier series. Expect a small difference under the load itself, of
the order of qm*(l/2)^2/2 on the moment (about 2e-3 relative for the
reference cases), and agreement elsewhere.


5. INPUT DATA ---------------------------------------------------------------------------------------
Input is read from two TAB-SEPARATED TEXT FILES per method folder. This
replaces the earlier Excel (.xlsx) input.

 - 'data_LAM.csv' : properties of the timber layers, one row per layer
	h [mm]     : height
	b [mm]     : width
	E [N/mm2]  : modulus of elasticity
	G [N/mm2]  : shear modulus (see the note below)

 - 'data_CON.csv' : properties of the connectors, one row per joint
	c [N/mm]   : stiffness of ONE connector
	s [mm]     : spacing between connectors along that joint

File format:
- first line = column headers, exactly as above;
- columns separated by a tabulation character;
- blank lines and lines starting with '#' are ignored, so the files can be
  commented;
- decimal mark '.' by default; pass decimal=',' to read_data() for files
  exported with a French/European locale.

Example data_LAM.csv (n = 3) and data_CON.csv (n - 1 = 2 joints):

	h	b	E	G                   c	s
	30	120	11000	650         1000	50
	30	120	11000	650         1000	50
	30	120	11000	650

Definition rules:
- Layer 1 is the BOTTOM layer, layer n the TOP one; layers must be listed in
  that order in data_LAM.csv.
- Joint 1,2 is the interface between layers 1 and 2, up to joint (n-1,n);
  data_CON.csv therefore has exactly (n-1) rows.
- Each joint may have its own spacing s: the connector-force routine handles
  joints with different connector counts.

read_data() validates the file and raises an explicit error instead of
silently returning wrong numbers, for instance:

	ValueError: Missing or non-numeric value(s) in 'data_LAM.csv' at data
	row(s) [2]. Check the separator (sep='\t') and the decimal mark
	(decimal='.').

NOTE ON G - IMPORTANT FOR COMPARISONS
G is used by KREUZINGER ONLY, where it enters the effective shear stiffness
GSef. The four other methods read G from data_LAM.csv and never use it:
their formulations contain no shear deformation of the layers themselves.
Running KREUZINGER with a realistic G (e.g. 650 MPa for spruce) while
comparing it with the other four therefore adds a physical term on one side
only, and the difference observed then mixes two causes: the methods
themselves, and that extra term. Setting G to a very large value (1e8, as in
the KREUZINGER reference data set) neutralises the layers' own shear
flexibility and puts all five methods on the same assumptions. Use a
realistic G only when running KREUZINGER on its own. For reference, going
from G = 1e8 to G = 650 increases the KREUZINGER deflection by about 1.2 %
on the reference case.

Converting legacy Excel files: run xlsx_to_csv.py once, from the repository
root with --recursive, or inside a method folder:

    python xlsx_to_csv.py --recursive

Internal parameters (in the script): under the INPUT DATA section of each
script, the user specifies the beam length, the loading parameters (type,
magnitude, position) and the section where stresses are evaluated. For the
FS modes of KREUZINGER, the Fourier truncation N_fs and the grid size
n_points are derived automatically from L_beam and l - do not hard-code
them, see README_KR.txt section 3.


6. OUTPUT AND RESULTS ---------------------------------------------------------------------------------------
Results are automatically exported into dedicated subfolders, tab-separated, in .csv format:

| Folder          | Content                                                 | Typical files                                                                                                       |
| --------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `RES_all`       | Global data: loads, internal forces, and displacements. | `all_METHOD.csv`                                                                                                     |
| `RES_interface` | Connector and interface forces.                         | `Ts_METHOD.csv` (interface shear flow [N/mm]), `Fconn_METHOD.csv` (individual connector forces).                     |
| `RES_stresses`  | Normal and shear stresses within the layers.            | `stresses_METHOD_SIGMA_zglobal.csv`, `..._TAU_zglobal.csv` (at the selected section), `..._SIGMA_MAX_L.csv`, `..._TAU_MAX_L.csv` (maxima along the span). |

Fconn_METHOD.csv holds one 'x_pos_i,j [mm]' / 'F_conn_i,j [N]' column pair
per joint, since joints with different spacing have different connector
positions; shorter joints are padded with NaN. Each script also prints, per
joint, the connector force of largest magnitude - signed, because the
interface shear flow changes sign along the span and a plain maximum would
miss the critical connector on the other half.

COMMON SIGN CONVENTIONS
All five methods now export their per-layer quantities with the SAME
conventions, so the all_*.csv files can be compared column by column:
	w   < 0 downwards (KREUZINGER exports w = -w_a, i.e. positive downwards)
	M   > 0 in sagging
	M_i > 0 for a positive global M
	N_i > 0 in tension, so the bottom layer is in tension under sagging
	sigma = -M_i*z/I_i  (compression at the top under sagging)
This required aligning HEIMESHOFF, which used the opposite pair of
conventions (negative M_i with sigma = +z*M_i/I_i). The stresses were
correct either way - the two sign flips cancelled - but the M_i column had
the opposite sign from the other methods, which is exactly the kind of trap
a comparison repository must not contain.

CONSISTENCY CHECKS
Every script ends with one or two automatic checks and prints OK or KO:
- sectional equilibrium, M(x) = sum_i M_i(x) + sum_i N_i(x)*(zG - z_i),
  together with sum_i N_i = 0 (GIRHAMMAR_generalised, HEIMESHOFF,
  HEIMESHOFF_extended);
- global reaction, the shear force at the supports against the total applied
  load (GIRHAMMAR, KREUZINGER);
- equality of the two equivalent beams' deflections, w_A = w_B (KREUZINGER).
A KO is a signal that the numerical settings, not the data, need attention -
the method READMEs explain what drives each one.


7. VALIDATION ---------------------------------------------------------------------------------------
The methods overlap in their domains of validity, which gives independent
cross-checks. These are the strongest evidence available that the
implementations are correct, and re-running them is the recommended
regression test after any modification:

| Cross-check                                     | Agreement      |
| ----------------------------------------------- | -------------- |
| GIRHAMMAR BVP vs GIRHAMMAR modal (UL, n = 2)    | 1.2e-08        |
| GI_generalised vs closed-form Girhammar (SINE)  | 1.3e-16        |
| GI_generalised vs HEIMESHOFF, n = 2 and 3       | EIeff 1.5e-16, N_i 2.1e-16 |
| HE_extended vs HEIMESHOFF, n = 2 and 3          | EIeff 1.5e-16, N_i 2.1e-16 |

Note that GI_generalised reproduces the closed-form Girhammar solution to
machine precision on the SINE case, while GIRHAMMAR's own BVP solver agrees
only to 3.0e-06 - the latter being limited by its solve_bvp tolerance, not
by the formulation. For a single sine mode, GI_generalised is the more
accurate of the two.

Sectional equilibrium and, where applicable, tau continuity across every
interface hold to machine precision (< 1e-12) for all methods, for symmetric
and asymmetric layups alike.

Physical bounds: for any loading, the deflection must lie between the
no-composite value (layers acting independently, sum of EI_i) and the
full-composite value. This is checked in the method READMEs and is a quick
sanity test on any new data set.


8. KNOWN LIMITATIONS ---------------------------------------------------------------------------------------
- Simply supported beam only; linear elasticity; small displacements;
  constant connector spacing along a given joint; no axial load (the F term
  present in the GIRHAMMAR formulation is set to 0 in every script).
- HEIMESHOFF is limited to n = 2 or 3 by this implementation, not by the
  gamma-method itself; use HEIMESHOFF_extended beyond that.
- GIRHAMMAR's FS modes rely on a piecewise polynomial smoothing of the
  derivatives whose validity requires the loaded patch length l to stay
  small compared with 1/alpha (about 314 mm for the reference data). See
  README_GI.txt section 3.B.
- KREUZINGER's FS modes are sensitive to the Fourier truncation and grid
  size; both are now derived from the geometry, but a run with hand-forced
  values can be silently wrong. See README_KR.txt section 3, Step 1.
- The gamma factors of HEIMESHOFF and HEIMESHOFF_extended are derived under
  a sinusoidal load assumption and reused as an approximation for the other
  loading modes.


9. LICENSE AND CITATION ---------------------------------------------------------------------------------------
These scripts are released under the BSD 3-Clause License. See the LICENSE
file at the repository root for the full text. In short, anyone may use,
modify and redistribute them, including commercially, provided that the
copyright notice and the disclaimer are preserved, and that the names of the
author and of the University of Mons are not used to endorse derived
products without prior written permission.

This licence is compatible with every dependency of the project, all of
which are permissive: NumPy, SciPy and pandas are BSD-3-Clause, openpyxl is
MIT, and matplotlib uses a BSD-compatible PSF-style licence. None of them
imposes any condition on the licence of code that merely imports them.

If you use these scripts in your research, please cite them:

    Duyck, A. (2026). Analytical Modeling Scripts of Layered Composite
    Timber Beams. University of Mons, Department of Structural Mechanics -
    Timber Research Group. https://doi.org/10.5281/zenodo.22826936


10. ACKNOWLEDGEMENTS ---------------------------------------------------------------------------------------
The analytical formulations implemented here, the modelling choices and the
original scripts are the work of the author.

Claude Opus 5 (Anthropic) was used as an assistant during the review and
consolidation of the code base prior to publication. That assistance covered:
- systematic review and debugging of the five methods, including the
  identification of blocking errors (an np.trapz / np.trapezoid conflict
  that prevented the code from running on any version of NumPy, an argument
  passed of the wrong type, an unpacked return value) and of several silent
  errors that produced plausible but incorrect results (inverted sign of the
  neutral-axis offsets for n = 2 in HEIMESHOFF, inverted sign of the shear
  branch masked by an abs() in KREUZINGER, an ill-posed matrix system in
  HEIMESHOFF_extended, aliasing of the Fourier series in KREUZINGER);
- numerical verification of each method against closed-form solutions,
  physical bounds and cross-checks between methods, and the design of the
  automatic consistency checks now present in every script;
- the migration of the input format from Excel to tab-separated text;
- the writing of the documentation, including this file and the five
  method-specific READMEs.

Every correction and every figure quoted in the documentation was verified
by execution before being adopted. The author reviewed and validated all
changes, and remains solely responsible for the content, the physical
assumptions and the results of these scripts.


11. REFERENCES ---------------------------------------------------------------------------------------
The main references used for the development and implementation of these scripts are:
- B. Heimeshoff, 'Zur Berechnung von Biegetragern aus nachgiebig miteinander verbundenen Querschnittsteilen im Ingenieurholzbau', Holz als Roh-und Werkstoff, vol. 45, no. 6, pp. 237-241, June 1987, doi: 10.1007/BF02616416.
- European Committee for Standardization, EN 1995-1-1:2005 Eurocode 5: Design of timber structures - Part 1-1: General - Common rules and rules for buildings, Annex B (informative), Jan. 2005.
- H. Kreuzinger, 'Platten, Scheiben und Schalen', in Bauen mit Holz, vol. 101, 1999.
- H. Kreuzinger and H. J. Blass, 'Calculation models for prefabricated wood-based loadbearing stressed skin panels for use in roofs', European Organisation for Technical Approvals (EOTA) TR 019. Accessed: Oct. 15, 2025. [Online]. Available: https://www.eota.eu/sites/default/files/uploads/Technical%20reports/tr019total.pdf
- U. A. Girhammar, 'A simplified analysis method for composite beams with interlayer slip', International Journal of Mechanical Sciences, vol. 51, no. 7, pp. 515-530, July 2009, doi: 10.1016/j.ijmecsci.2009.05.003.
- U. A. Girhammar and V. K. A. Gopu, 'Composite Beam-Columns with Interlayer Slip - Exact Analysis', J. Struct. Eng., vol. 119, no. 4, pp. 1265-1282, Apr. 1993, doi: 10.1061/(ASCE)0733-9445(1993)119:4(1265).
- L. Resch, 'Developpement d'elements de construction en bois de pays lamelles assembles par tourillons thermo-soudes', PHD Thesis, Nancy 1, 2009. Accessed: Nov. 21, 2023. [Online]. Available: https://www.theses.fr/2009NAN10142
- M. Wallner-Novak, J. Koppelhuber, and K. Pock, Cross-Laminated Timber Structural Design - Basic design and engineering principles according to Eurocode. Wien: ProHolz Austria, 2014.
- T. Bogensperger, G. Silly, and G. Schickhofer, 'Comparison of Methods of Approximate Verification Procedures for Cross Laminated Timber', holz.bau forschungs gmbh, Institute for Timber Engineering and Wood Technology, Graz University of Technology, MMSM 2.2.3 sfem_mat, Sep. 2012.


Python libraries : 
- C. R. Harris et al., 'Array programming with NumPy', Nature, vol. 585, no. 7825, pp. 357-362, Sept. 2020, doi: 10.1038/s41586-020-2649-2.
- P. Virtanen et al., 'SciPy 1.0: fundamental algorithms for scientific computing in Python', Nat Methods, vol. 17, no. 3, pp. 261-272, Mar. 2020, doi: 10.1038/s41592-019-0686-2.
- W. McKinney, 'Data Structures for Statistical Computing in Python', presented at the Python in Science Conference, Austin, Texas, 2010, pp. 56-61. doi: 10.25080/Majora-92bf1922-00a.
- J. D. Hunter, 'Matplotlib: A 2D Graphics Environment', Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, May 2007, doi: 10.1109/MCSE.2007.55.
- openpyxl: A Python library to read/write Excel 2010 xlsx/xlsm files. Accessed: Nov. 12, 2025. Available: https://openpyxl.readthedocs.io  (used by xlsx_to_csv.py only)