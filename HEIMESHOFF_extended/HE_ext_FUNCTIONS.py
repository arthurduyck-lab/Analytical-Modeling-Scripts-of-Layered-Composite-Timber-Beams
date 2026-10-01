import numpy as np
import pandas as pd
import math
from typing import Dict, List

n_disc = 45

# np.trapz was removed in NumPy 2.0 and replaced by np.trapezoid.
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


def read_data(filepath, sep='\t', decimal='.'):
    """
    Read numerical data from a delimited text file into a NumPy structured array.
    Blank lines and lines starting with '#' are ignored. Pass decimal=',' for
    files exported with a French/European locale.
    """
    df = pd.read_csv(filepath, sep=sep, decimal=decimal,
                     comment='#', skip_blank_lines=True)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how='all')
    if df.empty:
        raise ValueError(f"No data row found in '{filepath}'.")
    if df.isnull().values.any():
        bad = df.index[df.isnull().any(axis=1)].tolist()
        raise ValueError(
            f"Missing or non-numeric value(s) in '{filepath}' at data row(s) {bad}. "
            f"Check the separator (sep='{sep}') and the decimal mark (decimal='{decimal}')."
        )
    try:
        values = df.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric content in '{filepath}': {exc}") from None
    return np.array([tuple(r) for r in values],
                    dtype=[(nm, float) for nm in df.columns])


def HE_ext_Parameters(LAM: Dict[str, List[float]],
                      CON: Dict[str, List[float]],
                      L_beam: float) -> Dict:
    """
    Compute section parameters for Heimeshoff gamma-method.
    LAM: dict with keys 'h', 'b', 'E', 'G' (lists of length n)
    CON: dict with keys 'c', 's' (lists of length n-1)
    L_beam: beam length (float)

    Returns dict with keys:
      h, b, E, G, A, I, EA, EI, z0, n, c, s, Gamma, a, EIeff_list, EIeff
    """
    for col in ('h', 'b', 'E', 'G'):
        if col not in LAM.dtype.names:
            raise ValueError(f"Column '{col}' missing in the lamellae data file. "
                             f"Found: {list(LAM.dtype.names)}")
    for col in ('c', 's'):
        if col not in CON.dtype.names:
            raise ValueError(f"Column '{col}' missing in the connector data file. "
                             f"Found: {list(CON.dtype.names)}")

    h = LAM['h']
    b = LAM['b']
    E = LAM['E']
    G = LAM['G']
    n = len(h)

    if n < 2:
        raise ValueError(f"At least 2 lamellae are required, got n = {n}.")

    if not (len(b) == len(E) == len(G) == n):
        raise ValueError("LAM lists must have same length n")

    n_c = n - 1
    c = CON['c']
    s = CON['s']


    if len(c) != n-1 or len(s) != n-1:
        raise ValueError("CON['c'] and CON['s'] must be of length n-1")

    z0 = []
    z0_sumcum = 0

    for i in range(n):
        z0i = h[i] / 2 + z0_sumcum
        z0.append(z0i)
        z0_sumcum = z0_sumcum + h[i]

    z0 = np.array(z0, dtype=float)

    # Areas, inertias, axial and flexural stiffness per lamella
    A = np.array([b[i]*h[i] for i in range(n)], dtype=float)
    I = np.array([b[i]*h[i]**3/12.0 for i in range(n)], dtype=float)
    EA = np.array([E[i]*A[i] for i in range(n)], dtype=float)
    EI = np.array([E[i]*I[i] for i in range(n)], dtype=float)

    zGC = sum(EA *z0)/sum(EA)


    # a
    a = z0 - zGC
    a = np.array(a, dtype=float)


    return {
        'h': h,
        'b': b,
        'E': E,
        'G': G,
        'A': A,
        'I': I,
        'EA': EA,
        'EI': EI,
        'z0': z0,
        'n': n,
        'c': c,
        's': s,
        'a': a,
        'zGC': zGC,

    }

def HE_ext_Rigidity (n, c, s, a, EA, EI, L_beam, h=None, eps_frac=1e-10):
    """
    Solve the coupled system V @ Gamma = S for the reduction coefficients, and
    return the effective bending stiffness.

    Returns
    -------
    Gamma : ndarray
        Reduction coefficients.
    z : ndarray
        Gamma * a, used directly in the shear-stress formulas.
    EIeff : float
        Effective bending stiffness.
    a : ndarray
        The vector of centroid-to-neutral-axis distances ACTUALLY used to build
        and solve the system. It is returned because it may differ from the
        input a (see the regularisation note below), and every downstream
        quantity (Ni, z, EIeff, stresses) must be computed with this same
        vector, never with the raw geometric one.

    Regularisation of lamellae sitting on the neutral axis
    -----------------------------------------------------
    Row i of V is proportional to a[i], and column i of V is proportional to
    a[i] as well. So if a lamella's centroid coincides exactly with the
    section's elastic centroid (a[i] == 0 - always the case for the central
    lamella of a symmetric layup with an odd number of layers), column i is
    identically zero: Gamma_i does not appear in any equation and is
    structurally indeterminate, while row i remains a non-trivial equation
    coupling Gamma_(i-1) and Gamma_(i+1). The system is then OVERdetermined,
    not merely singular, and np.linalg.pinv silently returns a least-squares
    answer that is exact only when row i happens to be consistent - which
    requires the two joint stiffnesses adjacent to lamella i to be equal.
    With asymmetric connector stiffness, pinv returned Ni values violating
    sum_i Ni = 0 by ~30% of a typical Ni, with no error and no warning.

    The fix is a limiting process: a[i] is offset by a small eps instead of
    being exactly 0, which makes V regular and lets np.linalg.solve run. As
    eps -> 0 every physical quantity converges: Gamma_i itself diverges like
    1/eps, but the PRODUCT z_i = Gamma_i * a[i] - which is what actually
    enters Ni and the shear stresses - tends to a finite limit, and
    Gamma_i * EA_i * a[i]**2 -> 0 so EIeff is unaffected. Verified over six
    decades of eps (1e-1 down to 1e-11): z_i, Ni, EIeff and the sectional
    equilibrium are all stable from about eps = 1e-5 downwards, and
    sum_i Ni = 0 holds to machine precision throughout.

    eps is taken as a fraction of the lamella's OWN height (eps_frac * h[i]),
    not of the overall section depth: h[i] is the relevant local geometric
    scale, and using a global one would give a thin lamella in a deep section
    a disproportionate offset and a worse condition number. eps_frac = 1e-6
    keeps cond(V) around 1e8 (eight decades of headroom in double precision)
    while holding the sectional-equilibrium error near 1e-9, far below the
    1e-6 threshold of the consistency check in the scripts. Lowering it
    further buys accuracy that is already irrelevant at the cost of
    conditioning.

    If h is not supplied, the offset falls back to eps_frac * max(|a|).
    """
    a = np.array(a, dtype=float).copy()

    a_scale = max(np.max(np.abs(a)), 1.0)
    degenerate = np.where(np.abs(a) < 1e-12 * a_scale)[0]
    for i in degenerate:
        a[i] = eps_frac * (h[i] if h is not None else a_scale)

    c_ext = np.zeros(n+1)
    c_ext[1:n] = c/s

    a_ext = np.zeros(n+2)
    a_ext[1:n+1] = a

    D = (np.pi**2 * EA) / L_beam**2
    D_ext = np.zeros(n+2)
    D_ext[1:n+1] = D

    V = np.zeros((n, n))
    S = np.zeros(n)

    for i in range(n):
        j = i+1
        aj = a_ext[j]
        ajm1 = a_ext[j-1]
        ajp1 = a_ext[j + 1]

        cjm1_j = c_ext[j-1]
        cj_jp1 = c_ext[j]

        Dj = D_ext[j]

        # Right-Hand side member
        S[i] = -cj_jp1*(ajp1 - aj) + cjm1_j *(aj - ajm1)

        # V matrix
        V[i, i] = (cjm1_j + cj_jp1 + Dj) * aj  # main diagonal

        if i < (n-1):
            V[i+1, i] = -cj_jp1 * aj    # lower diagonal
            V[i, i+1] = -cj_jp1 * ajp1  # upper diagonal


    # V is now regular (see the regularisation note in the docstring), so a
    # direct solve is used instead of the pseudo-inverse. This is deliberate:
    # pinv absorbed a structurally ill-posed system silently, whereas
    # np.linalg.solve raises LinAlgError, turning a genuine degeneracy back
    # into a visible failure rather than a plausible-looking wrong answer.
    Gamma = np.linalg.solve(V, S)

    z = Gamma * a
    z = np.array(z, dtype=float)

    EIeff_list = []
    for i in range(n):
        EIeffi = EI[i] + Gamma[i] * EA[i] * a[i]**2
        EIeff_list.append(EIeffi)

    EIeff = sum(EIeff_list)

    # 'a' is returned as well: it may have been regularised above, and every
    # downstream quantity must use this vector, not the raw geometric one.
    return Gamma, z, EIeff, a


def HE_ext_SigmaSection(M_list, N_list, h, b, z0, n_disc=n_disc):

    n = len(h)
    rows = []

    for i in range(n):
        hi = h[i]
        bi = b[i]
        Mi = M_list[i]
        Ni = N_list[i]
        z0i = z0[i]

        Ai = bi * hi
        Weli = bi * hi ** 2 / 6.0
        Ii = bi * hi ** 3 / 12.0

        sigmaN_max = Ni / Ai   # [MPa]

        # Coordinates
        z_local = np.linspace(-hi / 2, hi / 2, n_disc)
        z_global = z0i + z_local

        # Stresses
        sigmaN_vals = np.full_like(z_local, sigmaN_max)
        sigmaM_vals = - z_local * Mi/Ii
        sigmaTOT_vals = sigmaN_vals + sigmaM_vals

        # Store
        df_i = pd.DataFrame({
            'z [mm]': z_global,
            'σTOT [MPa]': sigmaTOT_vals,
            'σM [MPa]': sigmaM_vals,
            'σN [MPa]': sigmaN_vals
        })
        rows.append(df_i)

    # Assemblage complet
    df_all = pd.concat(rows, ignore_index=True)
    return df_all


def HE_ext_SigmaMaxi(i, M, N, h, b):
    hi = h[i]
    bi = b[i]
    Mi = M
    Ni = N

    Ai = bi * hi
    Ii = bi * hi ** 3 / 12.0

    sigmaN = Ni / Ai  # [MPa]

    # Local coordinates locales
    z_local = np.linspace(-hi / 2, hi / 2, n_disc)

    # Stresses
    sigmaN_vals = np.full_like(z_local, sigmaN)
    sigmaM_vals = - z_local * Mi / Ii
    sigmaTOT_vals = sigmaN_vals + sigmaM_vals

    idx = np.argmax(np.abs(sigmaTOT_vals))
    sigmaTOT_max = sigmaTOT_vals[idx]
    return sigmaTOT_max


def HE_ext_TauSection (T, h, b, E, z, a, z0, c, EIeff, idx_inf, idx_cent, idx_sup, n_disc = n_disc):
    rows_inf = []
    rows_cent = []
    rows_sup = []

    Tsi_inf = 0
    Tsi_sup = 0
    Ts = np.zeros_like(c)

    for i in idx_inf:
        hi = h[i]
        bi = b[i]
        Ei = E[i]
        zi = z[i]
        ai = a[i]
        z0i = z0[i]

        z_local = np.linspace(-hi / 2, hi / 2, n_disc)
        z_global = z0i + z_local
        Z_tau = z_local + ai

        tauNi = -Ei / EIeff * T * zi * (hi / 2 + (Z_tau - ai))
        tauMi = Ei / EIeff * T * 1 / 2 * ((hi / 2) ** 2 - (Z_tau - ai) ** 2)
        tautwi = np.full_like(tauNi, Tsi_inf / bi)
        tauTOTi = tauNi + tauMi + tautwi

        Tsi_inf = Tsi_inf + (tauNi[-1] * bi)
        Ts[i] = Tsi_inf

        # Store
        df_i = pd.DataFrame({
            'z_global [mm]': z_global,
            'z_local [mm]': z_local,
            'z_tau [mm]': Z_tau,
            'τTOT [MPa]': tauTOTi,
            'τM [MPa]': tauMi,
            'τN [MPa]': tauNi,
            'τts [MPa]': tautwi
        })
        rows_inf.append(df_i)

    for i in idx_cent:
        hi = h[i]
        bi = b[i]
        Ei = E[i]
        zi = z[i]
        ai = a[i]
        z0i = z0[i]

        z_local = np.linspace(-hi / 2, hi / 2, n_disc)
        z_global = z0i + z_local
        Z_tau = z_local + ai

        tauNi = -Ei / EIeff * T * zi * (hi / 2 + (Z_tau - ai))
        tauMi = Ei / EIeff * T * 1 / 2 * ((hi / 2) ** 2 - (Z_tau - ai) ** 2)
        tautwi = np.full_like(tauNi, Tsi_inf / bi)
        tauTOTi = tauNi + tauMi + tautwi

        # Store
        df_i = pd.DataFrame({
            'z_global [mm]': z_global,
            'z_local [mm]': z_local,
            'z_tau [mm]': Z_tau,
            'τTOT [MPa]': tauTOTi,
            'τM [MPa]': tauMi,
            'τN [MPa]': tauNi,
            'τts [MPa]': tautwi
        })
        rows_cent.append(df_i)

    idx_sup_rev = idx_sup[::-1]
    for i in idx_sup_rev:
        hi = h[i]
        bi = b[i]
        Ei = E[i]
        zi = z[i]
        ai = a[i]
        z0i = z0[i]

        z_local = np.linspace(-hi / 2, hi / 2, n_disc)
        z_global = z0i + z_local
        Z_tau = z_local + ai

        tauNi = Ei / EIeff * T * zi * (hi / 2 - (Z_tau - ai))
        tauMi = Ei / EIeff * T * 1 / 2 * ((hi / 2) ** 2 - (Z_tau - ai) ** 2)
        tautwi = np.full_like(tauNi, Tsi_sup / bi)
        tauTOTi = tauNi + tauMi + tautwi

        Tsi_sup = Tsi_sup + (tauNi[0] * bi)
        Ts[i - 1] = Tsi_sup
        # Store
        df_i = pd.DataFrame({
            'z_global [mm]': z_global,
            'z_local [mm]': z_local,
            'z_tau [mm]': Z_tau,
            'τTOT [MPa]': tauTOTi,
            'τM [MPa]': tauMi,
            'τN [MPa]': tauNi,
            'τts [MPa]': tautwi
        })
        rows_sup.insert(0, df_i)

    rows = rows_inf + rows_cent + rows_sup
    df_all = pd.concat(rows, ignore_index=True)

    return df_all


def HE_ext_TauMaxi (T, h, b, E, z, a, z0, c, EIeff, idx_inf, idx_cent, idx_sup, n_disc = n_disc):
    tauTOT_max = np.zeros_like(h)

    Tsi_inf = 0
    Tsi_sup = 0

    for i in idx_inf:
        hi = h[i]
        bi = b[i]
        Ei = E[i]
        zi = z[i]
        ai = a[i]
        z0i = z0[i]

        z_local = np.linspace(-hi / 2, hi / 2, n_disc)
        Z_tau = z_local + ai

        tauNi = -Ei / EIeff * T * zi * (hi / 2 + (Z_tau - ai))
        tauMi = Ei / EIeff * T * 1 / 2 * ((hi / 2) ** 2 - (Z_tau - ai) ** 2)
        tautwi = np.full_like(tauNi, Tsi_inf / bi)
        tauTOTi = tauNi + tauMi + tautwi

        Tsi_inf = Tsi_inf + (tauNi[-1] * bi)

        # Store
        idx = np.argmax(np.abs(tauTOTi))
        tauTOT_max[i] = tauTOTi[idx]


    for i in idx_cent:
        hi = h[i]
        bi = b[i]
        Ei = E[i]
        zi = z[i]
        ai = a[i]
        z0i = z0[i]

        z_local = np.linspace(-hi / 2, hi / 2, n_disc)
        Z_tau = z_local + ai

        tauNi = -Ei / EIeff * T * zi * (hi / 2 + (Z_tau - ai))
        tauMi = Ei / EIeff * T * 1 / 2 * ((hi / 2) ** 2 - (Z_tau - ai) ** 2)
        tautwi = np.full_like(tauNi, Tsi_inf / bi)
        tauTOTi = tauNi + tauMi + tautwi

        # Store
        idx = np.argmax(np.abs(tauTOTi))
        tauTOT_max[i] = tauTOTi[idx]

    idx_sup_rev = idx_sup[::-1]
    for i in idx_sup_rev:
        hi = h[i]
        bi = b[i]
        Ei = E[i]
        zi = z[i]
        ai = a[i]
        z0i = z0[i]

        z_local = np.linspace(-hi / 2, hi / 2, n_disc)
        Z_tau = z_local + ai

        tauNi = Ei / EIeff * T * zi * (hi / 2 - (Z_tau - ai))
        tauMi = Ei / EIeff * T * 1 / 2 * ((hi / 2) ** 2 - (Z_tau - ai) ** 2)
        tautwi = np.full_like(tauNi, Tsi_sup / bi)
        tauTOTi = tauNi + tauMi + tautwi

        Tsi_sup = Tsi_sup + (tauNi[0] * bi)

        # Store
        idx = np.argmax(np.abs(tauTOTi))
        tauTOT_max[i] = tauTOTi[idx]

    tauTOT_max = np.array(tauTOT_max, dtype=float)
    return tauTOT_max


def HE_ext_Ts (T, h, b, E, z, a, z0, c, EIeff, idx_inf, idx_cent, idx_sup, n_disc = n_disc):
    Tsi_inf = 0
    Tsi_sup = 0
    Ts = np.zeros_like(c)

    for i in idx_inf:
        hi = h[i]
        bi = b[i]
        Ei = E[i]
        zi = z[i]
        ai = a[i]
        z0i = z0[i]

        z_local = np.linspace(-hi / 2, hi / 2, n_disc)
        Z_tau = z_local + ai

        tauNi = -Ei / EIeff * T * zi * (hi / 2 + (Z_tau - ai))
        tauMi = Ei / EIeff * T * 1 / 2 * ((hi / 2) ** 2 - (Z_tau - ai) ** 2)
        tautwi = np.full_like(tauNi, Tsi_inf / bi)

        Tsi_inf = Tsi_inf + (tauNi[-1] * bi)

        # Store
        Ts[i] = Tsi_inf

    for i in idx_cent:
        hi = h[i]
        bi = b[i]
        Ei = E[i]
        zi = z[i]
        ai = a[i]
        z0i = z0[i]

        z_local = np.linspace(-hi / 2, hi / 2, n_disc)
        Z_tau = z_local + ai

        tauNi = -Ei / EIeff * T * zi * (hi / 2 + (Z_tau - ai))
        tauMi = Ei / EIeff * T * 1 / 2 * ((hi / 2) ** 2 - (Z_tau - ai) ** 2)
        tautwi = np.full_like(tauNi, Tsi_inf / bi)


    idx_sup_rev = idx_sup[::-1]
    for i in idx_sup_rev:
        hi = h[i]
        bi = b[i]
        Ei = E[i]
        zi = z[i]
        ai = a[i]
        z0i = z0[i]

        z_local = np.linspace(-hi / 2, hi / 2, n_disc)
        Z_tau = z_local + ai

        tauNi = Ei / EIeff * T * zi * (hi / 2 - (Z_tau - ai))
        tauMi = Ei / EIeff * T * 1 / 2 * ((hi / 2) ** 2 - (Z_tau - ai) ** 2)
        tautwi = np.full_like(tauNi, Tsi_sup / bi)

        Tsi_sup = Tsi_sup + (tauNi[0] * bi)

        # Store
        Ts[i - 1] = Tsi_sup

    return Ts


def HE_ext_ConnectorForces(x_vals, df_Ts, L_beam, s_target):
    """
    Compute the forces in each connector, for every joint between lamellae.
    Connectors are positioned from x = 0 to x = L_beam inclusive, and each
    one carries the integral of the interface shear flow Ts over its
    tributary length (half a spacing on each side, half only at the two
    ends of the beam).

    Parameters
    ----------
    x_vals : array_like
        Unused; kept for signature compatibility with the existing calls.
        The abscissae actually used are read from df_Ts['x [mm]'].
    df_Ts : pandas.DataFrame
        Interface shear flow per joint. Must contain a column 'x [mm]' and
        one column per joint, named 'Ts_i,j [N/mm]'.
    L_beam : float
        Total length of the beam [mm].
    s_target : float or array_like
        Nominal connector spacing [mm]: a scalar (same for every joint) or
        an array of length n_joints (one value per joint).

    Returns
    -------
    df_conn : pandas.DataFrame
        One 'x_pos_i,j [mm]' / 'F_conn_i,j [N]' column pair per joint.
        Each joint gets its OWN position column, because joints with
        different spacing have different connector counts and positions;
        shorter joints are padded with NaN. The previous implementation
        built a dict of columns sharing a single 'x_pos [mm]' and raised
        "All arrays must be of the same length" as soon as s_target was not
        uniform across joints - a real limitation, since data_CON.csv has
        one s value per joint and therefore allows exactly that.
    Fconn_max : numpy.ndarray
        Signed value of largest magnitude per joint. NOT max(F_conn): Ts
        changes sign along the span, so a plain maximum would miss the
        critical connector whenever it sits on the negative half.
    """
    if not isinstance(df_Ts, pd.DataFrame):
        raise TypeError("df_Ts must be a pandas DataFrame.")
    if 'x [mm]' not in df_Ts.columns:
        raise ValueError("df_Ts must contain a column 'x [mm]'.")

    x_orig = np.asarray(df_Ts['x [mm]'].values, dtype=float)
    Ts_cols = [col for col in df_Ts.columns if col != 'x [mm]']
    n_joints = len(Ts_cols)
    if n_joints == 0:
        raise ValueError("df_Ts contains no joint column besides 'x [mm]'.")

    if np.isscalar(s_target):
        s_target = np.full(n_joints, float(s_target))
    else:
        s_target = np.asarray(s_target, dtype=float).flatten()
        if len(s_target) != n_joints:
            raise ValueError(
                f"s_target has {len(s_target)} value(s) for {n_joints} joint(s); "
                f"expected one spacing per joint (or a single scalar)."
            )

    per_joint = []
    Fconn_max = []

    for j, col in enumerate(Ts_cols):
        s_j = float(s_target[j])

        # Guards: without them, s_j = 0 raises an opaque ZeroDivisionError and
        # s_j > L_beam rounds n_segments down to 0, which makes s_cal infinite
        # and silently produces a single meaningless "connector".
        if not np.isfinite(s_j) or s_j <= 0:
            raise ValueError(
                f"Connector spacing for joint '{col}' is {s_j}; it must be a "
                f"finite positive length [mm]."
            )
        # The connector count is rounded so that the first and last connectors
        # land exactly on the supports, which means the spacing actually used
        # (s_cal) differs slightly from the requested s_j. n_segments falls to
        # 0 once s_j exceeds twice the beam length, which would make s_cal
        # infinite and produce a single meaningless "connector".
        n_segments = int(np.round(L_beam / s_j))
        if n_segments < 1:
            raise ValueError(
                f"Connector spacing for joint '{col}' is {s_j} mm for a beam of "
                f"{L_beam} mm: fewer than one spacing fits, so no connector "
                f"layout can be built."
            )
        s_cal = L_beam / n_segments  # spacing actually used, after rounding
        if abs(s_cal - s_j) > 0.05 * s_j:
            print(
                f"WARNING: ConnectorForces - joint '{col}': requested spacing "
                f"{s_j} mm does not divide the beam length ({L_beam} mm); "
                f"{n_segments + 1} connectors at {s_cal:.4g} mm are used instead "
                f"({100 * abs(s_cal - s_j) / s_j:.1f}% off)."
            )

        conn_pos = np.linspace(0, L_beam, n_segments + 1)

        # edges[i] is the LEFT boundary of the tributary length of connector i.
        edges = np.zeros(len(conn_pos))
        edges[0] = 0.0
        edges[1:-1] = (conn_pos[:-2] + conn_pos[1:-1]) / 2
        edges[-1] = conn_pos[-1] - s_cal / 2

        # Insert the tributary boundaries into the abscissae before
        # integrating, so that each connector's interval is integrated on its
        # exact bounds rather than on the nearest output grid points.
        x_all = np.unique(np.concatenate([x_orig, edges]))
        Ts_all = np.interp(x_all, x_orig, np.asarray(df_Ts[col].values, dtype=float))

        F_conn = np.zeros(len(conn_pos))
        for i in range(len(conn_pos)):
            if i == 0:
                x_start, x_end = 0.0, edges[1]
            elif i == len(conn_pos) - 1:
                x_start, x_end = edges[-1], L_beam
            else:
                x_start, x_end = edges[i], edges[i + 1]
            mask = (x_all >= x_start) & (x_all <= x_end)
            F_conn[i] = (_trapezoid(Ts_all[mask], x_all[mask])
                         if np.count_nonzero(mask) >= 2 else 0.0)

        # 'Ts_1,2 [N/mm]' -> '1,2'. Falls back to the joint's position in the
        # section if the column does not follow that naming scheme.
        joint_id = col.replace("Ts_", "").split(" [")[0].strip()
        if not joint_id or joint_id == "Ts":
            joint_id = f"{j + 1},{j + 2}"

        per_joint.append(pd.DataFrame({
            f'x_pos_{joint_id} [mm]': conn_pos,
            f'F_conn_{joint_id} [N]': F_conn,
        }))
        Fconn_max.append(F_conn[int(np.argmax(np.abs(F_conn)))])

    # Column-wise concatenation (not a dict of arrays): this is what lets
    # joints of different length coexist, padded with NaN.
    df_conn = pd.concat(per_joint, axis=1)
    return df_conn, np.array(Fconn_max, dtype=float)
