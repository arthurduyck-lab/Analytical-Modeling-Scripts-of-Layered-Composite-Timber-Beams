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


def HE_Parameters(LAM: Dict[str, List[float]],
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

    z0 = np.array(z0,dtype = float)

    # Areas, inertias, axial and flexural stiffness per lamella
    A = np.array([b[i] * h[i] for i in range(n)], dtype=float)
    I = np.array([b[i] * h[i] ** 3 / 12.0 for i in range(n)], dtype=float)
    EA = np.array([E[i] * A[i] for i in range(n)], dtype=float)
    EI = np.array([E[i] * I[i] for i in range(n)], dtype=float)

    # Gamma : one scalar per lamella
    Gamma = [1.0]*n

    if n == 2:
        Gamma[0] = 1.0 / (1.0 + (math.pi**2 * EA[0] * s[0]) / (c[0] * L_beam**2))
        Gamma[1] = 1.0
    elif n == 3:
        Gamma[0] = 1.0 / (1.0 + (math.pi**2 * EA[0] * s[0]) / (c[0] * L_beam**2))
        Gamma[1] = 1.0
        Gamma[2] = 1.0 / (1.0 + (math.pi**2 * EA[2] * s[1]) / (c[1] * L_beam**2))
    else:
        raise ValueError("n must be 2 or 3 for this implementation")

    Gamma = np.array(Gamma,dtype = float)

    # a : distance between the centroid of lamella i and the neutral axis
    # of the composite section (EN 1995-1-1 Annex B.6).
    #
    # a[1] is the SIGNED offset of the reference lamella from the neutral
    # axis, stored with the opposite sign of the a2 of EN 1995-1-1 so that
    # the "+ for lower lamellae / - for upper lamellae" rule applied in the
    # scripts gives the correct sign of Ni directly. a[0] and a[2] are the
    # (positive) EN 1995-1-1 magnitudes a1 and a3.
    #
    # The n = 2 case is the n = 3 formula with the third layer removed
    # (gamma3*EA3 = 0). Writing it as a separate expression, as was done
    # previously, silently used the opposite sign convention for a[0] and
    # a[1] and inverted every Ni: the bottom lamella came out in
    # compression under a sagging moment, and sectional equilibrium was off
    # by 50 to 76 %. EIeff was unaffected (a appears squared there), so the
    # deflection was correct and the error was invisible on w.
    a = [0.0] * n
    gEA = [Gamma[i] * EA[i] for i in range(n)]
    denom = 2.0 * sum(gEA)

    if n == 2:
        a[1] = (0.0 - gEA[0] * (h[0] + h[1])) / denom
        a[0] = (h[0] + h[1]) / 2.0 + a[1]
    else:  # n == 3, lamella 2 is the reference
        a[1] = (gEA[2] * (h[1] + h[2]) - gEA[0] * (h[0] + h[1])) / denom
        a[0] = (h[0] + h[1]) / 2.0 + a[1]
        a[2] = (h[1] + h[2]) / 2.0 - a[1]

    a = np.array(a,dtype = float)

    # EI effective contributions and total
    EIeff_list = [EI[i] + Gamma[i] * EA[i] * (a[i]**2) for i in range(n)]
    EIeff = sum(EIeff_list)

    z=[]
    for i in range(n):
        zi = Gamma[i] * a[i]
        z.append(zi)

    z = np.array(z,dtype = float)

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
        'Gamma': Gamma,
        'a': a,
        'z': z,
        'EIeff_list': EIeff_list,
        'EIeff': EIeff
    }


def HE_SigmaSection(M_list, N_list, h, b, z0, n_disc=n_disc):

    n = len(h)
    rows = []

    for i in range(n):
        hi = h[i]
        bi = b[i]
        Mi = M_list[i]
        Ni = N_list[i]
        z0i = z0[i]

        Ai = bi * hi
        Ii = bi * hi ** 3 / 12.0

        sigmaN_max = Ni / Ai   # [MPa]

        # Coordinates
        z_local = np.linspace(-hi / 2, hi / 2, n_disc)
        z_global = z0i + z_local

        # Stresses
        sigmaN_vals = np.full_like(z_local, sigmaN_max)
        sigmaM_vals = -z_local * Mi / Ii   # sigma = -M*z/I, as in GIRHAMMAR
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


def HE_SigmaMaxi(i, M, N, h, b):
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
    sigmaM_vals = -z_local * Mi / Ii   # sigma = -M*z/I, as in GIRHAMMAR
    sigmaTOT_vals = sigmaN_vals + sigmaM_vals

    idx = np.argmax(np.abs(sigmaTOT_vals))
    sigmaTOT_max = sigmaTOT_vals[idx]
    return sigmaTOT_max

def HE_tauwjk(j, k, b, E, A, Gamma, a, EIeff, T):
    tauwjk = T * E[j]*A[j]*Gamma[j]*a[j]/(EIeff*b[j])
    return tauwjk


def HE_TauSection(T, E, EIeff, h, b, A, Gamma, a, z, z0, idx_inf, idx_cent, idx_sup, n_disc = n_disc):
    n = len(h)
    rows = []

    for i in range(n):
        hi = h[i]
        bi = b[i]
        Ai = bi * hi
        Ei = E[i]
        zi = z[i]
        ai = a[i]
        z0i = z0[i]

        # Lower Lamellae
        if i in idx_inf :
            z_local = np.linspace(-hi / 2, hi / 2, n_disc)
            z_global = z0i + z_local
            Z_tau = z_local - ai

            tauNi = Ei/EIeff * T * zi * (hi/2 + (Z_tau + ai))
            tauMi = Ei/EIeff * T * 1/2 * ((hi/2)**2 - (Z_tau + ai)**2)
            tauwjk = 0
            tautwi = np.full_like(tauNi, tauwjk)
            tauTOTi = tauNi + tauMi + tautwi

        # Upper Lamellae
        elif i in idx_sup :
            z_local = np.linspace(-hi / 2, hi / 2, n_disc)
            z_global = z0i + z_local
            Z_tau = z_local + ai

            tauNi = Ei/EIeff * T * zi * (hi/2 - (Z_tau - ai))
            tauMi = Ei/EIeff * T * 1/2 * ((hi/2)**2 - (Z_tau - ai)**2)
            tauwjk = 0
            tautwi = np.full_like(tauNi, tauwjk)
            tauTOTi = tauNi + tauMi + tautwi

        # Central Lamella
        elif i in idx_cent :
            z_local = np.linspace(-hi / 2, hi / 2, n_disc)
            z_global = z0i + z_local
            Z_tau = z_local - ai

            tauNi = Ei/EIeff * T * zi * (hi/2 - (Z_tau - ai))
            tauMi = Ei/EIeff * T * 1/2 * ((hi/2)**2 - (Z_tau - ai)**2)
            tauwjk = HE_tauwjk(j=(i - 1), k=i, b=b, E=E, A=A, Gamma=Gamma, a=a, EIeff=EIeff, T=T)
            tautwi = np.full_like(tauNi, tauwjk)
            tauTOTi = tauNi + tauMi + tautwi

        else:
            raise ValueError(
                f"Lamella {i} belongs to none of idx_inf / idx_cent / idx_sup. "
                f"Check how lam_ref is defined in the calling script."
            )

        # Store
        df_i = pd.DataFrame({
            'z_global [mm]': z_global,
            'z_local [mm]': z_local,
            'z_tau [mm]': Z_tau,
            'τTOT [MPa]': tauTOTi,
            'τM [MPa]': tauMi,
            'τN [MPa]': tauNi,
            'τts [MPa]': tautwi,
        })
        rows.append(df_i)

    df_all = pd.concat(rows, ignore_index=True)
    return df_all


def HE_TauMaxi(i, T, E, EIeff, h, b, A, Gamma, a, z, z0, idx_inf, idx_cent, idx_sup, n_disc = 45):
    hi = h[i]
    bi = b[i]
    Ei = E[i]
    zi = z[i]
    ai = a[i]
    z0i = z0[i]

    # Lower Lamellae
    if i in idx_inf :
        z_local = np.linspace(-hi / 2, hi / 2, n_disc)
        z_global = z0i + z_local
        Z_tau = z_local - ai

        tauNi = Ei/EIeff * T * zi * (hi/2 + (Z_tau + ai))
        tauMi = Ei/EIeff * T * 1/2 * ((hi/2)**2 - (Z_tau + ai)**2)
        twjk = 0
        tautwi = np.full_like(tauNi, twjk)
        tauTOTi = tauNi + tauMi + tautwi

        idx = np.argmax(np.abs(tauTOTi))
        tauTOT_max = tauTOTi[idx]

    # Upper Lamellae
    elif i in idx_sup :
        z_local = np.linspace(-hi / 2, hi / 2, n_disc)
        z_global = z0i + z_local
        Z_tau = z_local + ai

        tauNi = Ei/EIeff * T * zi * (hi/2 - (Z_tau - ai))
        tauMi = Ei/EIeff * T * 1/2 * ((hi/2)**2 - (Z_tau - ai)**2)
        twjk = 0
        tautwi = np.full_like(tauNi, twjk)
        tauTOTi = tauNi + tauMi + tautwi

        idx = np.argmax(np.abs(tauTOTi))
        tauTOT_max = tauTOTi[idx]


    # Central Lamella
    elif i in idx_cent :
        z_local = np.linspace(-hi / 2, hi / 2, n_disc)
        z_global = z0i + z_local
        Z_tau = z_local - ai

        tauNi = Ei/EIeff * T * zi * (hi/2 - (Z_tau - ai))
        tauMi = Ei/EIeff * T * 1/2 * ((hi/2)**2 - (Z_tau - ai)**2)
        twjk = HE_tauwjk(j=(i - 1), k=i, b=b, E=E, A=A, Gamma=Gamma, a=a, EIeff=EIeff, T=T)
        tautwi = np.full_like(tauNi, twjk)
        tauTOTi = tauNi + tauMi + tautwi

        idx = np.argmax(np.abs(tauTOTi))
        tauTOT_max = tauTOTi[idx]

    return tauTOT_max


def HE_ConnectorForces(x_vals, df_Ts, L_beam, s_target):
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