import numpy as np
import pandas as pd
import math
from typing import Dict, List

n_disc = 45

# np.trapz was removed in NumPy 2.0 and replaced by np.trapezoid.
# This alias keeps the scripts runnable on both NumPy 1.x and 2.x.
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


def read_data(filepath, sep='\t', decimal='.'):
    """
    Read numerical data from a delimited text file into a NumPy structured array.

    First line holds the column names ('h', 'b', 'E', 'G' for the lamellae;
    'c', 's' for the connectors), following lines one record each. Blank lines
    and lines starting with '#' are ignored. Pass decimal=',' for files
    exported with a French/European locale.

    Columns are accessed as DATA['h'], DATA['b'], ... exactly as before.
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


def GI_gen_Parameters(LAM: Dict[str, List[float]],
                      CON: Dict[str, List[float]],
                      L_beam: float) -> Dict:
    """
    Compute section parameters for GIRHAMMAR generalised method.
    LAM: dict with keys 'h', 'b', 'E', 'G' (lists of length n)
    CON: dict with keys 'c', 's' (lists of length n-1)
    L_beam: beam length (float)

    Returns dict with keys:
      h, b, E, G, A, I, EA, EI, z0, n, c, s, K, v, vi_ip1, vim1_i
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

    K = np.array([c[i]/s[i] for i in range(n-1)], dtype=float)
    Kim1_i = []
    Ki_ip1 = []

    for i in range(n):
        if i == 0:
            Kim1_i.append(0)
            Ki_ip1.append(K[i])

        elif i == (n-1):
            Kim1_i.append(K[i-1])
            Ki_ip1.append(0)
        else :
            Kim1_i.append(K[i-1])
            Ki_ip1.append(K[i])

    Kim1_i = np.array(Kim1_i, dtype=float)
    Ki_ip1 = np.array(Ki_ip1, dtype=float)



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

    v = np.array([h[i]/2 for i in range(n)], dtype=float)
    vim1_i = []
    vi_ip1 = []
    for i in range(n):
        if i == 0:
            vim1_i.append(v[i])
            vi_ip1.append(v[i] + v[i+1])

        elif i == (n-1):
            vim1_i.append(v[i-1] + v[i])
            vi_ip1.append(v[i])
        else :
            vim1_i.append(v[i-1] + v[i])
            vi_ip1.append(v[i] + v[i+1])

    vim1_i = np.array(vim1_i, dtype=float)
    vi_ip1 = np.array(vi_ip1, dtype=float)


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
        'K': K,
        'Kim1_i': Kim1_i,
        'Ki_ip1': Ki_ip1,
        'v': v,
        'vim1_i': vim1_i,
        'vi_ip1': vi_ip1
    }

def GI_gen_MODAL (n, idx_s, EA, EI, q0, omega, v, vi_ip1, vim1_i, Ki_ip1, Kim1_i ):
    # System matrix and load vector
    A_sys = np.zeros((n + 1, n + 1), dtype=float)
    B_sys = np.zeros((n + 1, 1), dtype=float)

    # Equation A
    for i in range(n):
        A_sys[0, i] = EA[i]     # EA[i] * u0i
    A_sys[0, n] = 0.0           # 0 * w0
    B_sys[0, 0] = 0.0

    # Equation B
    row = 1
    for i in range(n):
        if i != idx_s :
            A_sys[row, i] += EA[i] * omega**2 + Kim1_i[i] + Ki_ip1[i]
            B_sys[row, 0] += 0.0

            if i - 1 >= 0:
                A_sys[row, i - 1] += -Kim1_i[i]
            if i + 1 < n:
                A_sys[row, i + 1] += -Ki_ip1[i]

            A_sys[row, n] += omega * (-Ki_ip1[i] * vi_ip1[i] + Kim1_i[i] * vim1_i[i])

            row += 1

    # Equation C
    sum_EI = np.sum(EI)

    A_sys[-1, :] = 0.0

    for i in range(n):
        # Left-hand side neighbouring term (i-1)
        if i - 1 >= 0:
            A_sys[-1, i - 1] += v[i] * Kim1_i[i] * omega

        # Central term (i)
        A_sys[-1, i] += v[i] * (Ki_ip1[i] * omega - Kim1_i[i] * omega)

        # Left-hand side neighbouring term (i+1)
        if i + 1 < n:
            A_sys[-1, i + 1] += -v[i] * Ki_ip1[i] * omega


    # Contribution of the term w0
    A_sys[-1, n] = -omega**4 * sum_EI - omega**2 * np.sum(v * (Ki_ip1 * vi_ip1 + Kim1_i * vim1_i))

    # Second part of the equation
    B_sys[-1, 0] = q0

    # Solving
    sol = np.linalg.pinv(A_sys) @ B_sys

    u0 = sol[:-1].flatten()
    w0 = sol[-1, 0]

    return (A_sys, B_sys, u0, w0)


def GI_gen_SigmaSection(M_list, N_list, h, b, z0, n_disc=n_disc):

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


def GI_gen_SigmaMaxi(i, M, N, h, b):
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


def GI_gen_TauSection(h,b,A,I, z0,dN, dM, idx_inf, idx_cent, idx_sup, n_disc=n_disc) :
    idx_infcent = np.concatenate((idx_inf, idx_cent))

    rows_inf = []
    rows_sup = []
    sum_dN_inf = 0
    sum_dN_sup = 0

    for i in idx_infcent:
        hi = h[i]
        bi = b[i]
        Ai = A[i]
        z0i = z0[i]

        Ii = I[i]


        z_local = np.linspace(-hi / 2, hi / 2, n_disc)
        z_global = z0i + z_local

        tauNi = 1/Ai * dN[i] * (hi/2+z_local)
        tauMi = 1/Ii * dM[i] * 1/2*((hi/2)**2 - (z_local)**2)
        tautwi = np.full_like(tauNi, 1/bi * sum_dN_inf)
        tauTOTi = tauNi + tauMi + tautwi

        sum_dN_inf += dN[i]

        # Store
        df_i = pd.DataFrame({
            'z_global [mm]': z_global,
            'z_local [mm]': z_local,
            'τTOT [MPa]': tauTOTi,
            'τM [MPa]': tauMi,
            'τN [MPa]': tauNi,
            'τts [MPa]': tautwi
        })

        rows_inf.append(df_i)

    idx_sup_rev = idx_sup[::-1]
    for i in idx_sup_rev:
        hi = h[i]
        bi = b[i]
        Ai = A[i]
        z0i = z0[i]
        Ii = I[i]


        z_local = np.linspace(-hi / 2, hi / 2, n_disc)
        z_global = z0i + z_local

        tauNi = - 1/Ai * dN[i] * (hi/2 - z_local)
        tauMi = 1/Ii * dM[i] * 1/2*((hi/2)**2 - (z_local)**2)
        tautwi = np.full_like(tauNi, 1/bi * sum_dN_sup)
        tauTOTi = tauNi + tauMi + tautwi

        sum_dN_sup += -dN[i]

        # Store
        df_i = pd.DataFrame({
            'z_global [mm]': z_global,
            'z_local [mm]': z_local,
            'τTOT [MPa]': tauTOTi,
            'τM [MPa]': tauMi,
            'τN [MPa]': tauNi,
            'τts [MPa]': tautwi
        })
        rows_sup.insert(0, df_i)


    rows = rows_inf + rows_sup
    df_all = pd.concat(rows, ignore_index=True)

    return df_all


def GI_gen_TauMaxi(h,b,A,I, z0,dN, dM, idx_inf, idx_cent, idx_sup, n_disc=n_disc) :
    idx_infcent = np.concatenate((idx_inf, idx_cent))

    tauTOT_max = np.zeros(len(h), dtype=float)

    sum_dN_inf = 0
    sum_dN_sup = 0

    for i in idx_infcent:
        hi = h[i]
        bi = b[i]
        Ai = A[i]
        z0i = z0[i]

        Ii = I[i]


        z_local = np.linspace(-hi / 2, hi / 2, n_disc)
        z_global = z0i + z_local

        tauNi = 1/Ai * dN[i] * (hi/2+z_local)
        tauMi = 1/Ii * dM[i] * 1/2*((hi/2)**2 - (z_local)**2)
        tautwi = np.full_like(tauNi, 1/bi * sum_dN_inf)
        tauTOTi = tauNi + tauMi + tautwi

        sum_dN_inf += dN[i]

        # Store
        idx = np.argmax(np.abs(tauTOTi))
        tauTOT_max[i] = tauTOTi[idx]


    idx_sup_rev = idx_sup[::-1]
    for i in idx_sup_rev:
        hi = h[i]
        bi = b[i]
        Ai = A[i]
        z0i = z0[i]
        Ii = I[i]


        z_local = np.linspace(-hi / 2, hi / 2, n_disc)
        z_global = z0i + z_local

        tauNi = - 1/Ai * dN[i] * (hi/2 - z_local)
        tauMi = 1/Ii * dM[i] * 1/2*((hi/2)**2 - (z_local)**2)
        tautwi = np.full_like(tauNi, 1/bi * sum_dN_sup)
        tauTOTi = tauNi + tauMi + tautwi

        sum_dN_sup += -dN[i]

        # Store
        idx = np.argmax(np.abs(tauTOTi))
        tauTOT_max[i] = tauTOTi[idx]


    tauTOT_max = np.array(tauTOT_max, dtype=float)
    return tauTOT_max


def GI_gen_Ts (h,b,A,I, z0,dN, dM, idx_inf, idx_cent, idx_sup, n_disc=n_disc) :

    Ts = np.zeros((len(h)-1))

    sum_dN_inf = 0
    sum_dN_sup = 0

    for i in idx_inf:
        hi = h[i]
        bi = b[i]
        Ai = A[i]
        z0i = z0[i]

        Ii = I[i]


        z_local = np.linspace(-hi / 2, hi / 2, n_disc)
        z_global = z0i + z_local

        tauNi = 1/Ai * dN[i] * (hi/2+z_local)
        tauMi = 1/Ii * dM[i] * 1/2*((hi/2)**2 - (z_local)**2)
        tautwi = np.full_like(tauNi, 1/bi * sum_dN_inf)
        tauTOTi = tauNi + tauMi + tautwi

        sum_dN_inf += dN[i]

        # Store
        Ts[i] = sum_dN_inf



    idx_sup_rev = idx_sup[::-1]
    for i in idx_sup_rev:
        hi = h[i]
        bi = b[i]
        Ai = A[i]
        z0i = z0[i]
        Ii = I[i]


        z_local = np.linspace(-hi / 2, hi / 2, n_disc)
        z_global = z0i + z_local

        tauNi = - 1/Ai * dN[i] * (hi/2 - z_local)
        tauMi = 1/Ii * dM[i] * 1/2*((hi/2)**2 - (z_local)**2)
        tautwi = np.full_like(tauNi, 1/bi * sum_dN_sup)
        tauTOTi = tauNi + tauMi + tautwi

        sum_dN_sup += -dN[i]

        # Store
        Ts[i -1] = sum_dN_sup


    Ts = np.array(Ts, dtype=float)
    return Ts


def GI_gen_ConnectorForces(x_vals, df_Ts, L_beam, s_target):
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
