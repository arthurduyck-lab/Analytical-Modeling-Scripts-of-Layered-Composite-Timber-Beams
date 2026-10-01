import numpy as np
import pandas as pd
import math

n_disc = 45

# np.trapz was removed in NumPy 2.0 and replaced by np.trapezoid.
# This alias keeps the scripts runnable on both NumPy 1.x and 2.x.
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


def read_data(filepath, sep='\t', decimal='.'):
    """
    Read numerical data from a delimited text file into a NumPy structured array.

    The file is a plain text table whose first line holds the column names
    ('h', 'b', 'E', 'G' for the lamellae; 'c', 's' for the connectors) and whose
    following lines hold one record per row. Blank lines and lines starting with
    '#' are ignored, so the file can be commented.

    Parameters
    ----------
    filepath : str
        Path to the data file (e.g. 'data_LAM.csv').
    sep : str, optional
        Column separator. Default is the tabulation character '\\t'.
    decimal : str, optional
        Decimal separator used in the file. Use ',' if the file was exported
        from a spreadsheet configured with a French/European locale.

    Returns
    -------
    DATA : numpy.ndarray
        Structured array with one named float field per column, so that the
        columns are accessed as DATA['h'], DATA['b'], ... exactly as before.
    """
    df = pd.read_csv(
        filepath,
        sep=sep,
        decimal=decimal,
        comment='#',
        skip_blank_lines=True,
    )

    # Strip stray spaces around the column names (frequent after a manual export)
    df.columns = [str(c).strip() for c in df.columns]

    # Drop fully empty rows, then check that nothing is missing
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
        raise ValueError(
            f"Non-numeric content in '{filepath}': {exc}"
        ) from None

    dtype = [(name, float) for name in df.columns]
    DATA = np.array([tuple(row) for row in values], dtype=dtype)
    return DATA


def GI_Parameters (LAM, CON) :
    """
    Compute section parameters (areas, inertia, centroid positions)
    for exact Girhammar method.
    """

    for col in ('h', 'b', 'E', 'G'):
        if col not in LAM.dtype.names:
            raise ValueError(
                f"Column '{col}' missing in the lamellae data file. "
                f"Found: {list(LAM.dtype.names)}"
            )
    for col in ('c', 's'):
        if col not in CON.dtype.names:
            raise ValueError(
                f"Column '{col}' missing in the connector data file. "
                f"Found: {list(CON.dtype.names)}"
            )

    h = LAM['h']
    b = LAM['b']
    E = LAM['E']
    G = LAM['G']
    n = len(h)

    # The exact Girhammar formulation implemented here (EAp, zcg_full, EI_full,
    # alpha, beta, gamma) is only valid for a two-layer cross-section.
    # Use GIRHAMMAR_generalised for n > 2.
    if n != 2:
        raise ValueError(
            f"The GIRHAMMAR method is restricted to n = 2 lamellae, but the data "
            f"file defines n = {n}. Use the GIRHAMMAR_generalised method instead."
        )

    n_c = n - 1
    c = CON['c']
    s = CON['s']
    if len(c) != n_c:
        raise ValueError(
            f"{len(c)} connector row(s) for {n} lamellae; expected {n_c}."
        )

    z0 = []
    z0_sumcum = 0
    r = []
    A = []
    I = []
    EA = []
    z0EA = []
    EI = []
    for i in range(n):
        z0i = h[i]/2 + z0_sumcum
        z0.append(z0i)
        z0_sumcum = z0_sumcum + h[i]

        ri = h[i]/2.0
        r.append(ri)

        Ai = b[i]*h[i]
        A.append(Ai)

        Ii = b[i] * h[i] ** 3 / 12
        I.append(Ii)

        EiAi = E[i] * Ai
        EA.append(EiAi)

        EiIi = E[i] * Ii
        EI.append(EiIi)

        z0iEiAi = z0i * EiAi
        z0EA.append(z0iEiAi)


    zCG = sum(z0EA)/sum(EA)  # Gravity center

    r_tot = sum(r)
    EA0 = sum(EA)
    EAp = math.prod(EA)

    zcg_full = EA[1]/EA0 * r_tot

    EI0 = sum(EI)
    EI_full = EI0 + EAp * r_tot ** 2 / EA0


    K = []
    for i in range(n_c):
        Ki = c[i] / s[i]
        K.append(Ki)

    alpha_carre = K[0] * (1 / EA[0] + 1 / EA[1] + r_tot ** 2 / EI0)
    alpha = math.sqrt(alpha_carre)
    beta = K[0] * r_tot / EI0
    gamma = K[0] * (1 / EA[1] + (EA[0] * r_tot ** 2) / (EA0 * EI0))

    return h, b, E, G, EA, EI, n, r, r_tot, EA0, EAp, z0, zCG, zcg_full, EI0, EI_full, K, s, alpha, alpha_carre, beta, gamma


def GI_ConnectorForces(x_vals, df_Ts, L_beam, s_target):
    """
        Compute the forces in each connector for all joints between lamellae.
        Connectors are positioned from x=0 to x=L_beam inclusive.

        Parameters
        ----------
        df_Ts : pandas.DataFrame
            DataFrame containing shear force distributions for each joint.
            Must contain a column 'x [mm]' and one column per joint: 'Ts_i,j [MPa]'.
        L_beam : float
            Total length of the beam [mm].
        s_target : float or array_like
            Nominal spacing of the connectors [mm].
            Can be a scalar (same value for all joints) or
            an array of size n_joints.

        Returns
        -------
        df_conn : pandas.DataFrame
            Columns:
            - 'Connector [/]'
            - 'x_pos [mm]'
            - one column per joint: 'F_conn_i,j [N]'
        """
    # --- Check input DataFrame ---
    if not isinstance(df_Ts, pd.DataFrame):
        raise TypeError("df_Ts must be a pandas DataFrame.")
    if 'x [mm]' not in df_Ts.columns:
        raise ValueError("df_Ts must contain a column 'x [mm]'.")

    x_orig = df_Ts['x [mm]'].values
    Ts_cols = [col for col in df_Ts.columns if col != 'x [mm]']
    n_joints = len(Ts_cols)

    # --- Normalize connector spacing ---
    if np.isscalar(s_target):
        s_target = np.full(n_joints, s_target)
    else:
        s_target = np.array(s_target).flatten()
        if len(s_target) != n_joints:
            raise ValueError("Length of s_target must match the number of joints.")

    results = {}

    for j, col in enumerate(Ts_cols):
        # --- spacing for current joint ---
        s_j = float(np.mean(s_target[j])) if hasattr(s_target, '__len__') else float(s_target)
        n_segments = int(np.round(L_beam / s_j))
        s_cal = L_beam / n_segments

        # --- connector positions ---
        conn_pos = np.linspace(0, L_beam, n_segments + 1)

        # --- define integration edges ---
        edges = np.zeros(len(conn_pos))
        edges[0] = 0  # first connector
        # edges[i] is the LEFT boundary of the tributary length of connector i
        edges[1:-1] = (conn_pos[:-2] + conn_pos[1:-1]) / 2  # internal connectors
        edges[-1] = conn_pos[-1] - s_cal / 2  # last connector covers half-length

        # --- combine original x values with integration edges ---
        x_all = np.unique(np.concatenate([x_orig, edges]))
        Ts_all = np.interp(x_all, x_orig, df_Ts[col].values)

        # --- integrate shear forces for each connector ---
        F_conn = np.zeros(len(conn_pos))
        for i, x_c in enumerate(conn_pos):
            if i == 0:
                x_start = 0
                x_end = edges[1]
            elif i == len(conn_pos) - 1:
                x_start = edges[-1]
                x_end = L_beam
            else:
                x_start = edges[i]
                x_end = edges[i + 1]

            mask = (x_all >= x_start) & (x_all <= x_end)
            if np.sum(mask) >= 2:
                F_conn[i] = _trapezoid(Ts_all[mask], x_all[mask])
            else:
                F_conn[i] = 0.0

        # --- store results ---
        results['Connector [/]'] = np.arange(1, len(conn_pos) + 1)
        results['x_pos [mm]'] = conn_pos
        joint_id = col.replace("Ts_", "").split(" [")[0].strip()
        joint_id = joint_id if joint_id and joint_id != "Ts" else f"{j + 1},{j + 2}"
        results[f'F_conn_{joint_id} [N]'] = F_conn

    df_conn = pd.DataFrame(results)
    return df_conn


def GI_SIGMA(i, N, M, h, b, n_disc=n_disc):
    i = i-1
    hi = h[i]
    bi = b[i]

    Ai = bi * hi
    Weli = bi*hi**2/6

    sigmaN_max = N / Ai
    sigmaM_max = -M / Weli

    z_vals = np.linspace(-hi / 2, hi / 2, n_disc)
    sigmaN_vals = [sigmaN_max]*n_disc
    sigmaM_vals = z_vals/(hi/2) * sigmaM_max

    MNi_sigma_zlocal = pd.DataFrame({
        'z [mm]': z_vals,
        'σN [MPa]': sigmaN_vals,
        'σM [MPa]': sigmaM_vals,
        'σTOT [MPa]': sigmaM_vals + sigmaN_vals
    })
    return MNi_sigma_zlocal


def GI_SIGMA_global (z0, zCG, Stress1_zlocal, Stress2_zlocal):

    Stress1_zglobal = pd.DataFrame({
        'z [mm]': Stress1_zlocal['z [mm]'] + z0[0] - zCG,
        'σN [MPa]': Stress1_zlocal['σN [MPa]'],
        'σM [MPa]': Stress1_zlocal['σM [MPa]'],
        'σTOT [MPa]': Stress1_zlocal['σTOT [MPa]']
    })

    Stress2_zglobal = pd.DataFrame({
        'z [mm]': Stress2_zlocal['z [mm]'] + z0[1] - zCG,
        'σN [MPa]': Stress2_zlocal['σN [MPa]'],
        'σM [MPa]': Stress2_zlocal['σM [MPa]'],
        'σTOT [MPa]': Stress2_zlocal['σTOT [MPa]']
    })

    frames = [Stress1_zglobal, Stress2_zglobal]
    SigmaTOT_zglobal = pd.concat(frames)

    return SigmaTOT_zglobal


def GI_TAU(i, T, Ts, h, b, E, r_tot, r, EI0, n_disc=n_disc):
    i = i-1
    hi = h[i]
    bi = b[i]
    Ei = E[i]
    ri = r[i]

    Ai = bi * hi


    z_vals = np.linspace(-hi / 2, hi / 2, n_disc)
    tau_vals = (T - Ts * r_tot) * Ei * (ri**2 - z_vals**2)/(2*EI0) + (Ts *(ri + z_vals))/Ai

    taui_zlocal = pd.DataFrame({
        'z [mm]': z_vals,
        'τ [MPa]': tau_vals
    })
    return taui_zlocal


def GI_TAU_global (z0, zCG, Stress1_zlocal, Stress2_zlocal):

    Stress1_zglobal = pd.DataFrame({
        'z [mm]': Stress1_zlocal['z [mm]'] + z0[0] - zCG,
        'τ [MPa]': Stress1_zlocal['τ [MPa]']
    })

    Stress2_zglobal = pd.DataFrame({
        'z [mm]': -Stress2_zlocal['z [mm]'] + z0[1] - zCG,
        'τ [MPa]': Stress2_zlocal['τ [MPa]'],
    })

    frames = [Stress1_zglobal, Stress2_zglobal]
    SigmaTOT_zglobal = pd.concat(frames)

    return SigmaTOT_zglobal




def GI_MODAL_from_FS(x_vals, qm, L, Ct, N, alpha, EI0, EI_full):
    """
    Reconstruct w and its derivatives from Fourier sine series coefficients Ct
    Ct: array of length N with Ct[t-1] = cos(t*pi*d1/L)-cos(t*pi*(d1+l)/L)+...
    Series term index t = 1..N
    q(x)  = (2/pi) * qm * sum_{t=1..N} ( Ct[t-1] / t ) * sin(k_t x)
    """
    x = x_vals
    t = np.arange(1, N+1)
    k = t * np.pi / L             # vector shape (N,)
    # amplitude A_t of sin(k x) in q(x)
    A_t = (2.0/np.pi) * qm * (Ct / t)   # shape (N,)
    # coefficient S_t = k^2/EI0 + alpha^2/EI_full
    S_t = (k**2) / EI0 + (alpha**2) / EI_full
    denom = (k**4) * (k**2 + alpha**2)   # shape (N,)
    # avoid division by zero (t starts at 1 so fine). guard nonetheless:
    denom[denom == 0] = np.finfo(float).eps

    # modal amplitude for w: B_t
    B_t = A_t * S_t / denom   # shape (N,)

    # now sum contributions. use broadcasting: sin_arg shape (N, len(x))
    arg = np.outer(k, x)           # shape (N, nx)
    sin_arg = np.sin(arg)
    cos_arg = np.cos(arg)

    # w(x)
    w = np.sum((B_t[:,None]) * sin_arg, axis=0)

    # derivatives:
    # w'  = sum B_t * k * cos(kx)
    w1 = np.sum((B_t * k)[:,None] * cos_arg, axis=0)
    # w'' = sum -B_t * k^2 * sin(kx)
    w2 = - np.sum((B_t * k**2)[:,None] * sin_arg, axis=0)
    # w''' = - sum B_t * k^3 * cos(kx)
    w3 = - np.sum((B_t * k**3)[:,None] * cos_arg, axis=0)
    # w'''' = sum B_t * k^4 * sin(kx)
    w4 = np.sum((B_t * k**4)[:,None] * sin_arg, axis=0)
    # w''''' = sum B_t * k^5 * cos(kx)
    w5 = np.sum((B_t * k**5)[:,None] * cos_arg, axis=0)

    # reconstruct q and q', q'' if needed (useful to compute T term)
    # q(x)  amplitude per term = A_t * sin(kx)
    q = np.sum((A_t[:,None]) * sin_arg, axis=0)
    qp = np.sum((A_t * k)[:,None] * cos_arg, axis=0)   # d/dx q = sum A_t k cos(kx)
    qpp = - np.sum((A_t * k**2)[:,None] * sin_arg, axis=0)

    # return dictionary
    return dict(w=w, w1=w1, w2=w2, w3=w3, w4=w4, w5=w5, q=q, qp=qp, qpp=qpp)


def GI_piecewise_derivatives(x_vals, y_vals, intervals, n_derivatives=1, margin=None, poly_order=7):
    """
    Computes successive derivatives of y_vals with respect to x_vals,
    piecewise-defined over given intervals.
    If margin is specified, a polynomial is fitted on the inner region
    (start + margin, end - margin) and the derivative over the entire
    interval is evaluated from this polynomial.

    Parameters
    ----------
    x_vals : ndarray
        x coordinates.
    y_vals : ndarray
        y values corresponding to x_vals.
    intervals : list of tuples
        List of intervals defined as (x_start, x_end, left_inclusive, right_inclusive).
    n_derivatives : int
        Number of successive derivatives to compute.
    margin : float, optional
        Margin excluded from both ends for polynomial fitting.
    poly_order : int
        Order of the polynomial used for fitting.

    Returns
    -------
    x_all : ndarray
        Concatenated x coordinates corresponding to all intervals.
    derivs : list of ndarray
        List of successive derivatives. derivs[0] = first derivative, etc.
    """
    x_parts = []
    y_parts = []

    for (start, end, left_inclusive, right_inclusive) in intervals:
        mask = (
            (x_vals > start if not left_inclusive else x_vals >= start)
            & (x_vals < end if not right_inclusive else x_vals <= end)
        )
        x_parts.append(x_vals[mask])
        y_parts.append(y_vals[mask])

    x_all = np.concatenate(x_parts)
    derivs = []
    y_current_parts = y_parts.copy()

    for n in range(n_derivatives):
        new_y_parts = []

        for i, (start, end, *_ ) in enumerate(intervals):
            x_part = x_parts[i]
            y_part = y_current_parts[i]

            if margin is not None and len(x_part) > 2 * poly_order:
                x_inner_mask = (x_part >= start + margin) & (x_part <= end - margin)
                if np.any(x_inner_mask):
                    x_inner = x_part[x_inner_mask]
                    y_inner = y_part[x_inner_mask]
                else:
                    x_inner, y_inner = x_part, y_part
            else:
                x_inner, y_inner = x_part, y_part

            # Polynomial fitting on the inner region.
            # np.polynomial.Polynomial.fit maps x onto [-1, 1] before solving the
            # least-squares problem. This matters here: the raw Vandermonde matrix
            # built on x in [0, L] is hopelessly ill-conditioned beyond poly_order
            # ~10 (cond ~ 1e31 for L = 1300 mm, order 10), which caps the order
            # that can usefully be requested. With the normalised domain the fit
            # stays well conditioned up to order ~20, which is what allows the 5th
            # derivative to be resolved accurately near the interval boundaries.
            # Do NOT replace this by np.polyfit.
            poly = np.polynomial.Polynomial.fit(x_inner, y_inner, poly_order)
            dydx_part = poly.deriv(1)(x_part)

            new_y_parts.append(dydx_part)

        y_current_parts = new_y_parts.copy()
        derivs.append(np.concatenate(new_y_parts))

    return derivs






