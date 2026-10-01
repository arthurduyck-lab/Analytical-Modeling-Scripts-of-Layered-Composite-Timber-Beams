import numpy as np
import pandas as pd

n_disc_val = 45

# np.trapz was removed in NumPy 2.0 and replaced by np.trapezoid.
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


def KR_Rigidity (LAM, CON) :

    """
    Compute the flexural and shear stiffness parameters for beams A and B according to the simplified Kreuzinger method.

    :param LAM:
        h [mm] = lamella height
        b [mm] = lamella width
        E [MPa] = bending modulus of elasticity
        G [MPa] = shear modulus
    :param CON:
        c [N/mm] = unit connector stiffness
        s [mm] = connector spacing

            Based on these two data points, K is calculated.
            It represents the stiffness of the connector along the lamella.
            This data is measured in N[mm.mm] and should not be confused with Ksurf, which takes into account
            the surface stiffness of the connector along the lamella and considers its width [N/mm².mm].

    :return:
        alpha [mm^(-2)] = Coupling parameter based on EIa, EIb and GSef
        GSef [N] = Effective shear stiffness
        EIa [Nmm^(2)] = Flexural stiffness of beam A
        EIb [Nmm^(2)] = Flexural stiffness of beam B
    """
    # === LAMELLAE DATA ===
    for col in ('h', 'b', 'E', 'G'):
        if col not in LAM.dtype.names:
            raise ValueError(f"Column '{col}' missing in the lamellae data file. "
                             f"Found: {list(LAM.dtype.names)}")
    for col in ('c', 's'):
        if col not in CON.dtype.names:
            raise ValueError(f"Column '{col}' missing in the connector data file. "
                             f"Found: {list(CON.dtype.names)}")
    if len(LAM['h']) < 2:
        raise ValueError(f"At least 2 lamellae are required, got n = {len(LAM['h'])}.")
    if len(CON['c']) != len(LAM['h']) - 1:
        raise ValueError(
            f"{len(CON['c'])} connector row(s) for {len(LAM['h'])} lamellae; "
            f"expected {len(LAM['h']) - 1}."
        )

    h = LAM['h']
    b = LAM['b']
    E = LAM['E']
    G = LAM['G']
    n = len(h)

    # === CONNECTORS DATA ===
    n_c = n - 1
    c = CON['c']
    s = CON['s']

    # === INERTIA ===
    # Intermediate lamellae parameters
    z0 = []
    z0_sumcum = 0
    A = []
    I = []
    EA = []
    z0EA = []
    EI = []
    for i in range(n):
        z0i = h[i]/2 + z0_sumcum
        z0.append(z0i)
        z0_sumcum = z0_sumcum + h[i]

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

    zG = sum(z0EA)/sum(EA)  # Gravity center
    a = sum(h) - h[0]/2 - h[-1]/2 # Distance between the centres of gravity of the upper and lower slats

    z = []  # Lever arm between zG and the centre of gravity of each lamella.
    EAz2 = []
    for i in range(n):
        zi = z0[i] - zG
        z.append(zi)

        EiAizi2 = EA[i] * zi ** 2
        EAz2.append(EiAizi2)

    # Intermediate connector parameters
    K=[]
    Kinv = []
    for i in range(n_c):
        Ki = c[i] / s[i]
        K.append(Ki)

        Ki_inv = 1/Ki
        Kinv.append(Ki_inv)

    # Calculation of inertia
    EIa = sum(EI)
    EIb = sum(EAz2)

    GSef_a = 1 / a ** 2
    GSef_b = h[0] / (2 * G[0] * b[0])
    GSef_c = 0
    for i in range(1, n-1):
        GSef_c += h[i] / (G[i] * b[i])
    GSef_d = h[-1] / (2 * G[-1] * b[-1])
    GSef_e = sum(Kinv)

    GSef = (GSef_a * (GSef_b + GSef_c + GSef_d + GSef_e)) ** (-1)
    alpha = GSef * (1/EIa + 1/EIb)

    return alpha, GSef, EIa, EIb


def KR_Parameters (LAM, CON) :
    """
    Compute section parameters (areas, inertia, centroid positions)
    for a laminated beam according to the Kreuzinger method.
    """

    for col in ('h', 'b', 'E', 'G'):
        if col not in LAM.dtype.names:
            raise ValueError(f"Column '{col}' missing in the lamellae data file. "
                             f"Found: {list(LAM.dtype.names)}")
    for col in ('c', 's'):
        if col not in CON.dtype.names:
            raise ValueError(f"Column '{col}' missing in the connector data file. "
                             f"Found: {list(CON.dtype.names)}")
    if len(LAM['h']) < 2:
        raise ValueError(f"At least 2 lamellae are required, got n = {len(LAM['h'])}.")
    if len(CON['c']) != len(LAM['h']) - 1:
        raise ValueError(
            f"{len(CON['c'])} connector row(s) for {len(LAM['h'])} lamellae; "
            f"expected {len(LAM['h']) - 1}."
        )

    h = LAM['h']
    b = LAM['b']
    E = LAM['E']
    G = LAM['G']
    n = len(h)

    n_c = n - 1
    c = CON['c']
    s= CON['s']

    z0 = []
    z0_sumcum = 0
    A = []
    I = []
    EA = []
    z0EA = []
    EI = []
    for i in range(n):
        z0i = h[i]/2 + z0_sumcum
        z0.append(z0i)
        z0_sumcum = z0_sumcum + h[i]

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

    zG = sum(z0EA)/sum(EA)
    a = sum(h) - h[0]/2 - h[-1]/2

    z = []
    EAz2 = []
    for i in range(n):
        zi = z0[i] - zG
        z.append(zi)

        EiAizi2 = EA[i] * zi ** 2
        EAz2.append(EiAizi2)

    return h, b, E, G, EA, EI, n, z0, zG, z, a, s


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


def KR_dMdx(M_vals, x_vals):
    """
    Shear force T = dM/dx.

    np.gradient uses a centred second-order difference in the interior but a
    one-sided FIRST-order difference at the two end points, so T(0) and T(L)
    carry an O(dx) error while every interior point is accurate to ~1e-12.
    That matters here because the Step 10 reaction check reads T exactly at
    those two points: for the uniform-load case it reported 699.30 N instead
    of 700.00 N, i.e. a 1.0e-3 relative error that failed the check while the
    solution itself was fine.

    The interior is left to np.gradient; the two end points are replaced by
    the derivative of a local cubic fitted to the first (and last) few nodes,
    which restores full accuracy there (700.0000 N for the same case).
    """
    T = np.gradient(M_vals, x_vals)
    k = min(8, len(x_vals) // 2)
    if k >= 4:
        pl = np.polynomial.Polynomial.fit(x_vals[:k], M_vals[:k], 3)
        pr = np.polynomial.Polynomial.fit(x_vals[-k:], M_vals[-k:], 3)
        T[0] = pl.deriv(1)(x_vals[0])
        T[-1] = pr.deriv(1)(x_vals[-1])
    return T


def KR_A_Sigma (MxA, h, E, z, EIa_val, n_disc = n_disc_val):
    """
       Compute normal stresses σ in beam A at top, middle, and bottom fibers.
       σ_i(z) = - (MxA / EIa) * E_i * z
    """

    n = len(h)

    # Fibre positions for each lamella (top, mid, bottom)
    z_top = h / 2.0
    z_mid = 0
    z_bot = - h / 2.0

    # Stresses Calculation
    factor = MxA / EIa_val
    sigma_top = -factor * np.array(E) * z_top
    sigma_mid = -factor * np.array(E) * z_mid
    sigma_bot = -factor * np.array(E) * z_bot

    A_Sigma = pd.DataFrame({
        'Lamelle [/]': np.arange(1, n+1),
        'σ_top [MPa]': sigma_top,
        'σ_mid [MPa]': sigma_mid,
        'σ_bot [MPa]': sigma_bot
    })

    z_vals = []
    sigma_vals = []
    for i in range(n):
        z_local = np.linspace(-h[i]/2,h[i]/2, n_disc)
        sigma_local = -factor * E[i] * z_local
        z_global = z[i] + z_local

        z_vals.extend(z_global)
        sigma_vals.extend(sigma_local)

    A_Sigma_zglobal = pd.DataFrame({
        'z [mm]': z_vals,
        'σ [MPa]': sigma_vals
    })

    return A_Sigma, A_Sigma_zglobal


def KR_A_Tau (TxA, h, E, z, EIa_val, n_disc = n_disc_val):
    """Compute shear stress τ distribution in beam A."""
    n = len(h)

    z_top = h / 2.0
    z_mid = 0
    z_bot = - h / 2.0

    # Stresses Calculation
    factor = TxA / EIa_val
    tau_top = -factor * np.array(E) * 1/2 * (z_top**2 - (np.array(h)/2)**2)
    tau_mid = -factor * np.array(E) * 1/2 * (z_mid**2 - (np.array(h)/2)**2)
    tau_bot = -factor * np.array(E) * 1/2 * (z_bot**2 - (np.array(h)/2)**2)

    A_Tau = pd.DataFrame({
        'Lamelle [/]': np.arange(1, n+1),
        'τ_top [MPa]': tau_top,
        'τ_mid [MPa]': tau_mid,
        'τ_bot [MPa]': tau_bot
    })

    z_vals = []
    tau_vals = []
    for i in range(n):
        z_local = np.linspace(-h[i]/2, h[i]/2, n_disc)
        tau_local = -factor * E[i] * 1/2 * (z_local**2 - (h[i]/2)**2)
        z_global = z[i] + z_local

        z_vals.extend(z_global)
        tau_vals.extend(tau_local)

    A_Tau_zglobal = pd.DataFrame({
        'z [mm]': z_vals,
        'τ [MPa]': tau_vals
    })

    return A_Tau, A_Tau_zglobal


def KR_B_Sigma (MxB, h, E, z, EIb_val, n_disc = n_disc_val):
    """Compute normal stresses σ for beam B (uniform across each lamella)."""
    n = len(h)

    factor = MxB / EIb_val
    sigma_uniforme = -factor * np.array(E) * np.array(z)

    sigma_top = sigma_uniforme
    sigma_mid = sigma_uniforme
    sigma_bot = sigma_uniforme

    B_Sigma = pd.DataFrame({
        'Lamelle [/]': np.arange(1, n+1),
        'σ_top [MPa]': sigma_top,
        'σ_mid [MPa]': sigma_mid,
        'σ_bot [MPa]': sigma_bot
    })

    z_vals = []
    sigma_vals = []
    for i in range(n):
        z_local = np.linspace(-h[i]/2,h[i]/2, n_disc)
        sigma_local = np.full_like(z_local, sigma_uniforme[i])
        z_global = z[i] + z_local

        z_vals.extend(z_global.tolist())
        sigma_vals.extend(sigma_local.tolist())

    B_Sigma_zglobal = pd.DataFrame({
        'z [mm]': z_vals,
        'σ [MPa]': sigma_vals
    })

    return B_Sigma, B_Sigma_zglobal


def KR_B_Tau (TxB, h, E, z, EIb_val, n_disc = n_disc_val):
    """Compute shear stress τ for beam B with interlamellar continuity."""
    n = len(h)
    factor = TxB / EIb_val

    # arrays to store summary values including continuity tau0
    tau_top = np.zeros(n, dtype=float)
    tau_mid = np.zeros(n, dtype=float)
    tau_bot = np.zeros(n, dtype=float)

    z_vals = []
    tau_vals = []
    tau0 = 0.0  # initial shear at bottom of first lamella

    for i in range(n):
        # local coordinate inside lamella i: from -h/2 (bottom) to +h/2 (top)
        z_local = np.linspace(-h[i] / 2.0, h[i] / 2.0, n_disc)

        # local shear distribution (uses tau0 as value at bottom of this lamella)
        # formula from your implementation
        tau_local = -factor * E[i] * z[i] * (h[i] / 2.0 + z_local) + tau0

        # global coordinates for plotting/export
        z_global = z[i] + z_local

        # store continuous distribution
        z_vals.extend(z_global.tolist())
        tau_vals.extend(tau_local.tolist())

        # extract and store summary values (bottom, mid, top) including tau0
        tau_bot[i] = tau_local[0]                       # bottom (z_local = -h/2)
        mid_idx = len(z_local) // 2
        tau_mid[i] = tau_local[mid_idx]                 # mid
        tau_top[i] = tau_local[-1]                      # top (z_local = +h/2)

        # update tau0 for next lamella (value at top of current lamella)
        tau0 = tau_local[-1]

    # build DataFrames
    B_Tau = pd.DataFrame({
        'Lamella [/]': np.arange(1, n + 1),
        'τ_top [MPa]': tau_top,
        'τ_mid [MPa]': tau_mid,
        'τ_bot [MPa]': tau_bot
    })

    B_Tau_zglobal = pd.DataFrame({
        'z [mm]': z_vals,
        'τ [MPa]': tau_vals
    })

    return B_Tau, B_Tau_zglobal



def sum_data(A_Array, B_Array):
    """
    Combine and sum two DataFrames (A and B) column-wise.
    Add suffixes for origin identification and a total column set.
    """

    if not A_Array.shape == B_Array.shape:
        raise ValueError("Shape mismatch between A_Array and B_Array.")

    TOT_Array = A_Array.copy()
    TOT_Array.iloc[:, 1:] = A_Array.iloc[:, 1:] + B_Array.iloc[:, 1:]
    TOT_Array.columns = [TOT_Array.columns[0]] + [c + " - TOT" for c in TOT_Array.columns[1:]]

    A_renamed = A_Array.copy()
    B_renamed = B_Array.copy()
    A_renamed.columns = [c + " - A" for c in A_renamed.columns]
    B_renamed.columns = [c + " - B" for c in B_renamed.columns]

    Combined = pd.concat([TOT_Array, A_renamed.iloc[:, 1:], B_renamed.iloc[:, 1:]], axis=1)

    return Combined


def FS_q(x, qm, L, d, l, N=10000):
    """
    Compute SF(q)(x) = (2/pi) * qm * sum_{theta=1..N} [ sin(theta*pi*x/L)/theta *
                    (cos(theta*d*pi/L) - cos(theta*(d+l)*pi/L)) ]

    Parameters
    ----------
    x : float or array
        Position along the beam.
    qm : float
        Load magnitude.
    L : float
        Beam length.
    d : float
        Start position of the loaded zone.
    l : float
        Length of the loaded zone.
    N : int, optional
        Number of series terms (approximation of infinity).

    Returns
    -------
    float or np.ndarray
        Computed SF(q)(x) values.
    """
    x = np.asarray(x, dtype=float)
    theta = np.arange(1, N + 1)
    coeff = np.cos(theta * d * np.pi / L) - np.cos(theta * (d + l) * np.pi / L)
    serie = np.zeros_like(x, dtype=float)

    for t in theta:
        serie += np.sin(t * np.pi * x / L) / t * (np.cos(t * d * np.pi / L) - np.cos(t * (d + l) * np.pi / L))

    return (2 / np.pi) * qm * serie


def FS_2q (x, qm, L, d1, d2, l, N=10000):
    """
    Compute SF(q)(x) = (2/pi) * qm * sum_{theta=1..N} [ sin(theta*pi*x/L)/theta *
                    (cos(theta*d1*pi/L) - cos(theta*(d1+l)*pi/L)+ cos(theta*d2*pi/L) - cos(theta*(d2+l)*pi/L)) ]

    Parameters
    ----------
    x : float or array
        Position along the beam.
    qm : float
        Load magnitude.
    L : float
        Beam length.
    d1 : float
        Start position of the loaded zone 1.
    d2 : float
        Start position of the loaded zone 2.
    l : float
        Length of the loaded zone.
    N : int, optional
        Number of series terms (approximation of infinity).

    Returns
    -------
    float or np.ndarray
        Computed SF(q)(x) values.
    """
    x = np.asarray(x, dtype=float)
    theta = np.arange(1, N + 1)
    coeff = np.cos(theta * d1 * np.pi / L) - np.cos(theta * (d1 + l) * np.pi / L) + np.cos(theta * d2 * np.pi / L) - np.cos(theta * (d2 + l) * np.pi / L)
    serie = np.zeros_like(x, dtype=float)

    for t in theta:
        serie += np.sin(t * np.pi * x / L) / t * (np.cos(t * d1 * np.pi / L) - np.cos(t * (d1 + l) * np.pi / L) + np.cos(t * d2 * np.pi / L) - np.cos(t * (d2 + l) * np.pi / L))

    return (2 / np.pi) * qm * serie


def KR_check_FS_sampling(x_vals, L_beam, N_fs, l_load=None):
    """
    Check that the x grid can represent the truncated Fourier series of the load.

    FS_q / FS_2q sum N_fs sine terms, the last of which has wavelength 2*L/N_fs.
    Sampling that on a grid of step dx > L/N_fs ALIASES the series: the load
    actually seen by the BVP solvers is not the load that was intended, and the
    solvers converge happily on the wrong problem. Nothing in solve_bvp's return
    flags this - success is reported either way.

    Measured on the 3Pts reference case (L = 2600 mm, l = 10 mm, N_fs = 10000),
    varying only n_points:

        n_points   dx      L/N_fs   dx <= L/N_fs   w_max
          1001    2.600    0.260        no        -2.6345   <- wrong sign
          2501    1.040    0.260        no         3.1875
          5001    0.520    0.260        no         6.4973
         10001    0.260    0.260        yes        7.1879   <- converged

    The same case run with N_fs = 500 gives 7.1879 for EVERY n_points from 1001
    up, i.e. the converged answer, because the criterion is then satisfied
    throughout. N_fs only has to resolve the loaded patch: a few times 2*L/l is
    ample (520 for the reference case). Raising N_fs beyond that buys nothing
    and forces a proportionally finer grid.

    Raises ValueError if the grid is too coarse; warns if N_fs looks too small
    to represent the patch itself.
    """
    x_vals = np.asarray(x_vals, dtype=float)
    dx = float(np.max(np.diff(x_vals)))
    dx_max = L_beam / N_fs

    if dx > dx_max * 1.01:
        n_needed = int(np.ceil(L_beam / dx_max)) + 1
        raise ValueError(
            f"Grid too coarse for the Fourier series of the load: dx = {dx:.4g} mm "
            f"but the series keeps terms down to a wavelength of "
            f"{2 * dx_max:.4g} mm, so dx must not exceed L/N_fs = {dx_max:.4g} mm. "
            f"Use n_points >= {n_needed} (currently {len(x_vals)}), or lower N_fs. "
            f"Without this the load is aliased and the solvers return a converged "
            f"but WRONG solution, with no error raised."
        )

    if l_load is not None and l_load > 0:
        N_min = 2.0 * L_beam / l_load
        if N_fs < N_min:
            print(
                f"WARNING: KR_check_FS_sampling - N_fs = {N_fs} is below "
                f"2*L/l = {N_min:.0f}: the Fourier series cannot resolve a loaded "
                f"patch of {l_load:.4g} mm and the load will be smeared out."
            )


def KR_Ts (x_vals, T_b_vals, h, E, z, b, EIb_val):
    """Compute shear forces in connectors along the beam for all lamellae."""

    n = len(h)
    m = len(x_vals)

    z_top = h / 2.0
    factor = T_b_vals / EIb_val

    tau0_i = np.zeros((m, n))
    Ts_i = np.zeros((m, n - 1))

    for i in range(n-1) :
        tau_i = -factor * E[i] * z[i] * (h[i] / 2.0 + z_top[i]) + tau0_i[:, i]
        Ts_i[:, i] = tau_i * b[i]
        tau0_i[:, i + 1] = tau_i

    Ts_vals = Ts_i

    data = {'x [mm]': x_vals}
    for i in range(n - 1):
        data[f'Ts_{i + 1},{i + 2} [N/mm]'] = Ts_i[:, i]

    df_Ts = pd.DataFrame(data)

    return Ts_vals, df_Ts


def KR_ConnectorForces(x_vals, df_Ts, L_beam, s_target):
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
        s_cal = L_beam / n_segments   # spacing actually used, after rounding
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


def KR_sigma_max_per_lamella(x_vals, A_Sigma_zglobal_all, B_Sigma_zglobal_all, h, z_centers):
    """
    Calculates the total maximum normal stresses (A + B) for each lamella and each position x.
    Parameters
    ----------
    x_vals : array_like
        Positions along the beam [mm].
    A_Sigma_zglobal_all : list of pd.DataFrame
        List of DataFrames of σ_A stresses for each x.
        Each DataFrame must contain “z [mm]” and “σ [MPa]”.
    B_Sigma_zglobal_all : list of pd.DataFrame
        List of DataFrames of σ_B stresses for each x.
        Same structure as A_Sigma_zglobal_all.
    h : array_like
        Thicknesses of the lamellae [mm].
    z_centers: array_like
        Global z coordinates of the centre of each lamella [mm].

    Returns
    -------
    df_sigma_max : pd.DataFrame
        Columns : x_vals, sigma_max1, sigma_max2, ...
    """

    n_lam = len(h)
    data = {'x [mm]': x_vals}

    # global vertical bounds of each slice
    z_bounds = []
    for i in range(n_lam):
        z_bottom = z_centers[i] - h[i]/2
        z_top = z_centers[i] + h[i]/2
        z_bounds.append((z_bottom, z_top))

    sigma_max_all = []
    zpos_max_all = []

    for idx, x in enumerate(x_vals):
        A_df = A_Sigma_zglobal_all[idx]
        B_df = B_Sigma_zglobal_all[idx]

        # Sum of stresses
        TOT_df = pd.DataFrame({
            'z [mm]': A_df['z [mm]'],
            'σ_tot [MPa]': A_df['σ [MPa]'] + B_df['σ [MPa]']
        })

        sigma_max_x = []
        zpos_max_x = []
        for (z_bottom, z_top) in z_bounds:
            mask = (TOT_df['z [mm]'] >= z_bottom) & (TOT_df['z [mm]'] <= z_top)
            if np.any(mask):
                sigma_slice = TOT_df.loc[mask, 'σ_tot [MPa]'].values  # numpy array of values
                z_slice = TOT_df.loc[mask, 'z [mm]'].values  # corresponding z positions

                # find index of value with largest absolute magnitude
                pos = int(np.argmax(np.abs(sigma_slice)))

                sigma_maxi = sigma_slice[pos]  # signed value (keeps sign)
                z_at_max = z_slice[pos]  # z where this value occurs

                sigma_max_x.append(sigma_maxi)
                # if you also want to store z position, collect it in a parallel list:
                zpos_max_x.append(z_at_max)
            else:
                sigma_max_x.append(np.nan)
                zpos_max_x.append(np.nan)
        sigma_max_all.append(sigma_max_x)
        zpos_max_all.append(zpos_max_x)

    sigma_max_all = np.array(sigma_max_all)
    for i in range(n_lam):
        data[f'σTOT_{i+1} [MPa]'] = sigma_max_all[:, i]

    df_sigma_max = pd.DataFrame(data)

    return df_sigma_max


def KR_tau_max_per_lamella(x_vals, A_tau_zglobal_all, B_tau_zglobal_all, h, z_centers):
    """
    Calculates the total maximum normal stresses (A + B) for each lamella and each position x.
    Parameters
    ----------
    x_vals : array_like
        Positions along the beam [mm].
    A_tau_zglobal_all : list of pd.DataFrame
        List of DataFrames of τ_A stresses for each x.
        Each DataFrame must contain “z [mm]” and “τ [MPa]”.
    B_tau_zglobal_all : list of pd.DataFrame
        List of DataFrames of τ_B stresses for each x.
        Same structure as A_tau_zglobal_all.
    h : array_like
        Thicknesses of the lamellae [mm].
    z_centers: array_like
        Global z coordinates of the centre of each lamella [mm].


    Returns
    -------
    df_tau_max : pd.DataFrame
        Columns : x_vals, tau_max1, tau_max2, ...
    """

    n_lam = len(h)
    data = {'x [mm]': x_vals}

    # global vertical bounds of each slice
    z_bounds = []
    for i in range(n_lam):
        z_bottom = z_centers[i] - h[i]/2
        z_top = z_centers[i] + h[i]/2
        z_bounds.append((z_bottom, z_top))

    tau_max_all = []
    zpos_max_all = []

    for idx, x in enumerate(x_vals):
        A_df = A_tau_zglobal_all[idx]
        B_df = B_tau_zglobal_all[idx]

        # Sum of stresses
        TOT_df = pd.DataFrame({
            'z [mm]': A_df['z [mm]'],
            'τ_tot [MPa]': A_df['τ [MPa]'] + B_df['τ [MPa]']
        })

        tau_max_x = []
        zpos_max_x = []
        for (z_bottom, z_top) in z_bounds:
            mask = (TOT_df['z [mm]'] >= z_bottom) & (TOT_df['z [mm]'] <= z_top)
            if np.any(mask):
                tau_slice = TOT_df.loc[mask, 'τ_tot [MPa]'].values  # numpy array of values
                z_slice = TOT_df.loc[mask, 'z [mm]'].values  # corresponding z positions

                # find index of value with largest absolute magnitude
                pos = int(np.argmax(np.abs(tau_slice)))

                tau_maxi = tau_slice[pos]  # signed value (keeps sign)
                z_at_max = z_slice[pos]  # z where this value occurs

                tau_max_x.append(tau_maxi)
                # if you also want to store z position, collect it in a parallel list:
                zpos_max_x.append(z_at_max)
            else:
                tau_max_x.append(np.nan)
                zpos_max_x.append(np.nan)
        tau_max_all.append(tau_max_x)
        zpos_max_all.append(zpos_max_x)

    tau_max_all = np.array(tau_max_all)
    for i in range(n_lam):
        data[f'τTOT_{i+1} [MPa]'] = tau_max_all[:, i]

    df_tau_max = pd.DataFrame(data)

    return df_tau_max