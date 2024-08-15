import numpy as np
import matplotlib.pyplot as plt
from math import asin, cos, sin, sqrt, pi, ceil
import re

from ahs_utils import plot_avg_density_2D

def index_convert(a, b, nx, ny):
    if a % 2 == 1:
        return (a-1) * ny + b - 1
    else:
        return a * ny - b
    
def reverse_index_convert(idx, nx, ny):
    # Adjust from zero-based to one-based indexing
    idx += 1
    a = np.ceil(idx / ny)
    # Determine if the row 'a' is odd or even and calculate column 'b'
    if a % 2 == 1:
        b = idx - (a - 1) * ny
    else:
        b = a * ny - idx + 1
    return int(a - 1), int(b - 1)


def get_coords(register):
    """
    Get the coordinates of all atoms in the register
    """
    
    coords_x = register.coordinate_list(0)
    coords_y = register.coordinate_list(1)
    coords = [[x, y] for (x, y) in zip(coords_x, coords_y)]
    return coords


def filterRowsCols(A, filterinds):
    N = A.shape[0]  # Assuming A is NxN

    # Create a full set of indices from 0 to N-1 (Python uses 0-based indexing)
    all_inds = np.arange(N)

    # Create a mask to keep rows and columns not in filterinds
    # Convert filterinds to 0-based by subtracting 1
    keep_mask = np.setdiff1d(all_inds, np.array(filterinds))

    # Filter out the rows and columns
    filteredA = A[np.ix_(keep_mask, keep_mask)]
    return filteredA

def get_dimer_bonds(Nx, Ny):
    dimers = []
    for ii in range(1, Nx + 1):
        for jj in range(1, Ny + 1):
            temp0 = ii + 1
            temp1 = jj + 1
            temp2 = jj - 1
            if ii % 2 == 1:  # Check if ii is odd
                if jj % 2 == 1:  # Check if jj is odd
                    if temp0 <= Nx and temp1 <= Ny:
                        u = index_convert(ii, jj, Nx, Ny)
                        v = index_convert(temp0, temp1, Nx, Ny)
                        dimers.append([u, v])
            else:  # ii is even
                if jj % 2 == 1:  # Check if jj is odd
                    if temp0 <= Nx and temp2 > 0:
                        u = index_convert(ii, jj, Nx, Ny)
                        v = index_convert(temp0, temp2, Nx, Ny)
                        dimers.append([u, v])
    return dimers

def LatticeFT(qx, qy, A, Nx, Ny):
    N = A.shape[0]
    assert Nx * Ny == N, "Nx * Ny must equal the number of elements in A"

    S = 0.0 + 0.0j  # Initialize S as a complex number
    for i in range(1, N + 1):  # Adjust range for 1-based to 0-based indexing
        for j in range(1, N + 1):
            ia, ib = reverse_index_convert(i, Nx, Ny)
            ja, jb = reverse_index_convert(j, Nx, Ny)
            phase = qx * (jb - ib) + qy * (ja - ia)
            S += np.exp(1j * phase) * A[i - 1, j - 1] / (N**2)  # Correct index and compute contribution

    return S



def get_R1_and_Rot(Dmin, alpha):
    if alpha <= pi/4:
        RotAngle = 0
        R1 = Dmin/sin(pi/4 - alpha/2)
    else:
        RotAngle = pi/4
        R1 = Dmin/sin(alpha/2)
    return R1, RotAngle

def SS_lattice_vertices_withFilteredExtraAtoms(Dmin, V2_over_V1, Nx, Ny, RotAngle=0, bd="x", ExcludeAtoms = []):
    alpha = 2 * asin(0.5/V2_over_V1**(1/6))
    assert 0 < alpha <= pi / 2
    psi = pi / 4 - alpha / 2
    
    #a, RotAnglex = get_R1_and_Rot(Dmin, alpha)
    a = Dmin/sin(pi/4 - alpha/2)
    
    #if RotAngle < 0:
    #    RotAngle = RotAnglex
    
    # Define translation vectors
    aodd = np.array([a * cos(psi), a * sin(psi)])
    bodd = np.array([a * sin(psi), a * cos(psi)])

    if bd == "x":
        Nx += 2
        Nsites = Nx * Ny
        vchain = np.zeros((Nx, Ny, 2))
        # First row
        for i in range(Nx - 1):
            vchain[i + 1, 0, 0] = vchain[i, 0, 0] + a * cos(psi)
            vchain[i + 1, 0, 1] = vchain[i, 0, 1] + (-1)**(i) * a * sin(psi)
        # Populate other columns
        for i in range(Nx):
            for j in range(Ny - 1):
                vchain[i, j + 1, 0] = vchain[i, j, 0] + (-1)**(j+1) * a * sin(psi)
                vchain[i, j + 1, 1] = vchain[i, j, 1] + a * cos(psi)
        # Shift to match extra-non extra
        for i in range(Nx):
            for j in range(Ny):
                vchain[i, j] -= aodd

    elif bd == "y":
        Ny += 2
        Nsites = Nx * Ny
        vchain = np.zeros((Nx, Ny, 2))
        # First column
        for i in range(Nx - 1):
            vchain[i + 1, 0, 0] = vchain[i, 0, 0] + a * cos(psi)
            vchain[i + 1, 0, 1] = vchain[i, 0, 1] + (-1)**(i+1) * a * sin(psi)
        # Populate other rows
        for i in range(Nx):
            for j in range(Ny - 1):
                vchain[i, j + 1, 0] = vchain[i, j, 0] + (-1)**(j) * a * sin(psi)
                vchain[i, j + 1, 1] = vchain[i, j, 1] + a * cos(psi)
        # Shift it down to match the positions of extra-non extra
        for i in range(Nx):
            for j in range(Ny):
                vchain[i, j] -= bodd
    else:
        raise ValueError("'bd' has to be 'x' or 'y'.")
    
    # Rotate Array
    xs = np.zeros(Nx*Ny)
    ys = np.zeros(Nx*Ny)
    
    for j in range(Ny):
        for i in range(Nx):
            p = index_convert(i+1, j+1, Nx, Ny)
            vx = vchain[i, j, 0]   
            vy = vchain[i, j, 1]
            xs[p] = vx*cos(RotAngle) - vy*sin(RotAngle)
            ys[p] = vx*sin(RotAngle) + vy*cos(RotAngle)
                    
    
    mask = np.ones(len(xs), dtype=bool)
    mask[ExcludeAtoms] = False
    Nsites = Nsites - len(ExcludeAtoms)
    return Nsites, xs[mask], ys[mask]


def get_SS_lattice_coords_filter_atoms(Nx: int, Ny: int, V2_over_V1: float, Delta_over_V1: float, Dmin: float, phase: str):
    
    assert phase in ["1/3", "2/5"]
    
    # Create a SS lattice with size (Nx+2) by Ny
    _, xs, ys = SS_lattice_vertices_withFilteredExtraAtoms(Dmin, V2_over_V1, Nx, Ny, RotAngle=0.0, bd="x", ExcludeAtoms=[])
    Nsites = len(xs)
    
    # Determine the indices of atoms to be excluded
    if phase == "1/3":
        ExcludeAtoms_left = [i for i in range(Ny) if np.mod(i, 3) != 0]
        ExcludeAtoms_right = [j for (i, j) in zip(range(Ny), range(Nsites-Ny, Nsites)) if np.mod(i, 3) != 0]
    else: # phase == "2/5"
        ExcludeAtoms_left = [i for i in range(Ny) if np.mod(i, 10) != 0 and np.mod(i, 10) != 5 and np.mod(i, 10) != 7]
        ExcludeAtoms_right = [j for (i, j) in zip(range(Ny), range(Nsites-Ny, Nsites)) if np.mod(i, 10) != 0 and np.mod(i, 10) != 5 and np.mod(i, 10) != 7]
    
    ExcludeAtoms = ExcludeAtoms_left + ExcludeAtoms_right
    mask = np.ones(len(xs), dtype=bool)
    mask[ExcludeAtoms] = False
    
    xs, ys = xs[mask], ys[mask]
    
    coords = []
    for x, y in zip(xs, ys):
        coords.append((x, y))
        
    FilterAtoms_left = range(Ny - len(ExcludeAtoms_left))
    FilterAtoms_right = range(len(coords) - len(FilterAtoms_left), len(coords))
    FilterAtoms = list(FilterAtoms_left) + list(FilterAtoms_right)
    
    return coords, FilterAtoms
        


def check_spacingVerticalMin(coords, spacingVerticalMin=2e-6):

    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            y_separation = coords[j][1] - coords[i][1]

            if y_separation == 0:
                continue

            if abs(y_separation) < spacingVerticalMin:
                raise ValueError(f"Site {i} and {j} have y-separation ({y_separation * 1e6} um) that is smaller than {spacingVerticalMin * 1e6} um.")

def check_spacingRadialMin(coords, spacingRadialMin=4e-6):

    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            radial_separation = np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)

            if radial_separation < spacingRadialMin:
                raise ValueError(f"Site {i} and {j} have radial separation ({radial_separation * 1e6} um) that is smaller than {spacingRadialMin * 1e6} um.")


def check_within_bounding_box(coords, width=75e-6, height=128e-6):
    x_max = max([coord[0] for coord in coords])
    x_min = min([coord[0] for coord in coords])
    y_max = max([coord[1] for coord in coords])
    y_min = min([coord[1] for coord in coords])

    biggest_x_distance = x_max - x_min
    biggest_y_distance = y_max - y_min

    if biggest_x_distance > width:
        raise ValueError(f"The geometry has width {biggest_x_distance * 1e6} um, which is wider than {width * 1e6} um.")

    if biggest_y_distance > height:
        raise ValueError(f"The geometry has width {biggest_y_distance * 1e6} um, which is taller than {height * 1e6} um.")


def check_geometry_fit_constraints(
    register,
    spacingVerticalMin=2e-6,
    spacingRadialMin=4e-6,
    width=75e-6, 
    height=128e-6
):
    coords = get_coords(register)

    check_spacingVerticalMin(coords, spacingVerticalMin=spacingVerticalMin)
    check_spacingRadialMin(coords, spacingRadialMin=spacingRadialMin)
    check_within_bounding_box(coords, width=width, height=height)
    
    
def CheckOmegaAndDelta(Omega, Delta):
    Delta_min  = -125000000.0
    Delta_max  =  125000000.0
    Omega_max  =  15800000.0
    
    
    if Omega < Omega_max:
        # print(" Omega is in the range")
        pass
    else:
        raise ValueError("Omega is bigger than maximum allowed Omega")
    
    if Delta_min < Delta < Delta_max:
        # print(" Delta is in the range")
        pass
    else:
        if (Delta_min > Delta):
            raise ValueError("Delta is smaller than minimum allowed Delta")
        else:
            raise ValueError("Delta is bigger than maximum allowed Delta")
            
def get_drive_params(V2_over_V1, Delta_over_V1, Dmin, Rb1, check_drive_validity=True):
    
    # Constants    
    C6 = 5.42e-24 # rad m^6/s
    
    # Find lattice angle
    alpha = 2 * np.arcsin(0.5/V2_over_V1**(1/6))
    assert 0 < alpha <= np.pi / 2 
    psi = np.pi/4 - alpha/2
    
    R1 = Dmin/np.sin(psi)
    V1 = C6/R1**6
    Rb = Rb1*R1
    
    Omega = C6/Rb**6
    Delta = Delta_over_V1*V1
    # print(Omega, Delta, Dmin)
    
    if check_drive_validity:
        CheckOmegaAndDelta(Omega, Delta)
    
    Delta_start = -5*Omega #
    Delta_end   = Delta    
    
    return Omega, Delta_start, Delta_end
            

def get_post_seqs(result):
    measurements = result.measurements
    return [meas.post_sequence for meas in measurements]


def get_Z_corr(postseq):
    corr = np.zeros((len(postseq), len(postseq)))
    for i in range(len(postseq)):
        for j in range(i, len(postseq)):
            corr[i,j] = postseq[i] * postseq[j]
            corr[j,i] = postseq[i] * postseq[j]

    return corr


def find_indices(Qx, Qy, Kx_list, Ky_list):
    indices = []
    for Kx, Ky in zip(Kx_list, Ky_list):
        idx_x = np.abs(Qx - Kx).argmin()
        idx_y = np.abs(Qy - Ky).argmin()
        indices.append((idx_x, idx_y))
    return indices

def get_neighbors(inds, Nx, Ny, radius):
    # Generate neighboring indices considering boundary conditions and the specified radius
    x, y = inds
    neighbors = [(x + dx, y + dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)
                 ]
    return neighbors

def sum_with_average_over_neighbors(FTq_NN, indices, Nx, Ny, radius):
    total = 0
    for inds in indices:
        # Get the value at the primary index
        main_value = FTq_NN[inds]
        neighbors = get_neighbors(inds, Nx, Ny, radius)
        # Sum over the primary index and its neighbors
        neighborhood_values = [FTq_NN[nx, ny] for (nx, ny) in neighbors]
        neighborhood_values.append(main_value)
        # Calculate the average for the neighborhood and sum it to the total
        total += np.mean(neighborhood_values)  # Averaging the neighborhood including the center point
    return total

def analyze(Nx, Ny, result, program, FilterAtoms_Phase, Delta_over_V1, Nq = 100, avg_radius = 8):
    
    measurements = result.measurements
    postsequences = []
    for meas in measurements:
        preseq = meas.pre_sequence
        postseq = meas.post_sequence
        
        if len(preseq) == sum(preseq):
            postsequences.append(postseq)
            
    postsequences = [1 - p for p in postsequences]
    register = program.register
    final_density = np.sum(postsequences, axis=0) / len(postsequences)
    
    NN_corr = np.sum([get_Z_corr(postseq) for postseq in postsequences], axis=0) / len(postsequences)
    
    NN_corr_filtered = filterRowsCols(NN_corr, FilterAtoms_Phase)
    
    qx = np.linspace(0, 2*np.pi, Nq)
    qy = np.linspace(0, 2*np.pi, Nq)
    Qx, Qy = np.meshgrid(qx, qy, indexing='ij')
    
    FTq_NN = LatticeFT(Qx, Qy, NN_corr_filtered, Nx, Ny)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))
    ax1.set_xlabel("x [μm]")
    ax1.set_ylabel("y [μm]")
    ax1.set_title("Rydberg density for $\Delta/V_1=$" + f"{np.round(Delta_over_V1, 2)}")
    ax1.axis('equal')
    plot_avg_density_2D(final_density, register, custom_axes = ax1);
    fn2 = ax2.pcolormesh(qx/(2*np.pi), qy/(2*np.pi), np.abs(FTq_NN), cmap='plasma')
    ax2.set_xlabel(r'$q_{x}/2\pi$')
    ax2.set_ylabel(r'$q_{y}/2\pi$')
    ax2.set_title("Structure factor for $\Delta/V_1=$" + f"{np.round(Delta_over_V1, 2)}")
    cb2 = fig.colorbar(fn2, ax=ax2, shrink=1.0, orientation="vertical", label=r"$|S_{\rm q}^{nn}|$")
    plt.show()
    
    SS13K_vecs = list([(2*np.pi/3, np.pi), (-2*np.pi/3+2*np.pi, np.pi), (np.pi, 2*np.pi/3), (np.pi, -2*np.pi/3+2*np.pi)])
    SS25K_vecs = list([(4*np.pi/5, np.pi), (-4*np.pi/5+2*np.pi, np.pi), (np.pi, 4*np.pi/5), (np.pi, -4*np.pi/5 + 2*np.pi)])    
    
    indices_SS13 = find_indices(qx, qy, *zip(*SS13K_vecs))
    indices_SS25 = find_indices(qx, qy, *zip(*SS25K_vecs))    
    
    Order_13 = sum_with_average_over_neighbors(FTq_NN, indices_SS13, Nx, Ny, avg_radius)
    Order_25 = sum_with_average_over_neighbors(FTq_NN, indices_SS25, Nx, Ny, avg_radius)
    
    return np.real(Order_13), np.real(Order_25)