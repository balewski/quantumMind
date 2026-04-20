__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

# original code is from Mark: MarkHirsbrunner@lbl.gov

'''Mark: Here's the functions I use to calculate two-point connected correlation functions. Here data would be the magnetization, displacement ranges from -ny +1 to ny - 1
(might be off by one, idk)
Then you iterate over displacements to get the correlation function, which should look like the attached plot. You can then fit that to an exponential to get the decay length
(Milan says to not include the displacement=0 point)

Vectorized form:

# Compute the density correlation matrix using np.cov with counts as weights
gij2 = np.cov(post_seq.T, aweights=counts, bias=True)

See: 02_Ordered_phases_in_Rydberg_systems.ipynb
'''
import numpy as np
from scipy.optimize import curve_fit
from pprint import pprint

#...!...!....................  it can be reaplced by cov-matrix
def dens_integrand(ind_1, ind_2, data):  # from Mark
    n1_data = data[:, ind_1]
    n2_data = data[:, ind_2]

    n1 = np.mean(n1_data)
    n2 = np.mean(n2_data)
    n1n2 = np.mean(n1_data * n2_data)

    return n1n2 - n1 * n2

#...!...!....................
def generate_pairs(ny, displacement):  # from Mark
    dy = displacement
    pairs = []
    if dy < 0:
        y_range = range(-dy, ny)
    else:
        y_range = range(0, ny - dy)
    for jj in y_range:
        pairs.append((jj, jj + dy))

    return pairs

#...!...!....................
def connected_two_point_corr_func(ny, displacement, data):  # from Mark
    pairs = generate_pairs(ny, displacement)
    g = 0
    for pair in pairs:
        g += integrand(pair[0], pair[1], data)
    g = g / len(pairs)
    return abg
        
#...!...!....................
def connected_two_point_corr_func_v2(ny, displacement, densCorr):
    pairs = generate_pairs(ny, displacement)
    g = 0
    for pair in pairs:
        g += densCorr[pair[0], pair[1]]
    g = g / len(pairs)
    return g

#...!...!....................
#Code from Mark
#Here's my very unsophisticated code for calculating the magnetization.
#The dictionary entry compiled_results["rydberg_occupation"] is the same array as you have, nshots x natoms

def do_magnetization(dataS):
    n_valid_shots,ny=dataS.shape
    magnetization = np.zeros_like(dataS)
    for shot_ind in range(n_valid_shots):
        #print('ss',shot_ind,dataS[shot_ind]);aa
        for yy in range(ny):
            n_i = dataS[shot_ind, yy]

            if yy == 0:
                n_j = dataS[shot_ind, yy + 1]

                magnetization[shot_ind, yy] =  ((-1) ** yy) * (n_i - n_j)
            elif yy == ny - 1:
                n_j = dataS[shot_ind, yy - 1]

                magnetization[shot_ind, yy] =  ((-1) ** yy) * (n_i - n_j)
            else:
                n_j1 = dataS[shot_ind, yy - 1]
                n_j2 = dataS[shot_ind, yy + 1]

                magnetization[shot_ind, yy] +=  ((-1) ** yy) * (2 * n_i - n_j1 - n_j2) / 2
    return magnetization
            

#...!...!....................
def fit_exponent_simple(XY):
        print('XY:',XY.shape)
        # Subset the data for fitting
        X_fit = XY[:,0]
        Y_fit = XY[:,1]

        print('X:',X_fit, Y_fit)
        # Define the exponential function to fit
        def exponential_func(x, a, b):
            return a * np.exp(b * x) 

        # Set initial values for the fit
        initial_values = [Y_fit[0], -1]  # Example initial values

        # Fit the exponential function to the data
        popt, pcov = curve_fit(exponential_func, X_fit, Y_fit, p0=initial_values)
        fmd={'func':'exp','a':float(popt[0]),'b':float(popt[1]),'max_site':XY.shape[0]}
        if 1:  # Print fit results
            print("Fit Parameters:")
            pprint(fmd)

        # Generate points for the fitted curve
        
        Y_fitted = exponential_func(X_fit, *popt)
        return fmd,X_fit,Y_fitted

      
