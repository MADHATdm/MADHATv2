#!/usr/bin/env python
# coding: utf-8

from scipy.stats import poisson
import numpy as np
import scipy.signal as SciSig
import scipy.sparse as SciSpa

import os
import time
import argparse
import logging

#default model and set files
default_models = ['dmtest.dat'] #, 'dmmumu.dat','dmtautau.dat', 'dmww.dat']
default_sets = ['set1.dat'] #, 'set0.dat', 'set1a.dat', 'set1b.dat', 'set1c.dat', 'set2.dat', 'set3.dat', 'set4.dat', 'set5.dat', 'set6.dat', 'set7.dat']
formatted_output = False # can be set to True with the --formatted_output (or -f) flag, or changed here

#target beta and tolerance, maximum iterations
beta_target = 0.95
beta_tolerance = 0.001
beta_background = 0.90 # Threshold for background model validation: if beta > this value when Phi_PP ≈ 0, it suggests observed counts are much lower than background predictions, indicating potential issues with the background model
MAX_CONVERGENCE_ITERATIONS = 200

#whether binning takes place, the type of weighting to take place
binning = 1 # 1 for 8bpd, 0 for 1bpd
weighting_type_flag = 1 # 0 for user defined weighting, 1 for N_B_mean weighting
weights_original = np.array([]) # User defined weights, will be initialized later if weighting_type_flag == 1

#parameters pertinent to the convergence aspect of the algorithm 
weight_raising_amount = 2
convergence_tolerance = 1e-4

#parameters related to different probability tolerances
bgd_prob_sum_tolerance = 1e-4
P_sig_ij_tol_denom = 1e4
P_bar_zero_out_threshold_denom = 1e4
energy_fraction_zero_out_threshold_denom = 1e4

NOBS_filename = "PMFdata/nobs1b.dat"
pmf_data_filename = "PMFdata/pmf1b.dat"
if binning == 1:
    NOBS_filename = "PMFdata/nobs8bpd.dat"
    pmf_data_filename = "PMFdata/pmf8bpd.dat"

logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s %(levelname)-8s %(name)s ┊ %(message)s",
    filename="output.log",
    filemode="w" # overwrite on each run; use "a" to append
)

header = """
#####################################################################################################################################################################
# Fermi_Pass8R3_239557417_681169985
###################################################################################OUTPUT############################################################################
# Mass(GeV)   Spectrum     Beta    Nbound    PhiPP(cm^3 s^-1 GeV^-2)   +dPhiPP            -dPhiPP            sigv(cm^3 s^-1)      +dsigv             -dsigv
"""

output_fmt = (
    "{mass:4.0f} "
    "{spectrum:17.8f} "
    "{beta:8.2f} "
    "{nbound:7.0f} "
    "{phipp:19.8e} "
    "{dphipp_p:25.8e} "
    "{dphipp_m:18.8e} "
    "{sigv:18.8e} "
    "{dsigv_p:20.8e} "
    "{dsigv_m:18.8e}"
)

def main(set_filename, model_filename, formatted_output=False):
    """
    Main analysis function places bounds on the dark matter annihilation cross section through a Fermi-LAT-data-based, 
    model-independent analysis of dark matter annihilation into gamma rays within dwarf galaxies/dwarf-like objects.
    This function performs the following steps:
    1. Loads model data (e.g., dmbb.dat) containing mass, integrated energy spectrum, and energy fractions.
    2. Loads set data (e.g., settest.dat) containing galaxy IDs and J-factors with uncertainties.
    3. Loads observed photon counts and exposures from the NOBS file.
    4. Loads probability mass function (PMF) data for photon counts.
    5. For each mass point:
        - Iterates over galaxies and energy bins.
        - Calculates observed photon counts, background and signal probabilities.
        - Computes confidence bounds for Phi_PP.
        - Handles edge cases where the background-only model is disfavored.
    6. Computes cross-section limits (sigv) and their uncertainties.
    Parameters:
        set_filename (str): Path to the set file containing galaxy IDs and J-factors.
        model_filename (str): Path to the model file containing mass and energy spectrum data.
        formatted_output (bool): If True, outputs results in a formatted manner with aligned columns; otherwise, outputs tab-separated values (same as MADHATv2).
    Outputs:
        Writes results to an output file in the 'Output/' directory, including:
            - Mass, integrated energy spectrum, beta, N_bounds, Phi_PP (central, upper, lower), 
              sigmav (central, upper, lower), and uncertainties.
        Logs progress and errors using the logger.
    Notes:
        - Requires adjustable parameters at the top of the madhat21.py file and helper functions (e.g., compute_P_bar, prune_P_bar, compute_beta) 
          located after the main function.
        - Handles both single-bin and multi-bin (e.g., 8bpd) analysis depending on the 'binning' flag.
    """
    # Initialize logger
    logger = logging.getLogger(__name__)

    energy_fractions = np.array([1])
    energy_bin_number = energy_fractions.size

    # Loading the model file, e.g. dmbb.dat
    # comments="#" ignores the header, dtype=float ensures that the data is read as float
    model_file_data = np.loadtxt(model_filename, comments="#", dtype=float)
    mass_number = model_file_data.shape[0] 
    masses = model_file_data[:, 0] 
    integrated_energy_spectrum_vals = model_file_data[:, 1]

    # For 8bpd, the energy fractions are in the remaining columns
    if binning == 1:
        energy_bin_number = model_file_data.shape[1] - 2 # Number of energy bins; rows - 2 (mass and integrated energy spectrum)
        energy_fractions_array = model_file_data[:, 2:] 

    # Loading the set file, e.g. settest.dat
    # np.atleast_2d ensures that the data is at least 2D (for example, for a single dwarf, there's only one row, it will be reshaped to 2D:[# # # #] --> [[# # # #]])
    set_data = np.atleast_2d(np.loadtxt(set_filename, comments="#", dtype=float)) # [[Dwarf ID, J, dJ+, dJ-][Dwarf ID, J, dJ+, dJ-]...]
    ID_arr, J_arr, dJp_arr, dJm_arr = set_data.T # Transpose to get the columns [[Dwarf ID, Dwarf ID...], [J, J...], [dJ+, dJ+...], [dJ-, dJ-...]]
    gal_number = ID_arr.size # Number of galaxies is the size of the number of IDs
    ID = ID_arr.astype(int) # Convert IDs to integers

    # Convert J-factors into 2D column vectors
    J_red_i = J_arr[:, None] 
    J_red_uncert_plus_i = dJp_arr[:, None]
    J_red_uncert_minus_i = dJm_arr[:, None]

    # Convert J-factors into correct units
    J_i_central = 10 ** J_red_i
    J_i_bigger = 10 ** (J_red_i + J_red_uncert_plus_i)
    J_i_smaller = 10 ** (J_red_i - J_red_uncert_minus_i)

    # Loading the NOBS file, e.g. nobs8b.dat
    nobs = np.atleast_2d(np.loadtxt(NOBS_filename, comments="#", dtype=float))
    rows = energy_bin_number * (ID.flatten() - 1) # Get the starting row corresponding to the IDs 
    exposures_i = nobs[rows, -1].reshape(-1, 1) # Reshape to get the exposures as a column vector (there is only one exposure per galaxy)
    i_idx = np.arange(gal_number)[:, None] # Create an array of indices for the galaxies --> [[0][1]...[gal_number-1]]   
    j_idx = np.arange(energy_bin_number)[None, :] # Create an array of indices for the energy bins --> [[0, 1, 2, ... , energy_bin_number-1]]                 
    N_O_ij_original = nobs[rows[i_idx] + j_idx, -2] # Get the N_O values for the galaxies and energy bins        

    # Loading the PMF data file, e.g. pmf8b.dat
    pmf_data = np.loadtxt(pmf_data_filename, comments="#", dtype=float)
    N_max_hard = pmf_data.shape[0] - 1 # Gets the maximum number of photon counts
    probs_all = pmf_data[:, 1:] # Get the probabilities for all photon counts, excluding the first column (which is the number of photons)
    id_zero = (ID.flatten() - 1).astype(int) # Get the IDs for the galaxies of interest, subtracting 1 to get the correct index
    cols = np.hstack([np.arange(d*energy_bin_number, (d+1)*energy_bin_number) for d in id_zero]) # Get the columns corresponding to the galaxies of interest
    probs = probs_all[:, cols] # Get the probabilities for the galaxies of interest
    pmf_data_original = probs.reshape((N_max_hard+1, len(id_zero), energy_bin_number)) # Reshape the probabilities to get the PMF data for the galaxies of interest

    # Initialize Phi_PP arrays
    Phi_PP_central_for_m = np.zeros(mass_number)
    Phi_PP_lower_for_m = np.zeros(mass_number)
    Phi_PP_upper_for_m = np.zeros(mass_number)

    for m in np.arange(mass_number):
        mass = masses[m]
        integrated_energy_spectrum = integrated_energy_spectrum_vals[m]

        print(f"Mass: {mass} GeV")
        logger.info(f'Mass: {mass} GeV')

        # For 8bpd, mask the energy fractions based on the threshold and remove the corresponding columns from N_O_ij and pmf_data
        if binning == 1:
            energy_fractions = energy_fractions_array[m, :]
            fraction_mask = energy_fractions < energy_fractions.max() / energy_fraction_zero_out_threshold_denom
            energy_fractions = np.delete(energy_fractions, fraction_mask)
            energy_bin_number = energy_fractions.size
            N_O_ij = np.delete(N_O_ij_original, fraction_mask, 1)
            pmf_data = np.delete(pmf_data_original, fraction_mask, 2)
            # For user defined weighting, update and normalize the weights after removing the columns
            if weighting_type_flag == 0:
                weights_intermediate = np.delete(weights_original, fraction_mask, 1)
                weights_intermediate = weights_intermediate / weights_intermediate.max()
        else:
            N_O_ij = N_O_ij_original
            pmf_data = pmf_data_original

        # For weighting_type_flag == 1, compute N_B_mean as in equation (5) [Reference arXiv:2401.05327]
        if weighting_type_flag == 1:
            N_B_mean = np.zeros((gal_number, energy_bin_number))
            for i in np.arange(gal_number):
                for j in np.arange(energy_bin_number):
                    for N_B_ij in np.arange(1, N_max_hard + 1):
                        N_B_mean[i, j] += N_B_ij * pmf_data[N_B_ij, i, j]
            weights_original = J_i_central * exposures_i * energy_fractions / N_B_mean # Equation (1) / (5) or (11) in [Reference arXiv:2401.05327]
        weights_original = weights_original / weights_original.max() # Normalize the weights

        # Initialize J-factor
        J_i = J_i_central
        Phi_PP_central = 0
        Phi_PP_J_bigger = 0
        Phi_PP_J_smaller = 0

        # Loop over each J-factor scenario (central, central + error, central - error)
        for j_n in np.arange(3):
            weights = weights_original
            if binning == 1 and weighting_type_flag == 0:
                weights = weights_intermediate
            weight_amp = 1
            if j_n == 1:
                J_i = J_i_bigger
            if j_n == 2:
                J_i = J_i_smaller

            Phi_PP_step_factor = 10
            Phi_PP = 1e-34
            loop_direction = 1
            Phi_PP_old = 0
            beta = 0
            currently_initial_weighting = 1
            weight_iteration = 0
            done = 0
            iteration = 0

            while done != 1:
                iteration += 1
                
                # Calculate the number of observed photons, using floor then sum to ensure integer counts
                N_O_bar = np.sum(np.floor(weights * N_O_ij)).astype(int)

                # Check if N_O_bar is zero, if so, increase weights
                if N_O_bar == 0:
                    while weight_iteration < 4:
                        weights *= 2
                        N_O_bar = np.sum(np.floor(weights * N_O_ij).astype(int))
                        weight_amp *= 2
                        weight_iteration += 1  
                        if N_O_bar != 0:
                            logger.debug(f"N_O_bar = 0, increased weights by a factor of {2**weight_iteration}")
                            print(f"N_O_bar = 0, increased weights by a factor of {2**weight_iteration}")
                            break
                if weight_iteration == 4 and N_O_bar == 0:
                    logger.error('Error: N_O_bar = 0 after increasing the weights by a factor of 16.')
                    print("Error: N_O_bar = 0 after increasing the weights by a factor of 16. Exiting iteration.")
                    break

                min_weight = weights.min()
                N_max = N_max_hard

                # If the minimum weight is greater than zero, adjust N_max based on N_O_bar and min_weight
                if min_weight != 0 and np.ceil(N_O_bar / min_weight) < N_max_hard:
                    N_max = np.ceil(N_O_bar / min_weight).astype(int)
                
                # Extract background probabilities from PMF data for the current N_max range
                # P_bgd_ij[n, i, j] = probability of observing n background photons in galaxy i, energy bin j
                P_bgd_ij = np.zeros((N_max + 1, gal_number, energy_bin_number))
                for i in np.arange(gal_number):
                    for j in np.arange(energy_bin_number):
                        P_bgd_ij[:, i, j] = pmf_data[np.arange(N_max + 1), i, j]

                # Calculate background probability cutoffs for computational efficiency
                # Find the point where cumulative background probability reaches (1 - tolerance)
                # Beyond this cutoff, probabilities are negligible and can be ignored
                cutoffs_N_B_upper = (N_max + 1) * np.ones((gal_number, energy_bin_number))
                for i in np.arange(gal_number):
                    for j in np.arange(energy_bin_number):
                        P_bgd_ij_sum = 0
                        N_B_ij = 0
                        # Sum probabilities until we reach (1 - bgd_prob_sum_tolerance) of total probability mass
                        while P_bgd_ij_sum < 1 - bgd_prob_sum_tolerance and N_B_ij <= N_max:
                            P_bgd_ij_sum += P_bgd_ij[N_B_ij, i, j]
                            N_B_ij += 1
                        cutoffs_N_B_upper[i, j] = N_B_ij

                cutoffs_N_B_upper = cutoffs_N_B_upper.astype(int) # Convert cutoffs to integers and calculate weighted maximum background count
                N_B_bar_max = weights.ravel().dot((cutoffs_N_B_upper-1).ravel())
                N_B_bar_max = np.floor(N_B_bar_max).astype(int)

                # Compute P_B_bar using the background probabilities and the cutoffs
                P_B_bar = compute_P_bar(N_B_bar_max, P_bgd_ij, cutoffs_N_B_upper, weights, gal_number, energy_bin_number, N_O_bar)

                # Initialize variables for convergence
                beta_low = 0
                Phi_PP_low = 0
                Phi_PP_low_found = 0
                beta_high = 0
                Phi_PP_high = 0
                Phi_PP_high_found = 0
                Phi_PP_stepping = 0
                convergence_iteration = 0
                max_convergence_flag = 0

                while Phi_PP_low_found == 0 or Phi_PP_high_found == 0:
                    # In certain dwarfs and mass/channel combinations, the background-only model itself appears to be strongly disfavored: when we set the initial Phi_PP
                    # to a negligible value (< 1e-60), the computed beta (the probability of observing ≤ number of observed photons under the background) is already ≥ ~0.9 or higher.
                    # This occurs when the actual observed photon counts are significantly lower than what the background model predicts, so most of the probability mass from the 
                    # background PMF is above the observed count. As a result, when the algorithm tries to tune Phi_PP lower, beta plateaus above beta_target, causing the convergence 
                    # loop to stall. The code now exits with the following error message in the log file for these cases: "Background-only model appears to be ruled out at 
                    # beta > threshold: 0.90, even with Phi_PP ≈ 0 and minimum weight > 1.0. This suggests that the observed photon count is lower than expected from the background model alone.
                    # Consider reviewing the background model for this target. Zeros have been added to the output file for this mass point."

                    # If the convergence iteration is zero, we check if the beta is already above the threshold. First, we set Phi_PP to a very small value (1e-60) and set the minimum weight to 1.0 (if min_weight > 0) or multiply the weights by 100 (if there is a weight with zero).
                    if convergence_iteration == 0:    
                        test_Phi_PP = 1e-60
                        if min_weight > 0:
                            min_weight_scale_factor = 1 / min_weight
                        else:
                            min_weight_scale_factor = 100

                        test_weights = weights * min_weight_scale_factor
                        test_weight_amp = weight_amp * min_weight_scale_factor # not used here, but kept for consistency
                        
                        N_S_mean_test = test_Phi_PP * J_i * exposures_i * energy_fractions
                        P_sig_arr_test, cutoffs_N_S_upper_test, N_S_bar_max_test = compute_P_sig(N_max, N_S_mean_test, gal_number, energy_bin_number, test_weights)
                        P_S_bar_test = compute_P_bar(N_S_bar_max_test, P_sig_arr_test, cutoffs_N_S_upper_test, test_weights, gal_number, energy_bin_number, N_O_bar)

                        test_beta = compute_beta(P_B_bar, P_S_bar_test)

                        if test_beta > beta_background:
                            logging.error(f"Background-only model appears to be ruled out at beta > threshold: {beta_background}, even with Phi_PP = 1e-60 and minimum weight > 1.0. This suggests that the observed photon count is lower than expected from the background model alone. Consider reviewing the background model for this target. Zeros have been added to the output file for this mass point.")
                            max_convergence_flag = 1
                            break

                    # Catches the case where beta is lower than 0.9, but the convergence loop still does not converge
                    if convergence_iteration > MAX_CONVERGENCE_ITERATIONS:
                        logging.error(f"Convergence loop did not converge after {MAX_CONVERGENCE_ITERATIONS} iterations. This suggests that the observed photon count is lower than expected from the background model alone. Consider reviewing the background model for this target. Zeros have been added to the output file for this mass point.")
                        max_convergence_flag = 1
                        break

                    # After the initial check/guess,
                    if Phi_PP_stepping == 1:
                        # If increasing the Phi_PP
                        if loop_direction == 1:
                            # If beta has overshot the target, reduce the step size and reverse the loop direction
                            if beta > beta_target:
                                loop_direction = -1
                                Phi_PP_step_factor = Phi_PP_step_factor ** (-1/2)
                            # Else, keep increasing the step size
                        # If decreasing the Phi_PP
                        else:
                            # If beta has undershot the target, reduce the step size and reverse the loop direction
                            if beta_target > beta:
                                loop_direction = 1
                                Phi_PP_step_factor = Phi_PP_step_factor ** (-1/2)
                            # Else, keep decreasing the step size
    
                        Phi_PP *= Phi_PP_step_factor

                    N_S_mean = Phi_PP * J_i * exposures_i * energy_fractions
                    P_sig_arr, cutoffs_N_S_upper, N_S_bar_max = compute_P_sig(N_max, N_S_mean, gal_number, energy_bin_number, weights)
                    P_S_bar = compute_P_bar(N_S_bar_max, P_sig_arr, cutoffs_N_S_upper, weights, gal_number, energy_bin_number, N_O_bar)

                    # Create sparse matrices for beta comparison
                    beta = compute_beta(P_B_bar, P_S_bar)

                    # Check if beta is within the target range. Note that both flags can be set to 1 simultaneously, because we use an interpolation to find the target Phi_PP
                    if np.abs(beta - beta_target) < beta_tolerance:
                        if beta <= beta_target:
                            beta_low = beta
                            Phi_PP_low = Phi_PP
                            Phi_PP_low_found = 1
                        if beta >= beta_target:
                            beta_high = beta
                            Phi_PP_high = Phi_PP
                            Phi_PP_high_found = 1

                    # For the first iteration, we determine the initial step size
                    if Phi_PP_stepping == 0:
                        if loop_direction == 1:
                            if currently_initial_weighting != 1:
                                potenital_new_Phi_PP_step_factor = 10**(1/(2*weight_amp))
                                if Phi_PP_step_factor < potenital_new_Phi_PP_step_factor:
                                    Phi_PP_step_factor = potenital_new_Phi_PP_step_factor
                            if beta > beta_target:
                                loop_direction = -1
                                Phi_PP_step_factor = Phi_PP_step_factor ** (-1)
                        else:
                            if currently_initial_weighting != 1:
                                potenital_new_Phi_PP_step_factor = 10**(-1/(2*weight_amp))
                                if Phi_PP_step_factor > potenital_new_Phi_PP_step_factor:
                                    Phi_PP_step_factor = potenital_new_Phi_PP_step_factor
                            if beta_target > beta:
                                loop_direction = 1
                                Phi_PP_step_factor = Phi_PP_step_factor ** (-1)
                        Phi_PP_stepping = 1

                    convergence_iteration += 1

                # Break out of the background loop if the maximum convergence flag is set
                if max_convergence_flag == 1:
                    break
                # If we have found both low and high Phi_PP values, we can interpolate to find the target Phi_PP
                if Phi_PP_high != Phi_PP_low:
                    slope = (Phi_PP_high - Phi_PP_low)/(beta_high - beta_low)
                    Phi_PP = Phi_PP_low + (slope*(beta_target - beta_low))

                # For the first iteration, we set the old Phi_PP to the current Phi_PP
                if currently_initial_weighting == 0:
                    convergence_metric = np.abs(Phi_PP - Phi_PP_old)/(Phi_PP_old*weight_raising_amount)
                    if convergence_metric < convergence_tolerance:
                        done = 1
                if currently_initial_weighting == 1:
                    currently_initial_weighting = 0
                if done == 0:
                    weight_amp_old = weight_amp
                    Phi_PP_old = Phi_PP
                    weight_amp += weight_raising_amount
                    weights = (weights * weight_amp) / weight_amp_old
                else:
                    done = 1
            # If the maximum convergence flag is set, we break out of the j_n loop
            if max_convergence_flag == 1:
                break
            if j_n == 0:
                Phi_PP_central = Phi_PP
            if j_n == 1:
                Phi_PP_J_bigger = Phi_PP
            if j_n == 2:
                Phi_PP_J_smaller = Phi_PP
        
        if Phi_PP_J_bigger > Phi_PP_J_smaller:
            logger.error("The \u03B2-level confidence bound of \u03A6_PP has been calculated to be higher when the J-factors include their additive uncertainties than when the J-factors include their subtractive uncertainties.")
            print("Error: the \u03B2-level confidence bound of \u03A6_PP has been calculated to be higher when the J-factors include their additive uncertainties than when the J-factors include their subtractive uncertainties.")

        # If the maximum convergence flag is set, we set the Phi_PP values to zero
        if max_convergence_flag == 1:
            Phi_PP_central = 0
            Phi_PP_J_bigger = 0
            Phi_PP_J_smaller = 0
            
        # Store the results for the current mass point
        Phi_PP_central_for_m[m] = Phi_PP_central
        Phi_PP_lower_for_m[m] = Phi_PP_J_bigger
        Phi_PP_upper_for_m[m] = Phi_PP_J_smaller

    # Compute the cross-section limits
    betas = np.ones(mass_number)*beta_target
    N_bounds = np.zeros(mass_number)
    plus_dPhiPP = Phi_PP_upper_for_m - Phi_PP_central_for_m
    minus_dPhiPP = Phi_PP_central_for_m - Phi_PP_lower_for_m
    sigmav = 8*np.pi*(masses**2)*Phi_PP_central_for_m/integrated_energy_spectrum_vals
    plus_dsigmav = 8*np.pi*(masses**2)*plus_dPhiPP/integrated_energy_spectrum_vals
    minus_dsigmav = 8*np.pi*(masses**2)*minus_dPhiPP/integrated_energy_spectrum_vals

    output_data = np.column_stack((masses, integrated_energy_spectrum_vals, betas, N_bounds, Phi_PP_central_for_m, plus_dPhiPP, minus_dPhiPP, sigmav, plus_dsigmav, minus_dsigmav))

    with open(set_filename, "r") as set_file:
        set_file_whole = set_file.readlines()

    output_filename_besides_beta_target_and_ext = os.path.splitext(os.path.basename(model_filename))[0] + os.path.splitext(os.path.basename(set_filename))[0]
    output_path = f'Output/{output_filename_besides_beta_target_and_ext}_{beta_target}.out'

    logger.info(header)
    print(header)

    if formatted_output:
        with open(output_path, "w") as f:
            f.write("".join(set_file_whole) + header)
            for row in output_data:
                line = output_fmt.format(
                    mass=row[0],
                    spectrum=row[1],
                    beta=row[2],
                    nbound=row[3],
                    phipp=row[4],
                    dphipp_p=row[5],
                    dphipp_m=row[6],
                    sigv=row[7],
                    dsigv_p=row[8],
                    dsigv_m=row[9],
                )
                f.write(line + "\n")
                logger.info(line)
                print(line)
    else:
        np.savetxt(f'Output/{output_filename_besides_beta_target_and_ext}_{beta_target}.out', output_data, fmt = '%.15g', delimiter = '\t', header="".join(set_file_whole) + header)
        for row in output_data:
            line = output_fmt.format(
                mass=row[0],
                spectrum=row[1],
                beta=row[2],
                nbound=row[3],
                phipp=row[4],
                dphipp_p=row[5],
                dphipp_m=row[6],
                sigv=row[7],
                dsigv_p=row[8],
                dsigv_m=row[9],
            )
            logger.info(line)
            print(line)

def compute_P_bar(N_bar_max, P_ij, cutoffs_N_upper, weights, gal_number, energy_bin_number, N_O_bar):
    """
    Computes the combined probability distribution P_bar for a set of galaxies and energy bins.

    This function aggregates the probability distributions of photon counts from multiple galaxies and energy bins,
    applying specified weights and cutoffs, and combines them via convolution to produce the overall probability
    distribution P_bar.

    Parameters:
        N_bar_max (int): Maximum number of photons in the combined distribution.
        P_ij (np.ndarray): Probability distribution array of shape (N, gal_number, energy_bin_number), where N is the
            maximum photon count per galaxy and energy bin.
        cutoffs_N_upper (np.ndarray): Array of shape (gal_number, energy_bin_number) specifying the upper cutoff for
            photon counts for each galaxy and energy bin.
        weights (np.ndarray): Array of shape (gal_number, energy_bin_number) specifying the weight to apply to each
            galaxy and energy bin.
        gal_number (int): Number of galaxies considered.
        energy_bin_number (int): Number of energy bins considered.
        N_O_bar (int): Maximum allowed length for the pruned array (array will be truncated to N_O_bar + 1 elements).
    
    Returns:
        np.ndarray: The pruned combined probability distribution P_bar as a 1D array of length N_bar_max + 1."""
    # 
    P_bar_comb = np.zeros(N_bar_max + 1)
    N_gal_arr_comb = np.arange(cutoffs_N_upper[0, 0])
    N_bar_gal_arr_comb = np.floor((N_gal_arr_comb * weights[0, 0])).astype(int)

    # If gal_number is 1 and energy_bin_number is 1, we can directly compute P_bar
    for n in N_gal_arr_comb:
        P_bar_comb[N_bar_gal_arr_comb[n]] += P_ij[n, 0, 0]
    if gal_number == 1 and energy_bin_number == 1:
        P_bar = P_bar_comb
    # Else, we need to convolve the probabilities
    else:
        i_floor = 1
        j_floor = 1
        if energy_bin_number > 1:
            i_floor = 0
        for i in np.arange(i_floor, gal_number):
            if i == 1:
                j_floor = 0
            for j in np.arange(j_floor, energy_bin_number):
                N_gal_arr_ij = np.arange(cutoffs_N_upper[i, j])
                N_bar_gal_arr_ij = np.floor(N_gal_arr_ij * weights[i, j]).astype(int)
                P_bar_gal_ij = np.zeros(np.floor((cutoffs_N_upper[i, j] - 1) * weights[i, j]).astype(int) + 1)
                for n in N_gal_arr_ij:
                    P_bar_gal_ij[N_bar_gal_arr_ij[n]] += P_ij[n, i, j]
                P_bar_comb = SciSig.convolve(P_bar_comb, P_bar_gal_ij)        
        P_bar = P_bar_comb
    return prune_P_bar(P_bar, N_O_bar)

def prune_P_bar(P_bar, N_O_bar):
    """
    Prunes and thresholds the input array P_bar based on the number of observed photons.

    Parameters:
        P_bar (np.ndarray): Input array to be pruned and thresholded.
        N_O_bar (int): Maximum allowed length for the pruned array (array will be truncated to N_O_bar + 1 elements).

    Returns:
        np.ndarray: The pruned and thresholded array.

    Notes:
        - If the size of P_bar exceeds N_O_bar + 1, it is truncated to that length.
        - Elements in P_bar that are less than (max(P_bar) / P_bar_zero_out_threshold_denom) are set to zero.
        - The variable P_bar_zero_out_threshold_denom is assumed to be defined in the enclosing scope.
    """

    if P_bar.size > N_O_bar + 1:
        P_bar = P_bar[:N_O_bar + 1]
    P_bar[P_bar < np.max(P_bar)/P_bar_zero_out_threshold_denom] = 0
    return P_bar

def compute_P_sig(N_max, N_S_mean, gal_number, energy_bin_number, weights):
    """
    Computes the signal probability array, upper cutoff indices, and weighted maximum signal count.

    Parameters:
        N_max (int): The maximum number of signal events.
        N_S_mean (np.ndarray): 2D array of shape (gal_number, energy_bin_number) containing the mean expected signal counts
            for each galaxy and energy bin.
        gal_number (int): The number of galaxies considered.
        energy_bin_number (int): The number of energy bins considered.
        weights (np.ndarray): Array of weights for each galaxy and energy bin, used to compute the weighted maximum signal count.

    Returns:
        P_sig_arr (np.ndarray): 3D array of shape (N_max + 1, gal_number, energy_bin_number) containing the Poisson probability
            3D array of shape (N_max + 1, gal_number, energy_bin_number) containing the Poisson probability
            mass function values for each possible signal count, galaxy, and energy bin.
        cutoffs_N_S_upper (np.ndarray): 2D integer array of shape (gal_number, energy_bin_number) containing the upper cutoff index for
            the signal count in each galaxy and energy bin, determined by a probability threshold.
        N_S_bar_max (int): The weighted maximum signal count, computed as the weighted sum of the upper cutoff indices minus one,
            floored to the nearest integer.
    Notes:
        - The function uses the Poisson probability mass function to compute the probabilities for each possible signal
          count from 0 to N_max for each galaxy and energy bin.
    """
    P_sig_arr = np.zeros((N_max + 1, gal_number, energy_bin_number))
    cutoffs_N_S_upper = (N_max + 1) * np.ones((gal_number, energy_bin_number))
    for i in np.arange(gal_number):
        for j in np.arange(energy_bin_number):
            cutoff_search_flag = 0
            P_sig_ij_tol = poisson.pmf(np.ceil(N_S_mean[i, j])-1, N_S_mean[i, j])/P_sig_ij_tol_denom
            for N_S_ij in np.arange(N_max + 1):
                if cutoff_search_flag == 1:
                    P_sig_ij_specific = poisson.pmf(N_S_ij, N_S_mean[i, j])
                    if P_sig_ij_specific < P_sig_ij_tol:
                        cutoff_search_flag = 2
                        cutoffs_N_S_upper[i, j] = N_S_ij
                    else:
                        P_sig_arr[N_S_ij, i, j] = P_sig_ij_specific
                if cutoff_search_flag == 0:
                    P_sig_ij_specific = poisson.pmf(N_S_ij, N_S_mean[i, j])
                    if P_sig_ij_specific >= P_sig_ij_tol:
                        P_sig_arr[N_S_ij, i, j] = P_sig_ij_specific
                        cutoff_search_flag = 1
    cutoffs_N_S_upper = cutoffs_N_S_upper.astype(int)
    N_S_bar_max = weights.ravel().dot((cutoffs_N_S_upper-1).ravel())
    N_S_bar_max = np.floor(N_S_bar_max).astype(int)

    return P_sig_arr, cutoffs_N_S_upper, N_S_bar_max

def compute_beta(P_B_bar, P_S_bar):
    """
    Computes the beta value based on using sparse matrix operations.

    This function constructs sparse matrices from the input arrays, computes their Kronecker product,
    and then calculates the sum of the lower triangular part of the resulting matrix. The final beta
    value is computed as 1 minus this sum.

    Args:
        P_B_bar (numpy.ndarray): A 1D array representing the first probability vector.
        P_S_bar (numpy.ndarray): A 1D array representing the second probability vector.

    Returns:
        float: The computed beta value.

    Notes:
        - The function uses SciSpa for matrix operations. Requires SciPy verion 1.13.0. Later versions may produce warnings or errors as the *_matrix functions are deprecated.
        - P_B_bar and P_S_bar are pruned w.r.t. the number of observed photons, so they should be the same size, meaning that the else is not expected to be executed.
    """
    column = 0
    row = 0
    if P_B_bar.size >= P_S_bar.size:
        column = SciSpa.csc_matrix(P_B_bar[::-1].reshape(P_B_bar.size,1))
        row = SciSpa.csr_matrix(P_S_bar)
    else:
        column = SciSpa.csc_matrix(P_S_bar[::-1].reshape(P_S_bar.size,1))
        row = SciSpa.csr_matrix(P_B_bar)
    beta_comp_calc_matrix = SciSpa.kron(column, row)
    beta_comp = SciSpa.coo_matrix.sum(SciSpa.tril(beta_comp_calc_matrix))
    beta_comp_calc_matrix = 0
    beta = 1 - beta_comp

    return beta

# Using the __name__ == "__main__" construct to allow for easy import of this script in other scripts without running the main function
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run madhat2 for multiple (set, model) combinations."
    )
    parser.add_argument(
        '--models', '-m', # Can be used as -m or --models
        nargs='+', # Accepts multiple model files
        type=lambda path: os.path.join("Input",path), # Converts the input to a path relative to the Input directory
        default=[os.path.join("Input",f) for f in default_models], # Default to the list of default models
        help='List of model files (e.g. Input/dmmumu.dat Input/dmtautau.dat ...)' 
    )
    parser.add_argument(
        '--sets', '-s', # Can be used as -s or --sets
        nargs='+', 
        type=lambda path: os.path.join("Input",path), 
        default=[os.path.join("Input",f) for f in default_sets],
        help='List of set files (e.g. Input/set1.dat Input/set2.dat ...)'
    )
    parser.add_argument(
        '--formatted_output', '-f', # Can be used as -f or --formatted_output
        default=formatted_output, # Default is picked from the output of the file
        help='If set, the output will be formatted with aligned columns; otherwise, it will be tab-separated values (same as MADHATv2).'
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    # Validate input files
    for file in args.models + args.sets:
        if not os.path.exists(file):
            logger.error(f"file not found: {file}")
            print(f"Error: file not found: {file}")
            exit(1)
    # Validate formatted_output argument, takes True, true, 1, yes, on as True and everything else as False        
    if not isinstance(args.formatted_output, bool):
        if str(args.formatted_output).lower() in ['true', '1', 'yes', 'on']:
            args.formatted_output = True
        else:
            args.formatted_output = False

    for set_file in args.sets:
        for model_file in args.models:
            logger.info(f"Running {set_file[6:]} with {model_file[6:]}")
            print(f"Running {set_file[6:]}  {model_file[6:]}")
            start = time.time()
            main(set_file, model_file, formatted_output=args.formatted_output)
            elapsed = time.time() - start
            logger.info(f"Done in {elapsed:.2f}s")
            print(f"done in {elapsed:.2f}s")