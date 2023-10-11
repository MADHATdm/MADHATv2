#!/usr/bin/env python
# coding: utf-8

import numpy as np

#target beta and related
beta_target = 0.95
beta_tolerance = 0.001

#whether binning takes place, the type of weighting to take place, and related
binning = 1
weighting_type_flag = 1
weights_original = np.array([[]])

#parameters pertinent to the convergence aspect of the algorithm 
weight_raising_amount = 2
convergence_tolerance = 1e-4

#parameters related to different probability tolerances
bgd_prob_sum_tolerance = 1e-4
P_sig_ij_tol_denom = 1e4
P_bar_zero_out_threshold_denom = 1e4
energy_fraction_zero_out_threshold_denom = 1e4

#filepaths
model_filepath = "Input/dmbb.dat"
set_filepath = "Input/set0.dat"
NOBS_filepath = "PMFdata/nobs8bpd.dat"
pmf_data_filepath = "PMFdata/pmf8bpd.dat"


from scipy.stats import poisson
import scipy.signal as SciSig
import scipy.sparse as SciSpa
import os

energy_fractions = np.array([1])
energy_bin_number = energy_fractions.size

with open(model_filepath, "r") as model_file:
    model_file_line = "#"
    while model_file_line.startswith("#"):
        model_file_line = model_file.readline()
    
    model_file_lines_minus_header = np.append(model_file_line, [line for line in model_file.readlines() if line.strip()])
    mass_number = np.size(model_file_lines_minus_header)
    masses = np.zeros(mass_number)
    integrated_energy_spectrum_vals = np.zeros(mass_number)
    if binning == 1:
        energy_bin_number = np.size(model_file_lines_minus_header[0].split()) - 2
        energy_fractions_array = np.zeros((mass_number, energy_bin_number))
    for m in np.arange(mass_number):
        masses[m] = model_file_lines_minus_header[m].split()[0]
        integrated_energy_spectrum_vals[m] = model_file_lines_minus_header[m].split()[1]
        if binning == 1:
            energy_fractions_array[m, :] = model_file_lines_minus_header[m].split()[2:]

Phi_PP_central_for_m = np.zeros(mass_number)
Phi_PP_lower_for_m = np.zeros(mass_number)
Phi_PP_upper_for_m = np.zeros(mass_number)

with open(set_filepath, "r") as set_file:
    set_file_line = "#"
    while set_file_line.startswith("#"):
        set_file_line = set_file.readline()
    set_file_lines_minus_header = np.append(set_file_line, [line for line in set_file.readlines() if line.strip()])
    gal_number = np.size(set_file_lines_minus_header)
    ID = np.zeros(gal_number, dtype = int)
    N_O_ij_original = np.zeros((gal_number, energy_bin_number))
    J_red_i = np.zeros((gal_number,1))
    J_red_uncert_plus_i = np.zeros((gal_number,1))
    J_red_uncert_minus_i = np.zeros((gal_number,1))
    exposures_i = np.zeros((gal_number,1))
    for i in np.arange(gal_number):
        ID[i] = set_file_lines_minus_header[i].split()[0]
        J_red_i[i] = set_file_lines_minus_header[i].split()[1]
        J_red_uncert_plus_i[i] = set_file_lines_minus_header[i].split()[2]
        J_red_uncert_minus_i[i] = set_file_lines_minus_header[i].split()[3]

with open(NOBS_filepath, "r") as NOBS_file:
    NOBS_file_line = "#"
    while NOBS_file_line.startswith("#"):
        NOBS_file_line = NOBS_file.readline()
    NOBS_file_lines_minus_header = np.append(NOBS_file_line, [line for line in NOBS_file.readlines() if line.strip()])
    for i in np.arange(gal_number):
        exposures_i[i] = NOBS_file_lines_minus_header[energy_bin_number*(ID[i]-1)].split()[-1]
        for j in np.arange(energy_bin_number):
            N_O_ij_original[i, j] = NOBS_file_lines_minus_header[energy_bin_number*(ID[i]-1) + j].split()[-2]

J_i_central = 10 ** J_red_i
J_i_bigger = 10 ** (J_red_i + J_red_uncert_plus_i)
J_i_smaller = 10 ** (J_red_i - J_red_uncert_minus_i)
with open(pmf_data_filepath, "r") as pmf_data_file:
    pmf_data_file_line = "#"
    while pmf_data_file_line.startswith("#"):
        pmf_data_file_line = pmf_data_file.readline()
    pmf_data_file_lines_minus_header = np.append(pmf_data_file_line, [line for line in pmf_data_file.readlines() if line.strip()])
    N_max_hard = np.size(pmf_data_file_lines_minus_header) - 1
    pmf_data_original = np.zeros((N_max_hard + 1, gal_number, energy_bin_number))
    for i in np.arange(gal_number):
        for j in np.arange(energy_bin_number):
            for N_B_ij in np.arange(N_max_hard + 1):
                pmf_data_original[N_B_ij, i, j] = pmf_data_file_lines_minus_header[N_B_ij].split()[((ID[i]-1)*energy_bin_number)+j+1]

for m in np.arange(mass_number):
    mass = masses[m]
    integrated_energy_spectrum = integrated_energy_spectrum_vals[m]
    if binning == 1:
        energy_fractions = energy_fractions_array[m, :]
        fraction_mask = energy_fractions < energy_fractions.max() / energy_fraction_zero_out_threshold_denom
        energy_fractions = np.delete(energy_fractions, fraction_mask)
        energy_bin_number = energy_fractions.size
        N_O_ij = np.delete(N_O_ij_original, fraction_mask, 1)
        pmf_data = np.delete(pmf_data_original, fraction_mask, 2)
        if weighting_type_flag == 0:
            weights_intermediate = np.delete(weights_original, fraction_mask, 1)
            weights_intermediate = weights_intermediate / weights_intermediate.max()
    else:
        N_O_ij = N_O_ij_original
        pmf_data = pmf_data_original

    if weighting_type_flag == 1:
        N_B_mean = np.zeros((gal_number, energy_bin_number))
        for i in np.arange(gal_number):
            for j in np.arange(energy_bin_number):
                for N_B_ij in np.arange(1, N_max_hard + 1):
                    N_B_mean[i, j] += N_B_ij * pmf_data[N_B_ij, i, j]
        weights_original = J_i_central * exposures_i * energy_fractions / N_B_mean
    weights_original = weights_original / weights_original.max()
    J_i = J_i_central
    Phi_PP_central = 0
    Phi_PP_J_bigger = 0
    Phi_PP_J_smaller = 0
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
        done = 0
        while done != 1:
            N_O_bar = weights.ravel().dot(N_O_ij.ravel())
            N_O_bar = np.floor(N_O_bar).astype(int)

            min_weight = weights.min()

            N_max = N_max_hard

            if min_weight != 0 and np.ceil(N_O_bar / min_weight) < N_max_hard:
                N_max = np.ceil(N_O_bar / min_weight).astype(int)
            P_bgd_ij = np.zeros((N_max + 1, gal_number, energy_bin_number))
            for i in np.arange(gal_number):
                for j in np.arange(energy_bin_number):
                    P_bgd_ij[:, i, j] = pmf_data[np.arange(N_max + 1), i, j]

            cutoffs_N_B_upper = (N_max + 1) * np.ones((gal_number, energy_bin_number))
            for i in np.arange(gal_number):
                for j in np.arange(energy_bin_number):
                    P_bgd_ij_sum = 0
                    N_B_ij = 0
                    while P_bgd_ij_sum < 1 - bgd_prob_sum_tolerance and N_B_ij <= N_max:
                        P_bgd_ij_sum += P_bgd_ij[N_B_ij, i, j]
                        N_B_ij += 1
                    cutoffs_N_B_upper[i, j] = N_B_ij
            cutoffs_N_B_upper = cutoffs_N_B_upper.astype(int)
            N_B_bar_max = weights.ravel().dot((cutoffs_N_B_upper-1).ravel())
            N_B_bar_max = np.floor(N_B_bar_max).astype(int)

            P_B_bar_length = N_B_bar_max + 1
            P_B_bar_comb = np.zeros(P_B_bar_length)
            N_B_gal_arr_comb = np.arange(cutoffs_N_B_upper[0, 0])
            N_B_bar_gal_arr_comb = np.floor((N_B_gal_arr_comb * weights[0, 0])).astype(int)
            for n in N_B_gal_arr_comb:
                P_B_bar_comb[N_B_bar_gal_arr_comb[n]] += P_bgd_ij[n, 0, 0]
            if gal_number == 1 and energy_bin_number == 1:
                P_B_bar = P_B_bar_comb
            else:
                i_floor = 1
                j_floor = 1
                if energy_bin_number > 1:
                    i_floor = 0
                for i in np.arange(i_floor, gal_number):
                    if i == 1:
                        j_floor = 0
                    for j in np.arange(j_floor, energy_bin_number):
                        N_B_gal_arr_ij = np.arange(cutoffs_N_B_upper[i, j])
                        N_B_bar_gal_arr_ij = np.floor(N_B_gal_arr_ij * weights[i, j]).astype(int)
                        P_B_bar_gal_ij = np.zeros(np.floor((cutoffs_N_B_upper[i, j] - 1) * weights[i, j]).astype(int) + 1)
                        for n in N_B_gal_arr_ij:
                            P_B_bar_gal_ij[N_B_bar_gal_arr_ij[n]] += P_bgd_ij[n, i, j]
                        P_B_bar_comb = SciSig.convolve(P_B_bar_comb, P_B_bar_gal_ij)        
                P_B_bar = P_B_bar_comb
                P_B_bar_length = P_B_bar.size
            if P_B_bar_length > N_O_bar + 1:
                P_B_bar = P_B_bar[:N_O_bar + 1]
                P_B_bar_length = N_O_bar + 1
            P_B_bar[P_B_bar < np.max(P_B_bar)/P_bar_zero_out_threshold_denom] = 0

            beta_low = 0
            Phi_PP_low = 0
            Phi_PP_low_found = 0
            beta_high = 0
            Phi_PP_high = 0
            Phi_PP_high_found = 0
            Phi_PP_stepping = 0
            while Phi_PP_low_found == 0 or Phi_PP_high_found == 0:
                if Phi_PP_stepping == 1:
                    if loop_direction == 1:
                        if beta > beta_target:
                            loop_direction = -1
                            Phi_PP_step_factor = Phi_PP_step_factor ** (-1/2)
                    else:
                        if beta_target > beta:
                            loop_direction = 1
                            Phi_PP_step_factor = Phi_PP_step_factor ** (-1/2)
                    Phi_PP *= Phi_PP_step_factor
                N_S_mean = Phi_PP * J_i * exposures_i * energy_fractions

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

                P_S_bar_length = N_S_bar_max + 1
                P_S_bar_comb = np.zeros(P_S_bar_length)
                N_S_gal_arr_comb = np.arange(cutoffs_N_S_upper[0, 0])
                N_S_bar_gal_arr_comb = np.floor((N_S_gal_arr_comb * weights[0, 0])).astype(int)
                for n in N_S_gal_arr_comb:
                    P_S_bar_comb[N_S_bar_gal_arr_comb[n]] += P_sig_arr[n, 0, 0]
                if gal_number == 1 and energy_bin_number == 1:
                    P_S_bar = P_S_bar_comb
                else:
                    i_floor = 1
                    j_floor = 1
                    if energy_bin_number > 1:
                        i_floor = 0
                    for i in np.arange(i_floor, gal_number):
                        if i == 1:
                            j_floor = 0
                        for j in np.arange(j_floor, energy_bin_number):
                            N_S_gal_arr_ij = np.arange(cutoffs_N_S_upper[i, j])
                            N_S_bar_gal_arr_ij = np.floor(N_S_gal_arr_ij * weights[i, j]).astype(int)
                            P_S_bar_gal_ij = np.zeros(np.floor((cutoffs_N_S_upper[i, j] - 1) * weights[i, j]).astype(int) + 1)
                            for n in N_S_gal_arr_ij:
                                P_S_bar_gal_ij[N_S_bar_gal_arr_ij[n]] += P_sig_arr[n, i, j]
                            P_S_bar_comb = SciSig.convolve(P_S_bar_comb, P_S_bar_gal_ij)        
                    P_S_bar = P_S_bar_comb
                    P_S_bar_length = P_S_bar.size
                if P_S_bar_length > N_O_bar + 1:
                    P_S_bar = P_S_bar[:N_O_bar + 1]
                    P_S_bar_length = N_O_bar + 1
                P_S_bar[P_S_bar < np.max(P_S_bar)/P_bar_zero_out_threshold_denom] = 0

                column = 0
                row = 0
                if P_B_bar_length >= P_S_bar_length:
                    column = SciSpa.csc_matrix(P_B_bar[::-1].reshape(P_B_bar_length,1))
                    row = SciSpa.csr_matrix(P_S_bar)
                else:
                    column = SciSpa.csc_matrix(P_S_bar[::-1].reshape(P_S_bar_length,1))
                    row = SciSpa.csr_matrix(P_B_bar)
                beta_comp_calc_matrix = SciSpa.kron(column, row)
                beta_comp = SciSpa.coo_matrix.sum(SciSpa.tril(beta_comp_calc_matrix))
                beta_comp_calc_matrix = 0
                beta = 1 - beta_comp
                if np.abs(beta - beta_target) < beta_tolerance:
                    if beta <= beta_target:
                        beta_low = beta
                        Phi_PP_low = Phi_PP
                        Phi_PP_low_found = 1
                    if beta >= beta_target:
                        beta_high = beta
                        Phi_PP_high = Phi_PP
                        Phi_PP_high_found = 1

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

            if Phi_PP_high != Phi_PP_low:
                slope = (Phi_PP_high - Phi_PP_low)/(beta_high - beta_low)
                Phi_PP = Phi_PP_low + (slope*(beta_target - beta_low))

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
        if j_n == 0:
            Phi_PP_central = Phi_PP
        if j_n == 1:
            Phi_PP_J_bigger = Phi_PP
        if j_n == 2:
            Phi_PP_J_smaller = Phi_PP

    if Phi_PP_J_bigger > Phi_PP_J_smaller:
        print("Error: the \u03B2-level confidence bound of \u03A6_PP has been calculated to be higher when the J-factors include their additive uncertainties than when the J-factors include their subtractive uncertainties.")

    Phi_PP_central_for_m[m] = Phi_PP_central
    Phi_PP_lower_for_m[m] = Phi_PP_J_bigger
    Phi_PP_upper_for_m[m] = Phi_PP_J_smaller

betas = np.ones(mass_number)*beta_target
N_bounds = np.zeros(mass_number)
plus_dPhiPP = Phi_PP_upper_for_m - Phi_PP_central_for_m
minus_dPhiPP = Phi_PP_central_for_m - Phi_PP_lower_for_m
sigmav = 8*np.pi*(masses**2)*Phi_PP_central_for_m/integrated_energy_spectrum_vals
plus_dsigmav = 8*np.pi*(masses**2)*plus_dPhiPP/integrated_energy_spectrum_vals
minus_dsigmav = 8*np.pi*(masses**2)*minus_dPhiPP/integrated_energy_spectrum_vals

output_data = np.column_stack((masses, integrated_energy_spectrum_vals, betas, N_bounds, Phi_PP_central_for_m, plus_dPhiPP, minus_dPhiPP, sigmav, plus_dsigmav, minus_dsigmav))

with open(set_filepath, "r") as set_file:
    set_file_whole = set_file.readlines()

output_filename_besides_beta_target_and_ext = os.path.splitext(os.path.basename(model_filepath))[0] + os.path.splitext(os.path.basename(set_filepath))[0]
np.savetxt(f'Output/{output_filename_besides_beta_target_and_ext}_{beta_target}.out', output_data, fmt = '%.15g', delimiter = '\t', header = "".join(set_file_whole) + f"""
#####################################################################################################################################################################
# Fermi_Pass8R3_239557417_681169985
###################################################################################OUTPUT############################################################################
#Mass(GeV)   Spectrum       Beta      Nbound        PhiPP(cm^3 s^-1 GeV^-2)   +dPhiPP             -dPhiPP            sigv(cm^3 s^-1)        +dsigv             -dsigv""")