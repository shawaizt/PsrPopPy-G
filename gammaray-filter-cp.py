#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 13:12:16 2022

@author: st3105
"""
import numpy as np
import pandas as pd
import healpy as hp
import scipy.stats as stats
from scipy.stats import truncnorm
import random
import math


"""
Set path for where the output from the synthesis is located (path) and where you want you output after gamma-ray selection
effects based on different luminosity models to be located (path_out).

Also provide path to where your sensitivity map for Fermi-LAT is located (path_fermi_map).
"""

path = '//home/user/multi-synthesis/'
path_out = '/home/user/multi-synthesis/'
path_fermi_map = '/home/user/multi-synthesis/3PC_sensitivity_v3.fits'

df_result = pd.read_csv(path+'all-radio.csv', encoding='unicode_escape')

epsilon = [math.acos(random.random()) for _ in range(len(df_result))]
df_result['epsilon'] = epsilon

#df_result['F1400'].astype(np.float64)

c=299792458.0 
beaming_frac = 1.0 #Only relevant for Gonthier+ 2018 models

cols = df_result.columns[df_result.dtypes.eq('object')]
df_result[cols] = df_result[cols].apply(pd.to_numeric, axis=1)


#Unit conversions:
df_result['dist_cm'] = df_result['dist'] * 3.086*10.0**21
df_result['pdot_gon'] = df_result['pdot_int'] / 10.0**(-21)
df_result['p_s'] = df_result['p_ms'] / 1000.0


#Read the Fermi-LAT sensitivity map and find the sensitivity at the position of each psrpoppy generated msp
sensitivity_map = hp.read_map(path_fermi_map)

gl = np.array(df_result['gl'])
gb = np.array(df_result['gb'])

sens_calc = lambda l, b : hp.get_interp_val(sensitivity_map, l, b, lonlat=True)
df_result['sens_limit'] = np.float64(sens_calc(gl, gb))


#Define functions to find luminosties using different models, flux and edot below.

def g_lum_2dplane(b, edot, p):
    """Luminosity using the 2D fundamental plane. Eq 6 in Kalapotharakos et. al. 2019.
    Output is in erg/s. Input B should be in Gauss, edot in erg/s. period in seconds"""
    r_star = 10**6 #cm
    b_star = ( (edot * c**3 * p**4)/(4*np.pi**4 * r_star**6) * 1.5 )**(0.5)
    g_lum = 10.0**15 * b_star**0.11 * edot**0.51
    return g_lum

def calc_ecut(method, edot):
    """Calcute the cut-off energy e_cut to be used in Fundamental Plane realtions.
    Input is edot in erg/s and output is e_cut in MeV"""
    if method == 'median':
        a_e = 0.18
        b_e = 2.83
        log_e_cut = a_e * np.log10(edot / (10**(34.5))) + b_e 
        e_cut = 10**(log_e_cut) #in MeV
    elif method == 'sample':
        mean = 3.513
        std_dev = 0.238
        amp = 0.935
        log_e_cut = np.random.normal(loc=mean, scale=std_dev, size=len(df_result)) * amp
        e_cut = 10**log_e_cut
#        e_cut = np.random.lognormal(1.09, 0.09, len(df_result))
    elif method == 'quad':
        a_e = -0.04968
        b_e = 3.51984
        c_e = -61.8024
        log_e = a_e * (np.log10(edot))**2 + b_e * np.log10(edot) + c_e #In GeV
        e_cut_gev = 10**(log_e)
        e_cut = e_cut_gev * 1000
    else:
        raise ValueError(f"Unsupported e_cut findind method: {method}. Please select median or sample")
    return e_cut
    
def g_lum_3dplane(b, edot, e_cut, p):
    """Luminosity using the 3D fundamental plane of eq 9 in Kalapotharakos et al 2022. 
    Eq 2.28 and table 3 in Ploeg et. al. 2020 used to get e_cut. We're using the median of e_cut to be e_cut.
    Output is in erg/s. Input B should be in Gauss, edot in erg/s, period in seconds, e_cut in MeV."""
    r_star = 10**6 #cm
    b_star = ( (edot * c**3 * p**4)/(4*np.pi**4 * r_star**6) * 1.5 )**(0.5)
    g_lum = 10.0**14.3 * e_cut**1.39 * b_star**0.12 * edot**0.39
    return g_lum


def g_lum_bpl(u, alpha1, alpha2, l_min, l_max, l_break):
    """Sample from a broken powerlaw to find luminosity. Outlined in Bartels et. al. (2018) paper in eq 5c.
    Using inverse transform sampling."""
    # First find the normalisation
   
    i1 = ((l_break)**(1-alpha1) - (l_min)**(1-alpha1)) / (1-alpha1)
    i2 = (l_break**(alpha2 - alpha1))*((l_max)**(1-alpha2) - (l_break)**(1-alpha2)) / (1-alpha2)    
    
    k = 1.0 / (i1 + i2)
    divison_val = i1 * k
    
    #following the method of https://github.com/grburgess/brokenpl_sample/blob/master/sample_broken_power_law.ipynb 
    #where the use bernolulli trials and booleans instead of an if/else.
    
    y = np.empty_like(u)
    idx = stats.bernoulli.rvs(divison_val, size=len(u)).astype(bool)

    # inverse transform sample the lower part for the "successes"
    y[idx] = np.power(
        u[idx] * (np.power(l_break, 1.0 - alpha1) - np.power(l_min, 1.0 - alpha1))
        + np.power(l_min, 1.0 - alpha1),
        1.0 / (1 - alpha1),
        )
    
    # inverse transform sample the upper part for the "failures"
    y[~idx] = np.power(
        u[~idx] * (np.power(l_max, 1.0 - alpha2) - np.power(l_break, 1.0 - alpha2))
        + np.power(l_break, 1.0 - alpha2),
        1.0 / (1 - alpha2),
        )
    return y

def g_lum_lognorm(l_min, l_max, mean, stdev, size):
    l_mi, l_ma = np.log10(l_min) , np.log10(l_max)
    a, b = (l_mi - mean) / stdev, (l_ma - mean) / stdev
    rv = truncnorm(a=a, b=b)
    l = 10.0**(rv.rvs(size=size) * stdev + mean) 
    return l
 
def g_flux_calc(g_lum, d):
    g_flux = g_lum / ((4.0*np.pi*beaming_frac)*d**2)
    return g_flux

def g_flux_calc_beamed(g_lum, d, alpha, epsilon):
    
    result = []
    for a, b, c, d in zip(g_lum, d, alpha, epsilon):
        if c < -d + 0.6109:
            result.append(a / (4.0*np.pi*1.9 * b**2))
        else:
            result.append(a / (4.0*np.pi*1.0 * b**2))
    return pd.Series(result)
     

def g_flux_beamed_song(g_lum, edot, dist):
    """
    Calculate the gamma-ray flux considering beaming factors from Song+2024.
    g_lum: gamma-ray luminosities
    edot: spindown power values
    dist: distances

    Out: gamma-ray flux values accounting for beaming
    """
    #make sure inputs to numpy arrays
    g_lum = np.array(g_lum)
    edot = np.array(edot)
    dist = np.array(dist)

    #make sure all inputs have the same length
    if not (len(g_lum) == len(edot) == len(dist)):
        raise ValueError("Input arrays must have the same length")

    #set beaming factor based on edot
    f_g = np.zeros_like(edot)
    f_g[edot > 4*10**32] = 1.0
    f_g[edot <= 4*10**32] = 0.0

    #calculate gamma-ray flux with random beaming factor
    gflux = np.zeros_like(g_lum)
    for i in range(len(g_lum)):
        if random.random() <= f_g[i]:  # Flux is beamed
            gflux[i] = g_lum[i] / (4 * np.pi * dist[i]**2)
        else:  # Flux is zero if not beamed
            gflux[i] = 0

    return gflux

def edot_calc(pdot, p):
    """Finds edot using HPA eq 3.6. Period should in seconds. Output is in erg/s """
    edot = 3.95*10.0**31 * (pdot/10.0e-15) * (p)**(-3)
    return edot

def g_lum_edot(edot):
    k = 10**33
    sqrt_term = edot / k 
    lum = k * (sqrt_term)**(1/2)
    return lum.astype(float)

#Initialize dictionary which will store the dataframes for each luminosity model

dfs_detected = {}
dfs_all = {}
results = {}

def detector(lum, all_df=False):
    """
    This is the "detector" function which outputs how many CPs would be detected for the selected 
    luminsoty model. It also gives a dataframe which has all the detected CPs.
    If all_df=True, it also outputs a dataframe which is same as df_result plus a coloumn with luminosities 
    for all CPs for the selected model.
    """
       
                
    if lum == 'bpl':
        t = np.random.uniform(0,1, len(df_result))
        df_result['L_gamma_ergs'] = g_lum_bpl(t, 0.97,2.60,10.0**30,10.0**37,10.0**33.24)
        df_result['g_flux'] = g_flux_calc(df_result['L_gamma_ergs'], df_result['dist_cm'])
        dfs_detected['bpl'] = df_result[df_result.g_flux >=  df_result['sens_limit']]
        results['bpl'] = "Number of detected pulsars for Broken Power Law L:", len(dfs_detected['bpl'])
        if all_df == True:
            dfs_all['bpl'] = df_result[df_result.g_flux >= 1.0e-97]
            return results, dfs_detected, dfs_all

    elif lum == '2dplane':
        df_result['L_gamma_ergs'] = g_lum_2dplane(df_result['B']*10.0**8, df_result['edot'], df_result['p_s'])
        df_result['log-l'] = np.log10(df_result['L_gamma_ergs'])
        df_result['log-l'] = df_result.apply(lambda row: row['log-l'] + np.random.normal(0.0, 0.2), axis=1)
        df_result['L_gamma_ergs'] = 10**df_result['log-l']
        df_result['g_flux'] = g_flux_calc(df_result['L_gamma_ergs'], df_result['dist_cm'])
        dfs_detected['2dplane'] = df_result[df_result.g_flux >=  df_result['sens_limit']]
        results['2dplane'] = "Number of detected pulsars for 2D Fundamental plane L:", len(dfs_detected['2dplane'])
        if all_df == True:
            dfs_all['2dplane'] = df_result[df_result.g_flux >= 1.0e-97]
            return results, dfs_detected, dfs_all
    
    elif lum == '3dplane':
        df_result['e_cut'] = calc_ecut('sample', df_result['edot'])
        df_result['L_gamma_ergs'] = g_lum_3dplane(df_result['B']*10.0**8, df_result['edot'], df_result['e_cut'], df_result['p_s'])
        df_result['log-l'] = np.log10(df_result['L_gamma_ergs'])
        df_result['log-l'] = df_result.apply(lambda row: row['log-l'] + np.random.normal(0.0, 0.2), axis=1)
        df_result['L_gamma_ergs'] = 10**df_result['log-l']
        df_result['g_flux'] = g_flux_calc(df_result['L_gamma_ergs'], df_result['dist_cm'])
        dfs_detected['3dplane'] = df_result[df_result.g_flux >=  df_result['sens_limit']]
        results['3dplane'] = "Number of detected pulsars for 3D Fundamental plane L:", len(dfs_detected['3dplane'])
        if all_df == True:
            dfs_all['3dplane'] = df_result[df_result.g_flux >= 1.0e-97]
            return results, dfs_detected, dfs_all
        
    elif lum == 'edot':
        df_result['L_gamma_ergs'] = g_lum_edot(df_result['edot'])
        df_result['log-l'] = np.log10(df_result['L_gamma_ergs'])
        df_result['log-l'] = df_result.apply(lambda row: row['log-l'] + np.random.normal(0.0, 0.2), axis=1)
        df_result['L_gamma_ergs'] = 10**df_result['log-l']
        df_result['g_flux'] = g_flux_beamed_song(df_result['L_gamma_ergs'], df_result['edot'], df_result['dist_cm'])
        dfs_detected['edot'] = df_result[df_result.g_flux >=  df_result['sens_limit']]
        results['edot'] = "Number of detected pulsars for Heuristic L:", len(dfs_detected['edot'])
        if all_df == True:
            dfs_all['edot'] = df_result[df_result.g_flux >= 1.0e-97]
            return results, dfs_detected, dfs_all
        
    elif lum == 'loglum':
        df_result['L_gamma_ergs'] = g_lum_lognorm(10.0**30, 10.0**37, 32.61, 0.63, len(df_result))
        df_result['g_flux'] = g_flux_calc(df_result['L_gamma_ergs'], df_result['dist_cm'])
        dfs_detected['loglum'] = df_result[df_result.g_flux >=  df_result['sens_limit']]
        results['loglum'] = "Number of detected pulsars for Log-normal L:", len(dfs_detected['loglum'])
        if all_df == True:
            dfs_all['loglum'] = df_result[df_result.g_flux >= 1.0e-97]
            return results, dfs_detected, dfs_all

    return results, dfs_detected


lums = {'2dplane', '3dplane', 'edot'}  

for l in lums:
    df_name = f'{l}'
    detector(df_name, all_df=False) 
    dfs_detected[df_name].to_csv(path_out+f'detc-{df_name}.csv', index=False)

results
#%%
