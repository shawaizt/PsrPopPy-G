#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 11 21:11:50 2025

@author: st3105
"""
import numpy as np
import random
from scipy.stats import weibull_min

c = 299792458 #m/s
k = (8 * np.pi**2 * (12*1000*100)**6) / (3 * (1.7*10**45) * (c)**3) # This is the one with R=12km, I=1.7*10^45 g cm^2, from Gonthier+'18

def draw_double_sided_exp(scale, origin=0.0):
    """Exponential distribution around origin, with scale height scale."""
    if scale == 0.0:
        return origin

    rn = random.random()
    sign = random.choice([-1.0, 1.0])

    return origin + sign * scale * np.log(rn)

def drawlnorm(mean, sigma):
    """Draw a random number from a log-normal distribution"""

    return 10.0**random.gauss(mean, sigma)

def powerlaw_bfield(minval, maxval, power):
    """Draw a value randomly from the powerlaw if Gonthier+18 is used to find B-field"""
    k = maxval**(power+1) - minval**(power+1)
    u = random.uniform(0, 1)
    y = (u*k + minval**(power+1))**(1/(power+1))
    return y

def p_of_t_fk06(p0, b8, age):
    """
    Equation 6 - Ridley & Lorimer 2010
    p0: the birth period in seconds
    b8: magnetic field in units of 10^8 G
    age: pulsar age in years
    
    Out: pulsar period at time=age in units of seconds
    """
    index_term = 3.0 - 1.0
    age_s = age * 3.15569e7  # convert to seconds
    kprime = k * (b8*10**8)**2

    brackets = p0**(index_term) + index_term * kprime * age_s

    return brackets**(1.0/index_term)

def p_of_t_gon18(p0, b8, age, alpha):
    """
    Equation 5 of Gonthier et al 2018
    Equation 6 - Ridley & Lorimer 2010
    p0: the birth period in seconds
    b8: magnetic field in units of 10^8 G
    age: pulsar age in years
    alpha: magnetic inclination angle in radians
    
    Out: pulsar period at time=age in units of seconds
    """
    
    r = 12e5 #cm
    c= 2.998e10 #cm/s
    I = 1.7e45 #g/cm^2
    
    age_s = age  * 3.15569e7  # convert to seconds
    b = b8*10**8 

    brackets = p0**2.0 + ((2.0 * np.pi**2.0 * r**6.0)/(c**3.0*I))*(1.0+(np.sin(alpha))**2.0)*b**2.0*age_s
    p_at_t = np.sqrt(brackets)
    return p_at_t

def pdot_fk06(p0, b8):
    """
    Equation 3 - Ridley & Lorimer 2010.
    p0: the birth period in seconds
    b8: magnetic field in units of 10^8 G
    
    Out: pulsar's period derivative in units of s/s
    """

    index_term = 2.0 - 3.0
    period_s =  p0
    kprime = k * (b8*10**8)**2
    return (kprime  * period_s**(index_term))

def pdot_gon18(p0, b8, age, alpha):
    """Pdot using the time derivative of eq 5 in Gonthier+'18.
    p0: the birth period in seconds
    b8: magnetic field in units of 10^8 G
    age: pulsar age in years
    alpha: magnetic inclination angle in radians
        
    Out: pulsar period derivate at time=age in units of s/s
    """
    r = 12e5 #cm
    c= 2.998e10 #cm/s
    I = 1.7e45 #g/cm^2
    
    np.sqrt((c**3 * I) / (np.pi**2 * r**6)) 
    np.sqrt(1.5543778010878183e+39)
    age_s = age  * 3.15569e7
    b = b8*10**8
    
    brackets = p0**2.0 + ((2.0 * np.pi**2.0 * r**6.0)/(c**3.0*I))*(1.0+(np.sin(alpha))**2.0)*b**2.0*age_s
    p = np.sqrt(brackets)
    pdot = 1/(2*p) * ((2.0 * np.pi**2.0 * r**6.0)/(c**3.0*I))*(1.0+(np.sin(alpha))**2.0)*b**2.0
    return pdot
    
def calc_dtrue(x, y, z):
    """Calculate true distance to pulsar from the sun. 
    Output is in kiloparsec (kpc)"""
    rsun = 8.5  # kpc
    return np.sqrt(x*x + (y-rsun)*(y-rsun) + z*z)


def shk_corr_vzonly(vz, d, p):
    """"Find the Schklovskii Correction for the pdots. 
    Assumes all three velocity component are equal to vz. This assumption doesn't affect results.
    d: distance in kpc 
    p: period in s. 
    """
    v_3d = ((vz*1000)**2 + (vz*1000)**2 + (vz*1000)**2)**0.5
    v_2d = (2/3)**0.5 * v_3d
    
    corr = 1/c * ((v_2d)**2)/(d*3.086*10**19) * p
    return corr

def calc_B(b_dist):
    """Find B-field either from a Power-law or Lognormal distribution. 
    b_dist: either "plaw" or "lognorm"
    
    Out: B-field in units of 10^8 G"""
    if b_dist == "plaw":
        b_8 = powerlaw_bfield(0.3, 10**3, -1.3) #changing the deafut min from 0.9 to 0.3
    elif b_dist == "lognorm":
        b_8 = (10**random.gauss(8.2, 0.3)) / (10**8) #testing different means. Canon: (8.4,0.5)
    else:
        raise ValueError("Unsupported B-field distribution selected: {}. Please select 'plaw' or 'lognorm'".format(b_dist))
    return b_8

def calc_p0(p_dist, p0_ln_mean=None, p0_ln_std=None,  b=None):
    """Find birth period using either Gonthier+18 deathline, DRL15 lognormal distribution, 
    Canononical pulsar gaussian distribution, or a weibull distribution.
    
    p_dist: either "gon18" or "drl15" or "cp"
    b: B-field in units of 10^8 G. Needed only for gon18.
    
    """
    if p_dist == "gon18":
        p0_ms = 0.18 * 10**(3*random.uniform(0, 2)/7) * (b**(6/7))
    elif p_dist == "drl15":
        p0_ms = np.random.lognormal(p0_ln_mean, p0_ln_std) 
    elif p_dist == "cp":
        p0_ms = random.gauss(0.3, 0.15) *1000       
    elif p_dist == 'weibull':
        p0_ms = weibull_min.rvs(1.546, loc=1.0447, scale=1.911, size=1)[0]
    else:
        raise ValueError("Unsupported period distribution type: {}. Please select drl15 or gon18.".format(p_dist))
    return p0_ms

def tm98_fraction(p):
    """
    Tauris and Manchester 1998 beaming fraction. 
    
    p: period in seconds.
    """
    periodterm = np.log10(p) - 1.0

    return 3. + 9. * periodterm**2.0

def luminosity_fk06(p, pdot, alpha=-1.5, beta=0.5, gamma=0.18):
    """ Equation 14 from  Ridley & Lorimer to find luminosity for a pulsar.
    Default alpha, beta and gamma are for CPs. For MSPs, use gamma=0.01.
    
    p: period in seconds
    pdot: period deivative in s/s
    
    Out: Radio luminosity in units of mJy kpc^2
    """
    # FInd the dither paramter to use in the equation
    delta_l = random.gauss(0.0, 0.8)

    # the equation
    logL = np.log10(gamma) + alpha*np.log10(p) + \
        beta * np.log10(pdot * 1.0e15) + delta_l
        
    return 10.0**logL

def xyz_to_lb(x_, y_, z_):
    """ Convert galactic xyz in kpc to l and b in degrees.
    x_: Galactic x coordinate in kpc
    y_: Galactic y coordinate in kpc
    z_: Galactic z coordinate in kpc
    
    Out: Galactic longtitude and latiude as (l,b) in degrees 
    """
    x, y, z = x_, y_, z_
    rsun = 8.5  # kpc

    # distance to pulsar
    d = np.sqrt(x*x + (rsun-y)*(rsun-y) + z*z)
    # radial distance
    b = np.arcsin(z/d)

    # take cosine
    dcb = d * np.cos(b)

    if y <= rsun:
        if np.fabs(x/dcb) > 1.0:
            l = 1.57079632679
        else:
            l = np.arcsin(x/dcb)
    else:
        if np.fabs(x/dcb) > 1.0:
            l = 0.0
        else:
            l = np.arccos(x/dcb)

        l += 1.57079632679
        if x < 0.:
            l -= 6.28318530718

    # convert back to degrees
    l = np.degrees(l)
    b = np.degrees(b)

    # convert to -180 < l < 180
    #if l > 180.0:
    #    l -= 360.0

    return l, b

def scatter_bhat(dm, scatterindex=-3.86, freq_mhz=1400.0):
    """Calculate Bhat+04 scattering timescale for freq in MHz.
    dm: pulsar dispersion measure in units of pc/cm^3
    
    Out: scattering timescale in ms
    """
    logtau = -6.46 + 0.154 * np.log10(dm)
    logtau += 1.07 * np.log10(dm)*np.log10(dm)
    logtau += scatterindex * np.log10(freq_mhz/1000.0)

    # return tau with power scattered with a gaussian, width 0.8
    return np.power(10.0, random.gauss(logtau, 0.8))

def scale_bhat(timescale, frequency, scaling_power=3.86):
    """Scale the scattering timescale from 1.4 GHz to any other frequency. Using Bhat+04.
    timescale: scattering timescale in ms
    frequencey: frequency to scale to in MHz
    
    Out: Scattering timescale at new frequency in ms
    """
    return timescale * (frequency/1400.0)**scaling_power

def s_1400(lum_1400, dtrue):
        """Calculate the flux of the pulsar given its luminosity and distance.
        lum_1400: Luminosity of the pulsar at 1400 MHz in units of mJy kpc^2
        dtrue: distance to the pulsar in kpc
        
        Out: Radio flux at 1400 MHz in mJy
        """
        return lum_1400 / dtrue / dtrue

def calcFlux(snr, beta, Trec, Tsky, gain, n_p, t_obs, bw, duty):
    """Calculate pulsar flux assuming radiometer equation.
    snr: signal to noise ratio
    beta: paramter to account for snr loss
    Trec: temparature of receiver in K
    Tsky: temparature of sky in K
    gain: telescope gain in K/Jy
    n_p: number of polarizations
    t_obs: integration time (length of time observed) in s 
    bw: bandwidth in MHz
    duty: pulse duty cycle in s
    
    Out: flux in mJy
    """

    signal = signalterm(beta, Trec, Tsky, gain, n_p, t_obs, bw, duty)

    return snr * signal

def calcSNR(flux, beta, Trec, Tsky, gain, n_p, t_obs, bw, duty):
    """Calculate the S/N ratio assuming radiometer equation
    flux: puslar flux in mJy
    beta: paramter to account for snr loss
    Trec: temparature of receiver in K
    Tsky: temparature of sky in K
    gain: telescope gain in K/Jy
    n_p: number of polarizations
    t_obs: integration time (length of time observed) in s 
    bw: bandwidth in MHz
    duty: pulse duty cycle in s
    
    Out: signal to noise ratio

    """

    signal = signalterm(beta, Trec, Tsky, gain, n_p, t_obs, bw, duty)

    return flux / signal

def signalterm(beta, Trec, Tsky, gain, n_p, t_obs, bw, duty):

    """Returns the rest of the radiometer equation (aside from
        SNR/Flux"""

    dterm = duty / (1.0 - duty)
    return beta * (Trec+Tsky) * np.sqrt(dterm) \
        / gain / np.sqrt(n_p * t_obs * bw)

def edot_calc(pdot, p):
    """Finds edot using HPA eq 3.6. Period should in seconds. Output is in erg/s """
    edot = 3.95*10**31 * (pdot/10e-15) * (p)**(-3)
    return edot