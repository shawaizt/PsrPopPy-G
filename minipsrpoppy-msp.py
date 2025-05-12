#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 09:18:27 2023

@author: Shawaiz Tabassum
"""
import numpy as np
import random
import os
import math
import ctypes as C
from scipy import stats
import pygedm
import pandas as pd
import python_functions as pyfunc

c = 299792458 #m/s
k = (8 * np.pi**2 * (12*1000*100)**6) / (3 * (1.7*10**45) * (c)**3) # This is the one with R=12km, I=1.7*10^45 g cm^2, from Gonthier+'18

#replace the following with the path where you want the output to be saved
path_out = '/home/user/multi-synthesis'

#replace the following with the path where radio surveys in the psrpopy format are located
surveys_path = '/home/user/Gamma-PsrPopPy/surveys/msp'

#replace the foloowing path wherever the libvxyz.so routine from psrpopppy is located in your machine
fortran_path = '/home/user/PsrPopPy2-master/lib/fortran'.encode() 

vxyzlib = C.CDLL(os.path.join(fortran_path, 'libvxyz.so'.encode()))
slalib = C.CDLL(os.path.join(fortran_path, 'libsla.so'.encode()))

ne2001lib = C.CDLL(os.path.join(fortran_path, 'libne2001.so'.encode()))
ne2001lib.dm_.restype = C.c_float

yklib = C.CDLL(os.path.join(fortran_path, 'libykarea.so'.encode()))
yklib.ykr_.restype = C.c_float
yklib.llfr_.restype = C.c_float

def vxyz(v,xyz,age):
    """Evolve a pulsar through galactic potential"""
    x, y, z = xyz[0], xyz[1], xyz[2]
    vx, vy, vz = v[0], v[1], v[2]
    age_Myr = age/1.0E6

    x, y, z = C.c_float(x), C.c_float(y), C.c_float(z)
    vx, vy, vz = C.c_float(vx), C.c_float(vy), C.c_float(vz)
    age_Myr = C.c_float(age_Myr)
    bound = C.c_long(0)

    # run the evolution code
    vxyzlib.vxyz_(C.byref(C.c_float(0.005)),
                  C.byref(x),
                  C.byref(y),
                  C.byref(z),
                  C.byref(age_Myr),
                  C.byref(vx),
                  C.byref(vy),
                  C.byref(vz),
                  C.byref(x),
                  C.byref(y),
                  C.byref(z),
                  C.byref(vx),
                  C.byref(vy),
                  C.byref(vz),
                  C.byref(bound)
                  )

    # convert the output C types to python numbers
    galCoords_e = np.array([x.value, y.value, z.value])
    vx_e = np.array([vx.value, vy.value, vz.value])
    return np.array([galCoords_e, vx_e])
        

def seed():
    return C.c_int(random.randint(1, 9999))

def ykr():
    """ Y&K Model"""
    return yklib.ykr_(C.byref(seed()))

def radec_to_lb(ra, dec):
    """Convert RA, Dec to l, b using SLA fortran.
    Be sure to return l in range -180 -> +180"""
    l = C.c_float(0.)
    b = C.c_float(0.)
    ra = C.c_float(ra)
    dec = C.c_float(dec)
    # call with arg = -1 to convert in reverse!
    slalib.galtfeq_(C.byref(l),
                    C.byref(b),
                    C.byref(ra),
                    C.byref(dec),
                    C.byref(C.c_int(-1)))
    if l.value > 180.:
        l.value -= 360.
    return l.value, b.value

def lb_to_radec(l, b):
    """Convert l, b to RA, Dec using SLA fortran (should be faster)."""
    ra = C.c_float(0.)
    dec = C.c_float(0.)
    l = C.c_float(l)
    b = C.c_float(b)
    # call with final arg 1 to do conversion in right direction!
    slalib.galtfeq_(C.byref(l),
                    C.byref(b),
                    C.byref(ra),
                    C.byref(dec),
                    C.byref(C.c_int(1)))
    return ra.value, dec.value

def spiralize_updated(r):
    """ Make spiral arms, as seen in Fuacher-Giguere & Kaspi 2006. Updated spiral arm parameters from YMW16."""
    
    # definitions
    k_list = [4.95, 5.46, 5.77, 5.37]
    r0_list = [3.35, 3.56, 3.71, 3.67]
    theta0_list = [0.77, 3.82, 2.09, 5.76]

    # select a spiral arm ( 1->4)
    arm = random.choice([0, 1, 2, 3])
    k = k_list[arm]
    r0 = r0_list[arm]
    theta0 = theta0_list[arm]

    # pick an angle
    theta = k * np.log(r/r0) + theta0

    # blurring angle
    angle = 2*np.pi * random.random() * np.exp(-0.35 * r)

    if random.random() < 0.5:
        angle = 0 - angle

    # modify theta
    theta += angle

    # blur in radial direction a little
    dr = np.fabs(random.gauss(0.0, 0.5 * r))
    angle = random.random() * 2.0 * np.pi
    dx = dr * np.cos(angle)
    dy = dr * np.sin(angle)

    x = r * np.sin(theta) + dx
    y = r * np.cos(theta) + dy

    return x, y

def galacticDistribute(radial_dist, zscale, age):
    """Select a galactic position. Chose radial distribution between 'gauss' or 'yk04' """
    # using spiral arms
    if radial_dist == "yk04":
        r0 = ykr()
    elif radial_dist == "gauss":
        r0 = np.fabs(random.gauss(0, 8.5))
    elif radial_dist == "fk06":
        lower_lim = 0.0
        mean = 7.04
        stdev = 1.83
        r0 = stats.truncnorm.rvs((lower_lim - mean) / stdev, (np.inf - mean) / stdev, mean, stdev)
    else:
        raise ValueError("Unsupported radial distribution selected: {}. Please select 'gauss', 'fk06 or 'yk04'".format(radial_dist))
    x, y = spiralize_updated(r0)

    # calculate z and r0
    z = pyfunc.draw_double_sided_exp(zscale)
    galCoords = np.array([x, y, z])
    return galCoords

def ne2001_get_smtau(dist, gl, gb):
    """Use NE2001 model to get the DISS scattering timescale"""
    dist = C.c_float(dist)

    # gl gb need to be in radians
    gl = C.c_float(np.radians(gl))
    gb = C.c_float(np.radians(gb))

    # call dmdsm and get the value out of smtau
    ndir = C.c_int(-1)
    sm = C.c_float(0.)
    smtau = C.c_float(0.)
    inpath = C.create_string_buffer(fortran_path)
    linpath = C.c_int(len(fortran_path))
    ne2001lib.dmdsm_(C.byref(gl),
                     C.byref(gb),
                     C.byref(ndir),
                     C.byref(C.c_float(0.0)),
                     C.byref(dist),
                     C.byref(C.create_string_buffer(' ')),
                     C.byref(sm),
                     C.byref(smtau),
                     C.byref(C.c_float(0.0)),
                     C.byref(C.c_float(0.0)),
                     C.byref(inpath),
                     C.byref(linpath)
                     )
    return sm.value, smtau.value

def ne2001_scint_time_bw(dist, gl, gb, freq):
    sm, smtau = ne2001_get_smtau(dist, gl, gb)
    if smtau <= 0.:
        scint_time = None
    else:
        # reference: eqn (46) of Cordes & Lazio 1991, ApJ, 376, 123
        # uses coefficient 3.3 instead of 2.3. They do this in the code
        # and mention it explicitly, so I trust it!
        scint_time = 3.3 * (freq/1000.)**1.2 * smtau**(-0.6)
    if sm <= 0.:
        scint_bw = None
    else:
        # and eq 48
        scint_bw = 223. * (freq/1000.)**4.4 * sm**(-1.2) / dist

    return scint_time, scint_bw

def readtskyfile():
    """Read in tsky.ascii into a list from which temps can be retrieved"""

    tskypath = os.path.join(fortran_path, 'lookuptables/tsky.ascii'.encode())
    tskylist = []
    with open(tskypath) as f:
        for line in f:
            str_idx = 0
            while str_idx < len(line):
                # each temperature occupies space of 5 chars
                temp_string = line[str_idx:str_idx+5]
                try:
                    tskylist.append(float(temp_string))
                except:
                    pass
                str_idx += 5

    return tskylist

class SurveyException(Exception):
    pass

class CoordinateException(Exception):
    pass

def makepointing(coord1, coord2, coordtype):

    if coordtype not in ['eq', 'gal']:
        raise CoordinateException('Wrong coordtype passed to Pointing')

    if coordtype == 'eq':
        # assume pointings in decimal degrees
        ra = coord1
        dec = coord2

        # convert to l and b :)
        gl, gb = radec_to_lb(ra, dec)

        if gl > 180.:
            gl -= 360.

    else:
        if coord1 > 180.:
            coord1 -= 360.

        gl = coord1
        gb = coord2

    return (gl, gb)

def makepointinglist(filename, coordtype):
    f = open(filename, 'r')

    gains = []
    tobs = []
    glgb = []

    for line in f:
        a = line.split()
        try:
            gains.append(float(a[2]))
            tobs.append(float(a[3]))
        except IndexError:
            pass

        glgb.append(makepointing(float(a[0]), float(a[1]), coordtype))

    f.close()

    return np.array(glgb), tobs, gains

class Survey:
    """Class to store survey parameters and methods"""
    def __init__(self, surveyName, pattern='gaussian'):
        """Read in a survey file and obtain the survey parameters"""

        # try to open the survey file locally first
        if os.path.isfile(surveyName):
            f = open(surveyName, 'r')
        else:
            try:
                # try to open file in lib
                # get path to surveys directory
                __libdir__ = os.path.dirname(surveys_path)
                filepath = os.path.join(__libdir__, 'surveys', surveyName)
                f = open(filepath, 'r')

            except IOError:
                # couldn't find the file
                s = 'File {0} does not exist!!!'.format(surveyName)
                raise SurveyException(s)

        self.surveyName = surveyName
        # initialise the pointings list to None
        # only change this is there is a list of pointings to be used
        self.pointingslist = None
        self.gainslist = None
        self.tobslist = None
        self.gainpat = pattern

        # adding AA parameter, so can scale s/n if the survey is
        # an aperture array
        self.AA = False

        # Parse the file line by line
        for line in f:
            # ignore any lines starting '#'
            if line.strip()[0] == '#':
                continue
            # otherwise, parse!
            a = line.split('!')

            # new feature - possible to have a list of positions in survey,
            # rather than a range of l,b or ra,dec
            if a[1].count('pointing list'):
                pointfname = a[0].strip()

                # try to open the pointing list locally
                if os.path.isfile(pointfname):
                    # pointfptr = open(pointfname, 'r')
                    filename = pointfname
                else:
                    try:
                        # try to open pointing file in the surveys dir
                        __dir__ = os.path.dirname(os.path.abspath(__file__))
                        __libdir__ = os.path.dirname(__dir__)
                        filepath = os.path.join(__libdir__,
                                                'surveys',
                                                pointfname)
                        # pointfptr = open(filepath, 'r')
                        filename = filepath
                    except:
                        s = 'File {0} does not exist!!!'.format(pointfname)
                        raise CoordinateException(s)

                if a[1].count('galactic'):
                    p_str = 'gal'
                elif a[1].count('equatorial'):
                    p_str = 'eq'
                else:
                    s = "Unknown coordinate type in survey file"
                    raise CoordinateException(s)

                self.pointingslist, \
                    self.tobslist, \
                    self.gainslist = makepointinglist(filename, p_str)
                """
                # read in the pointing list
                self.pointingslist = []
                # set coord conversion to be done, if any


                for line in pointfptr:
                    a = line.split()
                    if len(a) != 4:
                s = 'File {0} should have cols: gl/gb/gain/tobs'.format(
                                                                    pointfpath)
                        raise CoordinateException(s)
                    p = Pointing(float(a[0]),
                                 float(a[1]),
                                 p_str,
                                 float(a[2]),
                                 float(a[3])
                                 )
                    self.pointingslist.append(p)

                pointfptr.close()
                """

            elif a[1].count('survey degradation'):
                # beta
                self.beta = float(a[0].strip())
            elif a[1].count('antenna gain'):
                # gain
                self.gain = float(a[0].strip())
            elif a[1].count('integration time'):
                # tobs
                self.tobs = float(a[0].strip())
            elif a[1].count('sampling'):
                # tsamp
                self.tsamp = float(a[0].strip())
            elif a[1].count('system temperature'):
                # tsys
                self.tsys = float(a[0].strip())
            elif a[1].count('centre frequency'):
                # centre frequency
                self.freq = float(a[0].strip())
            elif a[1].strip().startswith('bandwidth'):
                # bandwidth
                self.bw = float(a[0].strip())
            elif a[1].count('channel bandwidth'):
                # bw_chan
                self.bw_chan = float(a[0].strip())
            elif a[1].count('polarizations'):
                # num polns
                self.npol = float(a[0].strip())
            elif a[1].count('half maximum'):
                # FWHM
                self.fwhm = float(a[0].strip())
            elif a[1].count('minimum RA'):
                # min RA
                self.RAmin = float(a[0].strip())
            elif a[1].count('maximum RA'):
                # max RA
                self.RAmax = float(a[0].strip())
            elif a[1].count('minimum DEC'):
                # min dec
                self.DECmin = float(a[0].strip())
            elif a[1].count('maximum DEC'):
                # mac dec
                self.DECmax = float(a[0].strip())
            elif a[1].count('minimum Galactic'):
                # min longitude
                self.GLmin = float(a[0].strip())
            elif a[1].count('maximum Galactic'):
                # max longitude
                self.GLmax = float(a[0].strip())
            elif a[1].count('minimum abs'):
                # min latitude
                self.GBmin = float(a[0].strip())
            elif a[1].count('maximum abs'):
                # max latitude
                self.GBmax = float(a[0].strip())
            elif a[1].count('coverage'):
                # coverage fraction
                self.coverage = float(a[0].strip())
                if self.coverage > 1.0:
                    self.coverage = 1.0
            elif a[1].count('signal-to-noise'):
                # SNR limit
                self.SNRlimit = float(a[0].strip())
            elif a[1].count('gain pattern'):
                # Gain pattern (can specify airy, default = gaussian)
                self.gainpat = a[0].strip()
            elif a[1].count('Aperture Array'):
                # turn on AA
                self.AA = True
            else:
                print("Parameter '", a[1].strip(), "' not recognized!")

        f.close()

        # get tsky array from file
        self.tskylist = readtskyfile()

    def __str__(self):
        """Method to define how to print the class"""
        s = "Survey class for {0}:".format(self.surveyName)
        s = '\n\t'.join([s, "beta = {0}".format(self.beta)])
        s = '\n\t'.join([s, "gain = {0}".format(self.gain)])
        s = '\n\t'.join([s, "tobs = {0} s".format(self.tobs)])
        s = '\n\t'.join([s, "tsamp = {0} ms".format(self.tsamp)])
        s = '\n\t'.join([s, "Tsys = {0} K".format(self.tsys)])
        s = '\n\t'.join([s, "Centre frequency = {0} MHz".format(self.freq)])
        s = '\n\t'.join([s, "Bandwidth = {0} MHz".format(self.bw)])
        s = '\n\t'.join([s, "Chan BW = {0} MHz".format(self.bw_chan)])
        s = '\n\t'.join([s, "Num polarisations = {0}".format(self.npol)])
        s = '\n\t'.join([s, "FWHM = {0} arcmin".format(self.fwhm)])
        s = '\n\t'.join([s, "SNR limit = {0}".format(self.SNRlimit)])

        return s

    def nchans(self):
        """ Returns the number of channels in the survey backend."""
        return self.bw / self.bw_chan

    def inRegion(self, gl, gb):
        """Test if pulsar is inside region bounded by survey."""
        # check if l, b are outside region first of all
        # print pulsar.gl, pulsar.gb, self.GLmax, self.GLmin
        if gl > 180.:
            gl -= 360.
        if gl > self.GLmax or gl < self.GLmin:
            return False
        if np.fabs(gb) > self.GBmax \
                or np.fabs(gb) < self.GBmin:
            return False

        # need to compute ra/dec of pulsar from the l and b (galtfeq)
        ra, dec = lb_to_radec(gl, gb)

        # are ra, dec outside region?
        if ra > self.RAmax or ra < self.RAmin:
            return False
        if dec > self.DECmax or dec < self.DECmin:
            return False

        # randomly decide if pulsar is in completed area of survey
        if random.random() > self.coverage:
            return False

        return True

    def inPointing(self, gl, gb):
        """Calculate whether pulsar is inside FWHM/2 of pointing position.
        Currently breaks as soon as it finds a match. !!!Could be a closer
        position further down the list!!!"""
        # initialise offset_deg to be a big old number
        # FWHM is in arcmin so always multiply by 60
        # http://wiki.scipy.org/Cookbook/KDTree
        offset_deg = 1.

        # loop over pointings
        for point in self.pointingslist:
            # do a really basic check first

            glterm = (gl - point.gl)**2
            gbterm = (gb - point.gb)**2
            offset_new = np.sqrt(glterm + gbterm)

            # if the beam is close enough, break out of the loop
            if offset_new < self.fwhm:
                offset_deg = offset_new
                self.gain = point.gain
                self.tobs = point.tobs
                break

        return offset_deg

    def inPointing_new(self, gl, gb):
        """Use numpy-foo to determine closest obs pointing"""

        p = (gl, gb)
        p = np.array(p)
        dists = np.sqrt(((self.pointingslist - p)**2).sum(1))
        # get the min of dists and its index
        offset_deg = np.min(dists)
        indx = np.argmin(dists)
        # set gain and tobs for that point - if given
        if self.gainslist:
            self.gain = self.gainslist[indx]
        if self.tobslist:
            self.tobs = self.tobslist[indx]

        return offset_deg

    def SNRcalc(self,
                period,
                dm,
                gl,
                gb,
                width_degree,
                lum1400,
                accelsearch=False,
                jerksearch=False,
                rratssearch=False):
        """Calculate the S/N ratio of a given pulsar in the survey"""
        # if not in region, S/N = 0

        # if we have a list of pointings, use this bit of code
        # haven't tested yet, but presumably a lot slower
        # (loops over the list of pointings....)

        # otherwise check if pulsar is in entire region
        if self.inRegion(gl, gb):
            # If pointing list is provided, check how close nearest
            # pointing is
            if self.pointingslist is not None:
                # convert offset from degree to arcmin
                offset = self.inPointing_new(gl, gb) * 60.0

            else:
                # calculate offset as a random offset within FWHM/2
                offset = self.fwhm * np.sqrt(random.random()) / 2.0
        else:
            return -2

        # Get degfac depending on self.gainpat
        
        degfac = np.exp(-2.7726 * offset * offset / (self.fwhm * self.fwhm))

        # calc dispersion smearing across single channel
        tdm = self._dmsmear(dm)

        # calculate bhat et al scattering time (inherited from GalacticOps)
        # in units of ms
    
        tscat = pyfunc.scatter_bhat(dm, -3.86, self.freq)

        # Calculate the effective width
        width_ms = width_degree * period / 360.0
        weff_ms = np.sqrt(width_ms**2 + self.tsamp**2 + tdm**2 + tscat**2)

        # calculate duty cycle (period is in ms)
        delta = weff_ms / period

        # if pulse is smeared out, return -1.0
        if delta > 1.0: #and pulsar.pop_time >= 1.0:
            # print width_ms, self.tsamp, tdm, tscat
            return -1

        # radiometer signal to noise
        
        sig_to_noise = pyfunc.calcSNR(pyfunc.s_1400(lum1400, d),
                                       self.beta,
                                       self.tsys,
                                       self.tskypy(gl,gb),
                                       self.gain,
                                       self.npol,
                                       self.tobs,
                                       self.bw,
                                       delta)
        
        # account for aperture array, if needed
        if self.AA and sig_to_noise > 0.0:
            sig_to_noise *= self._AA_factor(gl,gb)

        # return the S/N accounting for beam offset
        
        return sig_to_noise * degfac

    def _AA_factor(self, gl, gb):
        """ Aperture array factor """

        # need to compute ra/dec of pulsar from the l and b (galtfeq)
        ra, dec = lb_to_radec(gl, gb)

        offset_from_zenith = dec - (self.DECmax + self.DECmin)/2.0

        return np.cos(np.radians(offset_from_zenith))

    def _gpsFlux(self, psr, ref_freq):
        """Calculate the flux assuming GPS spectrum shape, spindex===b"""
        log_nu_1 = np.log10(ref_freq/1000.)
        log_nu_2 = np.log10(self.freq/1000.)
        gpsC = np.log10(psr.s_1400()) - (psr.gpsA * log_nu_1**2) \
                                        - psr.spindex * log_nu_1
        return 10.**(psr.gpsA * log_nu_2**2 + psr.spindex * log_nu_2 + gpsC)

    def _dmsmear(self, dm):
        """Calculate the smearing across a channel due to the pulsar DM"""
        return 8.3E6 * dm * self.bw_chan / np.power(self.freq, 3.0)

    def tskypy(self, gl, gb):
        """ Calculate tsky from Haslam table, scale to survey frequency"""
        # ensure l is in range 0 -> 360
        b = gb
        if gl < 0.:
            l = 360 + gl
        else:
            l = gl

        # convert from l and b to list indices
        j = b + 90.5
        if j > 179:
            j = 179

        nl = l - 0.5
        if l < 0.5:
            nl = 359
        i = float(nl) / 4.

        tsky_haslam = self.tskylist[180*int(i) + int(j)]
        # scale temperature before returning
        return tsky_haslam * (self.freq/408.0)**(-2.6)

    def scint(self, psr, snr):
        """ Add scintillation effects and modify the pulsar's S/N"""

        # calculate the scintillation strength (commonly "u")
        # first, calculate scint BW, assume Kolmogorov, C=1.16
        if hasattr(psr, 't_scatter'):
            tscat = pyfunc.scale_bhat(psr.t_scatter,
                                  self.freq,
                                  psr.scindex)
        else:
            tscat = pyfunc.scatter_bhat(psr.dm, psr.scindex, self.freq)
        # convert to seconds
        tscat /= 1000.

        scint_bandwidth = 1.16 / 2.0 / np.pi / tscat  # BW in Hz
        scint_bandwidth /= 1.0E6  # convert to MHz (self.freq is in MHz)

        scint_strength = np.sqrt(self.freq / scint_bandwidth)

        if scint_strength < 1.0:
            # weak scintillation
            # modulation index
            u_term = np.power(scint_strength, 1.666666)
            mod_indx = np.sqrt(u_term)

        else:
            # strong scintillation

            # m^2 = m_riss^2 + m_diss^2 + m_riss * m_diss
            # e.g. Lorimer and Kramer ~eq 4.44
            m_riss = np.power(scint_strength, -0.33333)

            # lorimer & kramer eq 4.44
            kappa = 0.15  # taking this as avrg for now

            # calculate scintillation timescale
            scint_ts, scint_bw = ne2001_scint_time_bw(psr.dtrue,
                                                         psr.gl,
                                                         psr.gb,
                                                         self.freq)

            # calc n_t and n_f
            if scint_ts is None:
                n_t = 1.
            else:
                n_t = self._calc_n_t(kappa, scint_ts)

            if scint_bw is None:
                n_f = 1.
            else:
                n_f = self._calc_n_f(kappa, scint_bw)

            # finally calc m_diss
            m_diss = 1. / np.sqrt(n_t * n_f)

            m_tot_sq = m_diss * m_diss + m_riss * m_riss + m_riss * m_diss

            # modulation index for strong scintillation
            mod_indx = np.sqrt(m_tot_sq)

        return self._modulate_flux_scint(snr, mod_indx)

    def _calc_n_t(self, kappa, delt_t):
        """Number of scintles sampled in time"""
        return 1. + kappa * self.tobs / delt_t

    def _calc_n_f(self, kappa, delt_f):
        """Number of scintles sampled in frequency"""
        return 1. + kappa * self.bw / delt_f

    def _modulate_flux_scint(self, snr, mod_indx):
        """Modify pulsar flux (actually S/N)
        according to the modulation index"""
        # flux and S/N are obviously proportional so it's simple to do this
        # sigma of scintillation
        sig_scint = mod_indx * snr
        return random.gauss(snr, sig_scint)
 
"""
Dictionary to store parameters for all generated pulsars

Units for each parameter list in the list that has units: 
    
p_s: s, 
p_t: s, 
B: 10^8 G, 
(vx,vy,vz): km/s, 
pdot_0: s/s, 
dm_arr: pc/cm^2, 
l1400_arr: mJy kpc^2, 
alpha_arr: radians
"""
results = {key: [] for key in [
    "p0", "p_s", "B", "V_x", "V_y", "V_z", "age",
    "X", "Y", "Z", "pdot_int", "DM",
    "L1400", "gl", "gb", "alpha_rad"
]}

#dictionary to store detected pulsar parameters
results_det = {key: [] for key in [
    "p0", "p_s", "B", "V_x", "V_y", "V_z", "age",
    "X", "Y", "Z", "pdot_int", "DM",
    "L1400", "gl", "gb", "alpha_rad"
]}

surveylist = ['DMB', 'PHSURV', 'PASURV', 'PMSURV_ST_EDITS', 'SWINHL', 'SWINIL', 'PALFA_MSP_65']
surveys = [Survey(s) for s in surveylist]
test = []

beaming_model = 'kramer98' #pick between 'tm98' or 'kramer98'
period_dist = 'drl15' #pick between 'drl15' or 'gon18'. If using 'gon18' edit the call to 'calc_p0" function by giving a b.
radial_distribution = 'fk06' #pick between 'fk06, 'yk04, 'gauss'.
max_age = 5.0e9 #yrs
z_scale = 0.20 #kpc
bv_stdev = 70.0 #birth velocity sigma to be used to draw from a gaussian with mean of 0
p0_ln_mean = 0.98 #mean for the lognormal distribution from which birth periods are sampled.
p0_ln_std = 0.52 #standard deviation for the lognormal distribution from which birth periods are sampled.
    
ndet = 0

while ndet < 92:
    
    #get a magnetic field based on the selected distribution.
    b8 = pyfunc.calc_B('lognorm') 

    #get a period based on the selected distribution
    p_ms = pyfunc.calc_p0(period_dist, p0_ln_mean=p0_ln_mean, p0_ln_std=p0_ln_std, b=None)

    #pass pulsars with unrealistic periods
    if p_ms > 10000 or p_ms < 0.0:
        continue
    
    #get an age by uniformly sampling from 0 to max age
    age_p = random.random() * max_age 

    #get a magnetic inclination angle
    alpha = math.acos(random.random()) 
    
    #evolve period using a spin-down model
    pt = pyfunc.p_of_t_gon18(p_ms/1000, b8, age_p, alpha) #period in seconds
    
    #Apply beaming model 
    if  beaming_model == 'tm98':
        fraction = pyfunc.tm98_fraction(pt)
    elif beaming_model == 'kramer98':
        fraction = random.uniform(0.5, 0.9)
    else: 
        raise ValueError(f"Unsupported beaming model: {beaming_model}. Please select drl15 or gon18.")
    
    if random.random() > fraction:
        continue
    
    #Find pdot
    pdot = pyfunc.pdot_gon18(pt, b8, age_p, alpha)

    #pass pulsars which have evolved to unrealistic periods or beyond the msp definition
    if pt < 0.0013 or pt > 0.030:
      continue
   
    #Apply deathline based on Bhattacharya+'92
    B = 3.94e19 * np.sqrt((pt*pdot)/(1+(np.sin(alpha))**2))
    
    if B/(pt)**2 < 0.17E12:
        continue
    
    #find pulse width
    width = (float(5.0)/100.) * (pt*1000)**0.9
    width = np.log10(width)
    width = pyfunc.drawlnorm(width, 0.3)
    width_degree = 360. * width / (pt*1000)
    
    #Get velocities and positions
    v = np.array([random.gauss(0.0, bv_stdev), random.gauss(0.0, bv_stdev), random.gauss(0.0, bv_stdev)]) #find 3d velocity 
    xyz = galacticDistribute(radial_distribution, z_scale, age_p) # x,y and z coordinates with a z-scale height and the selected radial distribution    
    
    #evolve the velocities and positions in the galactic potential
    evo = vxyz(v,xyz,age_p) 
    xyz_e = evo[0]
    v_e = evo[1]

    v_x = v_e[0]
    v_y = v_e[1]
    v_z = v_e[2]

    x = xyz_e[0]
    y = xyz_e[1]
    z = xyz_e[2]
    
    #Find luminosity
    l1400 = pyfunc.luminosity_fk06(pt, pdot, alpha=-1.5, beta=0.5, gamma=0.01) #I have been using 0.18, the defualt, for gamma. In our work though, we were using gamma = 0.01
    
    #find galactic coords and distance
    gl, gb = pyfunc.xyz_to_lb(x, y, z)
    d = pyfunc.calc_dtrue(x, y, z)
    
    if np.isnan(gl) or np.isnan(gb) or np.isnan(d):
        continue
    
    #find DM 
    dm = pygedm.dist_to_dm(gl, gb, d*1000, nu=1.4, method='ymw16')[0].value
    
    #set up dictionary with keys to allow storing pulsar
    synth_pulsar = {
        "p0": p_ms,
        "p_s": pt,
        "B": b8,
        "V_x": v_x,
        "V_y": v_y,
        "V_z": v_z,
        "age": age_p,
        "X": x,
        "Y": y,
        "Z": z,
        "pdot_int": pdot,
        "DM": dm,
        "L1400": l1400,
        "gl": gl,
        "gb": gb,
        "alpha_rad": alpha,
    }
    
    #store pulsar
    for key, value in synth_pulsar.items():
        if key in results:
            results[key].append(value)
        else:
            print(f"Warning: '{key}' not found in results")

    #set up dictionary with keys for detected pulsars
    
    for surv in surveys:
        SNR = surv.SNRcalc(pt*1000, dm, gl, gb, width_degree, l1400)
        if SNR > surv.SNRlimit:           
            ndet += 1
            for key, value in synth_pulsar.items():
                if key in results:
                    results_det[key].append(value)
                else:
                    print(f"Warning: '{key}' not found in results")      
            print(ndet)

#%%
#calculate distance
results["dist"] = pyfunc.calc_dtrue(np.array(results["X"]), np.array(results["Y"]), np.array(results["Z"])) #distance in kpc
results_det["dist"] = pyfunc.calc_dtrue(np.array(results_det["X"]), np.array(results_det["Y"]), np.array(results_det["Z"])) #distance in kpc

#calculate spindown power output
results["edot"] = pyfunc.edot_calc(np.array(results["pdot_int"]), np.array(results["p_s"])) #in ergs/s
results_det["edot"] = pyfunc.edot_calc(np.array(results_det["pdot_int"]), np.array(results_det["p_s"])) #in ergs/s

#apply shklovskii correction to pdots
pdot_shk = pyfunc.shk_corr_vzonly(np.array(results["V_z"]), np.array(results["dist"]), np.array(results["p_s"])) #the shklovskii correction to be added to intrinsic pdots,
pdot_shk_det = pyfunc.shk_corr_vzonly(np.array(results_det["V_z"]), np.array(results_det["dist"]), np.array(results_det["p_s"]))

results["pdot_obs"] = pdot_shk + np.array(results["pdot_int"]) #"Observed Pdots"
results_det["pdot_obs"] = pdot_shk_det + np.array(results_det["pdot_int"])

results["p_ms"] = np.array(results["p_s"])*1000 #evolved period in ms
results_det["p_ms"] = np.array(results_det["p_s"])*1000

#convert to dataframe and save as csv
df = pd.DataFrame(results)
df_det = pd.DataFrame(results_det)

#set path where output csvs are to be saved
df.to_csv(path_out+'/all-radio.csv', index=False)
df_det.to_csv(path_out+'/detect-radio.csv', index=False)