PsrPopPy-G
========


Multi-wavelength Pulsar Population Synthesis
-----------

I provide the code based on PsrPopPy2 that allow for a multi-wavelength population synthesis for Millisecond Pulsars (MSPs) and Canonical Pulsars (CPs) as done in Tabassum & Lorimer (2025, [arXiv:2504.02677](https://arxiv.org/abs/2504.02677)).

This done by first synthesizing a population accounting for radio selection effects and then applying gamma-ray selection effects. 

For MSPs, the radio synthesis is done using PsrPopPy2 framework with some modifications. Everything needed is provided in the scripts.

For CPs, a program based on the work of Johnston & Karastergiou (2017, [2017MNRAS.467.3493J](https://doi.org/10.1093/mnras/stx377); [arXiv:1702.03616](https://arxiv.org/abs/1702.03616)) should be first used to get a list of evolved CPs and then multi-wavelength selection effects can then be applied using the provided python scripts. A fortran program, adapted from code kindly provided by Simon Johnston, is made available for this purpose.
 
There are 5 python scripts:
1. `python_functions.py` : Contains all the PsrPopPy2 functions needed.
2. `minipsrpoppy-msp.py` : Used to generate MSPs accounting for radio selection effects.
3. `minipsrpoppy-cp.py` : Used to generate CPs accounting for radio selection effects
4. `gammaray-filter-msp.py` : Applies gamma-ray selection effects assuming several different gamma-ray luminosity models for MSPs.
5. `gammaray-filter-cp.py` : Applies gamma-ray selection effects assuming several different gamma-ray luminosity models for CPs.

The fortran program is in the `fortran-code` directory. This directory contains the main program `evolve_cp.f` and the two supplemental programs `normal.f` and `ran1.f`. These are written in f77 and can be compiled with gfortan.
 
Also provided are survey files for the surveys used in Tabassum & Lorimer 2025.

The Fermi-LAT All-sky sensitivity map can be obtained from the [Fermi-LAT Third Catalog of Gamma-ray Pulsars webpage](https://fermi.gsfc.nasa.gov/ssc/data/access/lat/3rd_PSR_catalog/).

An installation of PsrPopPy2 is required due to some fortran shared libraries which are used from it.

PsrPopPy & PsrPopPy2
--------

PsrPopPy is python implementation of PSRPOP (written by Duncan Lorimer, see [2006MNRAS.372..777L](https://doi.org/10.1111/j.1365-2966.2006.10887.x); [arXiv:astro-ph/0607640](https://arxiv.org/abs/astro-ph/0607640) for details) written by Sam Bates (2014, [2014MNRAS.439.2893B](https://doi.org/10.1093/mnras/stu157); [arXiv:1311.3427](https://arxiv.org/abs/1311.3427)). Its github repository and documentation can be found at http://samb8s.github.com/PsrPopPy/.

PsrPopPy2 added additional functionality to PsrPopPy. More details and installation instructions can be found at https://github.com/devanshkv/PsrPopPy2.

Usage
=====

MSPs:

Update the `minipsrpoppy-msp.py` script with the appropriate paths for `path_out`, `surveys_path` and `fortran_path`. Then run the script. It will output two CSV files; `all-radio.csv` and `detect-radio.csv`.
`all-radio.csv` contains all pulsars beaming towards Earth that were generated. `detect-radio.csv` contains only those pulsars which were detectable by the radio surveys specified, which should be 92 pulsars for with the default surveys.

Next, to apply gamma-ray selection effects, use the `gammaray-filter-msp.py` script. Update `path`, `path_out` and `path_fermi_map` with path to the outputs from running previous script, path where to store final outputs and path where the Fermi-LAT sensitivity map is located respectively. The Fermi-LAT All-sky sensitivity map can be obtained from the [3PC webpage](https://fermi.gsfc.nasa.gov/ssc/data/access/lat/3rd_PSR_catalog/). This script outputs CSV files which contain pulsars detectable by Fermi-LAT based on a specific gamma-ray luminosity model.

CPs:

`minipsrpoppy-cp.py` requires the path to a file containing pulsars evoloved using the fortran program in the `fortran-code` directory. We recommend producing enough CPs from this to allow for there to be enough CPs above the deathline and also enough CPs detectable by the surveys being modeled. We recommend at least 1 million pulsars.
After updating `evolved_cps_path`, `path_out`, `surveys_path` and `fortran_path`, you can run the script and get the same two output files as in the case of MSPs which were described above.

Gamma-ray selection effects can be applied in a similar manner to the MSP case by using 'gammaray-filter-cp.py' instead this time. The output is again a CSV file each for a specific gamma-ray luminosity model.
