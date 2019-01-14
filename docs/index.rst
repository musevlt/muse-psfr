PSFRec - PSF reconstruction for MUSE WFM-AO
===========================================

The PSFRec code allows to reconstruct a PSF for the MUSE WFM-AO mode, using
telemetry data from SPARTA.

Original code was written in IDL by `Thierry Fusco <thierry.fusco@onera.fr>`__
and `Benoit Neichel <benoit.neichel@lam.fr>`__. It was then ported to Python by
`Simon Conseil <simon.conseil@univ-lyon1.fr>`__.

The paper describing the original method can be found here:
http://adsabs.harvard.edu/abs/XXXXXXXXX (TODO)

.. contents::

Installation
============

PSFRec requires the following packages:

* Numpy
* Astropy
* SciPy 
* MPDAF
* Joblib
* Matplotlib (optional, for the PSF plot)
* Colorama (optional, for colored output)

The last stable release of PSFRec can be installed simply with pip::

    pip install psfrec

Or to install with optional dependencies::

    pip install psfrec[all]

Or into the user path with::

    pip install --user psfrec

Usage
=====

Command Line Interface
----------------------

PSFRec can be used from the command line, either with a set of seeing, GL, and
L0 values::

   $ psfrec --values 1,0.7,25
   PSFRec version 0.31
   Computing PSF Reconstruction from Sparta data
   Processing SPARTA table with 1 values, njobs=1 ...
   1/1 : seeing=1.00 GL=0.70 L0=25.00

   --------------------------------------------------------------------
   LBDA  5000 7000 9000
   FWHM  0.86 0.74 0.63
   BETA  2.58 2.27 1.94
   --------------------------------------------------------------------

   Results saved to psfrec.log

Or with a MUSE raw FITS file, that contains a ``SPARTA_ATM_DATA`` extension::

   $ psfrec raw/MUSE.2018-08-13T07:14:11.128.fits.fz
   PSFRec version 0.31
   OB MXDF-01-00-A 2018-08-13T07:39:21.835 Airmass 1.49-1.35
   Computing PSF Reconstruction from Sparta data
   Processing SPARTA table with 13 values, njobs=-1 ...
   4/13 : Using only 3 values out of 4 after outliers rejection
   4/13 : seeing=0.57 GL=0.75 L0=18.32
   Using three lasers mode
   1/13 : Using only 3 values out of 4 after outliers rejection
   1/13 : seeing=0.71 GL=0.68 L0=13.60
   Using three lasers mode
   6/13 : Using only 3 values out of 4 after outliers rejection
   6/13 : seeing=0.60 GL=0.75 L0=16.47
   Using three lasers mode
   ....

   OB MXDF-01-00-A 2018-08-13T07:39:21.835 Airmass 1.49-1.35
   --------------------------------------------------------------------
   LBDA  5000 7000 9000
   FWHM  0.57 0.46 0.35
   BETA  2.36 1.91 1.64
   --------------------------------------------------------------------

   Results saved to psfrec.log

More information use of the command line interface can be found with the
command ::

    psfrec -h

Python interface
----------------

Changelog
=========

.. include:: ../CHANGELOG

API
===

.. autofunction:: psfrec.compute_psf_from_sparta
