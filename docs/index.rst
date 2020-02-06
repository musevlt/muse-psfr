MUSE-PSFR - PSF reconstruction for MUSE WFM-AO
==============================================

.. ifconfig:: 'dev' in release

    .. warning::

        This documentation is for the version currently under development.

.. include:: ../README.rst

Original code was written in IDL by `Thierry Fusco <thierry.fusco@onera.fr>`__
and `Benoit Neichel <benoit.neichel@lam.fr>`__. It was ported to Python by
`Simon Conseil <simon.conseil@univ-lyon1.fr>`__.

.. contents::

Installation
============

MUSE-PSFR requires the following packages:

* Numpy
* Astropy
* SciPy
* MPDAF
* Joblib
* Matplotlib (optional, for the PSF plot)
* Colorama (optional, for colored output)

The last stable release of MUSE-PSFR can be installed simply with pip::

    pip install muse-psfr

Or to install with optional dependencies::

    pip install muse-psfr[all]

Or into the user path with::

    pip install --user muse-psfr

How it works
============

The algorithm is described in the article ([Fusco et al. in prep.]).

Inputs
------

The PSF reconstruction algorithm needs 3 values provided by SPARTA: the
*seeing*, the *Ground Layer fraction (GL)*, and the *outer-scale (L0)*. These
values can be provided directly as command-line arguments (see below), but the
typical use is to provide a raw MUSE file.

Since the GLAO commissioning, the MUSE raw files contain a FITS table
(``SPARTA_ATM_DATA``) containing the atmospheric turbulence profile estimated
by SPARTA. This table contains the values for each laser, with one row every
two minutes.

Number of reconstructed wavelengths
-----------------------------------

To reduce computation time, the ``muse-psfr`` command reconstructs the PSF at
three wavelengths: 500, 700, and 900 nm. But it is possible to reconstruct
the PSF at any wavelength, with the `~muse_psfr.compute_psf_from_sparta`
function.  This function reconstructs by default for 35 wavelengths between
490nm and 930nm (which can specified with the *lmin*, *lmax*, and *nl*
parameters).

Number of reconstructed direction
---------------------------------

Since the spatial variation is negligible over the MUSE field of view, the
reconstruction is done by default only at the center of field. This can be
changed in `~muse_psfr.compute_psf_from_sparta` with the *npsflin* parameter.


Usage
=====

Command Line Interface
----------------------

MUSE-PSFR can be used from the command line, either with a set of seeing, GL,
and L0 values:

.. command-output:: muse-psfr --no-color --values 1,0.7,25

Or with a MUSE raw FITS file which contains a ``SPARTA_ATM_DATA`` extension::

   $ muse-psfr raw/MUSE.2018-08-13T07:14:11.128.fits.fz
   MUSE-PSFR version 0.31
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

   Results saved to muse-psfr.log

More information use of the command line interface can be found with the
command:

.. command-output:: muse-psfr --help

By default it saves the computed values in a log file (``muse-psfr.log``). It
is also possible to save a FITS file with the fit results for all wavelengths
and all SPARTA rows with the ``--outfile`` option.

Python interface
----------------

The main entry point for the Python interface is the
`~muse_psfr.compute_psf_from_sparta` function. This function takes a file with
a SPARTA table,

.. plot::

   >>> from muse_psfr import compute_psf_from_sparta, create_sparta_table
   >>> tbl = create_sparta_table(seeing=1, L0=25, GL=0.7)
   >>> tbl.data
   FITS_rec([(25, 1, 0.7, 25, 1, 0.7, 25, 1, 0.7, 25, 1, 0.7)],
            dtype=(numpy.record, [('LGS1_L0', '<i8'), ('LGS1_SEEING', '<i8'), ('LGS1_TUR_GND', '<f8'), ('LGS2_L0', '<i8'), ('LGS2_SEEING', '<i8'), ('LGS2_TUR_GND', '<f8'), ('LGS3_L0', '<i8'), ('LGS3_SEEING', '<i8'), ('LGS3_TUR_GND', '<f8'), ('LGS4_L0', '<i8'), ('LGS4_SEEING', '<i8'), ('LGS4_TUR_GND', '<f8')]))
   >>> from astropy.io import fits
   >>> hdul = fits.HDUList([tbl])
   >>> out = compute_psf_from_sparta(hdul, lmin=500, lmax=900, nl=3, plot=True)
   Processing SPARTA table with 1 values, njobs=1 ...
   1/1 : seeing=1.00 GL=0.70 L0=25.00


Changelog
=========

.. include:: ../CHANGELOG

API
===

.. autofunction:: muse_psfr.compute_psf_from_sparta

.. autofunction:: muse_psfr.compute_psf

.. autofunction:: muse_psfr.create_sparta_table

.. autofunction:: muse_psfr.fit_psf_with_polynom
