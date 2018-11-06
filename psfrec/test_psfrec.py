import numpy as np
import os
from astropy.io import fits
from astropy.table import Table

from psfrec import compute_psf_from_sparta, plot_psf
from psfrec.run_psfrec import main


def create_test_table(testfile, nlines=1, bad_l0=False):
    # Create a fake SPARTA table with values for the 4 LGS
    seeing = 1.
    L0 = 25.
    Cn2 = [0.7, 0.3]

    tbl = [('LGS%d_%s' % (k, col), v) for k in range(1, 5)
           for col, v in (('SEEING', seeing), ('TUR_GND', Cn2[0]), ('L0', L0))]
    tbl = Table([dict(tbl)] * nlines)
    if bad_l0:
        tbl['LGS4_L0'] = 150

    hdu = fits.table_to_hdu(tbl)
    hdu.name = 'SPARTA_ATM_DATA'
    hdu.writeto(testfile, overwrite=True)


def test_reconstruction(tmpdir):
    testfile = os.path.join(str(tmpdir), 'sparta.fits')
    create_test_table(testfile)

    # Note: the case when npsflin=1 is tested below with test_script
    res = compute_psf_from_sparta(testfile, npsflin=3, lmin=490, lmax=930,
                                  nl=35, verbose=True)
    outfile = os.path.join(str(tmpdir), 'fitres.fits')
    res.writeto(outfile, overwrite=True)
    assert len(res) == 5
    # check that meta are correctly saved
    fit = Table.read(res['FIT1'])
    assert fit.meta['L0'] == 25.
    # check fit result
    fit = Table.read(res['FIT1'])
    assert np.allclose(fit['center'], 20)
    assert np.allclose(fit[1]['lbda'], 502.9, atol=1e-1)
    assert np.allclose(fit[1]['fwhm'], 0.84, atol=1e-2)


def test_bad_l0(tmpdir, capsys):
    testfile = os.path.join(str(tmpdir), 'sparta.fits')
    create_test_table(testfile, bad_l0=True)

    # Note: the case when npsflin=1 is tested below with test_script
    res = compute_psf_from_sparta(testfile, verbose=True)

    captured = capsys.readouterr()
    assert 'Missing 1 lasers' in captured.out.splitlines()

    outfile = os.path.join(str(tmpdir), 'fitres.fits')
    res.writeto(outfile, overwrite=True)
    assert len(res) == 5
    # check that meta are correctly saved
    fit = Table.read(res['FIT1'])
    assert fit.meta['L0'] == 25.
    # check fit result
    fit = Table.read(res['FIT1'])
    assert np.allclose(fit['center'], 20)
    assert np.allclose(fit[1]['lbda'], 502.9, atol=1e-1)
    assert np.allclose(fit[1]['fwhm'], 0.84, atol=1e-2)


def test_script(tmpdir):
    testfile = os.path.join(str(tmpdir), 'sparta.fits')
    create_test_table(testfile)

    logfile = os.path.join(str(tmpdir), 'psfrec.log')
    main([testfile, '--logfile', logfile])

    with open(logfile) as f:
        lines = f.read().splitlines()

    assert lines == [
        'OB None None Airmass 0.00-0.00',
        '--------------------------------------------------------------------',
        'LBDA 5000 7000 9000',
        'FWHM 0.84 0.75 0.68',
        'BETA 2.53 2.30 2.13',
        '--------------------------------------------------------------------'
    ]


def test_plot(tmpdir):
    import matplotlib
    matplotlib.use('agg', force=True)

    testfile = os.path.join(str(tmpdir), 'sparta.fits')
    create_test_table(testfile, nlines=2)

    res = compute_psf_from_sparta(testfile, verbose=True)
    outfile = os.path.join(str(tmpdir), 'fitres.fits')
    res.writeto(outfile, overwrite=True)

    fig = plot_psf(outfile)
    fig.savefig(str(tmpdir.join('fig.png')))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    test_reconstruction('.')
    fig = plot_psf('fitres.fits')
    plt.show()
