import numpy as np
import os
import pytest
from astropy.table import Table

from psfrec import compute_psf_from_sparta, plot_psf, create_sparta_table
from psfrec.run_psfrec import main


def test_reconstruction(tmpdir):
    testfile = os.path.join(str(tmpdir), 'sparta.fits')
    create_sparta_table(outfile=testfile)

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
    create_sparta_table(outfile=testfile, bad_l0=True)

    # Note: the case when npsflin=1 is tested below with test_script
    res = compute_psf_from_sparta(testfile, verbose=True)

    captured = capsys.readouterr()
    assert ('1/1 : Using only 3 values out of 4 after outliers rejection'
            in captured.out.splitlines())
    assert 'Using three lasers mode' in captured.out.splitlines()

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
    create_sparta_table(outfile=testfile)

    logfile = os.path.join(str(tmpdir), 'psfrec.log')
    main([testfile, '--logfile', logfile])

    with open(logfile) as f:
        lines = f.read().splitlines()

    assert lines[2:] == [
        'OB None None Airmass 0.00-0.00',
        '--------------------------------------------------------------------',
        'LBDA 5000 7000 9000',
        'FWHM 0.84 0.75 0.68',
        'BETA 2.53 2.30 2.13',
        '--------------------------------------------------------------------'
    ]

    with pytest.raises(SystemExit, match='no input file provided'):
        main([])

    with pytest.raises(SystemExit, match='--values must contain a list.*'):
        main(['--values', '0.1,0.2'])

    logfile = os.path.join(str(tmpdir), 'psfrec2.log')
    main(['--values', '1,0.7,25', '--logfile', logfile])

    with open(logfile) as f:
        lines = f.read().splitlines()

    assert lines[2:] == [
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
    create_sparta_table(outfile=testfile, nlines=2)

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
