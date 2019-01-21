import os
import pytest
from astropy.table import Table
from numpy.testing import assert_allclose

from psfrec import compute_psf_from_sparta, plot_psf, create_sparta_table
from psfrec.run_psfrec import main


def test_reconstruction(tmpdir):
    testfile = os.path.join(str(tmpdir), 'sparta.fits')
    create_sparta_table(outfile=testfile)

    # Note: the case when npsflin=1 is tested below with test_script
    res = compute_psf_from_sparta(testfile, npsflin=3, lmin=490, lmax=541.76,
                                  nl=5, verbose=True)
    outfile = os.path.join(str(tmpdir), 'fitres.fits')
    res.writeto(outfile, overwrite=True)
    assert len(res) == 5
    # check that meta are correctly saved
    fit = Table.read(res['FIT_ROWS'])
    assert_allclose(fit['L0'], 25)
    # check fit result
    assert_allclose(fit['center'], 20)
    assert_allclose(fit[1]['lbda'], 502.9, atol=1e-1)
    assert_allclose(fit[1]['fwhm'], 0.86, atol=1e-2)


def test_bad_l0(tmpdir, capsys):
    testfile = os.path.join(str(tmpdir), 'sparta.fits')
    create_sparta_table(outfile=testfile, bad_l0=True)

    # Note: the case when npsflin=1 is tested below with test_script
    res = compute_psf_from_sparta(testfile, lmin=490, lmax=541.76, nl=5,
                                  verbose=True)

    captured = capsys.readouterr()
    assert ('1/1 : Using only 3 values out of 4 after outliers rejection'
            in captured.out.splitlines())
    assert 'Using three lasers mode' in captured.out.splitlines()

    outfile = os.path.join(str(tmpdir), 'fitres.fits')
    res.writeto(outfile, overwrite=True)
    assert len(res) == 5
    # check that meta are correctly saved
    fit = Table.read(res['FIT_ROWS'])
    assert_allclose(fit['L0'], 25)
    # check fit result
    assert_allclose(fit['center'], 20)
    assert_allclose(fit[1]['lbda'], 502.9, atol=1e-1)
    assert_allclose(fit[1]['fwhm'], 0.86, atol=1e-2)

    # --------
    # Test no valid values
    testfile = os.path.join(str(tmpdir), 'sparta.fits')
    create_sparta_table(outfile=testfile, L0=1000)
    res = compute_psf_from_sparta(testfile, verbose=True)

    captured = capsys.readouterr()
    assert ('1/1 : No valid values, skipping this row'
            in captured.out.splitlines())


def test_script(tmpdir):
    testfile = os.path.join(str(tmpdir), 'sparta.fits')
    create_sparta_table(outfile=testfile)

    logfile = os.path.join(str(tmpdir), 'psfrec.log')
    main([testfile, '--no-color', '--logfile', logfile])

    with open(logfile) as f:
        lines = f.read().splitlines()

    assert lines[2:] == [
        'OB None None Airmass 0.00-0.00',
        '--------------------------------------------------------------------',
        'LBDA 5000 7000 9000',
        'FWHM 0.86 0.74 0.63',
        'BETA 2.58 2.27 1.94',
        '--------------------------------------------------------------------'
    ]

    with pytest.raises(SystemExit, match='no input file provided'):
        main([])

    with pytest.raises(SystemExit, match='--values must contain a list.*'):
        main(['--values', '0.1,0.2'])

    with pytest.raises(SystemExit, match='No results'):
        main(['--values', '1,0.7,1000'])

    logfile = os.path.join(str(tmpdir), 'psfrec2.log')
    main(['--no-color', '--values', '1,0.7,25', '--logfile', logfile])

    with open(logfile) as f:
        lines = f.read().splitlines()

    assert lines[2:] == [
        '--------------------------------------------------------------------',
        'LBDA 5000 7000 9000',
        'FWHM 0.86 0.74 0.63',
        'BETA 2.58 2.27 1.94',
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
