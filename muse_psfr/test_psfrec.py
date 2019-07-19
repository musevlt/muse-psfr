import os

import pytest
from astropy.io import fits
from astropy.table import Table
from numpy.testing import assert_allclose

from muse_psfr import (
    compute_psf_from_sparta,
    create_sparta_table,
    fit_psf_with_polynom,
    plot_psf,
)
from muse_psfr.cli import main


def test_reconstruction(tmpdir):
    tbl = create_sparta_table()
    hdul = fits.HDUList([tbl])

    # Note: the case when npsflin=1 is tested below with test_script
    res = compute_psf_from_sparta(hdul, npsflin=3, lmin=490, lmax=541.76, nl=5)
    assert len(res) == 5
    # check that meta are correctly saved
    fit = Table.read(res['FIT_ROWS'])
    assert_allclose(fit['L0'], 25)
    # check fit result
    assert_allclose(fit['center'], 20)
    assert_allclose(fit[1]['lbda'], 502.9, atol=1e-1)
    assert_allclose(fit[1]['fwhm'], 0.85, atol=1e-2)


def test_fit_poly(tmpdir):
    tbl = create_sparta_table()
    hdul = fits.HDUList([tbl])
    res = compute_psf_from_sparta(hdul, lmin=500, lmax=900, nl=9)
    fit = Table.read(res['FIT_ROWS'])
    res = fit_psf_with_polynom(fit['lbda'], fit['fwhm'][:, 0], fit['n'],
                               deg=(5, 5), output=1)
    assert_allclose(res['fwhm_pol'][0], 0.65, atol=1e-2)
    assert_allclose(res['beta_pol'][0], 0.78, atol=1e-2)
    # fit[1] at 550nm matches roughly res[8] (550.1)
    assert_allclose(res['beta_fit'][8], fit[1]['n'], atol=1e-2)
    assert_allclose(res['fwhm_fit'][8], fit[1]['fwhm'], atol=1e-2)


def test_reconstruction2(tmpdir):
    tbl = create_sparta_table()
    # Modify values for the first LGS
    tbl.data[0]['LGS1_L0'] = 20
    tbl.data[0]['LGS1_SEEING'] = 0.8
    tbl.data[0]['LGS1_TUR_GND'] = 0.5
    # and give a bad value to the 3rd LGS
    tbl.data[0]['LGS3_L0'] = 100
    hdul = fits.HDUList([tbl])

    # Note: the case when npsflin=1 is tested below with test_script
    res = compute_psf_from_sparta(hdul, npsflin=3, lmin=500, lmax=700,
                                  nl=3, mean_of_lgs=False)
    assert len(res) == 5
    # check that meta are correctly saved
    fit = Table.read(res['FIT_ROWS'])
    assert_allclose(fit[fit['lgs_idx'] == 1]['L0'], 20)
    assert_allclose(fit[fit['lgs_idx'] != 1]['L0'], 25)

    # check fit result
    assert_allclose(fit['center'], 20)
    assert_allclose(fit[fit['lbda'] == 500]['fwhm'][:, 0], [0.79, 0.86, 0.86],
                    atol=1e-2)


def test_bad_l0(tmpdir, caplog):
    testfile = os.path.join(str(tmpdir), 'sparta.fits')
    create_sparta_table(outfile=testfile, bad_l0=True)

    # Note: the case when npsflin=1 is tested below with test_script
    res = compute_psf_from_sparta(testfile, lmin=490, lmax=541.76, nl=5)

    assert (caplog.records[1].message ==
            '1/1 : Using only 3 values out of 4 after outliers rejection')
    assert caplog.records[3].message == 'Using three lasers mode'

    assert len(res) == 5
    # check that meta are correctly saved
    fit = Table.read(res['FIT_ROWS'])
    assert_allclose(fit['L0'], 25)
    # check fit result
    assert_allclose(fit['center'], 20)
    assert_allclose(fit[1]['lbda'], 502.9, atol=1e-1)
    assert_allclose(fit[1]['fwhm'], 0.86, atol=1e-2)


def test_bad_l0_invalid(tmpdir, caplog):
    # Test no valid values
    testfile = os.path.join(str(tmpdir), 'sparta.fits')
    create_sparta_table(outfile=testfile, L0=1000)
    compute_psf_from_sparta(testfile)

    assert caplog.records[1].message == '1/1 : No valid values, skipping this row'
    assert caplog.records[2].message == 'No valid values'


def test_script(tmpdir, caplog):
    with pytest.raises(SystemExit, match='no input file provided'):
        main([])

    with pytest.raises(SystemExit, match='--values must contain a list.*'):
        main(['--values', '0.1,0.2'])

    with pytest.raises(SystemExit, match='No results'):
        main(['--values', '1,0.7,1000'])

    # without colors, with values and logfile
    caplog.clear()
    logfile = os.path.join(str(tmpdir), 'muse-psfr2.log')
    main(['--no-color', '--values', '1,0.7,25', '--logfile', logfile])

    with open(logfile) as f:
        lines = f.read().splitlines()

    assert lines[2:] == [
        '--------------------------------------------------------------------',
        'LBDA 5000 7000 9000',
        'FWHM 0.85 0.73 0.62',
        'BETA 2.73 2.55 2.23',
        '--------------------------------------------------------------------'
    ]
    # sometimes this includes DEBUG logs, sometimes not .... not sure why
    records = [r for r in caplog.records if r.levelname != 'DEBUG']
    assert records[5].message == 'LBDA 5000 7000 9000'
    assert records[6].message == 'FWHM 0.85 0.73 0.62'
    assert records[7].message == 'BETA 2.73 2.55 2.23'

    # with colors and values
    caplog.clear()
    main(['--values', '1,0.7,25'])
    records = [r for r in caplog.records if r.levelname != 'DEBUG']
    assert 'LBDA' in records[5].message
    assert 'FWHM' in records[6].message
    assert 'BETA' in records[7].message
    assert '7000' in records[5].message
    assert '0.73' in records[6].message
    assert '2.55' in records[7].message


def test_script_with_file(tmpdir):
    testfile = os.path.join(str(tmpdir), 'sparta.fits')
    create_sparta_table(outfile=testfile)

    logfile = os.path.join(str(tmpdir), 'muse_psfr.log')
    outfile = os.path.join(str(tmpdir), 'out.fits')
    main([testfile, '--no-color', '--logfile', logfile, '--outfile', outfile])

    with fits.open(outfile) as hdul:
        assert [hdu.name for hdu in hdul] == [
            'PRIMARY', 'SPARTA_ATM_DATA', 'FIT_ROWS', 'FIT_MEAN', 'PSF_MEAN']

    with open(logfile) as f:
        lines = f.read().splitlines()

    assert lines[2:] == [
        'OB None None Airmass 0.00-0.00',
        '--------------------------------------------------------------------',
        'LBDA 5000 7000 9000',
        'FWHM 0.85 0.73 0.62',
        'BETA 2.73 2.55 2.23',
        '--------------------------------------------------------------------'
    ]


def test_plot(tmpdir):
    import matplotlib
    matplotlib.use('agg', force=True)

    testfile = os.path.join(str(tmpdir), 'sparta.fits')
    create_sparta_table(outfile=testfile, nlines=2)

    res = compute_psf_from_sparta(testfile)
    outfile = os.path.join(str(tmpdir), 'fitres.fits')
    res.writeto(outfile, overwrite=True)

    fig = plot_psf(res)
    fig.savefig(str(tmpdir.join('fig.png')))

    fig = plot_psf(outfile)
    fig.savefig(str(tmpdir.join('fig.png')))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    test_reconstruction('.')
    fig = plot_psf('fitres.fits')
    plt.show()
