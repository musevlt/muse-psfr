import numpy as np
import os
from astropy.io import fits
from astropy.table import Table

from psfrec import compute_psf_from_sparta, plot_psf


def test_reconstruction(tmpdir):
    # Create a fake SPARTA table with values for the 4 LGS
    seeing = 1.
    L0 = 25.
    Cn2 = [0.7, 0.3]
    tbl = [('LGS%d_%s' % (k, col), v) for k in range(1, 5)
           for col, v in (('SEEING', seeing), ('TUR_GND', Cn2[0]), ('L0', L0))]
    tbl = Table([dict(tbl)])
    hdu = fits.table_to_hdu(Table([dict(tbl)]))
    hdu.name = 'SPARTA_ATM_DATA'
    testfile = os.path.join(str(tmpdir), 'sparta.fits')
    hdu.writeto(testfile, overwrite=True)

    res = compute_psf_from_sparta(testfile, npsflin=3, seeing_correction=0.,
                                  lmin=490, lmax=930, nl=35, verbose=True)
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
    assert np.allclose(fit[1]['fwhm'], 0.651, atol=1e-2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    test_reconstruction('.')
    fig = plot_psf('fitres.fits')
    plt.show()
