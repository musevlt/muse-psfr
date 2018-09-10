import argparse
import io
import numpy as np
from astropy.io import fits
from mpdaf.obj import moffat_image
from psfrec import compute_psf_from_sparta


def conv_moffat(fwhm, beta, samp=0.28):
    f = fwhm / 0.2
    c = samp / 0.2
    moffat = moffat_image(shape=(31, 31), fwhm=(f, f), n=beta, unit_fwhm=None)
    conv = moffat.fftconvolve_gauss(fwhm=(c, c), unit_fwhm=None)
    fit = conv.moffat_fit(circular=True, unit_fwhm=None, unit_center=None,
                          verbose=False)
    return fit.fwhm[0] * 0.2, fit.n


def reconstruct_psf(rawname):
    print('Computing PSF Reconstruction from Sparta data')
    res = compute_psf_from_sparta(rawname)
    conv_fwhm = []
    conv_beta = []
    data = res['FIT_MEAN'].data
    for fwhm, beta in zip(data['fwhm'], data['n']):
        f, b = conv_moffat(fwhm[0], beta, samp=0.2 * 1.3)
        conv_fwhm.append(f)
        conv_beta.append(b)
    a, b = (0.35, 1.40)
    conv_beta_corr = np.array(conv_beta) * a + b
    return data['lbda'], conv_fwhm, conv_beta_corr


def main(args=None):
    parser = argparse.ArgumentParser(
        description='PSF Reconstruction version beta-1')
    parser.add_argument('raw', help='Observation Raw file name')
    parser.add_argument('--logfile', default='psfrec.log',
                        help='Name of log file')

    args = parser.parse_args(args)

    print('PSFRec version beta-1')

    hdr = fits.getheader(args.raw)
    header_line = ('OB %s %s Airmass %.2f-%.2f' % (
        hdr.get('HIERARCH ESO OBS NAME'),
        hdr.get('DATE'),
        hdr.get('HIERARCH ESO TEL AIRM START', 0),
        hdr.get('HIERARCH ESO TEL AIRM END', 0)
    ))
    print(header_line)

    lbda, fwhm, beta = reconstruct_psf(args.raw)
    lbref = np.array([500, 700, 900])
    fwhmref = np.interp(lbref, lbda, fwhm)
    betaref = np.interp(lbref, lbda, beta)

    f = io.StringIO()
    f.write(header_line + '\n')
    f.write('-' * 68 + '\n')
    f.write('LBDA %.0f %.0f %.0f\n' % tuple(lbref * 10))
    f.write('FWHM %.2f %.2f %.2f\n' % tuple(fwhmref))
    f.write('BETA %.2f %.2f %.2f\n' % tuple(betaref))
    f.write('-' * 68 + '\n')

    f.seek(0)
    print('\n' + f.read())

    if args.logfile is not None:
        f.seek(0)
        with open(args.logfile, 'a') as fd:
            fd.write(f.read())
        print('Results saved to %s' % (args.logfile))
