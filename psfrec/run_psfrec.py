import argparse
import io
import sys
from astropy.io import fits

from psfrec.psfrec import compute_psf_from_sparta, create_sparta_table
from psfrec.version import __version__


def reconstruct_psf(rawname, **kwargs):
    print('Computing PSF Reconstruction from Sparta data')
    res = compute_psf_from_sparta(rawname, **kwargs)
    data = res['FIT_MEAN'].data
    return data['lbda'], data['fwhm'][:, 0], data['n']


def main(args=None):
    parser = argparse.ArgumentParser(
        description='PSF Reconstruction version beta-1')
    parser.add_argument('raw', help='observation Raw file name', nargs='?')
    parser.add_argument('--values', help='Values of seeing, GL, L0, to use '
                        'instead of the raw file, comma-separated.')
    parser.add_argument('--logfile', default='psfrec.log',
                        help='Name of log file')
    parser.add_argument('--njobs', default=-1, type=int, help='number of '
                        'parallel jobs (by default use all CPUs)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='verbose flag')

    args = parser.parse_args(args)

    print('PSFRec version {}'.format(__version__))

    if args.values:
        values = [float(x) for x in args.values.split(',')]
        if len(values) != 3:
            sys.exit('--values must contain a list of 3 comma-separated '
                     'values for seeing, GL, and L0')
        header_line = None
        rawf = io.BytesIO()
        create_sparta_table(outfile=rawf, seeing=values[0], GL=values[1],
                            L0=values[2])
        rawf.seek(0)
    else:
        if args.raw is None:
            sys.exit('no input file provided')
        rawf = args.raw
        hdr = fits.getheader(rawf)
        header_line = ('OB %s %s Airmass %.2f-%.2f' % (
            hdr.get('HIERARCH ESO OBS NAME'),
            hdr.get('DATE'),
            hdr.get('HIERARCH ESO TEL AIRM START', 0),
            hdr.get('HIERARCH ESO TEL AIRM END', 0)
        ))
        print(header_line)

    lbda, fwhm, beta = reconstruct_psf(rawf, verbose=args.verbose, lmin=500,
                                       lmax=900, nl=3, n_jobs=args.njobs)
    f = io.StringIO()
    if header_line:
        f.write(header_line + '\n')
    f.write('-' * 68 + '\n')
    f.write('LBDA %.0f %.0f %.0f\n' % tuple(lbda * 10))
    f.write('FWHM %.2f %.2f %.2f\n' % tuple(fwhm))
    f.write('BETA %.2f %.2f %.2f\n' % tuple(beta))
    f.write('-' * 68 + '\n')

    f.seek(0)
    print('\n' + f.read())

    if args.logfile is not None:
        f.seek(0)
        with open(args.logfile, 'a') as fd:
            fd.write('\nFile: {}\n'.format(args.raw))
            fd.write(f.read())
        print('Results saved to %s' % (args.logfile))
