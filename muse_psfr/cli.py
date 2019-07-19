import argparse
import io
import logging
import sys

from astropy.io import fits

from muse_psfr import __version__, compute_psf_from_sparta, create_sparta_table

logger = logging.getLogger(__name__)


def main(args=None):
    parser = argparse.ArgumentParser(
        description=f'MUSE-PSFR version {__version__}')
    addarg = parser.add_argument
    addarg('raw', help='observation raw file name', nargs='?')
    addarg('--values', help='values of seeing, GL, L0, to use instead of the '
           'raw file, comma-separated')
    addarg('--logfile', default='muse_psfr.log', help='name of log file')
    addarg('-o', '--outfile', help='name of a FITS file in which the results '
           'are saved: table with individual and mean Moffat fits, and mean '
           'reconstructed PSF')
    addarg('--njobs', default=-1, type=int, help='number of parallel jobs '
           '(by default use all CPUs)')
    addarg('--verbose', '-v', action='store_true', help='verbose flag')
    addarg('--no-color', action='store_true', help='no color in output')
    addarg('--plot', action='store_true', help='plot reconstructed psf')
    addarg('--version', action='version', version='%(prog)s ' + __version__)

    args = parser.parse_args(args)
    logger.info('MUSE-PSFR version %s', __version__)

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
        logger.info(header_line)

    logger.info('Computing PSF Reconstruction from Sparta data')
    if args.verbose:
        _logger = logging.getLogger('muse_psfr')
        _logger.setLevel("DEBUG")
        _logger.handlers[0].setLevel("DEBUG")

    res = compute_psf_from_sparta(rawf, lmin=500, lmax=900, nl=3,
                                  n_jobs=args.njobs, plot=args.plot)
    if res:
        data = res['FIT_MEAN'].data
        lbda, fwhm, beta = data['lbda'], data['fwhm'][:, 0], data['n']
    else:
        sys.exit('No results')

    f = io.StringIO()
    if header_line:
        f.write(header_line + '\n')
    f.write('-' * 68 + '\n')

    try:
        import colorama  # noqa
    except ImportError:
        args.no_color = True

    lbda *= 10
    if args.no_color:
        f.write('LBDA %.0f %.0f %.0f\n' % tuple(lbda))
        f.write('FWHM %.2f %.2f %.2f\n' % tuple(fwhm))
        f.write('BETA %.2f %.2f %.2f\n' % tuple(beta))
    else:
        from colorama import Fore, Back, Style
        RED, GREEN, BLUE = Fore.RED, Fore.GREEN, Fore.BLUE
        begin_style = Back.BLACK + Style.BRIGHT + Fore.WHITE
        end_style = Fore.RESET + Style.NORMAL + Back.RESET
        f.write(
            f'{begin_style}'
            f'LBDA {BLUE}{lbda[0]:.0f} {GREEN}{lbda[1]:.0f} {RED}{lbda[2]:.0f}'
            f'{end_style}\n'
            f'{begin_style}'
            f'FWHM {BLUE}{fwhm[0]:.2f} {GREEN}{fwhm[1]:.2f} {RED}{fwhm[2]:.2f}'
            f'{end_style}\n'
            f'{begin_style}'
            f'BETA {BLUE}{beta[0]:.2f} {GREEN}{beta[1]:.2f} {RED}{beta[2]:.2f}'
            f'{end_style}\n'
        )
        f.write(Style.RESET_ALL)

    f.write('-' * 68 + '\n')

    f.seek(0)
    for line in f:
        logger.info(line.rstrip('\n'))

    if args.logfile is not None:
        f.seek(0)
        with open(args.logfile, 'a') as fd:
            fd.write('\nFile: {}\n'.format(args.raw))
            fd.write(f.read())
        logger.info('Results saved to %s' % args.logfile)

    if args.outfile is not None:
        res.writeto(args.outfile, overwrite=True)
        logger.info('FITS file saved to %s' % args.outfile)
