from psfrec import compute_psf_from_sparta
from mpdaf.obj import moffat_image
import numpy as np
from astropy.io import fits
import argparse


def conv_moffat(fwhm, beta, samp=0.28):
    f = fwhm/0.2
    c = samp/0.2
    moffat = moffat_image(shape=(31,31),fwhm=(f,f), n=beta, unit_fwhm=None)
    conv = moffat.fftconvolve_gauss(fwhm=(c,c), unit_fwhm=None)
    fit = conv.moffat_fit(circular=True, unit_fwhm=None, unit_center=None, verbose=False)
    return fit.fwhm[0]*0.2,fit.n

def psfrec(rawname):
    print('Computing PSF Reconstruction from Sparta data')
    res = compute_psf_from_sparta(rawname)
    conv_fwhm = []
    conv_beta = []
    data = res[f'FIT_MEAN'].data
    for fwhm,beta in zip(data['fwhm'],data['n']):
        f,b = conv_moffat(fwhm[0],beta,samp=0.2*1.3)
        conv_fwhm.append(f)
        conv_beta.append(b)
    a,b = (0.35,1.40)
    conv_beta_corr = np.array(conv_beta)*a + b
    return data['lbda'],conv_fwhm,conv_beta_corr

parser = argparse.ArgumentParser(description='PSF Reconstruction version beta-1')
parser.add_argument('-r', '--raw', default=False, help='Observation Raw file name')
parser.add_argument('--logfile', default='/diska/home/gto1/psfrec_logs/psfrec.log', help='Name of log file')

args = parser.parse_args()
args = vars(args)

print('PSFRec version beta-1')

rawname = args['raw']
raw = fits.open(rawname)
hdr = raw[0].header
print('OB %s %s Airmass %.2f-%.2f'%(hdr['HIERARCH ESO OBS NAME'], hdr['DATE'], hdr['HIERARCH ESO TEL AIRM START'],hdr['HIERARCH ESO TEL AIRM END']))
lbda,fwhm,beta = psfrec(rawname)
lbref = [500,700,900]
fwhmref = np.interp(lbref, lbda, fwhm)
betaref = np.interp(lbref, lbda, beta)
print(' ')
print('OB %s %s Airmass %.2f-%.2f'%(hdr['HIERARCH ESO OBS NAME'], hdr['DATE'], hdr['HIERARCH ESO TEL AIRM START'],hdr['HIERARCH ESO TEL AIRM END']))
print('----------------------------------------------------------------------')
print('LBDA %.0f %.0f %.0f'%(lbref[0]*10,lbref[1]*10,lbref[2]*10))
print('FWHM %.2f %.2f %.2f'%(fwhmref[0],fwhmref[1],fwhmref[2]))
print('BETA %.2f %.2f %.2f'%(betaref[0],betaref[1],betaref[2]))
print('---------------------------------------------------------------------- ')

if args['logfile'] is not None:
    with open(args['logfile'], 'a') as f:
        f.write(' \n')
        f.write('OB %s %s Airmass %.2f-%.2f\n'%(hdr['HIERARCH ESO OBS NAME'], hdr['DATE'], hdr['HIERARCH ESO TEL AIRM START'],hdr['HIERARCH ESO TEL AIRM END']))
        f.write('----------------------------------------------------------------------\n')
        f.write('LBDA %.0f %.0f %.0f\n'%(lbref[0]*10,lbref[1]*10,lbref[2]*10))
        f.write('FWHM %.2f %.2f %.2f\n'%(fwhmref[0],fwhmref[1],fwhmref[2]))
        f.write('BETA %.2f %.2f %.2f\n'%(betaref[0],betaref[1],betaref[2]))
        f.write('----------------------------------------------------------------------\n')
    print('Results sadded to log file %s'%(args['logfile']))
print('End of PSFrec (type return to quit)')
