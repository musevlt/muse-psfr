"""PSF reconstruction for MUSE WFM.

Original code was written in IDL by `Thierry Fusco <thierry.fusco@onera.fr>`__
and `Benoit Neichel <benoit.neichel@lam.fr>`__. It was ported to Python by
`Simon Conseil <simon.conseil@univ-lyon1.fr>`__.

Note: Many comments are still in french!

Le programme simul_psd_wfm crée un jeu de DSP (par direction du champ).

le programme psf_muse utilise ces DSP pour créer des PSF a chaque longueur
d'onde avec un pixel scale de 0.2 arcsec.

"""

import logging
import os
from math import gamma

import numpy as np
from astropy.convolution import Moffat2DKernel
from astropy.io import fits
from astropy.table import Column, Table, vstack
from joblib import Parallel, delayed
from mpdaf.obj import Cube
from numpy.fft import fft2, fftshift, ifft2
from scipy.interpolate import interpn
from scipy.signal import fftconvolve

MIN_L0 = 8   # minimum L0 in m
MAX_L0 = 30  # maximum L0 in m

logger = logging.getLogger(__name__)


def simul_psd_wfm(Cn2, h, seeing, L0, zenith=0., plot=False, npsflin=1,
                  dim=1280, three_lgs_mode=False, verbose=True):
    """ Batch de simulation de PSF WFM MUSE avec impact de NGS.

    Parameters
    ----------
    Cn2        = pondération du profil
    h          = altitude des couches (en m)
    seeing     = seeing @ zenith  (en arcsec @ 500nm)
    L0         = Grande echelle (en m)
    zenith     = (en degré)
    npsflin    = nombre de point (lineaire) dans le champ pour l'estimation
    des PSF (1 = au centre, 2 = au 4 coins, 3 = 9 PSF etc ...)
    dim        = Final dimension of the PSD

    """
    # STEP 0 : Définition des conditions
    # =============================================

    # Step 0.1 : turbulence
    # ---------
    Cn2 = np.array(Cn2)
    Cn2 /= Cn2.sum()

    h = np.array(h)
    vent = np.full_like(h, 12.5)

    # FIXME: currently set to IDL values for reproducability
    np.random.seed(12345)
    # arg_v = (np.random.rand(h.shape[0]) - 0.5) * np.pi  # wind dir.  [rad]
    arg_v = np.array([0.628163, -0.326497])  # from IDL

    # Step 0.2 : Systeme
    # ---------
    Dpup = 8.          # en m (diametre du telescope)
    # oc = 0.14          # normalisée [de 0 --> 1]
    altDM = 1.         # en m
    hsodium = 90000.   # en m

    # lambdalgs = 0.589  # en µm
    lambdaref = 0.5    # en µm
    nact = 24.         # nombre lineaire d'actionneurs
    nsspup = nact      # nombre lineaire d'actionneurs

    Fsamp = 1000.      # frequence d'échantillonnage [Hz]
    delay = 2.5        # retard en ms (lecture CCD + calcul)

    seplgs = 63.       # separation (en rayon) des LGS [arcsec]
    bruitLGS2 = 1.0    # radians de phase bord a bord de sspup @ lambdalgs

    if three_lgs_mode:
        if verbose:
            logger.info('Using three lasers mode')
        poslgs = np.array([[1, 1], [-1, -1], [-1, 1]], dtype=float).T
    else:
        poslgs = np.array([[1, 1], [-1, -1], [-1, 1], [1, -1]], dtype=float).T

    poslgs *= seplgs   # *cos(pi/4) # position sur une grille cartesienne
    law = "LSE"        # type de lois : lse ou mmse
    recons_cn2 = 1     # a priori sur Cn2 => ici GLAO
    recons_h = altDM   # a priori sur h   => ici GLAO

    # Step 0.3 : Direction d'estimation
    dirperf = direction_perf(npsflin, plot=plot, lgs=poslgs)

    # Step 0.4 : Paremetres numériques
    # ---------
    Dimpup = 40        # Taille de la pupille en pixel pour zone de correction
    # coefL0 = 1         # pour gerer les "L0 numériques"

    # Step 0.5 : Mise en oeuvre
    # ---------
    r0ref = seeing2r01(seeing, lambdaref, zenith)  # passage seeing --> r0
    hz = h / np.cos(np.deg2rad(zenith))  # altitude dans la direction de visée
    dilat = (hsodium - hz) / hsodium  # dilatation pour prendre en compte
    hz_lgs = hz / dilat               # la LGS
    hz_lgs -= altDM  # on prend en compte la conjugaison negative du DM

    # Step 0.6 : Summarize of parameters
    # ---------
    logger.debug('r0 0.5um (zenith)        = %.2f', seeing2r01(seeing, lambdaref, 0))
    logger.debug('r0 0.5um (line of sight) = %.2f', r0ref)
    logger.debug('Seeing   (line of sight) = %.2f', 0.987 * 0.5 / r0ref / 4.85)
    logger.debug('hbarre   (zenith)        = %.2f',
                 np.sum(h ** (5 / 3) * Cn2) ** (3 / 5))
    logger.debug('hbarre   (line of sight) = %.2f',
                 np.sum(hz ** (5 / 3) * Cn2) ** (3 / 5))
    logger.debug('vbarre                   = %.2f',
                 np.sum(vent ** (5 / 3) * Cn2) ** (3 / 5))

    # ========================================================================

    # longueur physique d'un ecran de ^hase issue de la PSD
    # => important pour les normalisations
    # L = dim / Dimpup * Dpup

    pitch = Dpup / nact    # pitch: inter-actuator distance [m]
    fc = 1 / (2 * pitch)   # pItch frequency (1/2a)  [m^{-1}]

    # STEP 1 : Simulation des PSF LGS (tilt inclus)
    # ===============================================
    # cube de DSP pour chaque direction d'interet - ZONE DE CORRECTION ONLY
    dsp = dsp4muse(Dpup, Dimpup, Dimpup * 2, Cn2, h, L0, r0ref, recons_cn2,
                   recons_h, vent, arg_v, law, nsspup, nact, Fsamp, delay,
                   bruitLGS2, lambdaref, poslgs, dirperf)

    # STEP 2: Calcul DSP fitting
    # ===============================================
    dspa = fftshift(psd_fit(dim, 2 * Dpup, r0ref, L0, fc))
    dspf = np.resize(dspa, (dsp.shape[0], dim, dim))

    # Finale
    sl = slice(dim // 2 - Dimpup, dim // 2 + Dimpup)
    dspf[:, sl, sl] = np.maximum(dspa[sl, sl], fftshift(dsp, axes=(1, 2)))

    return dspf * (lambdaref * 1000 / (2 * np.pi)) ** 2


def direction_perf(npts, field_size=60, plot=False, lgs=None, ngs=None,
                   ax=None):
    """Create a grid of points where the PSF is estimated."""
    x, y = (np.mgrid[:npts, :npts] - npts // 2) * field_size / 2
    dirperf = np.array([x, y]).reshape(2, -1)

    if plot:
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        champvisu = np.max(dirperf)
        ax.scatter(dirperf[0], dirperf[1], marker='o', s=10,
                   label='Reconstruction directions')
        if lgs is not None:
            champvisu = max(champvisu, lgs.max())
            ax.scatter(lgs[0], lgs[1], marker='*', s=60, label='LGS')
        if ngs is not None:
            champvisu = max(champvisu, ngs.max())
            ax.scatter(ngs[0], ngs[1], marker='*', s=40, label='NGS')

        ax.set_xlim((-1.25 * champvisu, 1.25 * champvisu))
        ax.set_ylim((-1.25 * champvisu, 1.25 * champvisu))
        ax.set_xlabel('arcsecond')
        ax.set_ylabel('arcsecond')
        ax.legend(loc='upper center')

    return dirperf


def seeing2r01(seeing, lbda, zenith):
    """seeing @ 0.5 microns, lambda en microns."""
    r00p5 = 0.976 * 0.5 / seeing / 4.85  # r0 @ 0.5 µm
    r0 = r00p5 * (lbda * 2) ** (6 / 5) * np.cos(np.deg2rad(zenith)) ** (3 / 5)
    return r0


def pupil_mask(radius, width, oc=0, inverse=False):
    """Calcul du masque de la pupille d'un télescope.

    radius = rayon du télescope (en pixels)
    largeur = taille du masque
    oc = taux d'occultation centrale linéaire
    """
    center = (width - 1) / 2
    x, y = np.ogrid[:width, :width]
    rho = np.hypot(x - center, y - center) / radius
    mask = ((rho < 1) & (rho >= oc))
    if inverse:
        mask = ~mask
    return mask.astype(int)


def calc_var_from_psd(psd, pixsize, Dpup):
    # Decoupage de la DSP pour eagle
    psdtemp = fftshift(psd) * pixsize ** 2

    # Calcul de Fp
    FD = 1 / Dpup
    boxsize = FD / pixsize

    mask = pupil_mask(boxsize / 2, psd.shape[0], inverse=True)
    return np.sum(psdtemp * mask)


def calc_mat_rec_glao_finale(f, arg_f, pitchs_wfs, pitchs_dm, poslgs,
                             sigr, DSP_tab_recons, h_recons, LSE=False):
    """Computes the reconstruction matrix WMAP, accounting for all
    reconstruction parameters::

        WMAP = Ptomo ## Wtomo

    residual DSP is computed after that by `calc_dsp_res_glao_finale`.

    Parameters
    ----------
    f = spatial frequencies array
    arg_f = F phase
    poslgs = Guide stars positions
    sigr = A priori on noise associated to each GS
    DSP_Tab_recons = A priori, DSP on estimated turbulent layers
    h_recons = Altitudes of reconstructed layers
    #Wflag : Choice of reconstructor W1 or W2
    Keyword Tomo : Pure tomo
    Popt : output : optimal projector for MCAO, used later for Aliasing
    LSE : LSE instead of MAP

    """
    f_x = f * np.cos(arg_f)
    f_y = f * np.sin(arg_f)
    s = f.shape[0]

    # WFS used is Shack.
    # Each WFS has its own cut off frequency

    # Construction of WFS transfert function
    # NB: Here wfs is the transfert function of a SH, but something else could
    # be written here. Pyramid / Curvature / direct phase sensing (wfs=1)
    pitchs_wfs = pitchs_wfs[:, None, None]  # to broadcast to (nb_gs, s, s)
    wfs = (2 * np.pi * 1j * f *
           np.sinc(pitchs_wfs * f_x) * np.sinc(pitchs_wfs * f_y))
    fc = 1 / (2 * pitchs_wfs)
    # where((f NE 0) and (abs(f_x) GE fc) OR (abs(f_y) GE fc), count)
    # FIXME missing parenthesis around | ?
    wfs[(f != 0) & (np.abs(f_x) >= fc) | (np.abs(f_y) >= fc)] = 0.

    # -----------------------------------------------------------
    # Construction of WHAP = PtomoWtomo
    # Size : Ngs x Ndm
    # -----------------------------------------------------------
    # Starting with WTomo
    # 2 writings, again accessible with W1 or W2
    # -----------------------------------------------------------

    # 3 under-matrices are needed
    # Palpha' (and its transposed)
    # Cb = a priori on noise for each GS
    # Cphi = a priori on turbulence profile

    # Brique 1 : M.Palpha'
    nb_gs = poslgs.shape[1]
    nb_h_recons = h_recons.size
    Mr = np.zeros((nb_h_recons, nb_gs, s, s), dtype=complex)
    for j in range(nb_gs):
        for i in range(nb_h_recons):
            # FIXME: 206265 c'est quoi ca ?
            ff_x = f_x * poslgs[0, j] * h_recons[i] * 60 / 206265
            ff_y = f_y * poslgs[1, j] * h_recons[i] * 60 / 206265
            Mr[i, j] = wfs[j] * np.exp(1j * 2 * np.pi * (ff_x + ff_y))

    wfs = None

    # Wtomo, with its two forms :
    # 1: Wtomo = ((Mrt#Cb_inv_recons#Mr + Cphi_inv_recons)^-1)Mrt#Cb_inv_recons
    # 2: Wtomo = Cphi_recons#Mrt(Mr#Cphi_recons#Mrt + Cb_recons)^-1
    # Size is Nbgs x NL'

    # -----------------------------------------------------------------------
    # Choice of W1 or W2 :
    # W2 works better with high SNR, a flag like this could be interesting :
    # If mean(sigr) LT 0.01 then Wflag = 'W2' ELSE Wflag = 'W1'
    # -----------------------------------------------------------------------

    # Construction of Cb_inv (a priori on noise)
    Cb_inv_recons = 1 / sigr

    # Cphi-1, a priori on turbulence layers, computed from DSP_tab_recons
    if not LSE:
        Cphi_inv_recons = np.zeros((nb_h_recons, nb_h_recons, s, s))
        for i in range(nb_h_recons):
            Cphi_inv_recons[i, i] = 1. / DSP_tab_recons[i]
        # Filtering of piston in reconstruction :
        Cphi_inv_recons[0, 0, 0, 0] = 0.

    # W1 = ((Mrt#Cb_inv_recons#Mr + Cphi_inv_recons)^-1)Mrt#Cb_inv_recons
    # ----------------------------------------------------------------------
    # Mrt#Cb_inv first
    res_tmp = np.zeros_like(Mr)
    for i in range(nb_gs):
        for j in range(nb_h_recons):
            res_tmp[j, i] += Mr[j, i].conj() * Cb_inv_recons[i]

    # Mrt#Cb_inv#Mr then :
    MAP = np.zeros((nb_h_recons, nb_h_recons, s, s), dtype=complex)
    for k in range(nb_gs):
        for i in range(nb_h_recons):
            for j in range(nb_h_recons):
                MAP[i, j] += res_tmp[j, k] * Mr[i, k]

    # to be inversed :
    if not LSE:
        MAP += Cphi_inv_recons
        Cphi_inv_recons = None

    # ---------------------------------------------------------------------
    # Without a priori, this is WLSE
    # ---------------------------------------------------------------------

    # Inversion of MAP matrix - Inversion frequency by frequency
    inv = np.zeros_like(MAP)
    tmp = np.zeros((nb_h_recons, nb_h_recons), dtype=complex)
    for j in range(s):
        for i in range(s):
            tmp = MAP[..., i, j]

            # inversion of each sub matrix
            if tmp.sum() != 0:
                if nb_h_recons > 1:
                    raise NotImplementedError
                    # FIXME: not ported yet! numpy.linalg.pinv ?
                    # condmax : Max acceptable conditionning in inversion for
                    # POPT computation
                    # condmax = 1e6
                    # la_tsvd(mat=tmp, inverse=tmp_inv, condmax=seuil,
                    #         silent=True)
                else:
                    tmp_inv = np.linalg.inv(tmp)

                if i == 0 and j == 0:
                    tmp_inv[:] = 0

                inv[..., i, j] = tmp_inv
    MAP = None

    # Last step W1 = inv#res_tmp
    W1 = np.zeros((nb_gs, nb_h_recons, s, s), dtype=complex)
    for i in range(nb_gs):
        for j in range(nb_h_recons):
            for k in range(nb_h_recons):
                W1[i, j] += inv[k, j] * res_tmp[k, i]

    return W1


def calc_dsp_res_glao_finale(f, arg_f, pitchs_wfs, poslgs, beta, sigv,
                             DSP_tab_vrai, h_vrai, h_dm, Wmap, td, ti, wind,
                             tempo=False, fitting=False, err_recons=None,
                             err_noise=None):
    """Calcule la DSP_res (incluant tout les termes d'erreurs classiques)
    pour TOUT type de WFAO.

    - Si on considere plusieurs etoiles Guides + plusieurs DMs + une
      optimisation moyenne dans un champs => On fait de la MCAO
    - Si on ne met qu'1 DM, et qu'on optimise dans 1 direction en
      particulier => On fait de la LTAO/MOAO
    - Si on optimsie sur un grand champs, mais qu'on a qu'1 miroir => on
      fait du GLAO.

    Bref, cette fonction, elle fait tout !

    Parameters
    ----------
    f = tableau des frequences spatiales
    arg_f = argument de f
    pitchs_wfs = tableau des pitchs WFS
    poslgs = position des etoiles Guides dans le champ ((x,y) en arcmin)
    Beta = position ou on evalue la performance ((x,y) en arcmin)
    sigv = bruit associé a chaque GS, utilisé dans le calcul de Cb.
    DSP_tab_vrai = DSPs couches a couches de la vraie turbulence introduite.
    h_vrai = altitudes des couches du vrai profil
    h_dm = altitude des DMs
    Wmap = Matrice de reconstruction Tomographique, from calc_mat_rec_finale
    td = delai
    ti = tableau des temps d'integration des WFS
    Wind = tableau vents

    """
    f_x = f * np.cos(arg_f)
    f_y = f * np.sin(arg_f)
    s = f.shape[0]
    nb_h_vrai = h_vrai.size
    nb_gs = poslgs.shape[1]

    # ici on ecrit tous les termes de la DSP residuelle :
    # 1. L'erreur de reconstruction
    # 2. La propagation du bruit
    # 3. Le servo-lag
    # 4. Le fitting

    if tempo:
        # logger.info('Servo-lag Error')
        pass
    else:
        wind = np.zeros((2, nb_h_vrai))
        ti = np.zeros(nb_gs)
        td = 0.

    # 1. L'erreur de reconstruction
    # -----------------------------------------------------------
    # faut ecrire :
    # (PbetaL - PbetaDM#WMAP#M.PalphaL)

    # On ecrit alors la matrice "model Vrai"
    # C'est la matrice M.PalphaL, elle contient tous les phaseurs pour le vrai
    # profil de turbulence.
    # Type de WFS :
    pitchs_wfs = pitchs_wfs[:, None, None]  # to broadcast to (nb_gs, s, s)
    wfs = (2 * np.pi * 1j * f *
           np.sinc(pitchs_wfs * f_x) * np.sinc(pitchs_wfs * f_y))
    fc = 1 / (2 * pitchs_wfs)
    # where((f != 0) and (abs(f_x) GT fc) OR (abs(f_y) GT fc), count)
    # FIXME missing parenthesis around | ? > vs >= ?
    wfs[(f != 0) & (np.abs(f_x) > fc) | (np.abs(f_y) > fc)] = 0.

    Mv = np.zeros((nb_h_vrai, nb_gs, s, s), dtype=complex)
    for i in range(nb_h_vrai):
        for j in range(nb_gs):
            ff_x = f_x * poslgs[0, j] * h_vrai[i] * 60 / 206265
            ff_y = f_y * poslgs[1, j] * h_vrai[i] * 60 / 206265
            www = np.sinc(wind[0, i] * ti[j] * f_x + wind[1, i] * ti[j] * f_y)
            Mv[i, j] = www * wfs[j] * np.exp(1j * 2 * (ff_x + ff_y) * np.pi)
    wfs = None

    # ensuite, faut ecrire PbetaL#
    # on considere que les ecrans on bougé de DeltaTxV
    # deltaT = (max(ti) + td)(0)
    deltaT = ti.max() + td
    # -----------------
    proj_beta = np.zeros((nb_h_vrai, s, s), dtype=complex)
    for j in range(nb_h_vrai):
        # on considere un shift en X,Y en marche arriere :
        proj_beta[j] = np.exp(
            1j * 2 * np.pi *
            (h_vrai[j] * 60 / 206265 * (beta[0] * f_x + beta[1] * f_y) -
             (wind[0, j] * deltaT * f_x + wind[1, j] * deltaT * f_y)))

    # et PbetaDM
    h_dm = np.atleast_1d(h_dm)
    nb_h_dm = h_dm.size
    proj_betaDM = np.zeros((nb_h_dm, s, s), dtype=complex)
    for j in range(nb_h_dm):
        proj_betaDM[j] = np.exp(1j * 2 * np.pi * h_dm[j] * 60 / 206265 *
                                (beta[0] * f_x + beta[1] * f_y))

    # ok, on ecrit donc le produit de toutes ces matrices :
    # PbetaDM#WMAP c'est un vecteur qui fait Ngs
    proj_tmp = np.zeros((nb_gs, s, s), dtype=complex)
    for i in range(nb_gs):
        proj_tmp[i] = np.sum(proj_betaDM * Wmap[i], axis=0)  # sum on nb_h_dm

    # Puis, on ecrit proj_tmp#Mv
    proj_tmp2 = np.zeros((nb_h_vrai, s, s), dtype=complex)
    for i in range(nb_h_vrai):
        proj_tmp2[i] = np.sum(proj_tmp * Mv[i], axis=0)  # sum on nb_gs

    # Puis (PbetaL - proj) ca sera le projecteur qu'on appliquera a Cphi pour
    # trouver l'erreur de reconstruction
    proj = proj_beta - proj_tmp2
    Mv = proj_tmp = proj_tmp2 = proj_beta = None

    # -----------------------------------------------------------------
    # MAINTENANT ON PEUT ECRIRE err_recons !!!!
    # -----------------------------------------------------------------

    # err_recons = proj#Cphi#proj_conj
    Cphi_vrai = DSP_tab_vrai
    err_recons = np.sum(proj * Cphi_vrai * proj.conj(), axis=0)
    err_recons[0, 0] = 0.
    err_recons = err_recons.real
    proj = None

    # --------------------------------------------------------------------
    # ET VOILA, ON A LA DSP D'ERREUR DE RECONSTRCUTION GLAO
    # --------------------------------------------------------------------

    # ####################################################################
    # MAINTENANT IL FAUT ECRIRE LA DSP DU BRUIT PROPAGEE A TRAVERS LE
    # RECONSTRUCTEUR GLAO
    # ####################################################################

    # That is Easy, ca s'ecrit : PbetaDM#Wmap#Cb#(PbetaDM#Wmap)T
    # Faut deja ecrire PbetaDM#Wmap
    proj_noise = np.zeros((nb_gs, s, s), dtype=complex)
    for i in range(nb_gs):
        proj_noise[i] = np.sum(proj_betaDM * Wmap[i], axis=0)  # sum on nb_h_dm

    # -----------------------------------------------------------------------
    # MAINTENANT ON PEUT ECRIRE err_noise !!!!
    # -----------------------------------------------------------------------

    # err_noise = proj_noise#Cb#proj_noise_conj
    Cb_vrai = sigv[:, None, None]
    err_noise = np.sum(proj_noise * Cb_vrai * proj_noise.conj(), axis=0)
    err_noise[0, 0] = 0.
    err_noise = err_noise.real

    # -----------------------------------------------------------------------
    # ET VOILA, ON A LA DSP D'ERREUR DE PROPAGATION DU BRUIT TOMOGRAPHIQUE
    # -----------------------------------------------------------------------

    dsp_res = err_recons + err_noise
    if fitting:
        return dsp_res

    fc = np.max(1 / (2 * pitchs_wfs))
    return np.where((f != 0) & (abs(f_x) <= fc) & (abs(f_y) <= fc), dsp_res, 0)


def dsp4muse(Dpup, pupdim, dimall, Cn2, hh, L0, r0ref, recons_cn2, h_recons,
             vent, arg_v, law, nsspup, nact, Fsamp, delay, bruitLGS2,
             lambdaref, poslgs, dirperf):

    # Passage en arcmin
    poslgs1 = poslgs / 60
    dirperf1 = dirperf / 60

    LSE = (law == 'LSE')  # mmse ou lse
    tempo = True          # erreur temporelle
    fitting = True        # fitting
    dimall = int(dimall)
    err_R0 = 1.
    cst = 0.0229

    # -------------------------------------------------------------------

    fx = np.fft.fftfreq(dimall, Dpup / pupdim)[:, np.newaxis]
    fy = fx.T
    f = np.sqrt(fx ** 2 + fy ** 2)
    with np.errstate(all='ignore'):
        arg_f = fy / fx
    arg_f[0, 0] = 0  # to get the same as idl (instead of NaN)
    arg_f = np.arctan(arg_f)

    # -------------------------------------------------------------------
    #  PSD turbulente
    # -------------------------------------------------------------------

    h_recons = np.atleast_1d(h_recons)
    recons_cn2 = np.atleast_1d(recons_cn2)
    DSP_tab_recons = (
        cst *
        (recons_cn2[:, None, None] ** (-3 / 5) * r0ref / err_R0) ** (-5 / 3) *
        (f ** 2 + (1 / L0) ** 2) ** (-11 / 6))

    hh = np.atleast_1d(hh)
    Cn2 = np.atleast_1d(Cn2)
    DSP_tab_vrai = (
        cst * (Cn2[:, None, None] ** (-3 / 5) * r0ref) ** (-5 / 3) *
        (f ** 2 + (1 / L0) ** 2) ** (-11 / 6))

    # -----------------------------------------------------
    # CALCUL DE LA MATRICE de commande GLAO
    # -----------------------------------------------------

    nb_gs = poslgs1.shape[1]
    pitchs_wfs = np.repeat(Dpup / nsspup, nb_gs)
    sig2 = np.repeat(bruitLGS2, nb_gs)
    fech_tab = np.repeat(Fsamp, nb_gs)

    pitchs_DM = Dpup / nact
    h_dm = 1.
    ti = 1 / fech_tab
    td = delay * 1.e-3

    Wmap = calc_mat_rec_glao_finale(f, arg_f, pitchs_wfs, pitchs_DM, poslgs1,
                                    sig2, DSP_tab_recons, h_recons, LSE=LSE)

    # DSP dans les differentes directions de reconstruction
    # =======================================================
    nb_dir_perf = dirperf1.shape[1]
    dsp = np.zeros((nb_dir_perf, dimall, dimall))
    wind = np.stack([vent * np.cos(arg_v), vent * np.sin(arg_v)])

    L = Dpup * dimall / pupdim  # taille de l'ecran en m.
    pixsize = 1. / L

    for bbb in range(nb_dir_perf):
        beta = dirperf1[:, bbb]
        # DSP tempo + noise-tomp + fitting
        dsp_res = calc_dsp_res_glao_finale(
            f, arg_f, pitchs_wfs, poslgs1, beta, sig2, DSP_tab_vrai,
            hh, h_dm, Wmap, td, ti, wind, tempo=tempo, fitting=fitting)
        dsp[bbb, :, :] = dsp_res

        resval = calc_var_from_psd(dsp_res, pixsize, Dpup)
        logger.debug("dirperf=%d, %.2f", bbb,
                     np.sqrt(resval) * lambdaref * 1e3 / (2 * np.pi))

    # The above was ported from IDL with inverse rows/columns convention, so we
    # transpose the array here.
    return np.moveaxis(dsp, -1, -2)


def psd_fit(dim, L, r0, L0, fc):
    dim = int(dim)
    fx, fy = fftshift((np.mgrid[:dim, :dim] - (dim - 1) / 2) / L, axes=(1, 2))
    f = np.sqrt(fx ** 2 + fy ** 2)

    out = np.zeros_like(f)
    cst = ((gamma(11 / 6)**2 / (2 * np.pi**(11 / 3))) *
           (24 * gamma(6 / 5) / 5)**(5 / 6))
    f_ind = (f >= fc)
    out[f_ind] = cst * r0**(-5 / 3) * (f[f_ind]**2 + (1 / L0)**2)**(-11 / 6)
    return out


def crop(arr, center, size):
    center, size = int(center), int(size)
    sl = slice(center - size, center + size)
    return arr[sl, sl]


def interpolate(arr, xout, method='linear'):
    """Function that mimics IDL's interpolate."""
    xin = np.arange(arr.shape[0])
    # xout = np.mgrid[:dimpsf, :dimpsf] * npixc / dimpsf
    if method == 'cubic':
        raise NotImplementedError('FIXME: use gridddata or spline ?')
    return interpn((xin, xin), arr, xout.T, method='linear').T


def psf_muse(psd, lambdamuse):
    if psd.ndim == 2:
        ndir = 1
        dim = psd.shape[0]
    elif psd.ndim == 3:
        ndir = psd.shape[0]
        dim = psd.shape[1]

    nl = lambdamuse.size

    D = 8
    samp = 2
    pup = pupil_mask(dim / 4, dim / 2, oc=0.14)

    dimpsf = 40
    pixscale = 0.2
    psfall = np.zeros((nl, dimpsf, dimpsf))

    FoV = (lambdamuse / (2 * D)) * dim / (4.85 * 1e3)   # = champ total
    npixc = (np.round(((dimpsf * pixscale * 2 * 8 * 4.85 * 1000) /
                       lambdamuse) / 2) * 2).astype(int)
    lbda = lambdamuse * 1.e-9

    for i in range(nl):
        if psd.ndim == 3:
            psf = np.zeros((ndir, npixc[i], npixc[i]))
            for j in range(ndir):
                psf_tmp = psd_to_psf(psd[j], pup, D, lbda[i], samp=samp, FoV=FoV[i])
                psf[j] = crop(psf_tmp, center=psf_tmp.shape[1] // 2,
                              size=npixc[i] // 2)
            psf = psf.mean(axis=0)
        else:
            psf = psd_to_psf(psd, pup, D, lbda[i], samp=samp, FoV=FoV[i])
            psf = crop(psf, center=psf.shape[0] // 2, size=npixc[i] // 2)

        psf /= psf.sum()
        np.maximum(psf, 0, out=psf)

        pos = np.mgrid[:dimpsf, :dimpsf] * npixc[i] / dimpsf
        psfall[i] = interpolate(psf, pos, method='linear')

    psfall /= psfall.sum(axis=(1, 2))[:, None, None]
    return psfall


def psd_to_psf(psd, pup, D, lbda, phase_static=None, samp=None, FoV=None,
               return_all=False):
    """Computation of a PSF from a residual phase PSD and a pupil shape.

    Programme pour prendre en compte la multi-analyse les geometries
    d'etoiles et la postion de la galaxie.

    FUNCTION psd_to_psf, dsp, pup, local_L, osamp

    PSD: 2D array with PSD values (in nm² per freq² at the PSF wavelength)
    pup: 2D array representing the pupill
    Samp: final PSF sampling (number of pixel in the diffraction). Min = 2 !
    FoV  : PSF FoV (in arcsec)
    lbda : PSF wavelength in m
    D = pupil diameter
    phase_static in nm

    """
    dim = psd.shape[0]
    npup = pup.shape[0]
    sampnum = dim / npup  # numerical sampling related to PSD vs pup dimension
    L = D * sampnum       # Physical size of the PSD

    if dim < 2 * npup:
        logger.info("the PSD horizon must be at least two time larger than "
                    "the pupil diameter")

    # from PSD to structure function
    convnm = 2 * np.pi / (lbda * 1e9)  # nm to rad
    bg = ifft2(fftshift(psd * convnm**2)) * (psd.size / L**2)

    # creation of the structure function
    Dphi = 2 * (bg[0, 0].real - bg.real)
    Dphi = fftshift(Dphi)

    # Extraction of the pupil part of the structure function
    sampin = samp if samp is not None else sampnum
    if samp < 2:
        logger.info('PSF should be at least nyquist sampled')

    # even dimension of the num psd
    dimnum = int(np.fix(dim * (sampin / sampnum) / 2)) * 2
    sampout = dimnum / npup  # real sampling

    if samp <= sampnum:
        ns = sampout * npup / 2
        sl = slice(int(dim / 2 - ns), int(dim / 2 + ns))
        Dphi2 = Dphi[sl, sl]
    else:
        Dphi2 = np.zeros(dimnum, dimnum) + (
            Dphi[0, 0] + Dphi[dim - 1, dim - 1] +
            Dphi[0, dim - 1] + Dphi[dim - 1, 0]) / 4
        sl = slice(int(dimnum / 2 - dim / 2), int(dimnum / 2 + dim / 2))
        Dphi2[sl, sl] = Dphi
        logger.warning('Sampling > Dim DSP / Dim pup => extrapolation !!! '
                       'We recommmend to increase the PSD size')

    logger.debug('input sampling: %.2f, output sampling: %.2f, max num sampling: %.2f',
                 sampin, sampout, sampnum)

    # increasing the FoV PSF means oversampling the pupil
    FoVnum = (lbda / (sampnum * D)) * dim / (4.85 * 1.e-6)
    if FoV is None:
        FoV = FoVnum
    overFoV = FoV / FoVnum

    if not np.allclose(FoV, FoVnum):
        dimover = int(np.fix(dimnum * overFoV / 2)) * 2
        xxover = np.arange(dimover) / dimover * dimnum
        Dphi2 = np.maximum(interpolate(Dphi2, xxover, method='cubic'), 0)

        npupover = int(np.fix(npup * overFoV / 2)) * 2
        xxpupover = np.arange(npupover) / npupover * npup
        pupover = np.maximum(interpolate(pup, xxpupover, method='cubic'), 0)
    else:
        dimover = dimnum
        npupover = npup
        pupover = pup

    if phase_static is not None:
        npups = phase_static.shape[0]
        if npups != npup:
            logger.info("pup and static phase must have the same number of pixels")
        if not np.allclose(FoV, FoVnum):
            phase_static = np.maximum(
                interpolate(phase_static, xxpupover, method='cubic'), 0)

    logger.debug('input FoV: %.2f, output FoV: %.2f, Num FoV: %.2f',
                 FoV, FoVnum * dimover / dimnum, FoVnum)

    if FoV > 2 * FoVnum:
        logger.warning(': Potential alisiang issue .. I recommend to create '
                       'initial PSD and pupil with a larger numbert of pixel')

    # creation of a diff limited OTF (pupil autocorrelation)
    tab = np.zeros((dimover, dimover), dtype=complex)
    if phase_static is not None:
        pupover = pupover * np.exp(1j * phase_static * 2 * np.pi / lbda)
    tab[:npupover, :npupover] = pupover

    dlFTO = fft2(np.abs(ifft2(tab))**2)
    dlFTO = fftshift(np.abs(dlFTO) / pup.sum())

    # creation of A OTF (aoFTO = np.exp(-Dphi2 / 2))
    Dphi2 *= - 0.5
    np.exp(Dphi2, out=Dphi2)

    # Computation of final OTF
    sysFTO = fftshift(Dphi2 * dlFTO)

    # Computation of final PSF
    sysPSF = np.real(fftshift(ifft2(sysFTO)))
    sysPSF /= sysPSF.sum()  # normalisation to 1

    if return_all:
        FoV = FoVnum * dimover / dim
        return sysPSF, sampout, FoV
    else:
        return sysPSF


def radial_profile(arr, binsize=1):
    """Adapted from
    https://github.com/keflavich/image_tools/blob/master/image_tools/radialprofile.py
    """
    x, y = np.ogrid[:arr.shape[0], :arr.shape[1]]
    r = np.hypot(x - int(arr.shape[0] / 2 + .5),
                 y - int(arr.shape[1] / 2 + .5))
    nbins = int(np.round(r.max() / binsize) + 1)
    maxbin = nbins * binsize
    bins = np.linspace(0, maxbin, nbins + 1)
    nr = np.histogram(r, bins)[0]
    radial_prof = np.histogram(r, bins, weights=arr)[0]
    bin_centers = (bins[1:] + bins[:-1]) / 2
    return bin_centers, radial_prof / nr


def plot_psf(filename, npsflin=1):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    if isinstance(filename, fits.HDUList):
        psf = filename['PSF_MEAN'].data
    else:
        psf = fits.getdata(filename, extname='PSF_MEAN')
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), tight_layout=True)
    ax1, ax2, ax3 = axes[0]
    im = ax1.imshow(psf[1], origin='lower', norm=LogNorm())
    fig.colorbar(im, ax=ax1)
    ax1.set_title('PSF')

    ax2.axis('off')

    seplgs = 63.       # separation (en rayon) des LGS [arcsec]
    poslgs = np.array([[1, 1], [-1, -1], [-1, 1], [1, -1]], dtype=float).T
    poslgs *= seplgs   # *cos(pi/4) # position sur une grille cartesienne
    direction_perf(npsflin, plot=True, lgs=poslgs, ax=ax3)

    ax1, ax2, ax3 = axes[1]
    center, radial_prof = radial_profile(psf[1])
    ax1.plot(center[1:], radial_prof[1:], lw=1)
    ax1.set_yscale('log')
    ax1.set_title('radial profile')

    fit = Table.read(filename, hdu='FIT_MEAN')
    ax2.plot(fit['lbda'], fit['fwhm'][:, 0])
    ax2.set_title(r'$FWHM(\lambda)$')
    ax3.plot(fit['lbda'], fit['n'])
    ax3.set_title(r'$\beta(\lambda)$')

    return fig


def fit_psf_cube(lbda, psfcube):
    """Fit a Moffat PSF on each wavelength plane of the psfcube."""
    res = [im.moffat_fit(unit_center=None, unit_fwhm=None,
                         circular=True, fit_back=False, verbose=False)
           for im in psfcube]
    res = Table(rows=[r.__dict__ for r in res])
    res.remove_columns(('ima', 'rot', 'cont', 'err_rot', 'err_cont'))
    res['fwhm'] *= 0.2
    res['err_fwhm'] *= 0.2
    res.add_column(Column(name='lbda', data=lbda), index=0)
    return res


def convolve_final_psf(lbda, seeing, GL, L0, psf):
    """Convolve with tip-tilt and MUSE PSF to get the final PSF."""

    # 1. Convolve with Tip-tilt, beta=2
    # ---------------------------------
    beta_tt = 2

    seeingHL = seeing * (1 - GL) ** (3. / 5.)

    r0HL = 0.976 * 0.5 / seeingHL / 4.85  # *(lambdaall[ll]/(5.*1.e-7))**(6/5.)

    # aiL0 = reform(correl_osos_num(0., CN2 =  1,PROFIL_H =1 , DIAM = 8,
    # DR0 = 1, NUM_ZERN1 =  2,NUM_ZERN2 = 2,HSOURCE = 1000000000000000000.,
    # GD_ECHELLE = L01[k]>10.))
    # aikolmo = reform(correl_osos_num(0., CN2 =  1,PROFIL_H = 1, DIAM = 8,
    # DR0 = 1, NUM_ZERN1 =  2,NUM_ZERN2 = 2,HSOURCE = 1000000000000000000.,
    # GD_ECHELLE =1000000000000000000000000000000000000000.))
    # coeffHL  = 2*(aiL0/aikolmo)

    # instead of computing the thing above, we use a pre-computed table which
    # gives directly coeffHL
    l0_ind, coeff = fits.getdata(os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'coeffL0.fits'))
    coeffHL = np.interp(L0, l0_ind, coeff)

    pixscale = 0.2
    fwhmTTopt = (np.sqrt(coeffHL * 0.97 * 6.88 *
                         (.5 * 1.e-6 / (2. * np.pi))**2 *
                         8**(-1 / 3.) * r0HL**(-5 / 3.)) /
                 (4.85 * 1.e-6) * 2.35 / pixscale)

    alpha_tt = fwhmTTopt / (2 * np.sqrt(2**(1. / beta_tt) - 1))

    # Mamp = 1  # amplitude
    # interneGTTopt = moffat(Npix, Npix, [0, Mamp, alpha_tt, alpha_tt,
    #                                     Npix / 2, Npix / 2, 0, beta_tt])

    nx, ny = psf.shape[1:]
    if nx % 2 == 0:
        nx += 1
    if ny % 2 == 0:
        ny += 1
    kernel = Moffat2DKernel(alpha_tt, beta_tt, x_size=nx, y_size=ny)
    psf = fftconvolve(psf, kernel.array[np.newaxis, :, :], mode='same')

    # 2. Convolve with MUSE PSF, Use polynomial approximation
    # -----------------------------------

    fwhm, beta_muse, _, _ = muse_intrinsic_psf(lbda)
    fwhm = fwhm / pixscale
    alpha_muse = fwhm / (2 * np.sqrt(2**(1. / beta_muse) - 1))
    psf_final = np.zeros_like(psf)
    for k, (alpha, beta) in enumerate(zip(alpha_muse, beta_muse)):
        kernel = Moffat2DKernel(alpha, beta, x_size=nx, y_size=ny)
        psf_final[k, :, :] = fftconvolve(psf[k, :, :], kernel, mode='same')

    return psf_final


def compute_psf(lbda, seeing, GL, L0, npsflin=1, h=(100, 10000), three_lgs_mode=False,
                verbose=True):
    """Reconstruct a PSF from a set of seeing, GL, and L0 values.

    Parameters
    ----------
    lbda : array
        Array of wavelength for which the PSF is computed (nm).
    npsflin : int
        Number of points where the PSF is reconstructed (on each axis).
    h : tuple of float
        Altitude of the ground and high layers (m).
    three_lgs_mode : bool
        If True, use only 3 LGS.
    verbose : bool
        If True (default) log informations

    """
    if verbose:
        logger.info('Compute PSF with seeing=%.2f GL=%.2f L0=%.2f', seeing, GL, L0)
    Cn2 = [GL, 1 - GL]
    psd = simul_psd_wfm(Cn2, h, seeing, L0, zenith=0., npsflin=npsflin,
                        dim=1280, three_lgs_mode=three_lgs_mode, verbose=verbose)

    # et voila la/les PSD.
    # Pour aller plus vite, on pourrait moyennee les PSD .. c'est presque
    # la meme chose que la moyenne des PSF ... et ca permet d'aller
    # npsflin^2 fois plus vite:
    # psd = psd.mean(axis=0)

    if npsflin == 1:
        psd = psd[0]

    # Passage PSD --> PSF
    psf = psf_muse(psd, lbda)

    # Convolve with MUSE PSF and Tip-tilt
    psf = convolve_final_psf(lbda, seeing, GL, L0, psf)

    # fit all planes with a Moffat and store fit parameters
    res = fit_psf_cube(lbda, Cube(data=psf, copy=False))
    res.meta.update({'SEEING': seeing, 'GL': GL, 'L0': L0})
    res['SEEING'] = seeing
    res['GL'] = GL
    res['L0'] = L0
    return res, psf


def compute_psf_from_sparta(filename, extname='SPARTA_ATM_DATA', npsflin=1,
                            lmin=490, lmax=930, nl=35, lbda=None,
                            h=(100, 10000), n_jobs=-1, plot=False, mean_of_lgs=True,
                            verbose=True):
    """Reconstruct a PSF from SPARTA data.

    Parameters
    ----------
    filename : str or `astropy.io.fits.HDUList`
        FITS file containing a SPARTA table.
    extname : str
        Name of the SPARTA extension (defaults to SPARTA_ATM_DATA).
    npsflin : int
        Number of points where the PSF is reconstructed (on each axis).
    lmin, lmax : float
        Wavelength range (nm).
    nl : int
        Number of wavelength planes to reconstruct.
    lbda : array
        Array of wavelength values. If not given it is computed from lmin, lmax
        and nl.
    h : tuple of float
        Altitude of the ground and high layers (m).
    n_jobs : int
        Number of parallel processes to process the rows of the SPARTA table.
    plot : bool
        If True, plots the configuration if the AO system (positions of the
        LGS and the directions of reconstruction).
    mean_of_lgs : bool
        If True (default), compute the mean seeing, GL and L0 over the
        4 lasers. Otherwise a PSF is reconstructed for each laser.
    verbose : bool
        If True (default), log informations

    """
    try:
        if isinstance(filename, fits.HDUList):
            hdul = filename
        else:
            hdul = fits.open(filename)

        tbl = Table.read(hdul[extname])
        out = fits.HDUList([fits.PrimaryHDU(), hdul[extname].copy()])
    finally:
        if isinstance(filename, str):
            hdul.close()

    if len(tbl) == 1:
        n_jobs = 1

    laser_idx = []
    to_compute = []
    nrows = len(tbl)
    if lbda is None:
        lbda = np.linspace(lmin, lmax, nl)

    if verbose:
        logger.info('Processing SPARTA table with %d values, njobs=%d ...', nrows, n_jobs)

    for irow, row in enumerate(tbl, start=1):
        # use the mean value for the 4 LGS for the seeing, GL, and L0
        values = np.array([[row['LGS%d_%s' % (k, col)]
                            for col in ('SEEING', 'TUR_GND', 'L0')]
                           for k in range(1, 5)])

        # check if there are some bad values, apparently the 4th value is
        # often crap. Check if  GL > MIN_L0 and L0 < MAX_L0
        check_non_null_laser = ((values[:, 1] > 0) &       # GL > 0
                                (values[:, 2] < MAX_L0) &  # L0 < MAX_L0
                                (values[:, 2] > MIN_L0))   # L0 > MIN_L0

        nb_gs = np.sum(check_non_null_laser)
        three_lgs_mode = nb_gs < 4

        if nb_gs == 0:
            if verbose:
                logger.info('%d/%d : No valid values, skipping this row', irow, nrows)
                logger.debug('Values:', values.tolist())
            continue
        elif nb_gs < 4:
            if verbose:
                logger.info('%d/%d : Using only %d values out of 4 after outliers '
                            'rejection', irow, nrows, nb_gs)

        if mean_of_lgs:
            seeing, GL, L0 = values[check_non_null_laser].mean(axis=0)
            laser_idx.append(-1)
            to_compute.append((lbda, seeing, GL, L0, npsflin, h, three_lgs_mode, verbose))
        else:
            for i in np.where(check_non_null_laser)[0]:
                seeing, GL, L0 = values[i]
                laser_idx.append(i + 1)
                to_compute.append((lbda, seeing, GL, L0, npsflin, h, three_lgs_mode, verbose))

    if len(to_compute) == 0:
        logger.warning('No valid values')
        return

    res = Parallel(n_jobs=n_jobs)(
        delayed(compute_psf)(*args) for args in to_compute)

    # get fit table and psf for each row
    tables, psftot = zip(*res)
    stats = [(tbl.meta['SEEING'], tbl.meta['GL'], tbl.meta['L0'])
             for tbl in tables]

    for irow, (tbl, lgs_idx) in enumerate(zip(tables, laser_idx), start=1):
        tbl['row_idx'] = irow
        tbl['lgs_idx'] = lgs_idx

    # store fit values for all rows in a big table
    tbl = vstack(tables, metadata_conflicts='silent')
    hdu = fits.table_to_hdu(tbl)
    hdu.header.remove('SEEING')
    hdu.header.remove('GL')
    hdu.header.remove('L0')
    hdu.name = 'FIT_ROWS'
    out.append(hdu)

    # compute the mean PSF and store PSF and fit parameters
    psftot = np.mean(psftot, axis=0)
    res = fit_psf_cube(lbda, Cube(data=psftot, copy=False))
    # and store the mean seeing, gl and L0
    seeing, GL, L0 = np.mean(stats, axis=0)
    res.meta.update({'SEEING': seeing, 'GL': GL, 'L0': L0})

    hdu = fits.table_to_hdu(res)
    hdu.name = 'FIT_MEAN'
    out.append(hdu)
    out.append(fits.ImageHDU(data=psftot, name='PSF_MEAN'))

    if plot:
        import matplotlib.pyplot as plt
        plot_psf(out, npsflin=npsflin)
        plt.show()

    return out


def create_sparta_table(nlines=1, seeing=1, L0=25, GL=0.7, bad_l0=False,
                        outfile=None):
    """Helper function to create a SPARTA table with the given seeing, L0, and
    GL values.
    """
    # Create a SPARTA table with values for the 4 LGS
    tbl = [('LGS%d_%s' % (k, col), float(v)) for k in range(1, 5)
           for col, v in (('SEEING', seeing), ('TUR_GND', GL), ('L0', L0))]
    tbl = Table([dict(tbl)] * nlines)
    if bad_l0:
        tbl['LGS4_L0'] = 150

    hdu = fits.table_to_hdu(tbl)
    hdu.name = 'SPARTA_ATM_DATA'

    if outfile is not None:
        hdu.writeto(outfile, overwrite=True)

    return hdu


def muse_intrinsic_psf(lbda):
    """Compute MUSE PSF polynomial approximation.

    Parameters
    ----------
    lbda : float or array of float
        wavelength in nm.

    Returns
    -------
    fwhm : array of float
    beta : array or float
    fwhm_std : float
    beta_std : float

    """
    pol_beta = [-0.83704697, 1.1337153, 0.0609222, -1.35581762,
                1.15237178, 2.2106042]
    pol_fwhm = [0.60467385, -1.58905792, 1.75293264, -1.0368302,
                0.21487023, 0.34851139]
    pol_beta_std = [0.18187424, -0.17841793, 0.30962616]
    pol_fwhm_std = [0.00707504, -0.0303464, 0.04596354]
    lb = (10 * lbda - 4750) / (9350 - 4750)
    fwhm = np.polyval(pol_fwhm, lb)
    beta = np.polyval(pol_beta, lb)
    fwhm_std = np.polyval(pol_fwhm_std, lb)
    beta_std = np.polyval(pol_beta_std, lb)
    return fwhm, beta, fwhm_std, beta_std


def fit_psf_with_polynom(lbda, fwhm, beta, deg=(5, 5), output=0):
    """Fit MUSE PSF fwhm and beta with polynoms.

    Parameters
    ----------
    lbda : array of float
        Wavelength in nm.
    fwhm: array of float
        Moffat FWHM in arcsec.
    beta: array of float
        Moffat beta parameter.
    deg: tuple
        (fwhm_deg, beta_deg), polynomial degre in fwhm and beta.
    output: int
         If set to 1, the fitted values are returned.

    Returns
    -------
    dict
        dictionary with fwhm_poly (array), beta_poly (array) if output=0
        and lbda_fit, fwhm_fit, beta_fit if output=1

    """
    lb = _norm_lbda(lbda, 475, 935)
    fwhm_pol = np.polyfit(lb, fwhm, deg[0])
    beta_pol = np.polyfit(lb, beta, deg[1])
    res = dict(fwhm_pol=fwhm_pol, beta_pol=beta_pol, lbda=lbda,
               lbda_lim=(475, 935))
    if output > 0:
        lbda_fit = np.linspace(475, 935, 50)
        lbf = _norm_lbda(lbda_fit, 475, 935)
        fwhm_fit = np.polyval(fwhm_pol, lbf)
        beta_fit = np.polyval(beta_pol, lbf)
        res['lbda_fit'] = lbda_fit
        res['fwhm_fit'] = fwhm_fit
        res['beta_fit'] = beta_fit
    return res


def _norm_lbda(lbda, lb1, lb2):
    nlbda = (lbda - lb1) / (lb2 - lb1) - 0.5
    return nlbda
