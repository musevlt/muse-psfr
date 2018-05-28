"""PSF reconstruction for MUSE WFM.

Le programme simul_psd_wfm crée un jeu de DSP (par direction du champ).

le programme psf_muse utilise ces DSP pour créer des PSF a chaque longueur
d'onde avec un pixel scale de 0.2 arcsec.

"""

import matplotlib.pyplot as plt
import numpy as np
from math import gamma
from numpy.fft import fft2, ifft2, fftshift
from scipy.interpolate import griddata


def simul_psd_wfm(Cn2, h, seeing, L0, zenith=0., visu=False, verbose=False,
                  npsflin=1, dim=1280):
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

    np.random.seed(12345)
    arg_v = (np.random.rand(h.shape[0]) - 0.5) * np.pi  # wind dir.  [rad]

    # Step 0.2 : Systeme
    # ---------
    Dpup = 8.          # en m (diametre du telescope)
    oc = 0.14          # normalisée [de 0 --> 1]
    altDM = 1.         # en m
    hsodium = 90000.   # en m

    lambdalgs = 0.589  # en µm
    lambdaref = 0.5    # en µm
    nact = 40.         # nombre lineaire d'actionneurs
    nsspup = 40.       # nombre lineaire d'actionneurs

    Fsamp = 1000.      # frequence d'échantillonnage [Hz]
    delay = 2.5        # retard en ms (lecture CCD + calcul)

    seplgs = 63.       # separation (en rayon) des LGS [arcsec]
    bruitLGS2 = 1.0    # radians de phase bord a bord de sspup @ lambdalgs
    poslgs = np.array([[1, 1], [-1, -1], [-1, 1], [1, -1]], dtype=float).T
    poslgs *= seplgs   # *cos(pi/4) # position sur une grille cartesienne
    law = "LSE"        # type de lois : lse ou mmse
    recons_cn2 = 1     # a priori sur Cn2 => ici GLAO
    recons_h = altDM   # a priori sur h   => ici GLAO

    # Step 0.3 : Direction d'estimation
    champ = 60
    dirperf = direction_perf(champ=champ, nblin=npsflin, visu=visu, lgs=poslgs)

    # Step 0.4 : Paremetres numériques
    # ---------
    Dimpup = 40        # Taille de la pupille en pixel pour zone de correction
    # coefL0 = 1         # pour gerer les "L0 numériques"

    # Step 0.5 : Mise en oeuvre
    # ---------
    r0ref = seeing2r01(seeing, lambdaref, zenith)  # passage seeing --> r0
    hz = h / np.cos(np.deg2rad(zenith))  # altitude dans la direction de visée
    dilat = (hsodium - hz) / hsodium  # dilatation pour prendre en compte la LGS
    hz_lgs = hz / dilat
    hz_lgs -= altDM  # on prend en compte la conjugaison negative du DM

    # Step 0.6 : Summarize of parameters
    # ---------
    if verbose:
        print('r0@0.5µm (zenith)        = ', seeing2r01(seeing, lambdaref, 0))
        print('r0@0.5µm (line of sight) = ', r0ref)
        print('Seeing   (line of sight) = ', 0.987 * 0.5 / r0ref / 4.85)
        print('hbarre   (zenith)        = ', np.sum(h ** (5 / 3) * Cn2) ** (3 / 5))
        print('hbarre   (line of sight) = ', np.sum(hz ** (5 / 3) * Cn2) ** (3 / 5))
        print('vbarre                   = ', np.sum(vent ** (5 / 3) * Cn2) ** (3 / 5))
        # print('theta0   (zenith)        = ',
        #       0.34 * seeing2r01(seeing, lambdaref, 0) /
        #       (np.sum(h**(5./3.)*Cn2))**(3./5.)/4.85*1.e6)
        # print('theta0   (line of sight) = ',
        #       0.34 * seeing2r01(seeing, lambdaref, zenith) /
        #       (np.sum(hz**(5./3.)*Cn2))**(3./5.)/4.85*1.e6)
        # print('t0       (line of sight) = ',
        #       0.34 * seeing2r01(seeing, lambdaref, zenith) /
        #       (np.sum(vent**(5./3.)*Cn2))**(3./5.)*1000.)

    # ========================================================================

    # longueur physique d'un ecran de ^hase issue de la PSD
    # => important pour les normalisations
    # L = dim / Dimpup * Dpup

    nact1 = 30.
    pitch = Dpup / nact1   # pitch: inter-actuator distance [m]
    fc = 1. / (2 * pitch)  # pItch frequency (1/2a)  [m^{-1}]

    # STEP 1 : Simulation des PSF LGS (tilt inclus)
    # ===============================================
    # cube de DSP pour chaque direction d'interet - ZONE DE CORRECTION ONLY
    dsp = dsp4muse(Dpup, Dimpup, Dimpup * 2, Cn2, h, L0, r0ref, recons_cn2,
                   recons_h, vent, arg_v, law, 1000, nsspup, nact, Fsamp,
                   delay, bruitLGS2, lambdaref, poslgs, dirperf,
                   verbose=verbose)

    # STEP 2: Calcul DSP fitting
    # ===============================================
    dspa = fftshift(psd_fit(dim, 2 * Dpup, r0ref, L0, fc))
    ns = dsp.shape[0]
    dspf = np.resize(dspa, (ns, dim, dim))

    # Finale
    sl = slice(dim // 2 - Dimpup, dim // 2 + Dimpup)
    dspf[:, sl, sl] = np.maximum(dspa[sl, sl], fftshift(dsp, axes=(1, 2)))

    return dspf * (lambdaref * 1000. / (2. * np.pi)) ** 2


def direction_perf(champ, nblin, visu=False, lgs=None, ngs=None):
    if nblin > 1:
        linear_tab = np.linspace(-1, 1, nblin) * champ / 2
        y, x = np.meshgrid(linear_tab, linear_tab, indexing='ij')
        dirperf = np.array([y, x]).reshape(2, -1)
    else:
        dirperf = np.zeros((2, 1))

    if visu:
        champvisu = np.max(dirperf)
        if lgs is not None:
            champvisu = max(champvisu, lgs.max())
        if ngs is not None:
            champvisu = max(champvisu, ngs.max())

        plt.scatter(dirperf[0], dirperf[1], marker='o', s=10)
        if lgs is not None:
            plt.scatter(lgs[0], lgs[1], marker='*', s=60)
        if ngs is not None:
            plt.scatter(ngs[0], ngs[1], marker='*', s=40)

        plt.xlim((-1.25 * champvisu, 1.25 * champvisu))
        plt.ylim((-1.25 * champvisu, 1.25 * champvisu))
        plt.xlabel('arcsecond')
        plt.ylabel('arcsecond')
        plt.show()

    return dirperf


def seeing2r01(seeing, lbda, zenith):
    """seeing @ 0.5 microns, lambda en microns."""
    r00p5 = 0.976 * 0.5 / seeing / 4.85  # r0 @ 0.5 µm
    r0 = r00p5 * (lbda / 0.5) ** (6 / 5) * np.cos(np.deg2rad(zenith)) ** (3 / 5)
    return r0


def pupil_mask(rt, width, oc=0, inverse=False):
    """Calcul du masque de la pupille d'un télescope.

    rt = rayon du télescope (en pixels)
    largeur = taille du masque
    oc = taux d'occultation centrale linéaire
    """
    center = (width - 1) / 2
    x, y = np.mgrid[:width, :width] - center
    rho = np.sqrt(x ** 2 + y ** 2) / rt
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


def calc_mat_rec_glao_finale(f, arg_f, pitchs_wfs, pitchs_dm, nb_gs, alpha,
                             sigr, DSP_tab_recons, h_recons, seuil,
                             condmax, LSE=False):  # h_dm, Wflag=None,
    """Computes the reconstruction matrix WMAP.

    accounting for all reconstruction parameters::

        WMAP = Ptomo ## Wtomo

    residual DSP is computed after that by `calc_dsp_res_glao_finale`.

    Parameters
    ----------
    f = spatial frequencies array
    arg_f = F phase
    Nb_gs = Guide star number
    alpha = Guide stars positions
    #theta = Optimisation directions
    sigr = A priori on noise associated to each GS
    DSP_Tab_recons = A priori, DSP on estimated turbulent layers
    h_recons = Altitudes of reconstructed layers
    #h_dm : DM altitude (if not pure tomo)
    condmax : Max acceptable conditionning in inversion for POPT computation
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
    wfs = np.zeros((nb_gs * s, s), dtype=complex)
    for j in range(nb_gs):
        ccc = (2 * np.pi * f * 1j *
               np.sinc(pitchs_wfs[j] * f_x) * np.sinc(pitchs_wfs[j] * f_y))
        fc = 1 / (2 * pitchs_wfs[j])
        # where((f NE 0) and (abs(f_x) GE fc) OR (abs(f_y) GE fc), count)
        # FIXME missing parenthesis around | ?
        ccc[(f != 0) & (np.abs(f_x) >= fc) | (np.abs(f_y) >= fc)] = 0.
        wfs[j * s:(j + 1) * s, :] = ccc
    ccc = None
    # NB: Here CCC is the transfert function of a SH, but something else could
    # be written here. Pyramid / Curvature / direct phase sensing (CCC=1)

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

    # Brique 1 :
    # M.Palpha'
    nb_h_recons = h_recons.size
    Mr = np.zeros((nb_h_recons * s, nb_gs * s), dtype=complex)
    for j in range(nb_gs):
        for i in range(nb_h_recons):
            # 206265 c'est quoi ca ?
            ff_x = f_x * alpha[0, j] * h_recons[i] * 60 / 206265
            ff_y = f_y * alpha[1, j] * h_recons[i] * 60 / 206265
            Mr[i * s:(i + 1) * s, j * s:(j + 1) * s] = \
                (wfs[j * s:(j + 1) * s, :] *
                 np.exp(1j * 2 * (ff_x + ff_y) * np.pi))

    # suppression of WFS
    wfs = 0

    # Transpose
    Mr_t = np.zeros((nb_gs * s, nb_h_recons * s), dtype=complex)
    for i in range(nb_h_recons):
        for j in range(nb_gs):
            Mr_t[j * s:(j + 1) * s, i * s:(i + 1) * s] = \
                Mr[i * s:(i + 1) * s, j * s:(j + 1) * s].conj()

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
    Cb_inv_recons = np.zeros((nb_gs * s, nb_gs * s), dtype=complex)
    for k in range(nb_gs):
        for i in range(nb_gs):
            if i == k:
                Cb_inv_recons[s * i:s * (i + 1), s * k:s * (k + 1)] = 1. / sigr[i]

    # Cphi-1, a priori on turbulence layers, computed from DSP_tab_recons
    Cphi_inv_recons = np.zeros((nb_h_recons * s, nb_h_recons * s))

    for k in range(nb_h_recons):
        for i in range(nb_h_recons):
            if i == k:
                Cphi_inv_recons[s * i:s * (i + 1), s * k:s * (k + 1)] = \
                    1. / DSP_tab_recons[s * i:s * (i + 1), :]

    # Filtering of piston in reconstruction :
    Cphi_inv_recons[0, 0] = 0.

    if LSE:
        Cphi_inv_recons *= 0.

    # W1 = ((Mrt#Cb_inv_recons#Mr + Cphi_inv_recons)^-1)Mrt#Cb_inv_recons
    # ----------------------------------------------------------------------
    # Mrt#Cb_inv first
    res_tmp = np.zeros_like(Mr_t)

    for i in range(nb_gs):
        for j in range(nb_h_recons):
            for k in range(nb_gs):
                res_tmp[i * s:(i + 1) * s, j * s:(j + 1) * s] += \
                    (Mr_t[k * s:(k + 1) * s, j * s:(j + 1) * s] *
                     Cb_inv_recons[i * s:(i + 1) * s, k * s:(k + 1) * s])

    Cb_inv_recons = 0
    # Mrt#Cb_inv#Mr then :
    model_r = np.zeros((nb_h_recons * s, nb_h_recons * s), dtype=complex)

    for k in range(nb_gs):
        for i in range(nb_h_recons):
            for j in range(nb_h_recons):
                model_r[i * s:(i + 1) * s, j * s:(j + 1) * s] += \
                    (res_tmp[k * s:(k + 1) * s, j * s:(j + 1) * s] *
                     Mr[i * s:(i + 1) * s, k * s:(k + 1) * s])

    # to be inversed :
    MAP = model_r + Cphi_inv_recons

    Cphi_inv_recons = None
    model_r = None
    # ---------------------------------------------------------------------
    # Without a priori, this is WLSE
    # ---------------------------------------------------------------------

    # Inversion of MAP matrix - Inversion frequency by frequency
    inv = np.zeros_like(MAP)
    tmp = np.zeros((nb_h_recons, nb_h_recons), dtype=complex)
    for j in range(s):
        for i in range(s):
            for k in range(nb_h_recons):
                for l in range(nb_h_recons):
                    tmp[k, l] = MAP[i + k * s, j + l * s]

            # inversion of each sub matrix
            if tmp.sum() != 0:
                if nb_h_recons > 1:
                    # FIXME: not ported yet! numpy.linalg.pinv ?
                    la_tsvd(mat=tmp, inverse=tmp_inv, condmax=seuil,
                            silent=True)
                else:
                    tmp_inv = np.linalg.inv(tmp)

                if i == 0 and j == 0:
                    tmp_inv[:] = 0

                for k in range(nb_h_recons):
                    for l in range(nb_h_recons):
                        inv[i + k * s, j + l * s] = tmp_inv[k, l]

    MAP = 0

    # Last step W1 = inv#res_tmp
    W1 = np.zeros((nb_gs * s, nb_h_recons * s), dtype=complex)
    for i in range(nb_gs):
        for j in range(nb_h_recons):
            for k in range(nb_h_recons):
                W1[i * s:(i + 1) * s, j * s:(j + 1) * s] += \
                    (inv[k * s:(k + 1) * s, j * s:(j + 1) * s] *
                     res_tmp[i * s:(i + 1) * s, k * s:(k + 1) * s])

    return W1


def calc_dsp_res_glao_finale(f, arg_f, pitchs_wfs, nb_gs, alpha, beta, sigv,
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
    f :
        tableau des frequences spatiales
    arg_f :
        argument de f
    pitchs_wfs :
        tableau des pitchs WFS
    nb_gs :
        Nbre d'etoiles Guides
    alpha :
        position des etoiles Guides dans le champs (en cartesien (x,y) et en
        arcmin)
    Beta :
        position ou on evalue la performance (en cartesien (x,y) et en arcmin)
    sigv :
        Bruit "Vrai", i.e., le bruit associé a chaque GS, qu'on utilise dans le
        calcul de Cb.
    DSP_tab_vrai :
        tableau qui contient les DSPs couches a couches de la vraie turbulence
        introduite.
    h_vrai :
        altitudes des couches du vrai profil
    h_dm :
        altitude des DMs
    Wmap :
        C'est la big Matrice de reconstruction Tomographique, elle doit sortir
        de calc_mat_rec_finale.pro
    td :
        delai
    ti :
        tableau des temps d'integration des WFS
    Wind :
        tableau vents

    """
    f_x = f * np.cos(arg_f)
    f_y = f * np.sin(arg_f)
    s = f.shape[0]
    nb_h_vrai = h_vrai.size

    # ici on ecrit tous les termes de la DSP residuelle :
    # 1. L'erreur de reconstruction
    # 2. La propagation du bruit
    # 3. Le servo-lag
    # 4. Le fitting

    if tempo:
        # print('Servo-lag Error')
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
    wfs = np.zeros((nb_gs * s, s), dtype=complex)
    for j in range(nb_gs):
        ccc = (2 * np.pi * f * 1j *
               np.sinc(pitchs_wfs[j] * f_x) * np.sinc(pitchs_wfs[j] * f_y))
        fc = 1 / (2 * pitchs_wfs[j])
        # where((f != 0) and (abs(f_x) GT fc) OR (abs(f_y) GT fc), count)
        ccc[(f != 0) & (np.abs(f_x) > fc) | (np.abs(f_y) > fc)] = 0.
        wfs[j * s:(j + 1) * s, :] = ccc

    Mv = np.zeros((nb_h_vrai * s, nb_gs * s), dtype=complex)
    for i in range(nb_h_vrai):
        for j in range(nb_gs):
            ff_x = f_x * alpha[0, j] * h_vrai[i] * 60 / 206265
            ff_y = f_y * alpha[1, j] * h_vrai[i] * 60 / 206265
            www = np.sinc(wind[0, i] * ti[j] * f_x + wind[1, i] * ti[j] * f_y)
            Mv[i * s:(i + 1) * s, j * s:(j + 1) * s] = (
                www * wfs[j * s:(j + 1) * s, :] *
                np.exp(1j * 2 * (ff_x + ff_y) * np.pi))
    wfs = None

    # ensuite, faut ecrire PbetaL#
    # on considere que les ecrans on bougé de DeltaTxV
    # deltaT = (max(ti) + td)(0)
    deltaT = ti.max() + td
    # -----------------
    proj_beta = np.zeros((s * nb_h_vrai, s), dtype=complex)
    for j in range(nb_h_vrai):
        # on considere un shift en X,Y en marche arriere :
        proj_beta[j * s:(j + 1) * s, :] = np.exp(
            1j * 2 * np.pi *
            (h_vrai[j] * 60 / 206265 * (beta[0] * f_x + beta[1] * f_y) -
             (wind[0, j] * deltaT * f_x + wind[1, j] * deltaT * f_y)))

    # ensuite, faut ecrire PbetaL
    # proj_beta = np.zeros((s*nb_h_vrai, s), dtype=complex)
    # for j in range(nb_h_vrai):
    # proj_beta(j*s:(j+1)*s-1, *) = np.exp(1j*2*np.pi*h_vrai[j]*60/206265*
    # (beta[0]*f_x+beta[1]*f_y))

    # et PbetaDM
    h_dm = np.atleast_1d(h_dm)
    nb_h_dm = h_dm.size
    proj_betaDM = np.zeros((s * nb_h_dm, s), dtype=complex)
    for j in range(nb_h_dm):
        proj_betaDM[j * s:(j + 1) * s, :] = np.exp(
            1j * 2 * np.pi * h_dm[j] * 60 / 206265 *
            (beta[0] * f_x + beta[1] * f_y))

    # ok, on ecrit donc le produit de toutes ces matrices :
    # PbetaDM#WMAP c'est un vecteur qui fait Ngs
    proj_tmp = np.zeros((nb_gs * s, s), dtype=complex)
    for i in range(nb_gs):
        for k in range(nb_h_dm):
            proj_tmp[i * s:(i + 1) * s, :] += (
                proj_betaDM[k * s:(k + 1) * s, :] *
                Wmap[i * s:(i + 1) * s, k * s:(k + 1) * s])

    # Puis, on ecrit proj_tmp#Mv
    proj_tmp2 = np.zeros((nb_h_vrai * s, s), dtype=complex)
    for i in range(nb_h_vrai):
        for k in range(nb_gs):
            proj_tmp2[i * s:(i + 1) * s, :] += (
                proj_tmp[k * s:(k + 1) * s, :] *
                Mv[i * s:(i + 1) * s, k * s:(k + 1) * s])

    # Puis (PbetaL - proj) ca sera le projecteur qu'on appliquera a Cphi pour
    # trouver l'erreur de reconstruction
    proj = proj_beta - proj_tmp2
    Mv = proj_tmp = proj_tmp2 = proj_beta = None

    # il faut son transposée : proj_T
    proj_conj = np.zeros((s, nb_h_vrai * s), dtype=complex)
    for j in range(nb_h_vrai):
        proj_conj[:, j * s:(j + 1) * s] = proj[j * s:(j + 1) * s, :].conj()

    # il manque juste a exprimer Cphi_vrai :
    Cphi_vrai = np.zeros((nb_h_vrai * s, nb_h_vrai * s))

    for i in range(nb_h_vrai):
        for k in range(nb_h_vrai):
            if i == k:
                Cphi_vrai[s * i:s * (i + 1), s * k:s * (k + 1)] = \
                    DSP_tab_vrai[s * i:s * (i + 1), :]

    # -----------------------------------------------------------------
    # MAINTENANT ON PEUT ECRIRE err_recons !!!!
    # -----------------------------------------------------------------

    # err_recons = proj#Cphi#proj_conj
    inter = np.zeros((s * nb_h_vrai, s), dtype=complex)

    for i in range(nb_h_vrai):
        for j in range(nb_h_vrai):
            inter[i * s:(i + 1) * s, :] += (
                proj[j * s:(j + 1) * s, :] *
                Cphi_vrai[i * s:(i + 1) * s, j * s:(j + 1) * s])

    err_recons = np.zeros((s, s), dtype=complex)
    for j in range(nb_h_vrai):
        tmp = inter[j * s:(j + 1) * s, :] * proj_conj[:, j * s:(j + 1) * s]
        err_recons += tmp

    err_recons[0, 0] = 0.
    err_recons = err_recons.real

    Cphi_vrai = proj = proj_conj = inter = None

    # --------------------------------------------------------------------
    # ET VOILA, ON A LA DSP D'ERREUR DE RECONSTRCUTION GLAO
    # --------------------------------------------------------------------

    # ####################################################################
    # MAINTENANT IL FAUT ECRIRE LA DSP DU BRUIT PROPAGEE A TRAVERS LE
    # RECONSTRUCTEUR GLAO
    # ####################################################################

    # That is Easy, ca s'ecrit : PbetaDM#Wmap#Cb#(PbetaDM#Wmap)T
    # Faut deja ecrire PbetaDM#Wmap

    proj_noise = np.zeros((nb_gs * s, s), dtype=complex)
    for i in range(nb_gs):
        for k in range(nb_h_dm):
            proj_noise[i * s:(i + 1) * s, :] += (
                proj_betaDM[k * s:(k + 1) * s, :] *
                Wmap[i * s:(i + 1) * s, k * s:(k + 1) * s])

    # Puis faut le transposer :
    proj_noise_conj = np.zeros((s, nb_gs * s), dtype=complex)
    for j in range(nb_gs):
        proj_noise_conj[:, j * s:(j + 1) * s] = \
            proj_noise[j * s:(j + 1) * s, :].conj()

    # Faut ecrire Cb_vrai :
    Cb_vrai = np.zeros((nb_gs * s, nb_gs * s), dtype=complex)
    for i in range(nb_gs):
        for k in range(nb_gs):
            if i == k:
                Cb_vrai[s * i:s * (i + 1), s * k:s * (k + 1)] = sigv[i]

    # -----------------------------------------------------------------------
    # MAINTENANT ON PEUT ECRIRE err_noise !!!!
    # -----------------------------------------------------------------------

    # err_noise = proj_noise#Cb#proj_noise_conj
    inter = np.zeros((s * nb_gs, s), dtype=complex)

    for i in range(nb_gs):
        for j in range(nb_gs):
            inter[i * s:(i + 1) * s, :] += (
                proj_noise[j * s:(j + 1) * s, :] *
                Cb_vrai[i * s:(i + 1) * s, j * s:(j + 1) * s])

    err_noise = np.zeros((s, s), dtype=complex)
    for j in range(nb_gs):
        err_noise += (inter[j * s:(j + 1) * s, :] *
                      proj_noise_conj[:, j * s:(j + 1) * s])

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
             vent, arg_v, seuil, LAW, nsspup, nact, Fsamp, delay, bruitLGS2,
             lambdaref, poslgs, dirperf, verbose=False):

    # Passage en arcmin
    poslgs1 = poslgs / 60
    dirperf1 = dirperf / 60

    LSE = (LAW == 'LSE')  # mmse ou lse
    tempo = True          # erreur temporelle
    fitting = True        # fitting
    dimall = int(dimall)
    err_R0 = 1.
    cst = 0.0229

    # -------------------------------------------------------------------
    # local_L = Dpup * dimall / pupdim  # taille de l'ecran en m.
    # original method:
    # fx = eclat(((np.arange(dimall) - dimall // 2) / local_L) *
    #            np.ones((dimall, 1)))
    # fy = np.transpose(fx)
    # another method:
    # fy, fx = eclat((np.mgrid[:dimall, :dimall] - dimall // 2) / local_L)

    fx = np.fft.fftfreq(dimall, Dpup / pupdim)[:, np.newaxis]
    fy = fx.T
    f = np.sqrt(fx ** 2 + fy ** 2)  # c'est f le tableau
    with np.errstate(all='ignore'):
        arg_f = fy / fx
    arg_f[0, 0] = 0  # to get the same as idl (instead of NaN)
    arg_f = np.arctan(arg_f)
    s = f.shape[0]   # taille lineaire du tableau f

    # -------------------------------------------------------------------
    #  PSD turbulente
    # -------------------------------------------------------------------

    h_recons = np.atleast_1d(h_recons)
    recons_cn2 = np.atleast_1d(recons_cn2)
    nb_h_recons = h_recons.size
    DSP_tab_recons = np.zeros((nb_h_recons * s, s))
    for i in range(nb_h_recons):
        DSP_tab_recons[s * i: s * (i + 1), :] = (
            cst * (recons_cn2[i] ** (-3 / 5) * r0ref / err_R0) ** (-5 / 3) *
            (f ** 2 + (1 / L0) ** 2) ** (-11 / 6))

    hh = np.atleast_1d(hh)
    Cn2 = np.atleast_1d(Cn2)
    DSP_tab_vrai = np.zeros((hh.size * s, s))
    for i in range(hh.size):
        DSP_tab_vrai[s * i: s * (i + 1), :] = (
            cst * (Cn2[i] ** (-3 / 5) * r0ref) ** (-5 / 3) *
            (f ** 2 + (1 / L0) ** 2) ** (-11 / 6))

    # -----------------------------------------------------
    # CALCUL DE LA MATRICE de commande GLAO
    # -----------------------------------------------------

    if verbose:
        print('seuil = ', seuil)

    nb_gs = poslgs1.shape[1]
    pitchs_wfs = np.repeat(Dpup / nsspup, nb_gs)
    sig2 = np.repeat(bruitLGS2, nb_gs)
    fech_tab = np.repeat(Fsamp, nb_gs)

    pitchs_DM = Dpup / nact
    h_dm = 1.
    condmax = 1e6
    ti = 1 / fech_tab
    td = delay * 1.e-3

    Wmap = calc_mat_rec_glao_finale(f, arg_f, pitchs_wfs, pitchs_DM, nb_gs,
                                    poslgs1, sig2, DSP_tab_recons, h_recons,
                                    seuil, condmax, LSE=LSE)

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
            f, arg_f, pitchs_wfs, nb_gs, poslgs1, beta, sig2, DSP_tab_vrai,
            hh, h_dm, Wmap, td, ti, wind, tempo=tempo, fitting=fitting)
        dsp[bbb, :, :] = dsp_res

        if verbose:
            resval = calc_var_from_psd(dsp_res, pixsize, Dpup)
            print(bbb, np.sqrt(resval) * lambdaref * 1e3 / 2. / np.pi)

    return dsp


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


def psf_muse(psd, lambdamuse, verbose=False):
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
    pixcale = 0.2
    psfall = np.zeros((nl, dimpsf, dimpsf))

    for i in range(nl):
        print('lambda {:.1f}'.format(lambdamuse[i]))
        FoV = (lambdamuse[i] / (2 * D)) * dim / (4.85 * 1e3)   # = champ total
        npixc = int(round((dimpsf * pixcale /
                           (lambdamuse[i] / (2 * 8) / 4.85 / 1000)) / 2) * 2)
        lbda = lambdamuse[i] * 1.e-9
        if ndir != 1:
            psf = np.zeros((ndir, npixc, npixc))
            for j in range(ndir):
                psf[j] = crop(psd_to_psf(psd[j], pup, D, lbda, samp=samp,
                                         FoV=FoV, verbose=verbose),
                              nc=npixc, mil=True)
            psf = psf.mean(axis=0)
        else:
            psf = psd_to_psf(psd, pup, D, lbda, samp=samp, FoV=FoV,
                             verbose=verbose)
            # psf = crop(psf, nc=npixc, mil=True)
            center = psf.shape[0] // 2
            size = npixc // 2
            sl = slice(center - size, center + size)
            psf = psf[sl, sl]

        psf /= psf.sum()
        np.clip(psf, 0, None, out=psf)

        # FIXME: griddata is slow!
        # psfall[i] = interpolate(psf, x, x, grid=True)
        x, y = np.mgrid[:dimpsf, :dimpsf] * npixc / dimpsf
        posin = np.mgrid[:npixc, :npixc].reshape(2, -1).T
        psfall[i] = griddata(posin, psf.ravel(), (x, y), method='linear')
        psfall[i] = psfall[i] / psfall[i].sum()

    return psfall


def psd_to_psf(psd, pup, D, lbda, phase_static=None, samp=None, FoV=None,
               verbose=False):
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
        print("the PSd horizon must at least two time larger than the "
              "pupil diameter")

    convnm = 2 * np.pi / (lbda * 1e9)  # nm to rad

    # from PSD to structure function
    bg = ifft2(fftshift(psd * convnm**2)) * psd.size / L**2

    # creation of the structure function
    Dphi = np.real(2 * (bg[0, 0] - bg))
    Dphi = fftshift(Dphi)

    # Extraction of the pupil part of the structure function

    sampin = samp if samp is not None else sampnum

    if samp < 2:
        print('PSF should be at least nyquist sampled')

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
        print('WARNING : Samplig > Dim DSP / Dim pup => extrapolation !!! '
              'We recommmend to increase the PSD size')

    if verbose:
        print('input sampling: {}, output sampling: {}, max num sampling: {}'
              .format(sampin, sampout, sampnum))

    # increasing the FoV PSF means oversampling the pupil
    FoVnum = (lbda / (sampnum * D)) * dim / (4.85 * 1.e-6)
    if FoV is None:
        FoV = FoVnum
    overFoV = FoV / FoVnum

    if not np.allclose(FoV, FoVnum):
        dimover = int(np.fix(dimnum * overFoV / 2)) * 2
        npupover = int(np.fix(npup * overFoV / 2)) * 2
        xxover = np.arange(dimover) / dimover * dimnum
        xxpupover = np.arange(npupover) / npupover * npup
        Dphi2 = interpolate(Dphi2, xxover, xxover, cubic=True, grid=True) > 0
        pupover = interpolate(pup, xxpupover, xxpupover, cubic=True, grid=True) > 0
    else:
        dimover = dimnum
        npupover = npup
        pupover = pup

    if phase_static is not None:
        npups = phase_static.shape[0]
        if npups != npup:
            print("pup and static phase must have the same number of pixels")
        if not np.allclose(FoV, FoVnum):
            phase_static_o = interpolate(phase_static, xxpupover, xxpupover,
                                         cubic=True, grid=True) > 0
        else:
            phase_static_o = phase_static

    if verbose:
        print('input FoV: {:.2f}, output FoV: {:.2f}, Num FoV: {:.2f}'
              .format(FoV, FoVnum * dimover / dimnum, FoVnum))

    if FoV > 2 * FoVnum:
        print('Warning : Potential alisiang issue .. I recommend to create '
              'initial PSD and pupil with a larger numbert of pixel')

    # creation of a diff limited OTF (pupil autocorrelation)
    tab = np.zeros((dimover, dimover), dtype=complex)
    if phase_static is not None:
        pupover = pupover * np.exp(1j * phase_static_o * 2 * np.pi / lbda)
    tab[:npupover, :npupover] = pupover

    dlFTO = fft2(np.abs(ifft2(tab))**2)
    dlFTO = fftshift(np.abs(dlFTO) / pup.sum())

    # creation of A OTF
    aoFTO = np.exp(-Dphi2 / 2)

    # Computation of final OTF
    sysFTO = aoFTO * dlFTO
    sysFTO = fftshift(sysFTO)

    # Computation of final PSF
    sysPSF = np.real(fftshift(ifft2(sysFTO)))
    sysPSF /= sysPSF.sum()  # normalisation to 1

    # FIXME: remove this or add option to return these values?
    samp = sampout
    FoV = FoVnum * dimover / dim
    return sysPSF


if __name__ == "__main__":
    visu = False
    verbose = True
    if visu:
        plt.ion()

    seeing = 1.
    L0 = 25.
    Cn2 = [0.7, 0.3]
    h = [500, 15000.]
    zenith = 0.
    npsflin = 3
    dim = 1280

    psd = simul_psd_wfm(Cn2, h, seeing, L0, zenith=zenith, visu=visu,
                        verbose=verbose, npsflin=npsflin, dim=dim)

    # et voila la/les PSD
    # on moyenne les PSD .. c'ets preque la meme chose que la moyenne des
    # PSF ... et ca permet d'aller npsflin^2 fois plus vite
    psdm = np.mean(psd, axis=0)

    from astropy.io import fits
    fits.writeto('psd_mean.fits', psdm, overwrite=True)

    if visu:
        from muse_analysis.plotutils import show_images_grid
        center = psdm.shape[0] // 2
        sl = slice(center - 10, center + 10)
        psd2 = psdm[sl, sl] / psdm.max()
        ref = fits.getdata('idl/psd_mean.fits')[sl, sl] / psdm.max()
        show_images_grid([ref, psd2, ref - psd2])
        plt.suptitle('IDL (left) vs Python (right)')
        plt.show()

    # Passage PSD --> PSF
    # ===================
    lambdamin = 490
    lambdamax = 930
    nl = 35
    lbda = np.linspace(lambdamin, lambdamax, nl)

    psf = psf_muse(psdm, lbda, verbose=verbose)  # < 1s par lambda sur mon PC
    fits.writeto('psf.fits', psf, overwrite=True)

    fits.printdiff('psd_mean.fits', 'psfmuse/psd_mean.fits')
    fits.printdiff('psf.fits', 'psfmuse/psf.fits')
