"""Test script

Le programme simul_psd_wfm crée un jeu de DSP (par direction du champ).

le programme psf_muse utilise ces DSP pour créer des PSF a chaque longueur
d'onde avec un pixel scale de 0.2 arcsec.

IDL notes:
----------

rgen(a, b, n, /le) => np.linspace(a, b, n)
rgen(a, b, n) => np.linspace(a, b, n, endpoint=False)

IDL> a = fltarr(2,  5)
IDL> a[1,2] = 1
IDL> a
       0.0000000       0.0000000
       0.0000000       0.0000000
       0.0000000       1.0000000
       0.0000000       0.0000000
       0.0000000       0.0000000

"""

import matplotlib.pyplot as plt
import numpy as np


def simul_psd_wfm(Cn2, h, seeing, L0, zenith=0., visu=False, verbose=False,
                  npsflin=1, dim=1280.):
    """ Batch de simulation de PSF WFM MUSE avec impact de NGS.

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
    poslgs = np.array([[1, 1], [-1, -1], [-1, 1], [1, -1]], dtype=float)
    poslgs *= seplgs   # *cos(pi/4) # position sur une grille cartesienne
    law = "LSE"        # type de lois : lse ou mmse
    recons_cn2 = 1     # a priori sur Cn2 => ici GLAO
    recons_h = altDM   # a priori sur h   => ici GLAO

    # Step 0.3 : Direction d'estimation
    champ = 60.
    # Affichage de la config
    dirperf = direction_perf(champ=champ, nblin=npsflin, visu=visu, lgs=poslgs)

    # Step 0.4 : Paremetres numériques
    # ---------
    Dimpup = 40.       # Taille de la pupille en pixel pour zone de correction
    coefL0 = 1         # pour gerer les "L0 numériques"
    dim1 = Dimpup * 2.

    # Step 0.5 : Mise en oeuvre
    # ---------
    r0ref = seeing2r01(seeing, lambdaref, zenith)  # passage seeing --> r0
    hz = h / np.cos(zenith * np.pi / 180.)  # altitude dans la direction de visée
    dilat = (hsodium - hz) / hsodium  # dilatation pour prendre en compte la LGS
    hz_lgs = (hz) / dilat
    hz_lgs = hz_lgs - altDM  # on prend en compte la conjugaison negative du
    # DM

    # Step 0.6 : Summarize of parameters
    # ---------
    # print('r0@0.5µm (zenith)        = ', seeing2r01(seeing, lambdaref, 0))
    # print('r0@0.5µm (line of sight) = ', seeing2r01(seeing, lambdaref, zenith))

    if verbose:
        print('r0@0.5µm (line of sight) = ', r0ref)
        print('Seeing   (line of sight) = ', 0.987 * 0.5 / r0ref / 4.85)
        print('hbarre   (zenith)        = ', np.sum(h ** (5. / 3.) * Cn2) ** (3. / 5.))
        print('hbarre   (line of sight) = ', np.sum(hz ** (5. / 3.) * Cn2) ** (3. / 5.))
        print('vbarre                   = ', np.sum(vent ** (5. / 3.) * Cn2) ** (3. / 5.))

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
    L = dim / Dimpup * Dpup

    nact1 = 30.
    pitch = Dpup / nact1   # pitch: inter-actuator distance [m]
    fc = 1. / (2 * pitch)  # pItch frequency (1/2a)  [m^{-1}]

    # STEP 1 : Simulation des PSF LGS (tilt inclus)
    # ===============================================
    # cube de DSP pour chaque direction d'interet - ZONE DE CORRECTION ONLY
    dsp = dsp4muse_s(Dpup, Dimpup, dim1, Cn2, h, float(L0), r0ref, recons_cn2,
                     recons_h, vent, arg_v, law, 1000, nsspup, nact, Fsamp,
                     delay, bruitLGS2, lambdaref, poslgs, dirperf,
                     verbose=verbose)

    __import__('pdb').set_trace()
    if len(dirperf) == 2:
        ns = 1
    else:
        ns = (size(dsp))[3]
    dspf = np.zeros((dim, dim, ns))

    # Step 2: Calcul DSP fitting
    # ------
    dspa = eclat(psd_fit_s(dim, 2 * Dpup, r0ref, L0, fc))

    # Finale
    for i in range(ns):
        dspaw = crop(dspa, nc=dim1, centre=[dim / 2, dim / 2], silent=True)
        dspf[:, :, i] = dspa
        indice = where(eclat(dsp[:, :, i]) > dspaw)
        dspaw[indice] = (eclat(dsp[:, :, i]))[indice]
        dspf[dim / 2 - dim1 / 2:dim / 2 + dim1 / 2,
             dim / 2 - dim1 / 2:dim / 2 + dim1 / 2, i] = dspaw

    return dspf * (lambdaref * 1000. / (2. * np.pi)) ** 2


def direction_perf(champ, nblin, visu=False, lgs=None, ngs=None):
    if nblin > 1:
        linear_tab1 = np.linspace(-1, 1, nblin) * champ / 2
        y, x = np.meshgrid(linear_tab1, linear_tab1, indexing='ij')
        dirperf = np.array([y, x]).reshape(2, -1).T
        # tt = 0
        # for pppx in range(nblin):
        #     for pppy in range(nblin):
        #         dirperf[:, tt] = [linear_tab1[pppx], linear_tab1[pppy]]
        #         tt = tt+1
    else:
        dirperf = np.zeros(2)

    if visu:
        champvisu = np.max(dirperf)
        if lgs is not None:
            champvisu = max(champvisu, lgs.max())
        if ngs is not None:
            champvisu = max(champvisu, ngs.max())

        plt.scatter(dirperf[:, 0], dirperf[:, 1], marker='o', s=10)
        if lgs is not None:
            plt.scatter(lgs[:, 0], lgs[:, 1], marker='*', s=60)
        if ngs is not None:
            plt.scatter(ngs[:, 0], ngs[:, 1], marker='*', s=40)

        plt.xlim((-1.25 * champvisu, 1.25 * champvisu))
        plt.ylim((-1.25 * champvisu, 1.25 * champvisu))
        plt.xlabel('arcsecond')
        plt.ylabel('arcsecond')
        plt.show()

    return dirperf


def seeing2r01(seeing, lbda, zenith):
    """seeing @ 0.5 microns, lambda en microns."""
    r00p5 = 0.976 * 0.5 / seeing / 4.85  # r0 @ 0.5 µm
    r0 = (r00p5 * (lbda / 0.5) ** (6. / 5.) *
          np.cos(zenith * np.pi / 180.) ** (3. / 5.))
    return r0


def eclat(imag, inverse=False):
    """eclate un tableau aux quatres coins"""
    sens = 1 if inverse else -1
    # to reproduce the same behavior as IDL, we need to compute "- (x//2)" and
    # not "-x//2" as it does not round to the same integer.
    if imag.ndim == 1:
        nl = imag.shape[0]
        gami = np.roll(imag, sens * (nl // 2))
    elif imag.ndim == 2:
        # image
        nl, nc = imag.shape
        gami = np.roll(imag, (sens * (nl // 2), sens * (nc // 2)), axis=(0, 1))
    elif imag.ndim == 3:
        # cube d'images
        _, nl, nc = imag.shape
        gami = np.roll(imag, (sens * (nl // 2), sens * (nc // 2)), axis=(1, 2))
    else:
        raise ValueError('ndim must be 1, 2 or 3')
    return gami


def calc_var_from_psd(psd, pixsize, DD, verbose=False):
    # Decoupage de la DSP pour eagle
    dim = psd.shape[0]
    psdtemp = eclat(psd) * pixsize ** 2.

    # Calcul de Fp
    FD = 1. / float(DD)
    boxsize = FD / pixsize
    __import__('pdb').set_trace()
    polaire2(RT=boxsize / 2., largeur=dim, entre4=True, masque=ppp)
    maskdsp = abs(ppp - 1)
    psdtemp = psdtemp * maskdsp

    return np.sum(psdtemp)


# Wflag=None, gfit=gfit,
def calc_mat_rec_glao_finale_s(f, arg_f, pitchs_wfs, pitchs_dm, nb_gs, alpha,
                               theta, sigr, DSP_tab_recons, h_recons, h_dm,
                               seuil, condmax, LSE=False):
    """
    This program computes the reconstruction matrix WMAP.
    accounting for all reconstruction parameters
    WMAP = Ptomo ## Wtomo
    residual DSP is computed after that by CALC_DSP_RES.PRO

    f = spatial frequencies array
    arg_f = F phase
    Nb_gs = Guide star number
    alpha = Guide stars positions
    theta = Optimisation directions
    sigr = A priori on noise associated to each GS
    DSP_Tab_recons = A priori, DSP on estimated turbulent layers
    h_recons = Altitudes of reconstructed layers
    h_dm : DM altitude (if not pure tomo)
    condmax : Max acceptable conditionning in inversion for POPT computation
    Wflag : Choice of reconstructor W1 or W2
    Keyword Tomo : Pure tomo
    Popt : output : optimal projector for MCAO, used later for Aliasing
    LSE : LSE instead of MAP

    """
    # nbre = nb_gs
    f_x = f * np.cos(arg_f)
    f_y = f * np.sin(arg_f)

    # out = np.zeros_like(f)
    s = f.shape[0]

    # WFS used is Shack.
    # Each WFS has its own cut off frequency
    # Construction of WFS transfert function
    wfs = np.zeros((nb_gs * s, s), dtype=complex)
    for j in range(nb_gs):
        ccc = (2 * np.pi * f *
               np.sinc(pitchs_wfs[j] * f_x) *
               np.sinc(pitchs_wfs[j] * f_y) * 1j)
        fc = 1. / (2. * pitchs_wfs[j])
        # where((f NE 0) and (abs(f_x) GE fc) OR (abs(f_y) GE fc), count)
        # FIXME missing parenthesis around | ?
        ccc[(f != 0) & (np.abs(f_x) >= fc) | (np.abs(f_y) >= fc)] = 0.
        wfs[j * s:(j + 1) * s, :] = ccc
    ccc = None
    # NB : Here CCC is the transfert function of a SH, but something else could
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
            ff_x = f_x * alpha[j, 0] * h_recons[i] * 60. / 206265.
            ff_y = f_y * alpha[j, 1] * h_recons[i] * 60. / 206265.
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

    # Writing 1 = Wtomo = ((Mrt#Cb_inv_recons#Mr + Cphi_inv_recons)^-1)Mrt#Cb_inv_recons
    # Writing 2 = Wtomo = Cphi_recons#Mrt(Mr#Cphi_recons#Mrt + Cb_recons)^-1
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

    Cphi_inv_recons = 0
    model_r = 0
    # ---------------------------------------------------------------------
    # Without a priori, this is WLSE
    # ---------------------------------------------------------------------

    __import__('pdb').set_trace()
    # Inversion of MAP matrix
    inv = np.zeros_like(MAP)
    # Inversion frequency by frequency

    tmp = np.zeros((nb_h_recons, nb_h_recons), dtype=complex)
    for j in range(s):
        for i in range(s):
            for k in range(nb_h_recons):
                for l in range(nb_h_recons):
                    tmp[k, l] = MAP[i + k * s, j + l * s]

            # inversion of each sub matrix
            if tmp.sum() != 0:
                if nb_h_recons > 1:
                    la_tsvd(mat=tmp, inverse=tmp_inv, condmax=seuil,
                            silent=True)
                else:
                    tmp_inv = invert(tmp)

                if i == 0. and j == 0.:
                    tmp_inv = tmp * 0.

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
    """
    Cette super fonction calcul la DSP_res (incluant tout les termes d'erreurs
    classiques) pour TOUT type de WFAO.

    Par exemple, si on considere plusieurs etoiles Guides + plusieurs DMs + une
    optimisation moyenne dans un champs => On fait de la MCAO

    Si on ne met qu'1 DM, et qu'on optimise dans 1 direction en particulier =>
    On fait de la LTAO/MOAO

    Si on optimsie sur un grand champs, mais qu'on a qu'1 miroir => on fait du
    GLAO.

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
    # nbre = nb_gs
    f_x = f * np.cos(arg_f)
    f_y = f * np.sin(arg_f)
    s = (size(f))(1)
    nb_h_vrai = n_elements(h_vrai)

    # ici on ecrit tous les termes de la DSP residuelle :
    # 1. L'erreur de reconstruction
    # 2. La propagation du bruit
    # 3. Le servo-lag
    # 4. Le fitting

    if tempo:
        print('Servo-lag Error')
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
        ccc = (2 * np.pi * f *
               np.sinc(pitchs_wfs(j) * f_x) *
               np.sinc(pitchs_wfs(j) * f_y) * 1j)
        fc = 1. / (2. * pitchs_wfs(j))
        # where((f != 0) and (abs(f_x) GT fc) OR (abs(f_y) GT fc), count)
        ccc[(f != 0) & (np.abs(f_x) > fc) | (np.abs(f_y) > fc)] = 0.
        wfs[j * s:(j + 1) * s, :] = ccc

    Mv = np.zeros((nb_h_vrai * s, nb_gs * s), dtype=complex)
    for i in range(nb_h_vrai):
        for j in range(nb_gs):
            ff_x = f_x * alpha[0, j] * h_vrai[i] * 60. / 206265.
            ff_y = f_y * alpha[1, j] * h_vrai[i] * 60. / 206265.
            www = np.sinc(wind[0, i] * ti[j] * f_x + wind[1, i] * ti[j] * f_y)
            Mv[i * s:(i + 1) * s, j * s:(j + 1) * s] = (
                www * wfs[j * s:(j + 1) * s, :] *
                np.exp(1j * 2 * (ff_x + ff_y) * np.pi))
    wfs = None

    # ensuite, faut ecrire PbetaL#
    # on considere que les ecrans on boug� de DeltaTxV
    deltaT = (max(ti) + td)(0)
    # -----------------
    proj_beta = np.zeros((s * nb_h_vrai, s), dtype=complex)
    for j in range(nb_h_vrai):
        # on considere un shift en X,Y en marche arriere :
        proj_beta[j * s:(j + 1) * s, :] = np.exp(
            1j * 2 * np.pi *
            (h_vrai[j] * 60. / 206265. * (beta[0] * f_x + beta[1] * f_y) -
             (wind[0, j] * deltaT * f_x + wind[1, j] * deltaT * f_y)))

    # ensuite, faut ecrire PbetaL
    # proj_beta = np.zeros((s*nb_h_vrai, s), dtype=complex)
    # for j in range(nb_h_vrai):
    # proj_beta(j*s:(j+1)*s-1, *) = np.exp(1j*2*np.pi*h_vrai[j]*60./206265.*
    # (beta[0]*f_x+beta[1]*f_y))

    # et PbetaDM
    nb_h_dm = n_elements(h_dm)
    proj_betaDM = np.zeros((s * nb_h_dm, s), dtype=complex)
    for j in range(nb_h_dm):
        proj_betaDM[j * s:(j + 1) * s, :] = np.exp(
            1j * 2 * np.pi * h_dm[j] * 60. / 206265. *
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
        proj_conj[:, j * s:(j + 1) * s] = conj(proj[j * s:(j + 1) * s, :])

    # il manque juste a exprimer Cphi_vrai :
    Cphi_vrai = np.zeros((nb_h_vrai * s, nb_h_vrai * s))

    for i in range(nb_h_vrai):
        for k in range(nb_h_vrai):
            if i == k:
                Cphi_vrai[s * i:s * (i + 1), s * k:s * (k + 1)] = \
                    DSP_tab_vrai[s * i:s * (i + 1), :]

    # -----------------------------------------------------------------
    # MAINTENANT ON PEUT ECRIRE Err_recons !!!!
    # -----------------------------------------------------------------

    # Err_recons = proj#Cphi#proj_conj
    inter = np.zeros((s * nb_h_vrai, s), dtype=complex)

    for i in range(nb_h_vrai):
        for j in range(nb_h_vrai):
            inter[i * s:(i + 1) * s, :] += (
                proj[j * s:(j + 1) * s, :] *
                Cphi_vrai[i * s:(i + 1) * s, j * s:(j + 1) * s])

    Err_recons = np.zeros((s, s), dtype=complex)
    for j in range(nb_h_vrai):
        tmp = np.zeros((s, s), dtype=complex)
        tmp = (inter[j * s:(j + 1) * s, :] *
               proj_conj[:, j * s:(j + 1) * s])
        Err_recons = Err_recons + tmp

    err_recons[0] = 0.
    err_recons = float(err_recons)

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
            conj(proj_noise[j * s:(j + 1) * s, :])

    # Faut ecrire Cb_vrai :
    Cb_vrai = np.zeros((nb_gs * s, nb_gs * s), dtype=complex)
    for i in range(nb_gs):
        for k in range(nb_gs):
            if i == k:
                Cb_vrai[s * i:s * (i + 1), s * k:s * (k + 1)] = sigv(i)

    # -----------------------------------------------------------------------
    # MAINTENANT ON PEUT ECRIRE Err_noise !!!!
    # -----------------------------------------------------------------------

    # Err_noise = proj_noise#Cb#proj_noise_conj
    inter = np.zeros((s * nb_gs, s), dtype=complex)

    for i in range(nb_gs):
        for j in range(nb_gs):
            inter[i * s:(i + 1) * s, :] += (
                proj_noise[j * s:(j + 1) * s, :] *
                Cb_vrai[i * s:(i + 1) * s, j * s:(j + 1) * s])

    Err_noise = np.zeros((s, s), dtype=complex)
    for j in range(nb_gs):
        tmp = np.zeros((s, s), dtype=complex)
        tmp = (inter[j * s:(j + 1) * s, :] *
               proj_noise_conj[:, j * s:(j + 1) * s])
        Err_noise = Err_noise + tmp

    err_noise[0] = 0.
    err_noise = float(err_noise)
    # -----------------------------------------------------------------------
    # ET VOILA, ON A LA DSP D'ERREUR DE PROPAGATION DU BRUIT TOMOGRAPHIQUE
    # -----------------------------------------------------------------------

    dsp_res = float(Err_recons + Err_noise)
    dsp_res[0, 0] = 0.

    if fitting:
        return dsp_res

    out_m = f * 0.0
    fc = max(1. / (2. * pitchs_wfs))
    f_ind = where((f != 0) & (abs(f_x) <= fc) & (abs(f_y) <= fc), count)
    if count != 0:
        out_m[f_ind] = dsp_res[f_ind]
    return out_m


def dsp4muse_s(Dpup, pupdim, dimall, Cn2, hh, L0, r0ref, recons_cn2, h_recons,
               vent, arg_v, seuil, LAW, nsspup, nact, Fsamp, delay, bruitLGS2,
               lambdaref, poslgs, dirperf, verbose=False):

    # Passage en arcmin
    poslgs1 = poslgs / 60.
    dirperf1 = dirperf / 60.

    if LAW == 'LSE':  # mmse ou lse
        LSE = True
    else:
        LSE = False

    # gfit = True       # generalized fitting
    tempo = True      # erreur temporelle
    fitting = True    # fitting

    # -------------------------------------------------------------------
    local_L = Dpup * dimall / pupdim  # taille de l'ecran en m.
    local_dim = int(dimall)
    # original method:
    # fx = eclat(((np.arange(local_dim) - local_dim // 2) / local_L) *
    #            np.ones((local_dim, 1)))
    # fy = np.transpose(fx)
    # another method:
    # fy, fx = eclat((np.mgrid[:local_dim, :local_dim] - local_dim // 2) / local_L)

    fx = np.fft.fftfreq(local_dim, local_L / local_dim)
    fy = fx.T
    f = np.sqrt(fx ** 2 + fy ** 2)  # c'est f le tableau
    with np.errstate(invalid='ignore'):
        arg_f = np.arctan(fy / fx)
    s = f.shape[0]  # taille lineaire du tableau f
    # -------------------------------------------------------------------
    err_R0 = 1.
    cst = 0.0229

    # -------------------------------------------------------------------
    #  PSD turbulente
    # -------------------------------------------------------------------

    h_recons = np.atleast_1d(h_recons)
    recons_cn2 = np.atleast_1d(recons_cn2)
    nb_h_recons = h_recons.size
    DSP_tab_recons = np.zeros((nb_h_recons * s, s))
    for i in range(nb_h_recons):
        DSP_tab_recons[s * i: s * (i + 1), :] = (
            cst *
            (recons_cn2[i] ** (-3. / 5.) * r0ref / err_R0) ** (-5. / 3.) *
            (f ** 2 + (1 / L0) ** 2) ** (-11. / 6.))

    hh = np.atleast_1d(hh)
    Cn2 = np.atleast_1d(Cn2)
    DSP_tab_vrai = np.zeros((hh.size * s, s))
    for i in range(hh.size):
        DSP_tab_vrai[s * i: s * (i + 1), :] = (
            cst *
            (Cn2[i] ** (-3. / 5.) * r0ref) ** (-5. / 3.) *
            (f ** 2 + (1 / L0) ** 2) ** (-11. / 6.))

    # -----------------------------------------------------
    # CALCUL DE LA MATRICE de commande GLAO
    # -----------------------------------------------------

    if verbose:
        print('seuil = ', seuil)

    Nb_gs = poslgs1.shape[0]
    pitchs_wfs = np.repeat(Dpup / nsspup, Nb_gs)
    sig2 = np.repeat(bruitLGS2, Nb_gs)
    fech_tab = np.repeat(Fsamp, Nb_gs)

    pitchs_DM = Dpup / nact
    h_dm = 1.
    condmax = 1e6
    ti = 1 / fech_tab
    td = delay * 1.e-3

    Wmap = calc_mat_rec_glao_finale_s(f, arg_f, pitchs_wfs, pitchs_DM, Nb_gs,
                                      poslgs1, poslgs1, sig2, DSP_tab_recons,
                                      h_recons, h_dm, seuil, condmax,
                                      LSE=LSE)  # gfit=gfit,

    # DSP dans les differentes directions de reconstruction
    # =======================================================
    nn = size(dirperf1)
    if nn[0] == 1:
        nb_dir_perf = 1
    else:
        nb_dir_perf = nn[2]

    dsp = np.zeros((nb_dir_perf, dimall, dimall))

    wind = np.transpose([[vent * np.cos(arg_v)], [vent * np.sin(arg_v)]])

    for bbb in range(nb_dir_perf):
        beta = [dirperf1(0, bbb), dirperf1(1, bbb)]
        # DSP tempo + noise-tomp + fitting
        dsp_res = calc_dsp_res_glao_finale(
            f, arg_f, pitchs_wfs, Nb_gs, poslgs1, beta, sig2, DSP_tab_vrai,
            hh, h_dm, Wmap, td, ti, wind, tempo=tempo, fitting=fitting,
            err_recons=err_recons, err_noise=err_noise)
        dsp[bbb, :, :] = dsp_res

        if verbose:
            L = Dpup * dimall / pupdim
            pixsize = 1. / float(L)
            resva1 = calc_var_from_psd(dsp_res, pixsize=pixsize, DD=Dpup)
            print(bbb, np.sqrt(resva1) * lambdaref * 1e3 / 2. / np.pi)

    return dsp


if __name__ == "__main__":
    seeing = 1.
    L0 = 25.
    Cn2 = [0.7, 0.3]
    h = [500, 15000.]
    zenith = 0.
    npsflin = 3
    dim = 1280.

    psd = simul_psd_wfm(Cn2, h, seeing, L0, zenith=zenith,
                        visu=False, verbose=True, npsflin=npsflin, dim=dim)

    # et voila la/les PSD
    # on moyenne les PSD .. c'ets preque la meme chose que la moyenen des
    # PSF ... et ca permet d'alller npsflin^2 fois plus vite
    psdm = np.sum(psd, axis=2) / npsflin ** 2

    # Passage PSD --> PSF
    # ===================
    lambdamin = 490.
    lambdamax = 930.
    nl = 35
    lambdamuse = np.linspace(lambdamin, lambdamax, nl)

    psf1 = psf_muse(psdm, lambdamuse)  # < 1s par lambda sur mon PC ...

    # psf2 = psf_muse(psd,lambdamuse)
    # i= 9
    # atv,[psf1[*,*,i],psf2[*,*,i], abs(psf1[*,*,i]-psf2[*,*,i])]
