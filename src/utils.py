import numpy as np

def dmpos_from_ids(ids, cell_size):
    """
    Vectorized version of dmpos_from_id.
    ids: array-like of integer IDs (1-based)
    cell_size: float
    Returns: positions array of shape (len(ids), 3)
    """
    ids = np.asarray(ids)
    Npart = ids.size

    # Constants
    B = 64
    N = 512
    NB = N // B             # 8 blocks per dimension
    block_particles = B**3  # 262144

    # -------------------------
    # Convert to zero-based ID
    # -------------------------
    ID0 = ids - 1

    # -------------------------
    # Block index + offset
    # -------------------------
    block_id = ( ID0 // block_particles ) // 2
    offset_in_block = ID0 % block_particles

    # --------------------------------
    # Compute block coordinates (bx,by,bz)
    # --------------------------------
    bz = block_id % NB
    by = (block_id // NB) % NB
    bx = (block_id // (NB * NB)) % NB

    # Block origins
    x0 = bx * B * cell_size
    y0 = by * B * cell_size
    z0 = bz * B * cell_size

    # -------------------------
    # Offset inside block
    # -------------------------
    iz = offset_in_block % B
    iy = (offset_in_block // B) % B
    ix = (offset_in_block // (B * B)) % B

    # -------------------------
    # Cell-centering offsets
    # -------------------------
    half = cell_size / 2
    x = x0 + half + ix * cell_size
    y = y0 + half + iy * cell_size
    z = z0 + half + iz * cell_size

    # -------------------------
    # Periodic wrapping (vectorized)
    # -------------------------
    boxsize = 50000.0
    x %= boxsize
    y %= boxsize
    z %= boxsize

    # Combine into (N,3) array
    return np.stack((x, y, z), axis=1)

def load_CAMELS_50_1P_parameters():
    pars = np.loadtxt("/mnt/ceph/users/camels/Sims/IllustrisTNG/L50n512/1P/CosmoAstroSeed_IllustrisTNG_L50n512_1P.txt", skiprows=1,\
                     dtype={'names': ('Name', 'Omega0', 'sigma8', 'WindEnergyIn1e51erg', 'RadioFeedbackFactor', 'VariableWindVelFactor',\
                                      'RadioFeedbackReiorientationFactor', 'OmegaBaryon', 'HubbleParam', 'n_s',\
                                      'MaxSfrTimescale', 'FactorForSofterEQS', 'IMFslope', 'SNII_MinMass_Msun',\
                                      'ThermalWindFraction', 'VariableWindSpecMomentum', 'WindFreeTravelDensFac',\
                                      'MinWindVel', 'WindEnergyReductionFactor', 'WindEnergyReductionMetallicity',\
                                      'WindEnergyReductionExponent', 'WindDumpFactor', 'SeedBlackHoleMass',\
                                      'BlackHoleAccretionFactor', 'BlackHoleEddingtonFactor', 'BlackHoleFeedbackFactor',\
                                      'BlackHoleRadiativeEfficiency', 'QuasarThreshold', 'QuasarThresholdPower', 'UVBH0beta',\
                                      'UVBH0Deltaz', 'UVBHepbeta', 'UVBHepDeltaz', 'SNIa_Rate_Norm', 'SNIa_Rate_DTD_power',\
                                      'SofteningComovingType01', 'seed'),
                            'formats': ('U100', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4',\
                                       'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4',\
                                       'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4')} )

    params = {}
    for par_i in range(1,36):
        params[par_i] = {}

        if par_i==15:
            names = ["1P_p1_0",
                    "1P_p{:d}_1".format(par_i),
                    "1P_p{:d}_2".format(par_i),
                    "1P_p{:d}_3".format(par_i),
                    "1P_p{:d}_4".format(par_i)]
        elif par_i==1:
            names = ["1P_p{:d}_n2".format(par_i),
                    "1P_p{:d}_n1".format(par_i),
                    "1P_p{:d}_0".format(par_i),
                    "1P_p{:d}_1".format(par_i),
                    "1P_p{:d}_2".format(par_i)]
        elif par_i==29:
            names = ["1P_p{:d}_n1".format(par_i),
                     "1P_p1_0",
                    "1P_p{:d}_1".format(par_i),
                    "1P_p{:d}_2".format(par_i),
                    "1P_p{:d}_3".format(par_i)]
        else:
            names = ["1P_p{:d}_n2".format(par_i),
                    "1P_p{:d}_n1".format(par_i),
                    "1P_p1_0",
                    "1P_p{:d}_1".format(par_i),
                    "1P_p{:d}_2".format(par_i)]

        for i in range(len(names)):
            for j in range(len(pars)):
                if pars[j][0] == names[i]:
                    params[par_i][i] = np.array([pars[j][n] for n in range(1,36)])

    return params


def load_CAMELS_50_1P_biases(SFR_out=False):

    b_gal = {}
    mstar = {}
    SFR = {}
    
    kd = 0.3
    basedir = "/mnt/home/fmaion/storage/projects/bias_priors/biases/redshift_0.00/1P/"
    
    for par_i in range(1,36):
        b_gal[par_i] = {}
        mstar[par_i] = {}
        SFR[par_i] = {}
        
        if par_i==15:
            names = ["1P_p1_0",
                    "1P_p{:d}_1".format(par_i),
                    "1P_p{:d}_2".format(par_i),
                    "1P_p{:d}_3".format(par_i),
                    "1P_p{:d}_4".format(par_i)]
        elif par_i==1:
            names = ["1P_p{:d}_n2".format(par_i),
                    "1P_p{:d}_n1".format(par_i),
                    "1P_p{:d}_0".format(par_i),
                    "1P_p{:d}_1".format(par_i),
                    "1P_p{:d}_2".format(par_i)]
        elif par_i==29:
            names = ["1P_p{:d}_n1".format(par_i),
                     "1P_p1_0",
                    "1P_p{:d}_1".format(par_i),
                    "1P_p{:d}_2".format(par_i),
                    "1P_p{:d}_3".format(par_i)]
        elif par_i==30 or par_i==32:
            names = ["1P_p1_0",
                    "1P_p{:d}_n1".format(par_i),
                    "1P_p{:d}_n2".format(par_i),
                    "1P_p{:d}_n3".format(par_i),
                    "1P_p{:d}_n4".format(par_i)]
        else:
            names = ["1P_p{:d}_n2".format(par_i),
                    "1P_p{:d}_n1".format(par_i),
                    "1P_p1_0",
                    "1P_p{:d}_1".format(par_i),
                    "1P_p{:d}_2".format(par_i)]
            
        for i in range(len(names)):
            tmp_bias = np.load(basedir+names[i]+".npy", allow_pickle=True)[0]

            b_gal[par_i][i] = tmp_bias['bgal']
            mstar[par_i][i] = tmp_bias['mstar']
            SFR[par_i][i] = tmp_bias['SFR']
    if SFR_out is True:
        return b_gal, mstar, SFR
    else:
        return b_gal, mstar

def sel_LRG_DESI(sim_name):
    import bacco
    
    basedir = "/mnt/ceph/users/camels/Sims/IllustrisTNG/L50n512/1P/" + sim_name
    
    snap = 73
    
    sigma8 = 0.8159
    ns     = 0.9667
    tau    = 0.0965
    sim_format = 'TNG'
    cm_50 = bacco.Simulation(verbose=False, basedir=basedir, halo_file='groups_%03d/fof_subhalo_tab_%03d'%(snap,snap),\
               tau=tau,ns = ns, sigma8=sigma8, numpart=512**3, sim_format=sim_format, use_orphans=False)
    
    cm_50.header['Seed'] = 18
    cm_50.header['InitialPhase'] = 0.0
    cm_50.header['FixedInitialAmplitude'] = False    

    assert np.abs( (1/cm_50.Cosmology.expfactor - 1) - 0.5 ) < 0.01
    
    sSFR_tot = cm_50.sub["SFR"] / ( 1e10 * cm_50.sub["MassType"][:,4] )
    mstar_tot = 1e10 * cm_50.sub["MassType"][:,4]
    
    presel = np.where( (-np.log10(sSFR_tot) > 9.0862) & (~np.isnan(sSFR_tot)) )[0]
    
    mstar_presel = 1e10 * cm_50.sub['MassType'][presel,4]
    sort = np.argsort(mstar_presel)[::-1]

    nbar_DESI = 1e-3
    N_DESI = 50**3 * nbar_DESI

    sel_DESI = presel[sort[:int(N_DESI)+1]]

    return sel_DESI

def sel_LRG_bfiles():

    b_gal, mstar, SFR = load_CAMELS_50_1P_biases(SFR_out=True)

    # sel_LRG = {}
    # for par_i in range(1,36):
    #     for j in range(5):
            

def sel_ELG_DESI(sim_name):
    import bacco
    
    basedir = "/mnt/ceph/users/camels/Sims/IllustrisTNG/L50n512/1P/" + sim_name
    
    snap = 61
    
    sigma8 = 0.8159
    ns     = 0.9667
    tau    = 0.0965
    sim_format = 'TNG'
    cm_50 = bacco.Simulation(verbose=False, basedir=basedir, halo_file='groups_%03d/fof_subhalo_tab_%03d'%(snap,snap),\
               tau=tau,ns = ns, sigma8=sigma8, numpart=512**3, sim_format=sim_format, use_orphans=False)
    
    cm_50.header['Seed'] = 18
    cm_50.header['InitialPhase'] = 0.0
    cm_50.header['FixedInitialAmplitude'] = False    

    assert np.abs( (1/cm_50.Cosmology.expfactor - 1) - 1 ) < 0.01
    
    sSFR_tot = cm_50.sub["SFR"] / ( 1e10 * cm_50.sub["MassType"][:,4] )
    mstar_tot = 1e10 * cm_50.sub["MassType"][:,4]
    
    presel = np.where( (-np.log10(sSFR_tot) < 9.0862) & (~np.isnan(sSFR_tot)) )[0]
    
    mstar_presel = 1e10 * cm_50.sub['MassType'][presel,4]
    sort = np.argsort(mstar_presel)[::-1]

    nbar_DESI = 1e-3
    N_DESI = 50**3 * nbar_DESI

    sel_DESI = presel[sort[:int(N_DESI)+1]]

    return sel_DESI
