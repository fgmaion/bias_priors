import numpy as np

def downsample(mstar, field, nbar, Nbins):
    '''
        Function for downsampling the amount of galaxies in each simulation.

        This helps standardize the training set.
    '''

    Mmin = 8.5
    Mmax = 12.5

    bins = np.linspace(Mmin, Mmax, Nbins+1)
    hist = np.digitize(mstar, bins)

    fsel = {}
    for i in range(Nbins):
        field_bin = field[hist==(i+1)]
        # TODO: In its current form, this can repeat the same galaxy in even in the case where we have enough objects
        # we should perhaps change it, since that means we are actually losing some information.
        try:
            rand_ints = np.random.randint(low=0, high=len(field_bin), size=int(nbar * (bins[i+1]-bins[i])) )
            fsel[i] = field_bin[rand_ints]
        except:
            print("There were no galaxies in bin {:d} of this sim".format(i))
            fsel[i] = np.array([])
    field_sel = np.concatenate([fsel[i] for i in range(Nbins)] )
    return field_sel


def combine_pars_sm_optimized(cosmo_astro, mstar):
    # This tells NumPy how many times to repeat each row of cosmo_astro
    repeats = np.array([len(mstar[i]) for i in range(len(mstar))])
    
    # 2. Repeat the cosmo_astro rows to match the mstar lengths
    # np.repeat(array, repeats, axis=0) does the heavy lifting
    pars_expanded = np.repeat(cosmo_astro, repeats, axis=0)

    # 3. Flatten the mstar list into a single 1D array
    mstar_1d = np.concatenate([mstar[i] for i in range(len(mstar))])

    # 4. Combine them (horizontal stack / column stack)
    return np.column_stack([pars_expanded, mstar_1d])

def log_params(data, pars_sel=None):
    ## Preprocess the parameters before feeding them to the NPE
    '''
        data: This function receives directly the data array imported from the CAMELS parameters file
    '''
    
    log_normalize = np.array([15,23,26,28])
    
    if pars_sel is None:
        params = np.zeros((1024, 35))

        for j in range(35):
            par = np.array([data[i][j+1] for i in range(1024)])
            if j in log_normalize:
                par = np.log10(par)
    
            params[:,j] = par
    else:
        params = np.zeros((1024, len(pars_sel)))
        for j in range(len(pars_sel)): 
            par = np.array([data[i][pars_sel[j]] for i in range(1024)])
            if pars_sel[j] in log_normalize:
                par = np.log10(par)
            params[:,j] = par

    return params


if __name__ == '__main__':

    Nsims = 1024

    import sys
    try:
        input_name = sys.argv[1]
    except:
        print("Running with full dataset, without downsampling or standardizing")

    if "input_name" in locals():
        assert input_name == "--downsample", "The option {input_name} was not recognized"
        d_sample = True
    
    #########################################################
    # Load the bias parameters for the relevant simulations #
    #########################################################
    
    mstar = {}
    bgal = {}
    
    for i in range(Nsims):
        b_SB = np.load("/mnt/home/fmaion/storage/projects/bias_priors/biases/redshift_0.00/SB35/SB35_{:d}.npy".format(i), allow_pickle=True)[0]
        
        mstar_temp = b_SB['mstar']
        mask = mstar_temp > 8.5
        
        bgal[i] = b_SB['bgal'][mask,:]
        mstar[i] = b_SB['mstar'][mask]
    
    mstar_sel = {}
    b1_sel = {}
    for i in range(Nsims):
        if d_sample:
            mstar_sel[i] = downsample(mstar[i], mstar[i], 200, 10)
            b1_sel[i] = downsample(mstar[i], bgal[i][:,0], 200, 10)
        else:
            mstar_sel[i] = mstar[i]
            b1_sel[i] = bgal[i][:,0]

    ########################################
    # Load the values of the 35 parameters #
    ########################################

    data = np.loadtxt("/mnt/ceph/users/camels/Sims/IllustrisTNG_extras/L50n512/SB35/CosmoAstroSeed_IllustrisTNG_L50n512_SB35.txt", skiprows=1,\
                     dtype={'names': ('Name', 
                                      'Omega0', #1
                                      'sigma8', #2
                                      'WindEnergyIn1e51erg', #3
                                      'RadioFeedbackFactor', #4
                                      'VariableWindVelFactor', #5
                                      'RadioFeedbackReiorientationFactor', #6
                                      'OmegaBaryon', #7
                                      'HubbleParam', #8
                                      'n_s', #9
                                      'MaxSfrTimescale', #10
                                      'FactorForSofterEQS', #11
                                      'IMFslope', #12
                                      'SNII_MinMass_Msun', #13
                                      'ThermalWindFraction', #14
                                      'VariableWindSpecMomentum', #15
                                      'WindFreeTravelDensFac', #16
                                      'MinWindVel', #17
                                      'WindEnergyReductionFactor', #18
                                      'WindEnergyReductionMetallicity', #19
                                      'WindEnergyReductionExponent', #20
                                      'WindDumpFactor', #21
                                      'SeedBlackHoleMass', #22
                                      'BlackHoleAccretionFactor', #23
                                      'BlackHoleEddingtonFactor', #24
                                      'BlackHoleFeedbackFactor', #25
                                      'BlackHoleRadiativeEfficiency', #26
                                      'QuasarThreshold', #27
                                      'QuasarThresholdPower', #28
                                      'UVBH0beta', #29
                                      'UVBH0Deltaz', #30
                                      'UVBHepbeta', #31
                                      'UVBHepDeltaz', #32
                                      'SNIa_Rate_Norm', #33
                                      'SNIa_Rate_DTD_power', #34
                                      'SofteningComovingType01', #35
                                      'seed'),
                            'formats': ('U100', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4',\
                                       'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4',\
                                       'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4')} )
    
    
    # TODO: Perhaps it would be nice to filter these parameters into a set of ~10, such that we don't have to pass this gigantic amount of information into the NN
    pars_sel = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16])

    cosmo_astro = log_params(data, pars_sel=pars_sel)

    b1_1d = np.concatenate([b1_sel[i] for i in range(Nsims)])
    x = b1_1d[:,np.newaxis]

    theta = combine_pars_sm_optimized(cosmo_astro, mstar_sel)
    if d_sample:
        res_out_name = "/mnt/home/fmaion/storage/projects/bias_priors/training_data/x_log_11_downsampled.npy"
        par_out_name = "/mnt/home/fmaion/storage/projects/bias_priors/training_data/theta_log_11_downsampled.npy"
    else:
        res_out_name = "/mnt/home/fmaion/storage/projects/bias_priors/training_data/x_log.npy"
        par_out_name = "/mnt/home/fmaion/storage/projects/bias_priors/training_data/theta_log.npy"
 
    np.save(res_out_name, x)
    print("Saved the results at {res_out_name}")

    np.save(par_out_name, theta)
    print("Saved the parameters at {par_out_name}")
