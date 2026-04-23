import bacco
import numpy as np
import bacco.probabilistic_bias as pb
import scipy
import os
import sys

bacco.configuration.update({'number_of_threads':16})

project_src = '/mnt/ceph/users/fmaion/projects/bias_priors/src'
sys.path.insert(0, project_src)
os.chdir(project_src)

import utils

def get_seed(basedir):

    file_name = basedir + "/ICs/2LPT.param"
    with open(file_name) as f:
        lines = f.readlines()
        seed = int(lines[24].split()[1])

    return seed

def get_biases(sim_i, kd, seed):
    snap = 90

    basedir = "/mnt/ceph/users/camels"
    dm_file   = basedir + "/Sims/IllustrisTNG_extras/L50n512/SB35/SB35_{:d}".format(sim_i) + "/snapdir_%03d/snap_%03d"%(snap,snap)
    halo_file = basedir + "/FOF_Subfind/IllustrisTNG_extras/L50n512/SB35/SB35_{:d}".format(sim_i) + "/groups_%03d/groups_%03d"%(snap,snap)

    sigma8 = 0.8159
    ns     = 0.9667
    tau    = 0.0965
    sim_format = 'TNG'
    cm_50 = bacco.Simulation(verbose=False, basedir=basedir, halo_file=halo_file,\
               dm_file=dm_file, tau=tau, ns=ns, sigma8=sigma8,\
               numpart=512**3, sim_format=sim_format, use_orphans=False)
    
    cm_50.header['Seed'] = seed 
    cm_50.header['InitialPhase'] = 0.0
    cm_50.header['FixedInitialAmplitude'] = False    

    sel = 1e10 * cm_50.sub['MassType'][:,4] > 1e8
    
    sort = np.argsort(cm_50.dm['ids'])
    
    dm_pos = cm_50.dm['pos'][sort]
    dm_vel = cm_50.dm['vel'][sort]
    dm_ids = cm_50.dm['ids'][sort]
    
    pos_tree = scipy.spatial.KDTree(dm_pos)

    # These are the variables that need to be measured on a Lagrangian grid
    D_variables = ("J2", "J2=2")
    D_pbm = pb.ProbabilisticBiasManager(cm_50, variables=D_variables, damping_scale=kd, ngrid=192, filt='sharpk')
    
    D_terms = ("J2", "J22", "J2=2", )
    D_model = D_pbm.setup_bias_model(pb.TensorBiasND, terms=D_terms, spatial_order=2)
    
    d_dm, id_dm = pos_tree.query(cm_50.sub['pos'][sel]) # This has to be improved, let's explicitly get the mostbound ID and then match it.
    q = utils.dmpos_from_ids(dm_ids[id_dm], 50000/512)
    print("Done! Lagrangian positions have been obtained")
    
    tr_q, tr_value, tr_mask = D_pbm._define_tracers(tracer_q=q/1e3)
    
    b_gal = D_model.bias_per_object(tr_value)
    
    mstar = np.log10(1e10*cm_50.sub['MassType'][sel,4])
    # mass bins
    
    return mstar, b_gal, tr_q, np.where(sel)[0]


seed_dir = "/mnt/ceph/users/camels/Sims/IllustrisTNG_extras/L50n512/SB35/"
kd = 0.3

for sim_i in range(1):
    print("Now measuring simulation {:d}".format(sim_i))
    
    dir_name = "SB35_{:d}".format(sim_i)
    seed = get_seed(seed_dir+dir_name)
    assert(seed==25000+sim_i)

    mstar, bgal, tr_q, gal_ids = get_biases(sim_i, kd, seed)

    np.save("/mnt/ceph/users/fmaion/projects/bias_priors/biases/"+dir_name, [{'kd':kd, 'mstar':mstar, 'bgal':bgal, 'tr_q':tr_q, 'gal_ids':gal_ids}])

