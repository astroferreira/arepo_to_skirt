
import os
import sys
import h5py 
import requests
import json as j
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import LambdaCDM
from sklearn.neighbors import KDTree

h = 0.6774
cosmo = LambdaCDM(H0=100*h, Om0=0.3089, Ob0=0.0486, Ode0=0.6911, Tcmb0=2.73)
snapZ = np.load('/data/snapTNG.npy')
headers = {"api-key":"ff352a2affacf64753689dd603b5b44e"}


def resample(all_df):

    old_stars_idx = np.where(all_df.age >= (100 * u.Myr).to(u.yr).value)
    old_stars = all_df  .iloc[old_stars_idx]


    nothing = np.array([])

    galaxev_df = old_stars[['x', 'y', 'z', 'h', 'mass', 'Z', 'age']]
    mappings_df = all_df[['x', 'y', 'z', 'h', 'SFR', 'Z', 'logC', 'P', 'fc', 'mass']].iloc[nothing]
    dust_ism_df = all_df[['x', 'y', 'z', 'h', 'SFR', 'Z', 'logC', 'P', 'fc', 'mass']].iloc[nothing]


    young_stars_idx = np.where((all_df.age < (100 * u.Myr).to(u.yr).value) & (all_df.SFR > 0))
    young_stars = all_df.iloc[young_stars_idx]

    mass_to_form = young_stars.SFR * (100 * u.Myr).to(u.yr).value

    nstars = young_stars.shape[0]
    for i in range(nstars):
        
        try_mass = mass_to_form.values[i]
        subparticle_masses = np.array([])

        resampling_mass = 0
        while(resampling_mass <= try_mass):
            sampled_mass = rndm(700, 1e6, -1.8, 1)
            subparticle_masses = np.append(subparticle_masses, sampled_mass)
            resampling_mass += sampled_mass

        #exclude last particle and normalize sum
        subparticle_masses = np.delete(subparticle_masses, -1)
        subparticle_masses /= subparticle_masses.sum()
        subparticle_masses *= try_mass

        n_resampled = len(subparticle_masses)
        print(n_resampled)
        min_formation_time = subparticle_masses / young_stars.iloc[i].SFR
        formation_time = np.random.uniform(1, (100*u.Myr).to(u.yr).value, size=len(min_formation_time))        
        #inheret position, smoothing length from parent particle

        xi = np.zeros(n_resampled) + young_stars.iloc[i].x
        yi = np.zeros(n_resampled) + young_stars.iloc[i].y
        zi = np.zeros(n_resampled) + young_stars.iloc[i].z
        hi = np.zeros(n_resampled) + young_stars.iloc[i].h
        Zi = np.zeros(n_resampled) + young_stars.iloc[i].Z
        P = np.zeros(n_resampled)  + young_stars.iloc[i].P
        SFRi = 10**(-7) * subparticle_masses        
        logC = 3/5 * np.log10((subparticle_masses*u.solMass).to(u.kg) / const.M_sun) + 2/5 * np.log10(P*u.Pa / const.k_B / (u.cm**-3 * u.K))
        fc = np.zeros_like(formation_time) + 0.2

        #this is the size of the HII region, its smoothing scale
        rhII = np.cbrt(8*subparticle_masses*10/(np.pi*young_stars.iloc[i].density))

        #this is the radius of the sphere where the positions are to be sampled around the parent position
        hi2 = np.sqrt(hi**2 - rhII**2)

        hi2[np.isnan(hi2)] = 0

        def sample_about(x, y, z, r):

            phi = np.random.uniform(0, 2*np.pi, len(x))
            costheta = np.random.uniform(-1,1, len(x))
            u = np.random.uniform(0,1, len(x))

            theta = np.arccos(costheta)
            ri = r * np.cbrt(u)
            xn = x + r * np.sin(theta) * np.cos(phi)
            yn = y + r * np.sin(theta) * np.sin(phi)
            zn = z + r * np.cos(theta)

            return xn, yn, zn

        xn, yn, zn = sample_about(xi, yi, zi, hi2)
        
        to_dusty_ism = np.where((formation_time <= min_formation_time))
        mappings_idx = np.where((formation_time > min_formation_time) & (formation_time <= (10*u.Myr).to(u.yr).value))
        galexev_idx = np.where((formation_time > min_formation_time)  & (formation_time  > (10*u.Myr).to(u.yr).value))
        
        galaxev_df = pd.concat([galaxev_df, pd.DataFrame(np.array([xi[galexev_idx], yi[galexev_idx], zi[galexev_idx], hi[galexev_idx], subparticle_masses[galexev_idx], Zi[galexev_idx], formation_time[galexev_idx]]).T, columns=['x', 'y', 'z', 'h', 'mass', 'Z', 'age'])])
        mappings_df = pd.concat([mappings_df, pd.DataFrame(np.array([xn[mappings_idx], yn[mappings_idx], zn[mappings_idx], rhII[mappings_idx], SFRi[mappings_idx], Zi[mappings_idx], logC[mappings_idx], P[mappings_idx], fc[mappings_idx], subparticle_masses[mappings_idx]]).T, columns=['x', 'y', 'z', 'h', 'SFR', 'Z', 'logC', 'P', 'fc', 'mass'])])
        dust_ism_df = pd.concat([mappings_df, pd.DataFrame(np.array([xn[to_dusty_ism], yn[to_dusty_ism], zn[to_dusty_ism], rhII[to_dusty_ism], SFRi[to_dusty_ism], Zi[to_dusty_ism], logC[to_dusty_ism], P[to_dusty_ism], fc[to_dusty_ism], subparticle_masses[to_dusty_ism]]).T, columns=['x', 'y', 'z', 'h', 'SFR', 'Z', 'logC', 'P', 'fc', 'mass'])])
    
    return galaxev_df, mappings_df, dust_ism_df

def rndm(a, b, g, size=1):

    def power_law(k_min, k_max, y, gamma):
        return ((k_max**(-gamma+1) - k_min**(-gamma+1))*y  + k_min**(-gamma+1.0))**(1.0/(-gamma + 1.0))

    scale_free_distribution = np.zeros(size, float)
    gamma = 1.8

    for n in range(size):
        scale_free_distribution[n] = power_law(a, b, np.random.uniform(0,1), gamma)
    
    return scale_free_distribution

def rotationMatricesFromInertiaTensor(I):
    """ Calculate 3x3 rotation matrix by a diagonalization of the moment of inertia tensor.
    Note the resultant rotation matrices are hard-coded for projection with axes=[0,1] e.g. along z. """

    # get eigen values and normalized right eigenvectors
    eigen_values, rotation_matrix = np.linalg.eig(I)

    # sort ascending the eigen values
    sort_inds = np.argsort(eigen_values)
    eigen_values = eigen_values[sort_inds]

    # permute the eigenvectors into this order, which is the rotation matrix which orients the
    # principal axes to the cartesian x,y,z axes, such that if axes=[0,1] we have face-on
    new_matrix = np.matrix( (rotation_matrix[:,sort_inds[0]],
                             rotation_matrix[:,sort_inds[1]],
                             rotation_matrix[:,sort_inds[2]]) )

    # make a random edge on view
    phi = np.random.uniform(0, 2*np.pi)
    theta = np.pi / 2
    psi = 0

    A_00 =  np.cos(psi)*np.cos(phi) - np.cos(theta)*np.sin(phi)*np.sin(psi)
    A_01 =  np.cos(psi)*np.sin(phi) + np.cos(theta)*np.cos(phi)*np.sin(psi)
    A_02 =  np.sin(psi)*np.sin(theta)
    A_10 = -np.sin(psi)*np.cos(phi) - np.cos(theta)*np.sin(phi)*np.cos(psi)
    A_11 = -np.sin(psi)*np.sin(phi) + np.cos(theta)*np.cos(phi)*np.cos(psi)
    A_12 =  np.cos(psi)*np.sin(theta)
    A_20 =  np.sin(theta)*np.sin(phi)
    A_21 = -np.sin(theta)*np.cos(phi)
    A_22 =  np.cos(theta)

    random_edgeon_matrix = np.matrix( ((A_00, A_01, A_02), (A_10, A_11, A_12), (A_20, A_21, A_22)) )

    # prepare return with a few other useful versions of this rotation matrix
    r = {}
    r['face-on'] = new_matrix
    r['edge-on'] = np.matrix( ((1,0,0),(0,0,1),(0,-1,0)) ) * r['face-on'] # disk along x-hat
    r['edge-on-smallest'] = np.matrix( ((0,1,0),(0,0,1),(1,0,0)) ) * r['face-on']
    r['edge-on-y'] = np.matrix( ((0,0,1),(1,0,0),(0,-1,0)) ) * r['face-on'] # disk along y-hat
    r['edge-on-random'] = random_edgeon_matrix * r['face-on']
    r['phi'] = phi
    r['identity'] = np.matrix( np.identity(3) )

    return r

def periodic_distance(subhalo_position, particle_coordinates):

    L = 302600
    
    dist = particle_coordinates - shPos
    print(dist)
    
    idx = np.where(abs(dist) > L/2)
    
    if(len(idx[0]) == 0):
        
        r = np.sqrt(dist[:,0]**2 + dist[:,1]**2 + dist[:,2]**2)
        
        return r
    
    dist = correct_periodic_distances(dist)
                
    return np.sqrt(dist[0]**2 + dist[1]**2 + dist[2]**2)

def correct_periodic_distances(distances):
    
    L = 302600
    
    for i, particle in enumerate(distances):
        for j, di in enumerate(particle):
            if(di>L/2):
                distances[i][j] = di - L

            if(di<-L/2):
                distances[i][j] = di + L

    return distances

def get(path, filename=None, params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically
    
    if 'content-disposition' in r.headers:
        print(r.headers['content-disposition'].split("filename=")[1])
        with open('/data/TNG_cutouts/'+filename+'.hdf5', 'wb') as f:
            f.write(r.content)
        return '/data/TNG_cutouts/'+filename+'.hdf5' # return the filename string
    
    return r

def verify_if_already_downloaded(name):

    cutout = os.path.exists(f'/data/TNG_cutouts/halo_{name}.hdf5')
    json = os.path.exists(f'/data/TNG_cutouts/{name}.json')
    stars = os.path.exists(f'/data/skirts_data/{name}_stars.dat')
    starbursting = os.path.exists(f'/data/skirts_data/{name}_starbursting.dat')
    dust = os.path.exists(f'/data/skirts_data/{name}_dust.dat')

    return cutout & json & stars & starbursting & dust

def find_faceon_rotation(gas, shPos, rHalf):
    rad_gas = periodic_distance(shPos, gas['Coordinates'][:] * a / cosmo.h)
    wGas = np.where((rad_gas <= 2.0*rHalf) & (gas['StarFormationRate'][:] > 0.0) )[0]

    masses = gas['Masses'][:][wGas] / cosmo.h
    xyz = gas['Coordinates'][:][wGas,:] * a / cosmo.h
    xyz = np.squeeze(xyz)

    if xyz.ndim == 1:
        xyz = np.reshape( xyz, (1,3) )

    for i in range(3):
        xyz[:,i] -= shPos[i]

    xyz = correct_periodic_distances(xyz)

    I = np.zeros( (3,3), dtype='float32' )

    I[0,0] = np.sum( masses * (xyz[:,1]*xyz[:,1] + xyz[:,2]*xyz[:,2]) )
    I[1,1] = np.sum( masses * (xyz[:,0]*xyz[:,0] + xyz[:,2]*xyz[:,2]) )
    I[2,2] = np.sum( masses * (xyz[:,0]*xyz[:,0] + xyz[:,1]*xyz[:,1]) )
    I[0,1] = -1 * np.sum( masses * (xyz[:,0]*xyz[:,1]) )
    I[0,2] = -1 * np.sum( masses * (xyz[:,0]*xyz[:,2]) )
    I[1,2] = -1 * np.sum( masses * (xyz[:,1]*xyz[:,2]) )
    I[1,0] = I[0,1]
    I[2,0] = I[0,2]
    I[2,1] = I[1,2]

    rotation = rotationMatricesFromInertiaTensor(I)['face-on']
    return rotation

def io(name):

    snap = int(name.split('_')[0])
    subfind = int(name.split('_')[1])

    url = f'https://www.tng-project.org/api/TNG300-1/snapshots/{snap}/subhalos/{subfind}/'

    json = get(url)
        
    rHalf = json['halfmassrad_gas'] * a / cosmo.h
    shPos = np.array([json['pos_x'], json['pos_y'], json['pos_z']]) * a / cosmo.h

    halo_url = json['cutouts']['parent_halo']
    filename = f'halo_{name}'

    if(~os.path.exists(f'/data/TNG_cutouts/{filename}.hdf5')):
        filename = get(halo_url, filename=filename)

    halo = h5py.File(filename, 'r')

    return halo, json, rHalf, shPos

def smoothing_length(x, y, z, Nk=64):
    X = np.vstack([x, y, z]).T
    tree = KDTree(X, leaf_size=2)
    dist, ind = tree.query(X, k=Nk) 
    return dist[:,-1]

def load_particles(stellar, gas, shPos, snap, resampling=True):

    def load_stellar(stellar, shPos, snap):

        age_a = stellar['GFM_StellarFormationTime'][:]
        stars = np.where(age_a > 0)[0] # exclude stellar wind entries
        ages = cosmo.lookback_time(1/stellar['GFM_StellarFormationTime'][:][stars] - 1) - cosmo.lookback_time(snapZ.T[snap][1])
        
        rotated = rotation.dot((stellar['Coordinates'][:][stars] * a / cosmo.h - shPos).T).T
        coords = rotated * 1000 
        x = np.squeeze(np.array(coords.T[0], np.float))
        y = np.squeeze(np.array(coords.T[1], np.float))
        z = np.squeeze(np.array(coords.T[2], np.float))

        h = smoothing_length(x, y, z, Nk=128)

        mass = 10**10 * stellar['GFM_InitialMass'][:][stars] / cosmo.h
        metalicity = stellar['GFM_Metallicity'][:][stars]


        starburst_threhold = (10 * u.Myr).to(u.yr)
        

        '''
            Stellar particles in Illustris don't have a density associated to it (or parent gas particle), so we use the density estimated by the smoothing length
            https://arxiv.org/pdf/1203.5667.pdf
        '''

        H_abundance = stellar['GFM_Metals'][:][stars][:, 0]
        rho = stellar['SubfindDensity'][:][stars] * (10**10 * u.solMass) * cosmo.h**2 / ((a*u.kpc)**3)
        nH = (H_abundance * rho.to(u.kg/u.cm**3) / const.m_p) # neutral hydrogen number density
        
        #normalization for polytropic pressure
        norm = 10**3 * u.cm**(-3) * u.K * 1/((0.1 * u.cm**-3 * const.m_p / 0.75)**(4/3) / const.k_B)
        gamma = 4/3
        P = (norm * rho**(gamma)).to(u.Pa)
        logC = 3/5 * np.log10((mass*u.solMass).to(u.kg) / const.M_sun) + 2/5 * np.log10(P/const.k_B / (u.cm**-3 * u.K))

        #infer SFR from initial_mass
        SFR = mass / starburst_threhold.to(u.yr).value # assume constant SFR
        fc = np.zeros_like(SFR) + 0.2

        # this is the table with all stellar particles without distinction. We separate them in GALEXEV and MAPPINGS
        # when the resampling process takes place
        df_all = pd.DataFrame(np.array([x, y, z, h, mass, metalicity, ages.to(u.yr).value, H_abundance, rho, nH, P, logC, SFR, fc]).T,
                            columns=['x', 'y', 'z', 'h', 'mass', 'Z', 'age', 'Hab', 'density', 'nH', 'P', 'logC', 'SFR', 'fc'], dtype=object)

        return df_all                        

    def load_gas(gas, shPos):
        rotated = rotation.dot((gas['Coordinates'][:]*a/cosmo.h - shPos).T)
        coords = rotated.T*1000

        x = np.squeeze(np.array(coords.T[0], np.float))
        y = np.squeeze(np.array(coords.T[1], np.float))
        z = np.squeeze(np.array(coords.T[2], np.float))
        h = smoothing_length(x, y, z, Nk=64)
        mass = gas['Masses'][:] * (10**10 * u.solMass) / cosmo.h
        metalicity = gas['GFM_Metallicity'][:]
        ages = np.zeros_like(metalicity) + 1 # age is not used for gas particles
        H_abundance = gas['GFM_Metals'][:][:, 0]

        rho = gas['Density'][:] * (10**10 * u.solMass) * cosmo.h**2 / ((a*u.kpc)**3)
        rho = rho.to(u.solMass / u.pc**3)
        nH = (H_abundance * rho.to(u.kg/u.cm**3) / const.m_p).value
        SFR = gas['StarFormationRate'][:]
        norm = 10**3 * u.cm**(-3) * u.K * 1/((0.1 * u.cm**-3 * const.m_p / 0.75)**(4/3) / const.k_B)
        gamma = 4/3
        P = (norm * rho**(gamma)).to(u.Pa)
        logC = 3/5 * np.log10((mass).to(u.kg) / const.M_sun) + 2/5 * np.log10(P/const.k_B / (u.cm**-3 * u.K))
        fc = np.zeros_like(SFR) + 0.2

        df_all = pd.DataFrame(np.array([x, y, z, h, mass, metalicity, ages, H_abundance, rho, nH, P, logC, SFR, fc]).T,
                          columns=['x', 'y', 'z', 'h', 'mass', 'Z', 'age', 'Hab', 'density', 'nH', 'P', 'logC', 'SFR', 'fc'], dtype=object)
    
        return df_all

    """
        With All Combined
    """
    df_stellar = load_stellar(stellar, shPos, snap)
    df_gas     = load_gas(gas, shPos)

    df_all = pd.concat([df_stellar, df_gas])

    print(df_all)

    within_fov = np.where((abs(df_all.x) < FOV*1000/2) & (abs(df_all.y) <  FOV*1000/2))

    df_all = df_all.iloc[within_fov]
    
    galaxev_df, mappings_df, dust_ism_df = [], [], []
    if(resampling):
        galaxev_df, mappings_df, dust_ism_df = resample(df_all)
    else:
        starburst_threshold = (10 * u.Myr).to(u.yr).value
        starbursting = np.where((df_all.age <= starburst_threshold) & ((df_all.SFR > 0) | (df_all.nH > 0.1))) #MAPPINGS
        regular = np.where(df_all.age > starburst_threshold) #GALAXEV
        galaxev_df = df_all[['x', 'y', 'z', 'h', 'mass', 'Z', 'age']].iloc[regular]
        mappings_df = df_all[['x', 'y', 'z', 'h', 'SFR', 'Z', 'logC', 'P', 'fc', 'mass']].iloc[starbursting]
        print(galaxev_df)
        print(mappings_df)

    return galaxev_df, mappings_df, dust_ism_df

def load_dusty_ism(gas, mappings_df, dust_ism_df, shPos):

    rotated = rotation.dot((gas['Coordinates'][:]*a/cosmo.h - shPos).T).T

    coords = rotated*1000
    x = np.squeeze(np.array(coords.T[0], np.float))
    y = np.squeeze(np.array(coords.T[1], np.float))
    z = np.squeeze(np.array(coords.T[2], np.float))
    
    rho = (gas['Density'][:]  * 10**10 * u.solMass /(a**3*u.kpc**3)).to(u.solMass/u.pc**3)*cosmo.h**2
    masses = gas['Masses'][:] * (10**10 * u.solMass) / cosmo.h
    sfr = gas['StarFormationRate'][:]
    nH = (gas['GFM_Metals'][:, 0] * rho.to(u.kg/u.cm**3) / const.m_p).value
    h = smoothing_length(x, y, z)
    Z = gas['GFM_Metallicity'][:]

    contributing = np.where((abs(x) < FOV/2*1000) & (abs(y) < FOV/2*1000) & ((sfr > 0) | (nH > 0.1)))

    df_dust = pd.DataFrame(np.array([x, y, z, h, masses, Z]).T, columns=['x', 'y', 'z', 'h', 'mass', 'Z']).iloc[contributing]

    ghost_particles = mappings_df[['x', 'y', 'z', 'h', 'mass', 'Z']]

    ghost_mass = ghost_particles.mass.values * -10
    ghost_region = ghost_particles.h.values * 3

    ghost_particles['mass'] = ghost_mass
    ghost_particles['h'] = ghost_region

    print(ghost_particles)
    if len(dust_ism_df) > 0:
        return pd.concat([df_dust, ghost_particles, dust_ism_df[['x', 'y', 'z', 'h', 'mass', 'Z']]])
    else:
        return df_dust

def estimate_sfr_gas_fraction(gas, json):

    sfr_gas_mass = gas['Masses'][:][index].sum()
    barionic_mass = json['mass_gas'] + json['mass_stars']
    f_sfr_gas = sfr_gas_mass/barionic_mass

    json['f_sfr_gas'] = f_sfr_gas

    with open(f'/data/TNG_cutouts/{name}.json', 'w') as outfile:
        j.dump(json, outfile)

    return f_sfr_gas*100

def calculate_frame(FOV, pixel_scale, target_z):
    FOV = FOV * u.kpc
    wave_bins = 330
    arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(target_z)
    size = np.ceil(arcsec_per_kpc.value * FOV / pixel_scale)
    n_photons = 50*size**2 / wave_bins
    return n_photons.value, size.value, pixel_scale, FOV.to(u.pc)/size


def output_ski(name, nphotons, FOV, npixels):

    FOV = (FOV * u.kpc).to(u.pc)

    MINFOV = f'{-int(FOV.value/2)} pc'
    MAXFOV = f'{int(FOV.value/2)} pc'
    STRFOV = f'{int(FOV.value)} pc'
    
    with open('91base.ski', 'r') as f: 
        with open(f'{name}.ski','w') as w:
            content = f.read()
            content = content.replace('@@@@', name)
            content = content.replace('@@@NPHOTONS@@@', str(int(nphotons)))
            content = content.replace('@@@MINFOV@@@', MINFOV)
            content = content.replace('@@@MAXFOV@@@', MAXFOV)
            content = content.replace('@@@FOV@@@', STRFOV)
            content = content.replace('@@@NPIXELS@@@', str(int(npixels)))
            w.write(content)
    
    

redshift_snap = 0.1

FOV = 70


a = cosmo.scale_factor(redshift_snap)

if __name__ == '__main__':

    name = sys.argv[1]
    snap = int(name.split('_')[0])
    subfind = int(name.split('_')[1])

    halo, json, rHalf, shPos = io(name)    

    gas = halo['PartType0']
    stellar = halo['PartType4']

    """
    Find the rotation matrix that translated the x, y plane to faceon view.
    If it is not possible to estimate it, return the identity matrix.
    """
    try:
        rotation = find_faceon_rotation(gas, shPos, rHalf)
    except:
        rotation = np.identity(3)

    galaxev_df, mappings_df, dust_ism_df = load_particles(stellar, gas, shPos, snap)
 
    
    galaxev_df.to_csv(f'{name}_stars.dat', sep=' ', header=False, index=False)
    mappings_df[['x', 'y', 'z', 'h', 'SFR', 'Z', 'logC', 'P', 'fc']].to_csv(f'{name}_starbursting.dat', sep=' ', header=False, index=False)

    df_dust = load_dusty_ism(gas, mappings_df, dust_ism_df, shPos)
    df_dust.to_csv(f'{name}_dust.dat', sep=' ', header=False, index=False)

    #print(estimate_sfr_gas_fraction(gas, json))

    n_photons, n_pixels, pixel_scale, pc_per_pix = calculate_frame(FOV, 0.03, 0.5)
    output_ski(name, n_photons, FOV, n_pixels)

