import numpy as np
from astropy.cosmology import LambdaCDM
from sklearn.neighbors import KDTree

h = 0.6774
cosmo = LambdaCDM(H0=100*h, Om0=0.3089, Ob0=0.0486, Ode0=0.6911, Tcmb0=2.73)

def rotationMatricesFromInertiaTensor(I):
    """kindly provided by Dylan Nelson on https://www.tng-project.org/data/forum/topic/223/subhalo-face-on-vector-values/
    Calculate 3x3 rotation matrix by a diagonalization of the moment of inertia tensor.
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

def periodic_distance(subhalo_position, particle_coordinates, shPos):

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


def find_faceon_rotation(gas, stellar, shPos, rHalf, a):
    rad_gas = periodic_distance(shPos, gas['Coordinates'][:] * a / cosmo.h, shPos)
    rad_stellar = periodic_distance(shPos, stellar['Coordinates'][:] * a / cosmo.h, shPos)
    wGas = np.where((rad_gas <= 2.0*rHalf) & (gas['StarFormationRate'][:] > 0.0) )[0]
    wStellar = np.where((rad_stellar <= rHalf))[0]

    print('WGAS', len(wGas))
    print('wStellar', len(wStellar))

    if len(wGas) > len(wStellar):
        masses = gas['Masses'][:][wGas] / cosmo.h
        xyz = gas['Coordinates'][:][wGas,:] * a / cosmo.h
    else:
        masses = stellar['Masses'][:][wStellar] / cosmo.h
        xyz = stellar['Coordinates'][:][wStellar,:] * a / cosmo.h
    
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