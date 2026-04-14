import scipy.constants as const
import numpy as np
from tqdm import tqdm
from scipy.integrate import solve_ivp

from line_profiler import profile

#Universal constants
hbar = const.hbar
h = const.h
c = const.c
g = 9.80665  # m/s^2
eps_0 = const.epsilon_0
pi = const.pi
m_e = const.m_e
a_0 = const.physical_constants['Bohr radius'][0]
e = const.e

# Parameters of the system
m_yb = 2.8384645678e-25 #171Yb mass inkg
wl_latt = 532e-9  # m (Yb 1^S_0--> 3^P_1)
wl_tw = 485e-9
k_latt = 2 * np.pi / wl_latt  # 1/m
k_tw = 2 * np.pi / wl_tw  # 1/m

#Unitary vectors along x,y,z axes
e_x = np.array([1.0, 0.0, 0.0])
e_y = np.array([0.0, 1.0, 0.0])
e_z = np.array([0.0, 0.0, 1.0])

####################################################### GENERAL ############################################################

def nm_to_Hz(wl):
    """
    Convert wavelength to frequency.

    :param wl: wavelength in meters
    """
    return  2.99792458e17/wl

def au_to_SI(polarizability_au):
    """
    Convert polarizability from atomic units to SI units.

    :param polarizability_au: polarizability in atomic units
    """
    prop_const = m_e*e**2*a_0**4/(hbar**2)
    return polarizability_au * prop_const

def J_to_Hz(E):
    """
    Convert energy from joules to hertz.
    """
    return E/h


def gaussian_beam(x, y, z, P, w0, wavelength, z0=0):
    """
    Intensity profile of a Gaussian beam.
    
    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param P: Power of the laser beam
    :param w0: Beam waist
    :param wavelength: Wavelength of the light
    """
    z_R = pi * w0**2 / wavelength
    w_z = w0 * (1 + ((z-z0) / z_R)**2)**0.5
    r = (x**2 + y**2)**0.5
    I0 = (2 * P) / (pi * w_z**2) 
    exp_spatial = np.exp(-2 * r**2 / w_z**2)
    I = I0 * exp_spatial
    return I

def gaussian_beam_rotated(x, y, z, P, w0, wavelength, z0=0):
    """
    Intensity profile of a Gaussian beam.
    
    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param P: Power of the laser beam
    :param w0: Beam waist
    :param wavelength: Wavelength of the light
    """
    x, y, z = y, z, x
    z_R = pi * w0**2 / wavelength
    w_z = w0 * (1 + ((z-z0) / z_R)**2)**0.5
    r = (x**2 + y**2)**0.5
    I0 = (2 * P) / (pi * w_z**2) 
    exp_spatial = np.exp(-2 * r**2 / w_z**2)
    I = I0 * exp_spatial
    return I

def w_z(z, w0, wavelength, z0=0):
    """
    Beam radius as a function of z

    :param w0: Beam waist
    :param wavelength: Wavelength of the light
    """
    zR = pi*w0**2/wavelength
    w=w0*np.sqrt(1+((z-z0)/zR)**2)
    return w

def w_x(x, w0, wavelength, x0=0):
    """
    Beam radius as a function of x

    :param w0: Beam waist
    :param wavelength: Wavelength of the light
    """
    xR = pi*w0**2/wavelength
    w=w0*np.sqrt(1+((x-x0)/xR)**2)
    return w

def recoil_energy(m, wavelength):
    """
    Calculate the recoil energy of an atom.

    :param m: mass of the atom
    :param wavelength: wavelength of the light
    """
    k = 2 * pi / wavelength
    E_r = (hbar * k)**2 / (2 * m)
    return E_r

def recil_velocity(m, wavelength):
    """
    Calculate the recoil velocity of an atom.

    :param m: mass of the atom
    :param wavelength: wavelength of the light
    """
    k = 2 * pi / wavelength
    v_r = hbar * k / m
    return v_r

def grad_I(x, y, z, P, w0, wavelength, z0=0):
    """
    Gradient of the intensity of a Gaussian beam.

    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param P: Power of the laser beam
    :param w0: Beam waist
    :param wavelength: Wavelength of the light
    :param z0: Position of the waist
    """
    A = 2*P/(pi*w0**2)
    B = pi * w0**2 / wavelength
    C = w0**2
    grad_x = -4*x*A/(C*(1+z**2/B**2))*(1+z**2/B**2)**(-1)*np.exp(-2*(x**2+y**2)/(C*(1+z**2/B**2)))
    grad_y = -4*y*A/(C*(1+z**2/B**2))*(1+z**2/B**2)**(-1)*np.exp(-2*(x**2+y**2)/(C*(1+z**2/B**2)))
    grad_z = 2*A*z/(B**2*(1+z**2/B**2)**2)*np.exp(-2*(x**2+y**2)/(C*(1+z**2/B**2)))*(-1+2*(x**2+y**2)/(C*(1+z**2/B**2)))
    return np.array([grad_x, grad_y, grad_z])

def grad_I_rotated(x, y, z, P, w0, wavelength, z0=0):
    """
    Gradient of the intensity of a Gaussian beam.

    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param P: Power of the laser beam
    :param w0: Beam waist
    :param wavelength: Wavelength of the light
    :param z0: Position of the waist
    """
    A = 2*P/(pi*w0**2)
    B = pi * w0**2 / wavelength
    C = w0**2
    x, y, z = y, z, x
    grad_x = 2*A*z/(B**2*(1+z**2/B**2)**2)*np.exp(-2*(x**2+y**2)/(C*(1+z**2/B**2)))*(-1+2*(x**2+y**2)/(C*(1+z**2/B**2)))
    grad_y = -4*x*A/(C*(1+z**2/B**2))*(1+z**2/B**2)**(-1)*np.exp(-2*(x**2+y**2)/(C*(1+z**2/B**2)))
    grad_z = -4*y*A/(C*(1+z**2/B**2))*(1+z**2/B**2)**(-1)*np.exp(-2*(x**2+y**2)/(C*(1+z**2/B**2)))
    return np.array([grad_x, grad_y, grad_z])

#################################################### EQUATIONS OF MOTION #########################################################

#--------------------------------------------------- Initialization --------------------------------------------------------------
def random_points_in_sphere(radii, n):
    """Generate n random points uniformly distributed inside an ellipsoid
       with semi-axes a, b, c along x,y,z axes respectively.
       Returns array shape (n,3).
    """
    u = np.random.normal(size=(n,3))
    norms = np.linalg.norm(u, axis=1)
    norms[norms==0] = 1.0
    dirs = u / norms[:, None]
    r = np.random.rand(n) ** (1/3)
    pts = dirs * r[:, None] * np.array([radii,radii,radii])#-np.array([0.021, 0, 0])
    return pts

def sample_mb_velocity(T, mass, size):
    """Sample 3D Maxwell-Boltzmann velocities (numpy vectorized).
       Returns array shape (size,3).
    """
    # Draw each velocity component from normal distribution with std = sqrt(kB T / m)
    kB = const.k
    sigma = np.sqrt(kB * T / mass)
    vx=np.random.normal(0.0, sigma, size=size)
    vy=np.random.normal(0.0, sigma, size=size)
    vz=np.random.normal(0.0, sigma, size=size)
    return np.stack((vx, vy, vz), axis=-1)

#--------------------------------------------------- evolution functions ------------------------------------------------------------

def f_lattice(t, vec, Re_alpha, P, w01, w02, wavelength, z01=0, z02=0):
    # Extract positions (x,y,z) and velocities (vx,vy,vz)
    pos = vec[0:3]
    v = vec[3:6]
    # Acceleration
    grad_U = grad_U_L_rotated(pos[0], pos[1], pos[2], Re_alpha, P, w01, w02, wavelength, z01, z02)
    gravity = g*e_x
    a = - grad_U / m_yb #- gravity
    # Derivative of the state vector: [v, a]
    vec_dev = np.hstack((v, a)) 
    return vec_dev

def f_lattice_and_tweezer(vec, t, Re_alpha, P, w01, w02, wavelength, z01=0, z02=0):
    t0 = 0.1 #s
    # Extract positions (x,y,z) and velocities (vx,vy,vz)
    pos = vec[0:3]
    v = vec[3:6]
    # Acceleration
    if t<t0:
        grad_U = grad_U_L(pos[0], pos[1], pos[2], Re_alpha, P, w01, w02, wavelength, z01, z02)
        gravity = g*e_y
        a = -grad_U / m_yb - gravity
    else:
        grad_U_lattice = grad_U_L(pos[0], pos[1], pos[2], Re_alpha, P, w01, w02, wavelength, z01, z02)
        grad_U_tweezer = grad_U_T(pos[0], pos[1], pos[2], Re_alpha, P, w01, w02, wavelength, z01, z02)
        grad_U = grad_U_lattice + grad_U_tweezer
        gravity = g*e_y 
        a = -grad_U / m_yb - gravity
    # Derivative of the state vector: [v, a]
    vec_dev = np.hstack((v, a)) 
    return vec_dev

#--------------------------------------------------- Evaluation of the physical processes ---------------------------------------------
def atom_loading_MOT_lattice(max_t, Re_alpha, P, w01, w02, wavelength, z01, z02, radii, N_atoms, T):
    np.random.seed(10)
    #Initial conditions
    init_pos = random_points_in_sphere(radii, N_atoms)                 # Positions of each atoms over time                       [shape: (N_atoms, 3)]
    init_vel = sample_mb_velocity(T, m_yb, N_atoms)                    # Velocities of each atom over time                       [shape: (N_atoms, 3)]
    init_vec = np.hstack((init_pos, init_vel))                         # Initial state vector: [x, y, z, vx, vy, vz]             [shape: (N_atoms, 6)]
    velocities=[]                                                   # Velocities of each atom over time                       [shape: (N_atoms, 3, times)]
    positions=[]                                                    # Positions of each atoms over time                       [shape: (N_atoms, 3, times)]
    idx_lost_atoms = []
    energies = []
    
    args = [Re_alpha, P, w01, w02, wavelength, z01, z02]

    for i in tqdm(range(N_atoms)):
        #E_init = energy(init_vec[i][0], init_vec[i][1], init_vec[i][2], init_vec[i][3], init_vec[i][4], init_vec[i][5], Re_alpha, P, w01, w02, wavelength, z01, z02)
        sol = solve_ivp(f_lattice, [0,max_t], init_vec[i], method='DOP853', t_eval=np.linspace(0, max_t, 1000), args=args)
        times = sol.t
        vec=sol.y
        x, y, z = vec[0], vec[1], vec[2]
        vx, vy, vz = vec[3], vec[4], vec[5]
        E = energy(x, y, z, vx, vy, vz, Re_alpha, P, w01, w02, wavelength, z01, z02)
        lost = E > 0
        if lost.any():
            idx_lost_atoms.append(i)
        positions.append(np.array([x, y, z]))
        velocities.append(np.array([vx, vy, vz]))
        energies.append(E)
    return times, np.array(velocities), np.array(positions), np.array(energies), np.array(idx_lost_atoms)


######################################################## OPTICAL LATTICE ########################################################

def energy(x, y, z, vx, vy, vz, Re_alpha, P, w01, w02, wavelength, z01 = 0, z02 = 0):
    v2 = vx**2 + vy**2 + vz**2                                                                                       #shape: (N_atoms, len(t_eval))
    kinetik = 0.5 * m_yb * v2                                                                                        #shape: (N_atoms, len(t_eval))
    potential_lattice = optical_dipole_trap_2_beams_rotated(x, y, z, 0, Re_alpha, P, w01, w02, wavelength, z01, z02) #shape: (N_atoms, len(t_eval))
    #gravity = m_yb * g * x                                                                                           #shape: (N_atoms, len(t_eval))
    energy = kinetik + potential_lattice #+ gravity                                                                   # Energy of each atom over time                                                                 [shape: (N_atoms, len(t_eval))]
    return energy

def energy_event(t, vec, Re_alpha, P, w01, w02, wavelength, E_init, z01 = 0, z02 = 0):
    E = energy(vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], Re_alpha, P, w01, w02, wavelength, z01, z02)
    if E-E_init > 1e-9:
        event = 0
    else:
        event = 1
    return event

def two_gaussian_beams(z, t, I1, I2, w1, w2, wavelength):
    """
    Interference pattern of the intensity of two counter-propagating Gaussian beams with different frequencies and same wavelength.
    
    :param z: Coordinate of the beam propagation direction
    :param t: Time
    :param I1: Intesity of beam 1
    :param I2: Intesity of beam 2
    :param w1: Frequency of beam 1
    :param w2: Frequency of beam 2
    :param k: Wavevector
    """
    k = 2 * pi / wavelength
    I=I1+I2+2*np.sqrt(I1*I2)*np.cos(2*k*z- (w1 - w2)*t)
    return I

def two_gaussian_beams_rotated(x, t, I1, I2, w1, w2, wavelength):
    """
    Interference pattern of the intensity of two counter-propagating Gaussian beams with different frequencies and same wavelength.
    
    :param x: Coordinate of the beam propagation direction
    :param t: Time
    :param I1: Intesity of beam 1
    :param I2: Intesity of beam 2
    :param w1: Frequency of beam 1
    :param w2: Frequency of beam 2
    :param k: Wavevector
    """
    k = 2 * pi / wavelength
    I=I1+I2+2*np.sqrt(I1*I2)*np.cos(2*k*x- (w1 - w2)*t)
    return I

def optical_dipole_trap_2_beams(x, y, z, t, Re_alpha, P, w01, w02, wavelength, w1, w2, z01=0, z02=0):
    """
    Potential of a moving optical lattice formed by two counter-propagating Gaussian beams with different frequencies.

    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param t: time
    :param Re_alpha: Real part of the atomic polarizability
    :param P: Power of the laser beam
    :param w0: Beam waist
    :param z01: Position of the waist of the first beam
    :param z02: Position of the waist of the second beam
    :param wavelength: Wavelength of the light
    :param w1: Frequency of the first beam
    :param w2: Frequency of the second beam
    """
    Delta_w = w1 - w2
    k = 2 * pi / wavelength
    I_1 = gaussian_beam(x, y, z, P, w01, wavelength, z01)
    I_2 = gaussian_beam(x, y, -z, P, w02, wavelength, z02)
    U_0 = 0.5 * Re_alpha / (eps_0 * c) * (I_1 + I_2 + 2*np.sqrt(I_1 * I_2))
    U_latt = 2*Re_alpha / (eps_0 * c) * np.sqrt(I_1 * I_2)
    U=-U_0+U_latt*(np.sin(k*z-Delta_w*t/2))**2
    return U

def optical_dipole_trap_2_beams_rotated(x, y, z, t, Re_alpha, P, w01, w02, wavelength, w1, w2, z01=0, z02=0):
    """
    Potential of a moving optical lattice formed by two counter-propagating Gaussian beams with different frequencies.

    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param t: time
    :param Re_alpha: Real part of the atomic polarizability
    :param P: Power of the laser beam
    :param w0: Beam waist
    :param z01: Position of the waist of the first beam
    :param z02: Position of the waist of the second beam
    :param wavelength: Wavelength of the light
    :param w1: Frequency of the first beam
    :param w2: Frequency of the second beam
    """
    Delta_w = w1 - w2
    k = 2 * pi / wavelength
    I_1 = gaussian_beam_rotated(x, y, z, P, w01, wavelength, z01)
    I_2 = gaussian_beam_rotated(-x, y, z, P, w02, wavelength, z02)
    U_0 = 0.5 * Re_alpha / (eps_0 * c) * (I_1 + I_2 + 2*np.sqrt(I_1 * I_2))
    U_latt = 2*Re_alpha / (eps_0 * c) * np.sqrt(I_1 * I_2)
    z = x
    U=-U_0+U_latt*(np.sin(k*z-Delta_w*t/2))**2
    return U

def lattice_depth_2_beams(x, y, z, Re_alpha, P, w01, w02, wavelength, z01, z02):
    """
    Calculate the lattice depth formed by two counter-propagating Gaussian beams.
    
    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param Re_alpha: Real part of the atomic polarizability
    :param P: Power of the laser beam
    :param w01: Beam waist of the first beam
    :param w02: Beam waist of the second beam
    :param wavelength: Wavelength of the light
    :param z01: Position of the waist of the first beam
    :param z02: Position of the waist of the second beam
    """
    I_1 = gaussian_beam(x, y, z, P, w01, wavelength, z01)
    I_2 = gaussian_beam(x, y, -z, P, w02, wavelength, z02)
    U_latt = 2*Re_alpha / (eps_0 * c) * np.sqrt(I_1 * I_2)
    return U_latt

def lattice_depth_2_beams_rotated(x, y, z, Re_alpha, P, w01, w02, wavelength, z01, z02):
    """
    Calculate the lattice depth formed by two counter-propagating Gaussian beams.
    
    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param Re_alpha: Real part of the atomic polarizability
    :param P: Power of the laser beam
    :param w01: Beam waist of the first beam
    :param w02: Beam waist of the second beam
    :param wavelength: Wavelength of the light
    :param z01: Position of the waist of the first beam
    :param z02: Position of the waist of the second beam
    """
    I_1 = gaussian_beam_rotated(x, y, z, P, w01, wavelength, z01)
    I_2 = gaussian_beam_rotated(-x, y, z, P, w02, wavelength, z02)
    U_latt = 2*Re_alpha / (eps_0 * c) * np.sqrt(I_1 * I_2)
    return U_latt

def U_0_latt_2_beams_rotated(x, y, z, Re_alpha, P, w01, w02, wavelength, z01, z02):
    """
    Calculate the lattice depth formed by two counter-propagating Gaussian beams.
    
    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param Re_alpha: Real part of the atomic polarizability
    :param P: Power of the laser beam
    :param w01: Beam waist of the first beam
    :param w02: Beam waist of the second beam
    :param wavelength: Wavelength of the light
    :param z01: Position of the waist of the first beam
    :param z02: Position of the waist of the second beam
    """
    I_1 = gaussian_beam_rotated(x, y, z, P, w01, wavelength, z01)
    I_2 = gaussian_beam_rotated(-x, y, z, P, w02, wavelength, z02)
    U_0 = 0.5 * Re_alpha / (eps_0 * c) * (I_1 + I_2 + 2*np.sqrt(I_1 * I_2))
    return U_0

def U0_2_beams(x, y, z, Re_alpha, P, w01, w02, wavelength, z01, z02):
    """
    Calculate the non-interference part of the potential formed by two counter-propagating Gaussian beams.
    
    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param Re_alpha: Real part of the atomic polarizability
    :param P: Power of the laser beam
    :param w01: Beam waist of the first beam
    :param w02: Beam waist of the second beam
    :param wavelength: Wavelength of the light
    :param z01: Position of the waist of the first beam
    :param z02: Position of the waist of the second beam
    """
    I_1 = gaussian_beam(x, y, z, P, w01, wavelength, z01)
    I_2 = gaussian_beam(x, y, -z, P, w02, wavelength, z02)
    U_0 = -0.5 * Re_alpha / (eps_0 * c) * (I_1 + I_2 + 2*np.sqrt(I_1 * I_2))
    return U_0

def lattice_acceleration_z(x, y, z, t, Re_alpha, P, w01, w02, m, wavelength, w1, w2, z01=0, z02=0):
    """
    Lattice acceleration of a moving optical lattice formed by two counter-propagating Gaussian beams with different frequencies.

    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param t: time
    :param Re_alpha: Real part of the atomic polarizability
    :param P: Power of the laser beam
    :param w0: Beam waist
    :param z01: Position of the waist of the first beam
    :param z02: Position of the waist of the second beam
    :param wavelength: Wavelength of the light
    :param w1: Frequency of the first beam
    :param w2: Frequency of the second beam
    """
    Delta_w = w1 - w2
    k = 2 * pi / wavelength
    I_1 = gaussian_beam(x, y, z, P, w01, wavelength, z01)
    I_2 = gaussian_beam(x, y, -z, P, w02, wavelength, z02)
    U_latt = 2*Re_alpha / (eps_0 * c) * np.sqrt(I_1 * I_2)
    a = -U_latt*k*np.sin(2*k*z-Delta_w*t)/m
    adiabaticity = a*m/(U_latt*k)<1
    return a, adiabaticity

def U_L(x, y, z, Re_alpha, P, w01, w02, wavelength, z01=0, z02=0):
    """
    Lattice potential of a static lattice formed by two counter-propagating Gaussian beams with equal frequencies.

    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param Re_alpha: Real part of the atomic polarizability
    :param P: Power of the laser beam
    :param z01: Position of the waist of the first beam
    :param z02: Position of the waist of the second beam
    :param wavelength: Wavelength of the light
    :param w01: Beam waist of the first beam
    :param w02: Beam waist of the second beam
    """
    k = 2 * pi / wavelength
    I_1 = gaussian_beam(x, y, z, P, w01, wavelength, z01)
    I_2 = gaussian_beam(x, y, -z, P, w02, wavelength, z02)
    U_latt = 2*Re_alpha / (eps_0 * c) * np.sqrt(I_1 * I_2)
    U_0 = Re_alpha / (2*eps_0 * c) * (I_1 + I_2 + 2*np.sqrt(I_1 * I_2))
    U = - U_0 + U_latt*(np.sin(k*z))**2
    return U

def grad_U_L(x, y, z, Re_alpha, P, w01, w02, wavelength, z01=0, z02=0):
    """
    Gradient of the lattice potential of a static lattice formed by two counter-propagating Gaussian beams with equal frequencies.

    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param Re_alpha: Real part of the atomic polarizability
    :param P: Power of the laser beam
    :param w01: Beam waist of the first beam
    :param w02: Beam waist of the second beam
    :param wavelength: Wavelength of the light
    :param z01: Position of the waist of the first beam
    :param z02: Position of the waist of the second beam
    """
    k = 2 * pi / wavelength
    I_1 = gaussian_beam(x, y, z, P, w01, wavelength, z01)
    I_2 = gaussian_beam(x, y, -z, P, w02, wavelength, z02)
    # gradient of U_0
    const = Re_alpha/(2*eps_0 * c)
    I_1_grad = grad_I(x, y, z, P, w01, wavelength, z01)
    I_2_grad = grad_I(x, y, -z, P, w02, wavelength, z02)
    grad_sqrt_I1I2 = np.array([
        (I_2*I_1_grad[0]+I_1*I_2_grad[0])/2*np.sqrt(I_1*I_2),
        (I_2*I_1_grad[1]+I_1*I_2_grad[1])/2*np.sqrt(I_1*I_2),
        (I_2*I_1_grad[2]+I_1*I_2_grad[2])/2*np.sqrt(I_1*I_2)
    ])
    grad_U_0 = const*(I_1_grad + I_2_grad + 2*grad_sqrt_I1I2)
    #gradient of U_latt*sin^2(k*z)
    U_latt = 2*Re_alpha / (eps_0 * c) * np.sqrt(I_1 * I_2)
    v1 = U_latt*np.array([k*np.sin(2*k*z), 0, 0])
    v2 = (np.sin(k*z))**2*2*Re_alpha / (eps_0 * c) * grad_sqrt_I1I2
    grad_U_Latt_sin = v1 + v2
    grad_U = -grad_U_0 + grad_U_Latt_sin
    return grad_U

def grad_U_L_rotated(x, y, z, Re_alpha, P, w01, w02, wavelength, z01=0, z02=0):
    """
    Gradient of the lattice potential of a static lattice formed by two counter-propagating Gaussian beams with equal frequencies.

    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param Re_alpha: Real part of the atomic polarizability
    :param P: Power of the laser beam
    :param w01: Beam waist of the first beam
    :param w02: Beam waist of the second beam
    :param wavelength: Wavelength of the light
    :param z01: Position of the waist of the first beam
    :param z02: Position of the waist of the second beam
    """
    k = 2 * pi / wavelength
    const = Re_alpha/(2*eps_0 * c)
    I_1 = gaussian_beam_rotated(x, y, z, P, w01, wavelength, z01)
    I_2 = gaussian_beam_rotated(-x, y, z, P, w02, wavelength, z02)
    I_1_grad = grad_I_rotated(x, y, z, P, w01, wavelength, z01)
    I_2_grad = grad_I_rotated(-x, y, z, P, w02, wavelength, z02)
    grad_sqrt_I1I2 = np.array([
        (I_2*I_1_grad[0]+I_1*I_2_grad[0])/2*np.sqrt(I_1*I_2),
        (I_2*I_1_grad[1]+I_1*I_2_grad[1])/2*np.sqrt(I_1*I_2),
        (I_2*I_1_grad[2]+I_1*I_2_grad[2])/2*np.sqrt(I_1*I_2)
    ])
    grad_U_0 = const*(I_1_grad + I_2_grad + 2*grad_sqrt_I1I2)
    #gradient of U_latt*sin^2(k*z)
    U_latt = 2*Re_alpha / (eps_0 * c) * np.sqrt(I_1 * I_2)
    z = x
    v1 = U_latt*np.array([k*np.sin(2*k*z), 0, 0])
    v2 = (np.sin(k*z))**2*2*Re_alpha / (eps_0 * c) * grad_sqrt_I1I2
    grad_U_Latt_sin = v1 + v2
    grad_U = -grad_U_0 + grad_U_Latt_sin
    return grad_U

def lattice_velocity(w1, w2, wavelength):
    """
    Lattice velocity

    :param w1: Frequency of the first beam
    :param w2: Frequency of the second beam
    :param wavelength: Wavelength of the light
    """
    Delta_w = w1 - w2
    k = 2 * pi / wavelength
    v = Delta_w / (2 * k)
    return v

########################################################## OPTICAL TWEEZERS ##########################################################

def optical_dipole_trap_1_beam(x, y, z, Re_alpha, P, w0, wavelength, z0=0):
    """
    Potential of a moving optical lattice formed by two counter-propagating Gaussian beams with different frequencies.

    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param t: time
    :param Re_alpha: Real part of the atomic polarizability
    :param P: Power of the laser beam
    :param w0: Beam waist
    :param z01: Position of the waist of the first beam
    :param z02: Position of the waist of the second beam
    :param wavelength: Wavelength of the light
    :param w1: Frequency of the first beam
    :param w2: Frequency of the second beam
    """
    I = gaussian_beam(x, y, z, P, w0, wavelength, z0)
    U = -0.5 * Re_alpha / (eps_0 * c) * I
    return U

def potential_depth_1_beam(Re_alpha, P, w0):
    """
    Calculate the lattice depth formed by two counter-propagating Gaussian beams.
    
    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param Re_alpha: Real part of the atomic polarizability
    :param P: Power of the laser beam
    :param w0: Beam waist of the first beam
    :param wavelength: Wavelength of the light
    :param z0: Position of the waist of the beam
    """
    U_0 = P*Re_alpha / (pi * w0**2 * eps_0 * c)
    return U_0

def trapping_frequencies(P, Re_alpha, w0, wavelength, m):
    """
    Calculate the trapping frequencies of an optical dipole trap formed by a single Gaussian beam.

    :param P: Power of the laser beam
    :param Re_alpha: Real part of the atomic polarizability
    :param w0: Beam waist
    :param wavelength: Wavelength of the light
    :param m: Mass of the atom
    """
    wr = np.sqrt(4*P*Re_alpha/(eps_0*c*pi*w0**4*m))
    wz = np.sqrt(2*P*Re_alpha*wavelength**2/(eps_0*c*pi**3*w0**6*m))
    return wr, wz

def grad_U_T(x, y, z, Re_alpha, P, w0, wavelength, z0 = 0):
    """
    Gradient of the potential of a single Gaussian beam.

    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param Re_alpha: Real part of the atomic polarizability
    :param P: Power of the laser beam
    :param w0: Beam waist of the single beam
    :param wavelength: Wavelength of the light
    :param z0: Position of the waist of the beam
    """
    grad_I_ = grad_I(x, y, z, P, w0, wavelength, z0)
    grad_U = -0.5 * Re_alpha / (eps_0 * c) * grad_I_
    return grad_U

##################################################################### OPTICS ########################################################

def f_theta(theta, wavelength=532e-9, A=25e-3, pixel_size=8e-6):
    """
    Calculate the focal length of the Moiré lens.
    
    :param theta: angle between the two phase plates [rad]
    :param wavelength: wavelength of the light [m]
    :param A: Aperture of the lens [m]
    :param pixel_size: pixel size of the phase plates [m]
    """
    f = np.where(theta==0, np.inf, A*pixel_size*(1+4*pi**2)**0.5/(4*theta*wavelength))
    return f

def lens_system(f1, f2, d=None, wavelength=532e-9, w0=1800e-6, M2=1):
    """
    Calculate the beam shape after two lenses, assuming that the input beam is collimated.
    
    :param f1: focal length of the first lens [m]
    :param f2: focal length of the second lens [m]
    :param d: distance between the two lenses [m]
    """
    # Rayleigh range of collimated beam
    zR = np.pi * w0**2 / (M2 * wavelength)

    # Complex beam parameter at lens 1
    q = 1j * zR 

    # ---- Lens 1 ----
    q = q / (1 - q / f1)

    # ---- Free space between lenses ----
    if d is None:
        d = f2
    q = q + d

    # ---- Lens 2 ----
    q = q / (1 - q / f2)

    # ---- Extract beam parameters ----
    z0_2 = -np.real(q)
    zR_2 = np.imag(q)
    w0_2 = np.sqrt(M2 * wavelength * zR_2 / pi)
    return w0_2, z0_2



if __name__ == "__main__":

    # Parameters 3D MOT
    T_D = 4.4e-6 #K Doppler temperature for 1S0 → 3P1 transition of Yb
    T = 5*T_D # ????
    radii = 50e-6 #m 3D MOT radius
    # Parameters lattice
    #Parameters
    Re_alpha_lat = au_to_SI(270)                                        #A^2s^4/kg
    P = 1                                                               #W
    w01, w02 = 50e-6, 50e-6                                             #m
    wavelength = 532e-9                                                 #m
    w1, w2 = 2*np.pi*nm_to_Hz(wavelength), 2*np.pi*nm_to_Hz(wavelength) #Hz

    z = np.linspace(-600e-6, 600e-6, 500)                                #m
    x = np.linspace(-7*wavelength, 7*wavelength, 500)                  #m

    N_atoms = 1
    t, vels_overtime, pos_overtime = atom_loading_MOT_lattice(2, Re_alpha_lat, P, w01, w02, wavelength, 0, 0, radii, N_atoms, T)
    print("done")