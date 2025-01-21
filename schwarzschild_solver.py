import scipy
import numpy as np

def B_nu(T, nu):
    """
    Planck function
    
    In:
    T [K]: temperature
    nu [cm-1]; multiply all nu's by 100 to convert to m-1 in formula, then multiply
        # B by 100 for units of W/m^2/sr/cm-1 rather than m-1
        
    Returns:
    Planck function in units of W/m^2/sr/cm-1
    
    """
    k_B = scipy.constants.k # Boltzmann constant
    h = scipy.constants.h # Planck constant
    c = scipy.constants.speed_of_light # speed of light in a vacuum
    return ((2*h*(c**2)*((100*nu)**3))/(np.exp((h*c*(100*nu))/(k_B*T)) - 1))*100

## assuming the following atmospheric orientation: ground at index 0, TOA at index -1

### RTE-RRTMG implementation
def layer_source(T, nu):
    """
    Planck function in layer

    In: 
    T [K]: temperature
    nu [cm-1]: wavenumber
    """
    
    lay_source = np.empty(len(T))
    for i in range(len(T)):
        lay_source[i] = B_nu(T[i], nu)
        
    return lay_source

def level_source(T, nu):
    """
    Planck function at layer boundaries/half levels
    assume mean temperature between levels

    In: 
    T [K]: temperature
    nu [cm-1]: wavenumber
    """
    
    level_temps =  np.empty(len(T) + 1)
    level_temps[0] = T[0]
    level_temps[-1] = T[-1]
    
    internal_temps = 0.5*(T[1:] + T[:-1])
    level_temps[1:-1] = internal_temps
    
    level_source = np.empty(len(T) + 1)
    for i in range(len(internal_temps)):
        level_source[i] = B_nu(level_temps[i], nu)
    
    level_source_up = level_source[1:]
    level_source_dn = level_source[:-1]
    
    return level_source_up, level_source_dn
        
def lw_source_noscat(tau, T, nu):
    """
    Planck function into and out of layer

    In: 
    tau (unitless): layer optical depth
    T [K]: temperature
    nu [cm-1]: wavenumber
    """
    
    # calculate linear-in-tau planck functions
    lay_source = layer_source(T, nu)
    lev_source_up, lev_source_dn = level_source(T, nu)
    
    # tau threshhold is quartic root of machine epsilon
    tau_thresh = np.sqrt(np.sqrt(np.finfo(float).eps))
    
    # transmissivity
    Trans = np.exp(-tau)
    
    # empty arrays of length tau
    source_dn = np.empty(len(tau))
    source_up = np.empty(len(tau))
    
    for i in range(len(tau)):
        if tau[i] > tau_thresh:
            fact = ((1-Trans[i])/tau[i]) - Trans[i]
            
        else:
            fact = tau[i] * (0.5 + tau[i] * (-(1/3) + tau[i] * (1/8)))
            
        source_dn[i] = (1 - Trans[i]) * lev_source_dn[i] + 2 * fact * (lay_source[i] - lev_source_dn[i])
        source_up[i] = (1 - Trans[i]) * lev_source_up[i] + 2 * fact * (lay_source[i] - lev_source_up[i])
        
    return source_up, source_dn
    
    
def lw_solver_noscat_oneangle(tau, T, nu, sfc_emis):
    """
    Solve the twostream longwave radiative transfer equation, assuming
    no scattering 

    In: 
    tau (unitless): layer optical depth
    T [K]: temperature
    nu [cm-1]: wavenumber
    sfc_emis: surface emissivity
    """

    source_up, source_dn = lw_source_noscat(tau, T, nu)
    
    # transmissivity
    Trans = np.exp(-tau)
    
    radn_dn = np.zeros(len(tau) + 1)
    radn_up = np.zeros(len(tau) + 1)    
    
    # lw_transport_noscat_dn
    for i in range(0, len(tau) - 1)[::-1]:
        radn_dn[i] = Trans[i]*radn_dn[i + 1] + source_up[i]
        
    sfc_albedo = 1 - sfc_emis
    radn_up[0] = radn_dn[0]*sfc_albedo + sfc_emis * B_nu(T[0], nu)
    
    for i in range(1, len(tau) + 1):
        radn_up[i] = Trans[i - 1]*radn_up[i - 1] + source_dn[i - 1]

    flux_up = np.pi*radn_up
    flux_dn = np.pi*radn_dn
    return flux_up, flux_dn

### a different implementation
def Transmittance(tau_1, tau_2):
    """
    Transmittance between two optical depth levels
    """
    return np.exp(-np.abs(tau_1 - tau_2))


def flux_up(T, p, tau, nu):
    """
    Solve Schwartzschild's equations
    Inputs from ground to TOA
    
    In: 
    T [K]: temperature profile 
    p [Pa or hPa]: pressure profile
    tau: cumulative optical depth from TOA
    nu [cm-1]: wavenumbers
    
    Returns:
    flux up (W/m^2) across wavembers
    
    """
    e = 1
    
    if len(p) == 1:
        return np.pi*e*(Transmittance(tau[-1], tau[0]))*B_nu(T[0], nu)
    
    transmittance_matrix = np.zeros((len(p), len(nu)))
    for i in range(len(p)):
        transmittance_matrix[i, :] = Transmittance(tau[-1], tau[i]) 

    Planck_matrix = np.zeros((len(p), len(nu)))
    for i in range(len(p)):
        Planck_matrix[i, :] = B_nu(T[i], nu)

    dtransdp = np.gradient(transmittance_matrix, axis = 0).T/np.gradient(p)
    
    return np.pi*e*(Transmittance(tau[-1], tau[0]))*B_nu(T[0], nu) + np.pi*np.trapz(np.multiply(dtransdp.T, Planck_matrix), p, axis = 0)

def flux_up_profile(T, p, tau, nu):
    """
    Calculate full profile of upwards fluxes
    """
    fluxes_up = np.zeros((len(p), len(nu)))
    for idx in range(len(p)):
        idx = idx + 1
        fluxes_up[idx-1, :] = flux_up(T[:idx], p[:idx], tau[:idx], nu)
    return fluxes_up

def flux_down(T, p, tau, nu):
    """
    Solve Schwartzschild's equations
    Inputs from TOA to ground
    
    In: 
    T [K]: temperature profile 
    p [Pa or hPa]: pressure profile
    tau: cumulative optical depth from TOA
    nu [cm-1]: wavenumbers
    
    Returns:
    flux down (W/m^2) across wavembers
    
    """

    if len(p) == 1:
        return np.zeros(len(nu))
        
    transmittance_matrix = np.zeros((len(p), len(nu)))
    for i in range(len(p)):
        transmittance_matrix[i, :] = Transmittance(tau[-1], tau[i]) 

    Planck_matrix = np.zeros((len(p), len(nu)))
    for i in range(len(p)):
        Planck_matrix[i, :] = B_nu(T[i], nu)

    dtransdp = np.gradient(transmittance_matrix, axis = 0).T/np.gradient(p)
    return np.pi*np.trapz(np.multiply(dtransdp.T, Planck_matrix), p, axis = 0)

def flux_down_profile(T, p, tau, nu):
    """
    Calculate flux down profile
    """
    
    fluxes_down = np.zeros((len(p), len(nu)))
    for idx in range(len(p)):
        idx = idx + 1
        fluxes_down[idx-1, :] = flux_down(T[::-1][:idx], p[::-1][:idx], tau[::-1][:idx], nu)
    return fluxes_down[::-1]

