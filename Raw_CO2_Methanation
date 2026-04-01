# Reactor 2D for CO2 Methanation using PH3 model (pseudohomogeneous with radial dispersion)

# Created by João PB "PlumBum" Nunes 
# Last updated on 1st April

# Credit to Kai et al 1987; Guyer et al 2009; Koschany et al 2016; Champom et al 2019; Farsi et al 2020; 

#This code is the full code, it is its RAW format and not very well divided.

#---------------------------------------------------------------------------------------------------------#

#Imports
from fipy import *
import numpy as np
import matplotlib.pyplot as plt
from fipy.tools import numerix
import pandas as pd
from fipy.terms import ImplicitSourceTerm
import hvplot.pandas
import panel as pn
import pickle


#Random equations for non-NaN results
def safe(x):
    return numerix.maximum(x, 1e-8)

def safe_exp(x):
    return numerix.exp(numerix.clip(x, -100, 100))

def safe_div(numer, denom, eps=1e-12):
    return numer / numerix.maximum(denom, eps)

def radial_avg(Cvar, Nr, Nz):
    """Return radial average along reactor length (axis=0 is radial)."""
    return np.mean(Cvar.reshape(Nr, Nz), axis=0)

#Species and their values
species = ["H2", "CO2", "CH4", "H2O", "CO"]

nu = {
    "H2":  [-4, -1, -3],
    "CO2": [-1, -1,  0],
    "CH4": [ 1,  0,  1],
    "H2O": [ 2,  1,  1],
    "CO":  [ 0,  1, -1]
}

#Atomic composition per species (atoms per molecule)
atoms = {
    "H2":  {"H": 2, "C": 0, "O": 0},
    "CO2": {"H": 0, "C": 1, "O": 2},
    "CO":  {"H": 0, "C": 1, "O": 1},
    "CH4": {"H": 4, "C": 1, "O": 0},
    "H2O": {"H": 2, "C": 0, "O": 1},
}

#Molar masses [kg/mol]
MW = {
    "H2": 2.016e-3,
    "CO2": 44.01e-3,
    "CO": 28.01e-3,
    "CH4": 16.04e-3,
    "H2O": 18.015e-3
}

#Kinectic Models - Champom 2019; Koschany 2016; Kai 1987; Farsi 2020
def champom_kinetics(p, T, k):
    T = numerix.clip(T, 300, 2000)
    ln10 = numerix.log(10.0)

    kco2  = k["k0co2"]  * safe_exp(-k["eaco2"] / (R*T))
    krwgs = k["k0rwgs"] * safe_exp(-k["earwgs"] / (R*T))
    kco   = k["k0co"]   * safe_exp(-k["eaco"] / (R*T))

    Keq = 137* T**(-3.998) * safe_exp(158700/(R*T))
    Kwgs = 1.0 / numerix.exp((-2.4198 + 3.855e-4 * T + 2180.9 / T) * ln10)
    Kmeth = numerix.exp((4.1002e-5 * T**2 - 0.08025 * T + 39.6039) * ln10)
    
    Kco  = k["kads0co"]  * safe_exp(-k["deltahco"]  / (R*T))
    Kh2  = k["kads0h2"]  * safe_exp(-k["deltahh2"]  / (R*T))
    Kh2o = k["kads0h2o"] * safe_exp(-k["deltahh2o"] / (R*T))
    Kco2 = k["kads0co2"] * safe_exp(-k["deltahco2"] / (R*T))

    denom = 1 + Kh2 * p["H2"] + Kco2 * p["CO2"] + Kh2o * p["H2O"] + Kco * p["CO"]

    driving1 = safe_div(1 - (p["CH4"] * p["H2O"]**2) / (p["CO2"] * p["H2"]**4 * Keq), 1)
    driving1 = numerix.clip(driving1, 0, 1)
    driving2 = safe_div(1 - (p["CO"] * p["H2O"]) / (p["CO2"] * p["H2"] * Kwgs), 1)
    driving2 = numerix.clip(driving2, 0, 1)
    driving3 = safe_div(1 - (p["CH4"] * p["H2O"]) / (p["CO"] * p["H2"]**3 * Kmeth), 1)
    driving3 = numerix.clip(driving3, 0, 1)

    r1 = kco2 * Kh2 * Kco2 * p["H2"] * p["CO2"] * driving1 / numerix.maximum(denom**2, 1e-12)
    r2 = krwgs * Kco2 * p["CO2"] * driving2 / numerix.maximum(denom, 1e-12)
    r3   = kco * Kco2 * Kh2 * p["CO"] * p["H2"] * driving3 / numerix.maximum(denom**2, 1e-12)
    
    return [r1, r2, r3]

def koz_kinetics(p, T, k):
    T = numerix.clip(T, 300, 2000)

    kf = k["k0"] * safe_exp((k["Ea"] / R) * ( (1 / 555) - (1 / T)))
    
    Keq = 137 * T**(-3.998) * safe_exp(158700 / (R * T))
    
    K_OH  = safe_exp(k["Aoh"]  + (k["Boh"] / T))
    K_H2  = safe_exp(k["Ah2"]  + (k["Bh2"] / T))
    K_mix = safe_exp(k["Amix"] + (k["Bmix"] / T))

    driving = safe_div(1 - (p["CH4"] * p["H2O"]**2)/(p["CO2"] * p["H2"]**4 * Keq),1)
    driving = numerix.clip(driving,0,1)

    num = kf * p["H2"]**0.5 * p["CO2"]**0.5 * driving
    den = (1 + K_OH*(p["H2O"] / p["H2"]**0.5) + K_H2*p["H2"]**0.5 + K_mix*p["CO2"]**0.5)**2
    r = safe_div(num, den)
    return [r, 0.0, 0.0]

def kai_kinetics(p, T, k):
    T = numerix.clip(T, 300, 2000)

    kf = k["k0_kai"] * safe_exp(-k["Ea_kai"] / (R * T))
 
    Kh2kai  = safe_exp(k["kads_h2_kai"]  + (k["deltaH_h2_kai"] / (R * T)))
    Kco2kai  = safe_exp(k["kads_co2_kai"]  + (k["deltaH_co2_kai"] / (R * T)))
    Kh2okai = safe_exp(k["kads_h2o_kai"] + (k["deltaH_h2o_kai"] / (R * T)))

    denom = 1 + Kh2kai * (p["H2"]**(1/2)) + Kco2kai * (p["CO2"]**(1/2)) + Kh2okai * p["H2O"]

    r = kf * (p["H2"]**(1/2)) * (p["CO2"]**(1/3)) / numerix.maximum(denom**2, 1e-12)

    return [r, 0.0, 0.0]

def farsi_kinetics(p, T, k):

    # Temperature limits
    T = numerix.clip(T, 300.0, 2000.0)

    # === Arrhenius constants ===
    k1 = k["k0_1"] * numerix.exp((k["ea_1"] / R) * ( (1 / 555) - (1 / T)))
    k2 = k["k0_2"] * numerix.exp((k["ea_2"] / R) * ( (1 / 555) - (1 / T)))

    # === Adsorption constant (H2O) ===
    Kh2o = k["K_H2O_Farsi"] * numerix.exp(
        (k["ea_H2O_Farsi"] / R) * ((1.0 / 555.0) - (1.0 / T))
    )

    # === Equilibrium constants (FiPy-safe) ===
    ln10 = numerix.log(10.0)

    Kwgs = 1.0 / numerix.exp(
        (-2.4198 + 3.855e-4 * T + 2180.9 / T) * ln10
    )

    Kmeth = numerix.exp(
        (4.1002e-5 * T**2 - 0.08025 * T + 39.6039) * ln10
    )

    # === Total pressure (local) ===
    Ptot = sum(p[s] for s in p)
    
    denom = 1.0 + Kh2o * p["H2O"]
    denom2 = numerix.maximum(denom**2, 1e-12)

    drivingWGS = 1.0 - (
        (p["CO"] * p["H2O"]) /
        (p["CO2"] * p["H2"]**4 * Kwgs + 1e-20)
    )
    drivingWGS = numerix.clip(drivingWGS, 0.0, 1.0)

    drivingMeth = 1.0 - (((p["CH4"] * p["H2O"]) * Ptot**2) / (p["CO"] * p["H2"]**3 * Kmeth + 1e-20))
    drivingMeth = numerix.clip(drivingMeth, 0.0, 1.0)

    r2 = k1 * (p["CO2"]**0.5) * (p["H2"]**0.5) * drivingWGS / denom2
    r3 = k2 * p["CO"] * (p["H2"]**0.5) * drivingMeth / denom2

    return [0.0, r2, r3]

#Champom kinetics Parameters
kin_champom = dict(
    k0co2=1900000, eaco2=110000,
    k0rwgs=29666.66667, earwgs=97100,
    k0co=3716666.667, eaco=97300,
    kads0co=0.00239, deltahco=40600,
    kads0h2=0.000052, deltahh2=52000,
    kads0h2o=0.609, deltahh2o=14500,
    kads0co2=1.07, deltahco2=9720
)

#Koz kinetics Parameters
kin_koz = dict(
    k0=3.46e-4,
    Ea=77.5e3,
    Aoh=4.16, Boh=-2694.25,
    Ah2=-2.16, Bh2=745.73,
    Amix=-2.3, Bmix=1202.79
)

#Kai kinetics Parameters
kin_kai = dict(
    k0_kai=(9.32e3*(10**(2*(-5/6)))*10**(-3)),
    Ea_kai=72.5e3,
    kads_h2_kai=3.77e-10, deltaH_h2_kai=-90.2e3,
    kads_co2_kai=1.43e-3, deltaH_co2_kai=-29.5e3,
    kads_h2o_kai=2.75e-6, deltaH_h2o_kai=-64.3e3
)

#Farsi kinetics Parameters
kin_farsi = dict(
    k0_1=0.1425e-3, ea_1=166.55e3,
    k0_2=11.5451e-3, ea_2=60.98e3,
    K_H2O_Farsi=0.6782, ea_H2O_Farsi=11.44e3,
)
         
#Constants / Parameters
R = 8.314
rho_b = 2450
eps = 0.4
Dr = 1e-5
lambda_e = 0.8
dH = [-165e3, 41e3, -206e3]
rhoCp = 1.5e6            
dp  = 4.0e-3           
A   = 0.01              
mu_gas = 2.5e-5  #Viscosity [Pa·s]

#Molar masses [kg/mol]
MW = {
    "H2": 2.016e-3,
    "CO2": 44.01e-3,
    "CO": 28.01e-3,
    "CH4": 16.04e-3,
    "H2O": 18.015e-3
}

def update_pressure_ergun(P, C, T, u, dz):
    # Total concentration
    Ctot = sum(C[sp] for sp in C)

    # Mole fractions
    y = {sp: C[sp] / Ctot for sp in C}

    # Mean molar mass
    Mbar = sum(y[sp] * MW[sp] for sp in C)

    # Gas density
    rho = P * Mbar / (R * T)

    # Ergun pressure gradient
    dPdz = -(
        150.0 * (1 - eps)**2 / eps**3 * mu_gas * u / dp**2
        + 1.75 * (1 - eps) / eps**3 * rho * u**2 / dp
    )

    # Explicit axial march (in-place!)
    P.setValue(P.old + dPdz * dz)

    # Pressure floor (IN-PLACE)
    P.setValue(numerix.maximum(P.value, 1e3))


def atomic_total(C, atoms, element):
    total = 0.0
    for sp in C:
        n = atoms[sp][element]
        if n > 0:
            total += n * C[sp]
    return total

def enforce_atomic_conservation(C, atoms, eps=1e-20):

    # Reference totals (before correction)
    H0 = atomic_total(C, atoms, "H")
    C0 = atomic_total(C, atoms, "C")
    O0 = atomic_total(C, atoms, "O")

    # Current totals
    H1 = atomic_total(C, atoms, "H")
    C1 = atomic_total(C, atoms, "C")
    O1 = atomic_total(C, atoms, "O")

    # Correction factors (cellwise)
    fH = H0 / (H1 + eps)
    fC = C0 / (C1 + eps)
    fO = O0 / (O1 + eps)

    # Apply corrections
    for sp in C:
        aH = atoms[sp]["H"]
        aC = atoms[sp]["C"]
        aO = atoms[sp]["O"]

        correction = (
            (fH ** (aH / 2.0)) *
            (fC ** aC) *
            (fO ** aO)
        )

        C[sp].value[:] *= correction.value

        # Positivity guard
        C[sp].value[:] = numerix.maximum(C[sp].value, eps)


def enforce_constant_pressure(C, T, Pbar, species):
    Ctot = sum(C[sp] for sp in species)
    Ctot_ref = Pbar / (R * T)
    factor = Ctot_ref / (Ctot + 1e-30)

    for sp in species:
        C[sp].setValue(C[sp] * factor)


def update_velocity_mass(P, C, u, C_inlet):

    # Total mass concentration (rho = sum_i C_i * M_i)
    rho = sum(C[sp].value * MW[sp] for sp in species)      # kg/m^3 at current z
    rho_in = sum(C_inlet[sp] * MW[sp] for sp in species)  # kg/m^3 at inlet

    # Update velocity to conserve mass
    u.value[:] = u.value[0] * (rho_in / rho)

    
#Reactor
def run_reactor(H2_CO2, Twall, u0, Pbar, kinetics, kin_params, kin_name, Tin):

    #Geometry of reactor and mesh
    Rr, Lz = 0.1, 1.0
    Nr, Nz = 100, 100
    mesh = CylindricalGrid2D(dr=Rr/Nr, dz=Lz/Nz, nr=Nr, nz=Nz)
    dz = Lz / Nz
    dt = dz / u0

    #Conditions of inlet
    yCO2 = 1 / (1 + H2_CO2)
    yH2  = H2_CO2 * yCO2
    
    #Concentration at the beginning
    Cin = {
        "CO2": Pbar * yCO2 / (R * Tin),
        "H2":  Pbar * yH2  / (R * Tin),
        "CH4": 1e-8,
        "CO":  1e-8,
        "H2O": 1e-8
    }

    
    # Conc, Temp and Pressure as in the mesh
    C = {sp: CellVariable(mesh=mesh, value=Cin[sp], hasOld=True)
         for sp in species}
    
    T = CellVariable(mesh=mesh, value=Tin, hasOld=True)

    P = CellVariable(mesh=mesh, value=Pbar, hasOld=True)

    u = CellVariable(mesh=mesh, value=u0, hasOld=True)
    u_face = FaceVariable(mesh=mesh, rank=1)

    # axial velocity only → z-direction
    u_face[0] = 0.0
    u_face[1] = u.arithmeticFaceValue

    
    #BCs of Conc and Temp + Axial parameters
    for sp in species:
        C[sp].constrain(Cin[sp], mesh.facesLeft)      
        C[sp].faceGrad.constrain(0, mesh.facesRight) 
        C[sp].faceGrad.constrain(0, mesh.facesTop)   
        C[sp].faceGrad.constrain(0, mesh.facesBottom)

    T.constrain(Tin, mesh.facesLeft)
    T.constrain(Twall, mesh.facesRight)
    T.faceGrad.constrain(0, mesh.facesTop)
    T.faceGrad.constrain(0, mesh.facesBottom)

    P.constrain(Pbar, mesh.facesLeft)
    P.constrain(Pbar, mesh.facesRight)
    
    u.constrain(u0, mesh.facesLeft)
    u.constrain(u0, mesh.facesRight)

    
    Qaxial = []
    Tmax_axial = []

    #Axial starts
    for k in range(Nz):
        for sp in species:
            C[sp].updateOld()
        T.updateOld()
        P.updateOld()
        u.updateOld()
        
        #Recompute kinetics at this axial plane
        p = {s: safe(C[s] * R * T) for s in species}
        Ptot = sum(p[s] for s in species)
        rates = kinetics(p, T, kin_params)
        
      #MB of Species
        for sp in species:
            Rsp = sum(nu[sp][i] * rates[i] for i in range(3))

              #Fail-save for pressure at moment 0
            if k == 0:
                print(
                    f"[{kin_name}] "
                    f"p_H2 = {float(p['H2'].value.mean()):.4g} | "
                    f"p_CO2 = {float(p['CO2'].value.mean()):.4g} | "
                    f"p_H2O = {float(p['H2O'].value.mean()):.4g} | "
                    f"p_CO = {float(p['CO'].value.mean()):.4g} | "
                    f"p_CH4 = {float(p['CH4'].value.mean()):.4g} | "
                    f"Rate = {float(Rsp.value.mean()):.4g}"
                    )


            eq = (
                TransientTerm(coeff=eps, var=C[sp]) 
                + ConvectionTerm(coeff=u_face, var=C[sp])
                ==
                DiffusionTerm(coeff=eps * Dr, var=C[sp])
                + ImplicitSourceTerm(coeff=rho_b * Rsp, var=C[sp])
            )
            eq.solve(dt=dt)

        enforce_atomic_conservation(C, atoms)

        u_face[1] = u.arithmeticFaceValue

        update_pressure_ergun(P, C, T, u, dz)
   
        enforce_constant_pressure(C, T, Pbar, species)

        update_velocity_mass(P, C, u, Cin)

        
        #EB of Species
        Qrxn = sum(-dH[i] * rates[i] for i in range(3))

        energy = (
            TransientTerm(coeff=rhoCp, var=T)
            + ConvectionTerm(coeff=u_face * rhoCp, var=T)
            ==
            DiffusionTerm(coeff=lambda_e, var=T)
            + ImplicitSourceTerm(coeff=rho_b * Qrxn, var=T)
        )
        energy.solve(dt=dt)

        #Axial heat release (radial average at plane k)
        # reshape fields
        Qmat = Qrxn.value.reshape(Nz, Nr)
        Tmat = T.value.reshape(Nz, Nr)
        
        # axial averages
        Qaxial.append(Qmat[k, :].mean())
        Tmax_axial.append(Tmat[k, :].max())

        #Atoms check
        if k % 10 == 0:
            H = atomic_total(C, atoms, "H").value.mean()
            Cc = atomic_total(C, atoms, "C").value.mean()
            O = atomic_total(C, atoms, "O").value.mean()
        
            print(f"[z={k}] H={H:.4e}, C={Cc:.4e}, O={O:.4e}")

        #Are we done yet?
        if k % 50 == 0:
            print(f"z = {k*dz:6.3f} m | Tmax = {T.value.max():7.1f} K")
            print(
                    f"[{kin_name}] "
                    f"p_H2 = {float(p['H2'].value.min()):.4g} | "
                    f"p_CO2 = {float(p['CO2'].value.min()):.4g} | "
                    f"p_H2O = {float(p['H2O'].value.max()):.4g} | "
                    f"p_CO = {float(p['CO'].value.mean()):.4g} | "
                    f"p_CH4 = {float(p['CH4'].value.max()):.4g} | "
                    f"Rate = {float(Rsp.value.max()):.4g}"
                    )
            Ptot = sum(C[sp].value.mean() * R * T.value.mean() for sp in species)
            print(f"Total pressure = {float(Ptot):.4g}")

    #Outlet conditions and information
    CO2_out = C["CO2"].value[-Nr:].mean()
    CH4_out = C["CH4"].value[-Nr:].mean()

    #Conversion and seletivity
    X = (Cin["CO2"] - CO2_out) / Cin["CO2"]
    S = CH4_out / (Cin["CO2"] - CO2_out + 1e-12)

    return X, S, T.value, np.array(Qaxial), np.array(Tmax_axial), Nr, Nz, Rr, Lz


#The condtions that I can change
ratios = [4]
Tin_list = [673]
Twalls = [600]
vels = [0.1]
pressures = [7.5]

#PARAMETRIC SWEEP
results = []
profiles = []
Lz = 1.0
case = 0

#Knowing how many combinations do we have for the kinetics HOW MANY DO WE HAVE =?
total = len(ratios) * len(Twalls) * len(vels) * len(pressures) * len(Tin_list)

#The beginning of the reaction altering ratio, Tw, u and P
for r in ratios:
    for Tw in Twalls:
        for u in vels:
            for P in pressures:
                for Tin in Tin_list:
                    case += 1
                    print(f"▶ Case {case} / {total} | H2/CO2={r:.2f}, T={Tw:.2f}, u={u:.2f}, P={P:.2f}, Tin={Tin:.2f}")
    
                    Xc, Sc, Tc, Qaxialc, Tmaxc, Nr, Nz, Rr, Lz = run_reactor(r, Tw, u, P,
                                         champom_kinetics, kin_champom, "Champom", Tin)
                    Xk, Sk, Tk, Qaxialk, Tmaxk, Nr, Nz, Rr, Lz = run_reactor(r, Tw, u, P,
                                         koz_kinetics, kin_koz, "Koschany", Tin)
                    Xkai, Skai, Tkai, Qaxialkai, Tmaxkai, Nr, Nz, Rr, Lz = run_reactor(r, Tw, u, P,
                                         kai_kinetics, kin_kai, "Kai", Tin)
                    Xfar, Sfar, Tfar, Qaxialfar, Tmaxfar, Nr, Nz, Rr, Lz = run_reactor(r, Tw, u, P,
                                                     farsi_kinetics, kin_farsi, "Farsi", Tin)
                    profiles.append({ "kinetic": "Champom","H2_CO2": r,"Twall": Tw,"u0": u,"P": P,"Qaxial": Qaxialc,"Tmax": Tmaxc,"Tfield": Tc,
                                    "Nr": Nr,"Nz": Nz,"Rr": Rr,"Lz": Lz,"Tin": Tin})
                    
                    profiles.append({"kinetic": "Koz","H2_CO2": r,"Twall": Tw,"u0": u,"P": P,"Qaxial": Qaxialk,"Tmax": Tmaxk,"Tfield": Tk,
                                    "Nr": Nr,"Nz": Nz,"Rr": Rr,"Lz": Lz,"Tin": Tin})
                    
                    profiles.append({"kinetic": "Kai","H2_CO2": r,"Twall": Tw,"u0": u,"P": P,"Qaxial": Qaxialkai,"Tmax": Tmaxkai,"Tfield": Tkai,
                                    "Nr": Nr,"Nz": Nz,"Rr": Rr,"Lz": Lz,"Tin": Tin})
                    
                    profiles.append({"kinetic": "Farsi","H2_CO2": r,"Twall": Tw,"u0": u,"P": P,"Qaxial": Qaxialfar,"Tmax": Tmaxfar,"Tfield": Tfar,
                                    "Nr": Nr,"Nz": Nz,"Rr": Rr,"Lz": Lz,"Tin": Tin})
    
                    results.append([r, Tw, u, P, Xc, Sc, Xk, Sk, Xkai, Skai, Xfar, Sfar])

#Results and profiles
profiles = pd.DataFrame(profiles)
results = pd.DataFrame(
    results,
    columns=["H2_CO2", "Twall", "u0", "P",
             "X_champom", "S_champom", "X_Koz", "S_Koz","X_Kai", "S_Kai","X_far", "S_far" ]
)

results

# Add Tin to results (already in sweep loop)
results["Tin"] = Tin_list[0]  # or save per sweep if multiple


# PLOTS
# Convert conversion and selectivity to percent
results["X_ch_pct"] = results["X_champom"] * 100
results["X_koz_pct"] = results["X_Koz"] * 100
results["X_kai_pct"] = results["X_Kai"] * 100
results["X_far_pct"] = results["X_far"] * 100
results["S_ch_pct"] = results["S_champom"] * 100
results["S_koz_pct"] = results["S_Koz"] * 100
results["S_kai_pct"] = results["S_Kai"] * 100
results["S_far_pct"] = results["S_far"] * 100

#Plot styling 
plt.rcParams.update({
    "figure.figsize": (3.35, 2.8),
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "axes.grid": False,
    "figure.dpi": 300
})

COLORS = {"Champom": "black", "Koz": "dimgray", "Kai": "silver", "Farsi": "slategray"}
MARKERS = {"Champom": "o", "Koz": ".", "Kai": ",", "Farsi": "s"}
LINESTYLES = {"Champom": "-", "Koz": "--", "Kai": ":", "Farsi": "-."}

#Generic sweep plot
def plot_sweep(ax, x, y, label):
    ax.plot(
        x, y,
        marker=MARKERS[label],
        linestyle=LINESTYLES[label],
        color=COLORS[label],
        label=label
    )

#Conversion vs Selectivity
def plot_selectivity_conversion(results):
    plt.figure(figsize=(3.35, 2.8))
    plt.scatter(results.X_ch_pct, results.S_ch_pct, marker="o", color="black", edgecolor="black", label="Champom")
    plt.scatter(results.X_koz_pct, results.S_koz_pct, marker=".",color="dimgray", edgecolor="black", label="Koz")
    plt.scatter(results.X_kai_pct, results.S_kai_pct, marker=",",color="silver", edgecolor="black", label="Kai")
    plt.scatter(results.X_far_pct, results.S_far_pct, marker="s",color="slategray", edgecolor="black", label="Farsi")
    plt.xlabel("CO$_2$ conversion / %")
    plt.ylabel("CH$_4$ selectivity / %")
    plt.title("Selectivity vs Conversion")
    plt.legend(frameon=False)
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.show()

#Conversion sweeps for every parameter that was changed
def plot_conversion_sweeps(results):
    fixed_sets = results[["Twall", "u0"]].drop_duplicates().sort_values(["Twall", "u0"])
    for _, row in fixed_sets.iterrows():
        Tw, u = row.Twall, row.u0
        sub = results[(results.Twall == Tw) & (results.u0 == u)]
    
        fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2), sharey=True)
    
        # (a) Pressure
        plot_sweep(axes[0, 0], sub.P, sub.X_ch_pct, "Champom")
        plot_sweep(axes[0, 0], sub.P, sub.X_koz_pct, "Koz")
        plot_sweep(axes[0, 0], sub.P, sub.X_kai_pct, "Kai")
        plot_sweep(axes[0, 0], sub.P, sub.X_far_pct, "Farsi")
        axes[0, 0].set_xlabel("Pressure [bar]")
        axes[0, 0].set_ylabel("CO$_2$ conversion [%]")
        axes[0, 0].set_title("(a) Pressure")
    
        # (b) Wall temperature
        plot_sweep(axes[0, 1], sub.Twall, sub.X_ch_pct, "Champom")
        plot_sweep(axes[0, 1], sub.Twall, sub.X_koz_pct, "Koz")
        plot_sweep(axes[0, 1], sub.Twall, sub.X_kai_pct, "Kai")
        plot_sweep(axes[0, 1], sub.Twall, sub.X_far_pct, "Farsi")
        axes[0, 1].set_xlabel("Wall temperature [K]")
        axes[0, 1].set_title("(b) Temperature")
    
        # (c) Velocity
        plot_sweep(axes[1, 0], sub.u0, sub.X_ch_pct, "Champom")
        plot_sweep(axes[1, 0], sub.u0, sub.X_koz_pct, "Koz")
        plot_sweep(axes[1, 0], sub.u0, sub.X_kai_pct, "Kai")
        plot_sweep(axes[1, 0], sub.u0, sub.X_far_pct, "Farsi")
        axes[1, 0].set_xlabel("Superficial velocity [m s$^{-1}$]")
        axes[1, 0].set_ylabel("CO$_2$ conversion [%]")
        axes[1, 0].set_title("(c) Velocity")
    
        # (d) Ratio
        plot_sweep(axes[1, 1], sub.H2_CO2, sub.X_ch_pct, "Champom")
        plot_sweep(axes[1, 1], sub.H2_CO2, sub.X_koz_pct, "Koz")
        plot_sweep(axes[1, 1], sub.H2_CO2, sub.X_kai_pct, "Kai")
        plot_sweep(axes[1, 1], sub.H2_CO2, sub.X_far_pct, "Farsi")
        axes[1, 1].set_xlabel("H$_2$/CO$_2$ ratio [-]")
        axes[1, 1].set_title("(d) Ratio")
    
        # Shared legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    
        fig.suptitle(f"Twall = {Tw} K, u = {u} m s$^{{-1}}$", fontsize=9)
        plt.tight_layout(rect=[0, 0.15, 1, 1])
        plt.show()

        
#Axial heat release and Tmax for each kinetic
def plot_axial_heat_and_T(profile):

    Tin = float(profile["Tin"])
    Nr = int(profile["Nr"])
    Nz = int(profile["Nz"])
    Rr = float(profile["Rr"])
    Lz = float(profile["Lz"])

    # Safely get Qaxial or Qc
    Q = profile.get("Qaxial")
    Q = np.atleast_1d(np.array(Q, dtype=float).ravel())

    Tmax = np.atleast_1d(np.array(profile["Tmax"], dtype=float).ravel())

    # Expand scalar if needed
    if Q.size == 1 and Tmax.size > 1:
        Q = np.full_like(Tmax, Q[0])
    if Tmax.size == 1 and Q.size > 1:
        Tmax = np.full_like(Q, Tmax[0])

    if len(Q) != len(Tmax):
        # If lengths mismatch, interpolate to match
        z_Q = np.linspace(0, Lz, len(Q))
        z_T = np.linspace(0, Lz, len(Tmax))
        Q = np.interp(z_T, z_Q, Q)
        z = z_T
    else:
        z = np.linspace(0, Lz, len(Q))

    fig, ax1 = plt.subplots(figsize=(3.35, 2.8))
    ax1.plot(z, Q / 1e6, color="black", linewidth=1.5, label="Heat release")
    ax1.set_xlabel("Axial coordinate z [m]")
    ax1.set_ylabel("Volumetric heat release [MW m$^{-3}$]")

    ax2 = ax1.twinx()
    ax2.plot(z, Tmax, "--", color="firebrick", linewidth=1.5, label="T$_{max}$")
    ax2.set_ylabel("Maximum temperature [K]", color="firebrick")
    ax2.tick_params(axis="y", labelcolor="firebrick")

    # Combined legend
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, frameon=False)

    title = (
        f"{profile['kinetic']} | H2/CO2={profile['H2_CO2']}, "
        f"Twall={profile['Twall']} K, u={profile['u0']} m/s, P={profile['P']} bar"
    )
    ax1.set_title(title)
    ax1.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.show()


#Radial hotspot profile
def plot_radial_hotspot(profile):
    
    Tin = float(profile["Tin"])
    Nr = int(profile["Nr"])
    Nz = int(profile["Nz"])
    Rr = float(profile["Rr"])
    Lz = float(profile["Lz"])

    Tmax = np.atleast_1d(np.array(profile["Tmax"], dtype=float).ravel())
    Tc = np.atleast_1d(np.array(profile["Tfield"], dtype=float).ravel())


    # Find axial index of hotspot
    k_hot = np.argmax(Tmax)
    Tmat = Tc.reshape(Nz, Nr)
    r = np.linspace(0, Rr, Nr)

    plt.figure(figsize=(3.35, 2.8))
    plt.plot(r * 1e3, Tmat[k_hot, :], color="black", linewidth=1.5)
    plt.xlabel("Radial coordinate r [mm]")
    plt.ylabel("Temperature [K]")
    plt.title(f"Radial profile at hotspot ({profile['kinetic']}, z ≈ {k_hot / Nz * Lz:.2f} m)")
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.show()



def plot_all_profiles_combined(results, profiles):
    
    #Conversion sweeps
    plot_conversion_sweeps(results)

    #Selectivity vs conversion
    plot_selectivity_conversion(results)

    #Axial + radial profiles
    fixed_sets = profiles[["H2_CO2", "Twall", "u0", "P"]].drop_duplicates()
    
    for _, row in fixed_sets.iterrows():
        H2_CO2 = row.H2_CO2
        Twall = row.Twall
        u0 = row.u0
        P = row.P

        # Filter profiles for this parameter set
        subset = profiles[
            (profiles.H2_CO2 == H2_CO2) &
            (profiles.Twall == Twall) &
            (profiles.u0 == u0) &
            (profiles.P == P)
        ]

        # --- Combined kinetics plot ---
        plt.figure(figsize=(5, 3.5))
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        for _, prof in subset.iterrows():
            kinetic = prof["kinetic"]
            Tin = float(prof["Tin"])
            Nz = int(prof["Nz"])
            Nr = int(prof["Nr"])
            Lz = float(prof["Lz"])
            Rr = float(prof["Rr"])
            
            Tmax = np.atleast_1d(np.array(prof["Tmax"]))
            Tfield = np.atleast_1d(np.array(prof["Tfield"]))
            Tavg = Tfield.reshape(Nz, Nr).mean(axis=1)
            ΔT = Tmax - Tin
            Qaxial = np.atleast_1d(np.array(prof["Qaxial"]))
            z = np.linspace(0, Lz, len(Tmax))

            ax1.plot(
                z, Tmax, '-', color=COLORS[kinetic], label=f'{kinetic} Tmax',
                marker=MARKERS[kinetic], markevery=max(Nz // 8, 1)
            )
            ax1.plot(z, Tavg, '--', color=COLORS[kinetic], label=f'{kinetic} Tavg')
            ax1.plot(z, ΔT, ':', color=COLORS[kinetic], label=f'{kinetic} ΔT')
            ax2.plot(z, Qaxial, '-.', color=COLORS[kinetic], label=f'{kinetic} Qaxial')

        ax1.set_xlabel("Axial coordinate z [m]")
        ax1.set_ylabel("Temperature [K]")
        ax2.set_ylabel("Reaction heat Q [W/m³]")
        ax1.set_title(f"H2/CO2={H2_CO2}, Twall={Twall}, u={u0}, P={P}")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, frameon=False, ncol=2)

        ax1.grid(True, linestyle=":", linewidth=0.5)
        plt.tight_layout()
        plt.show()

        # --- Radial hotspot profiles for each kinetic ---
        for _, prof in subset.iterrows():
            Tin = float(prof["Tin"])
            Nz = int(prof["Nz"])
            Nr = int(prof["Nr"])
            Lz = float(prof["Lz"])
            Rr = float(prof["Rr"])
            
            Tmax = np.atleast_1d(np.array(prof["Tmax"]))
            Tc = np.atleast_1d(np.array(prof["Tfield"]))

            if Tc.size != Nr * Nz:
                Tc = np.resize(Tc, (Nz * Nr))

            k_hot = np.argmax(Tmax)
            Tmat = Tc.reshape(Nz, Nr)
            r = np.linspace(0, Rr, Nr)

            plt.figure(figsize=(3.35, 2.8))
            plt.plot(r * 1e3, Tmat[k_hot, :], color=COLORS[prof["kinetic"]], linewidth=1.5,
                     label=f"{prof['kinetic']}")
            plt.xlabel("Radial coordinate r [mm]")
            plt.ylabel("Temperature [K]")
            plt.title(f"Radial profile at hotspot (z ≈ {k_hot / Nz * Lz:.2f} m)")
            plt.grid(True, linestyle=":", linewidth=0.5)
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.show()



plot_all_profiles_combined(results, profiles)

def plot_two_panel_temperature_qaxial_sweep_bigger(profiles):

    # Unique combinations
    unique_sets = profiles[["H2_CO2", "Twall", "u0", "P"]].drop_duplicates()

    for _, row in unique_sets.iterrows():
        H2_CO2 = row.H2_CO2
        Twall = row.Twall
        u0 = row.u0
        P = row.P

        # Filter profiles for this parameter set
        subset = profiles[
            (profiles.H2_CO2 == H2_CO2) &
            (profiles.Twall == Twall) &
            (profiles.u0 == u0) &
            (profiles.P == P)
        ]

        # ---- Bigger figure ----
        fig, axes = plt.subplots(2, 1, figsize=(6.8, 7.0), sharex=True)

        # Secondary y-axis for Qaxial
        ax2 = axes[1].twinx()
        ax2.set_ylabel("Volumetric heat release [MW m$^{-3}$]", color="firebrick")
        ax2.tick_params(axis="y", labelcolor="firebrick")
        ax2.grid(False)

        for _, prof in subset.iterrows():
            kinetic = prof["kinetic"]
            Tmax = np.atleast_1d(np.array(prof["Tmax"]))
            Qaxial = np.atleast_1d(np.array(prof["Qaxial"]))
            Lz = float(prof["Lz"])
            Tin = float(prof["Tin"])
            Nz = len(Tmax)
            z = np.linspace(0, Lz, Nz)

            # ---- Top panel: Tmax ----
            axes[0].plot(
                z, Tmax,
                color=COLORS[kinetic],
                linestyle=LINESTYLES[kinetic],
                marker=MARKERS[kinetic],
                markevery=max(len(z)//8, 1),
                label=kinetic
            )

            # ---- Bottom panel: ΔT ----
            axes[1].plot(
                z, Tmax - Tin,
                color=COLORS[kinetic],
                linestyle=LINESTYLES[kinetic],
                marker=MARKERS[kinetic],
                markevery=max(len(z)//8, 1),
                label=f"{kinetic} ΔT"
            )

            # ---- Qaxial ----
            ax2.plot(
                z, Qaxial / 1e6,  # convert W/m³ → MW/m³
                color=COLORS[kinetic],
                linestyle='-.',
                alpha=0.7,
                label=f"{kinetic} Qaxial"
            )

        # ---------- Formatting ----------
        axes[0].set_ylabel("Maximum temperature [K]")
        axes[0].set_title(f"(a) Tmax | H2/CO2={H2_CO2}, Twall={Twall} K, u={u0} m/s, P={P} bar")

        axes[1].set_ylabel("Temperature rise $T_{max}-T_{in}$ [K]")
        axes[1].set_xlabel("Axial coordinate z [m]")
        axes[1].set_title("(b) Normalized temperature rise + Qaxial")
        axes[1].set_xlim(0, Lz)

        for ax in axes:
            ax.grid(True, linestyle=":", linewidth=0.5)

        # ---------- Legend outside ----------
        lines, labels = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(
            lines + lines2,
            labels + labels2,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False
        )

        plt.tight_layout()
        plt.show()


plot_two_panel_temperature_qaxial_sweep_bigger(profiles)


def interactive_two_panel_temperature(profiles, kinetic_filter=None):
    
    # Filter by kinetic if requested
    df = profiles.copy()
    if kinetic_filter:
        df = df[df["kinetic"] == kinetic_filter]

    # Flatten Tmax into a long DataFrame for hvplot
    rows = []
    for _, row in df.iterrows():
        Tmax_arr = np.atleast_1d(np.array(row["Tmax"]))
        Nz = len(Tmax_arr)
        z = np.linspace(0, float(row["Lz"]), Nz)
        Tin = float(row["Tin"])
        for zi, Ti in zip(z, Tmax_arr):
            rows.append({
                "z": zi,
                "Tmax": Ti,
                "DeltaT": Ti - Tin,
                "H2_CO2": row["H2_CO2"],
                "Twall": row["Twall"],
                "u0": row["u0"],
                "P": row["P"],
                "kinetic": row["kinetic"]
            })

    df_long = pd.DataFrame(rows)

    # ---------------- Panel (a): Tmax ----------------
    plot_Tmax = df_long.hvplot.line(
        x="z",
        y="Tmax",
        by="kinetic",
        width=900,
        height=400,
        hover_cols=["H2_CO2", "Twall", "u0", "P"],
        title="(a) Maximum Temperature along Reactor",
        widget_location="top"
    )

    # ---------------- Panel (b): DeltaT ----------------
    plot_dT = df_long.hvplot.line(
        x="z",
        y="DeltaT",
        by="kinetic",
        width=900,
        height=400,
        hover_cols=["H2_CO2", "Twall", "u0", "P"],
        title="(b) Temperature Rise ΔT = Tmax - Tin",
        widget_location="top"
    )

    # Stack vertically
    layout = pn.Column(plot_Tmax, plot_dT)
    return layout


interactive_two_panel_temperature(profiles)

# --- Interactive version with dropdown filters ---
def interactive_two_panel_with_filters(profiles):

    # Dropdown options
    kinetic_options = list(profiles["kinetic"].unique())
    H2_CO2_options = list(profiles["H2_CO2"].unique())
    Twall_options = list(profiles["Twall"].unique())
    u0_options = list(profiles["u0"].unique())
    P_options = list(profiles["P"].unique())

    # Panel widgets
    kinetic_w = pn.widgets.Select(name="Kinetic", options=kinetic_options, value=kinetic_options[0])
    H2_CO2_w = pn.widgets.Select(name="H2/CO2 ratio", options=H2_CO2_options, value=H2_CO2_options[0])
    Twall_w = pn.widgets.Select(name="Wall Temp [K]", options=Twall_options, value=Twall_options[0])
    u0_w = pn.widgets.Select(name="Velocity [m/s]", options=u0_options, value=u0_options[0])
    P_w = pn.widgets.Select(name="Pressure [bar]", options=P_options, value=P_options[0])

    @pn.depends(kinetic=kinetic_w, H2_CO2=H2_CO2_w, Twall=Twall_w, u0=u0_w, P=P_w)
    def update_plot(kinetic, H2_CO2, Twall, u0, P):
        # Filter the DataFrame
        df = profiles[
            (profiles["kinetic"] == kinetic) &
            (profiles["H2_CO2"] == H2_CO2) &
            (profiles["Twall"] == Twall) &
            (profiles["u0"] == u0) &
            (profiles["P"] == P)
        ]

        if df.empty:
            return pn.pane.Markdown(
                "No data available for this combination of filters.",
                style={"color": "red", "font-size": "16px"}
            )

        # Flatten Tmax
        rows = []
        for _, row in df.iterrows():
            Tmax_arr = np.atleast_1d(np.array(row["Tmax"]))
            Nz = len(Tmax_arr)
            Lz = float(row["Lz"])
            Tin = float(row["Tin"])
            z = np.linspace(0, Lz, Nz)
            for zi, Ti in zip(z, Tmax_arr):
                rows.append({
                    "z": zi,
                    "Tmax": Ti,
                    "DeltaT": Ti - Tin
                })

        df_long = pd.DataFrame(rows)

        # Plot Tmax and DeltaT
        plot_Tmax = df_long.hvplot.line(
            x="z",
            y="Tmax",
            width=900,
            height=400,
            title="Maximum Temperature along Reactor",
        )

        plot_dT = df_long.hvplot.line(
            x="z",
            y="DeltaT",
            width=900,
            height=400,
            title="Temperature Rise ΔT = Tmax - Tin",
        )

        return pn.Column(plot_Tmax, plot_dT)

    # Layout
    controls = pn.Row(kinetic_w, H2_CO2_w, Twall_w, u0_w, P_w)
    layout = pn.Column(controls, update_plot)
    return layout

# Create and serve the interactive layout
interactive_layout = interactive_two_panel_with_filters(profiles)
interactive_layout.servable()

interactive_two_panel_with_filters(profiles)