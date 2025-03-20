//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//! \brief Initializes a vertically integrated protoplanetary disk with various options
//! for dust evolution

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <cstdlib>    // srand
#include <cstring>    // strcmp()
#include <fstream>
#include <iostream>   // endl
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"
// #include "../units/units.hpp"

namespace {
void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
Real DenProfileCyl(const Real rad, const Real phi, const Real z);
Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z, const Real St, const Real eps);
Real SspeedProfileCyl(const Real rad, const Real phi, const Real z);
Real VelProfileCyl(const Real rad, const Real phi, const Real z);
Real afr_ini(const Real rad, const Real phi, const Real z);
Real Stokes_vol(Real size, Real rhog, Real cs, Real OmK);
Real Stokes_int(Real size, Real Sig);
Real log_size(int n, Real amax, Real amin);
Real mean_size(Real amax, Real amin, Real qd);
Real eps_bin(int bin, Real amax, Real epstot, Real qd);
Real dv_turb(Real tau_mx, Real tau_mn, Real t0, Real v0, Real ts, Real vs, Real reynolds);
Real dv_tot(Real a_0, Real a_1, Real dp, Real rhog, Real cs, Real omega);
void planet_acc_plummer(Real R, Real phi, Real z, Real time, Real* aR, Real* aphi, Real* az);
void planet_acc_power(Real R, Real phi, Real z, Real time, Real* aR, Real* aphi, Real* az);
void dust_pert_eq_pow_BL19(Real rad, Real phi, Real z, Real St0, Real St1, Real eps0, Real eps1, Real* vrg, Real* vphig, Real* vrd0, Real* vphid0, Real* vrd1, Real* vphid1);

void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &stopping_time, AthenaArray<Real> &nu_dust,
      AthenaArray<Real> &cs_dust, int is, int ie, int js, int je, int ks, int ke);
void MyViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &w,
                    const AthenaArray<Real> &bc,
                    int is, int ie, int js, int je, int ks, int ke);
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time_array);
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);

// problem parameters which are useful to make global to this file
Real year, au, mp, M_sun, gm0, G, rc, pSig, q, prho, hr_au, hr_r0, gamma_gas, pert, tcool_orb, delta_ini, v_frag, alpha_turb, alpha_gas, M_in, R_in, R_out, a_in, r_sm;
Real dfloor;
Real Omega0;
Real a_min, a_max_ini, q_dust, eps_ini, eps_floor, rho_m, mue;
Real R_min, R_max, dsize_in, dsize_out, t_damp_in, t_damp_out;
Real unit_len, unit_vel, unit_rho, unit_time, unit_sig;
Real R_p, Mp_s;
Real ms, r0, rchar, mdisk, period_ratio;
Real R_inter, C_in, C_out;
Real t_damp, th_min, th_max, dsize;
bool beta_cool, dust_cool, ther_rel, isotherm, MMSN, coag, infall, damping, planet, planet_power, Benitez, ind_term;
} // namespace

// User-defined boundary conditions for disk simulations
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get switches for different physics
  MMSN   = pin->GetBoolean("problem","MMSN");
  ther_rel     = pin->GetBoolean("problem","therm_relax");
  beta_cool    = pin->GetBoolean("problem","beta_cool");
  dust_cool    = pin->GetBoolean("problem","dust_cool");
  isotherm     = pin->GetBoolean("problem","isotherm");
  coag         = pin->GetBoolean("problem","coag");
  infall       = pin->GetBoolean("problem","infall");
  planet       = pin->GetBoolean("problem","planet");
  planet_power = pin->GetBoolean("problem","planet_power");
  damping      = pin->GetBoolean("problem","damping");
  Benitez      = pin->GetBoolean("problem","Benitez");
  ind_term     = pin->GetBoolean("problem","ind_term");

  // Get parameters for gravitatonal potential of central point mass
  au         = 1.495978707e13; // astronomical unit
  mp         = 1.67262192e-24; // proton mass in gram
  M_sun      = 1.989e33; // solar mass in gram
  year       = 3.154e7; // year in seconds
  G          = 6.67259e-8; // gravitational constant

  ms         = pin->GetReal("problem","ms_in_msol") * M_sun; // stellar mass
  r0         = pin->GetReal("problem","r0_in_au") * au; // reference radiud
  rchar      = pin->GetReal("problem","rc_in_au") * au; // characteristic disk radius
  mdisk      = pin->GetReal("problem","md_in_ms") * ms; // disk mass
  gm0        = pin->GetOrAddReal("problem","ms_in_msol",0.0);
  period_ratio = pin->GetOrAddReal("problem","period_ratio",0.0);
  pSig       = pin->GetReal("problem","beta_Sig");
  q          = pin->GetReal("problem","beta_T");
  pert       = pin->GetReal("problem","perturb");
  prho       = pSig - 0.5*(q+3.0);
  hr_au      = pin->GetReal("problem","hr_at_au");
  tcool_orb  = pin->GetReal("problem","t_cool");
  a_max_ini  = pin->GetReal("problem","dust_amax");
  a_min      = pin->GetReal("problem","dust_amin");
  eps_ini    = pin->GetReal("problem","eps_ini");
  q_dust     = pin->GetReal("problem","dust_q");
  rho_m      = pin->GetReal("problem","dust_rho_m");
  delta_ini  = pin->GetReal("problem","dust_d_ini");
  v_frag     = pin->GetReal("problem","v_frag");
  mue        = pin->GetReal("problem","mue");
  alpha_turb = pin->GetReal("problem","alpha_turb");
  alpha_gas  = pin->GetReal("problem","alpha_gas");
  a_in       = pin->GetReal("problem","a_in");
  R_min      = pin->GetReal("mesh","x1min");
  R_max      = pin->GetReal("mesh","x1max");
  Mp_s       = pin->GetReal("problem","Mp_s");
  r_sm       = pin->GetReal("problem","r_sm") * std::pow((Mp_s/(1.+Mp_s)/3.), 1./3.);
  R_inter    = pin->GetReal("problem","R_int") * au/r0;
  C_in       = pin->GetReal("problem","C_in");
  C_out      = pin->GetReal("problem","C_out");
  eps_floor  = pin->GetReal("problem","eps_floor");

  th_min = pin->GetReal("mesh","x2min");
  th_max = pin->GetReal("mesh","x2max");
  dsize  = pin->GetReal("problem","dsize");
  t_damp = pin->GetReal("problem","t_damp");


  // Define the code units - needed for Stokes number calculation
  unit_sig = mdisk*(2.+pSig) / (2.*PI*rchar*rchar);  // column density at rchar
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0)
    if(!MMSN)
      unit_rho = unit_sig;
    else
      unit_rho = 1700.*std::pow(r0/au, pSig);
  else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0)
    unit_rho = unit_sig / (std::sqrt(2.*PI)*hr_au*au*std::pow(rchar/au, 0.5*(q + 3.0))); // unit density (density at r0)
  unit_len  = r0;
  unit_time = 1./std::sqrt(ms * G / std::pow(unit_len,3.0));
  unit_vel  = unit_len/unit_time;

  rc         = pin->GetReal("problem","rc_in_au") * au / unit_len;
  hr_r0      = hr_au * std::pow(unit_len/au, 0.5*(q+1.0));
  M_in       = pin->GetReal("problem","M_in")*M_sun/year;
  R_in       = pin->GetReal("problem","R_in")*au;
  R_out      = pin->GetReal("problem","R_out")*au;

  // Get parameters of initial pressure and cooling parameters
  gamma_gas = pin->GetReal("hydro","gamma");
  Real float_min = std::numeric_limits<float>::min();
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(float_min)));

  Omega0 = pin->GetOrAddReal("orbital_advection","Omega0",0.0);

  // Enroll Source Terms
  EnrollUserExplicitSourceFunction(MySource);

    // Enroll Viscosity Function
  EnrollViscosityCoefficient(MyViscosity);

  // Enroll user-defined dust stopping time
  EnrollUserDustStoppingTime(MyStoppingTime);
  // Enroll user-defined dust diffusivity
  EnrollDustDiffusivity(MyDustDiffusivity);

  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
  }
  if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DiskInnerX2);
  }
  if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DiskOuterX2);
  }
  return;
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  // User-defined output variables
  AllocateUserOutputVariables(2);

  // Store initial condition and maximum particle size for internal use
  AllocateRealUserMeshBlockDataField(5); // rhog, cs, vphi, trelax
  Real orb_defined;
  (porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;
  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;
  int dk = NGHOST;
  if (block_size.nx3 == 1) dk = 0;

  // Store initial condition in meshblock data -> avoid recalculation at later stages/in boundary conditions 
  // Initial condition is axisymmetric -> 2D arrays
  ruser_meshblock_data[0].NewAthenaArray(block_size.nx3+2*NGHOST, block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST); // gas density
  ruser_meshblock_data[1].NewAthenaArray(block_size.nx3+2*NGHOST, block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST); // soundspeed
  ruser_meshblock_data[2].NewAthenaArray(block_size.nx3+2*NGHOST, block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST); // azimuthal velocity
  ruser_meshblock_data[3].NewAthenaArray(block_size.nx3+2*NGHOST, block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST); // power law exponent
  ruser_meshblock_data[4].NewAthenaArray(block_size.nx3+2*NGHOST, block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST); // trelax

  Real den, cs, vg_phi, rad, phi, z, x1, x2, x3;
  Real amean, St_mid, OmK, eps, den_dust, den_mid, as, sig_s, ns, tcool, rhod_tot, amax, q_d, aint, eps0, eps1, rhod0, rhod1, a0, a1, sigma, St0, St1;
  for (int k=is-NGHOST; k<=ke+NGHOST; ++k) {
    x3 = pcoord->x3v(k);
    for (int j=js-NGHOST; j<=je+NGHOST; ++j) {
      x2 = pcoord->x2v(j);
    #pragma omp simd
      for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {
        x1 = pcoord->x1v(i);

        // Calculate initial condition
        GetCylCoord(pcoord,rad,phi,z,i,j,0);
        den    = DenProfileCyl(rad,phi,z);
        cs     = SspeedProfileCyl(rad, phi, z);
        vg_phi = VelProfileCyl(rad,phi,z);
        if (porb->orbital_advection_defined)
          vg_phi -= vK(porb, x1, x2, x3);

        // Assign ruser_meshblock_data 0-2
        ruser_meshblock_data[0](k, j, i) = den;
        ruser_meshblock_data[1](k, j, i) = cs;
        ruser_meshblock_data[2](k, j, i) = vg_phi;
        ruser_meshblock_data[3](k, j, i) = q_dust;

        amax  = a_max_ini;
        aint = std::sqrt(amax*a_min);
        q_d   = ruser_meshblock_data[3](k, j, i);

        eps0   = eps_bin(0, amax, eps_ini, q_dust);
        eps1   = eps_bin(1, amax, eps_ini, q_dust);

        a1 = mean_size(aint,  amax, q_dust); 
        a0 = mean_size(a_min, aint, q_dust);
        
        sigma = std::pow(rad/rc, pSig) * std::exp(-std::pow(rad/rc, 2.0+pSig)) * unit_sig;
        St0 = Stokes_int(a0, sigma);
        St1 = Stokes_int(a1, sigma);
        
        Real h = cs/std::pow(rad,-1.5);
        rhod0 = DenProfileCyl_dust(rad, phi, z, St0, eps0);
        rhod1 = DenProfileCyl_dust(rad, phi, z, St1, eps1);
        rhod_tot = rhod0 + rhod1;

        as    = a_min * (q_d+3.)/(q_d+4.) * (std::pow(amax/a_min,q_d+4.)-1.)/(std::pow(amax/a_min,q_d+3.)-1.); // Sauter mean radius 
        sig_s = PI*SQR(as);      // Sauter mean radius collision cross section
        ns    = rhod_tot * unit_rho / (4./3.*PI*rho_m*std::pow(as, 3.0)); // Sauter mean number density
        tcool = std::min(50., std::sqrt(PI/8.) * gamma_gas/(gamma_gas-1.) / (ns*sig_s*cs*unit_vel) / unit_time * std::pow(rad,-1.5)) / std::pow(rad,-1.5);

        ruser_meshblock_data[4](k, j, i) = tcool;
      }
    }
  }
return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  srand(221094);
  Real rad(0.0), phi(0.0), z(0.0);
  Real den, vel, cs;
  Real a0,a1,aint,St0,St1,Hd0,Hd1,eps0,eps1,a_int_ini,afr,amax,rhod0,rhod1,sigma;
  Real ran_rho, ran_vx1, ran_vx2, ran_vx3;
  Real x1, x2, x3;

  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;
  //  Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    x3 = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      x2 = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {

        // Generate random perturbations (from -1 to 1)
        ran_rho = 2.0 * (((Real) rand() / RAND_MAX) - 0.5);
        ran_vx1 = 2.0 * (((Real) rand() / RAND_MAX) - 0.5);
        ran_vx2 = 2.0 * (((Real) rand() / RAND_MAX) - 0.5);
        ran_vx3 = 2.0 * (((Real) rand() / RAND_MAX) - 0.5);

        x1 = pcoord->x1v(i);
        GetCylCoord(pcoord,rad,phi,z,i,j,k); // convert to cylindrical coordinates

        // compute initial conditions in cylindrical coordinates
        den = ruser_meshblock_data[0](k, j, i);
        cs  = ruser_meshblock_data[1](k, j, i);
        vel = ruser_meshblock_data[2](k, j, i);

        // assign initial conditions for density and pressure (perturb profile)
        phydro->u(IDN,k,j,i) = (1.0 + pert*ran_rho) * den;
        phydro->u(IPR,k,j,i) = SQR(cs) * phydro->u(IDN,k,j,i);

        // assign initial conditions for momenta (perturb profiles)
        phydro->u(IM1,k,j,i) = ran_vx1*pert * cs * den;
        phydro->u(IM2,k,j,i) = ran_vx2*pert * cs * den;;
        phydro->u(IM3,k,j,i) = (1.0 + ran_vx3*pert) * den*vel;

        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i)  = SQR(cs)*phydro->u(IDN,k,j,i)/(gamma_gas - 1.0);
          phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
                                       + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
        }
        if (NDUSTFLUIDS > 0) {
          amax = a_max_ini;
          aint = std::sqrt(amax*a_min);

          eps0   = eps_bin(0, amax, eps_ini, q_dust);
          eps1   = eps_bin(1, amax, eps_ini, q_dust);

          a1 = mean_size(aint,  amax, q_dust); 
          a0 = mean_size(a_min, aint, q_dust);
          
          sigma = std::pow(rad/rc, pSig) * std::exp(-std::pow(rad/rc, 2.0+pSig)) * unit_sig;
          St0 = Stokes_int(a0, sigma);
          St1 = Stokes_int(a1, sigma);
          
          Real h = cs/std::pow(rad,-1.5);
          rhod0 = DenProfileCyl_dust(rad, phi, z, St0, eps0);
          rhod1 = DenProfileCyl_dust(rad, phi, z, St1, eps1);

          amax = (fabs(rhod1/den-eps_floor)<1e-5*eps_floor) ? 1e-4 : a_max_ini;

          int dust_id = 0;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;
          pdustfluids->df_cons(rho_id, k, j, i) = rhod0 * (1.0 + pert*ran_rho);
          pdustfluids->df_cons(v1_id,  k, j, i) = rhod0 * ran_vx1*pert * cs;
          pdustfluids->df_cons(v2_id,  k, j, i) = rhod0 * ran_vx2*pert * cs;
          pdustfluids->df_cons(v3_id,  k, j, i) = rhod0 * (1.0 + ran_vx3*pert) * vel;

          dust_id = 1;
          rho_id  = 4*dust_id;
          v1_id   = rho_id + 1;
          v2_id   = rho_id + 2;
          v3_id   = rho_id + 3;
          pdustfluids->df_cons(rho_id, k, j, i) = rhod1 * (1.0 + pert*ran_rho);
          pdustfluids->df_cons(v1_id,  k, j, i) = rhod1 * ran_vx1*pert * cs;
          pdustfluids->df_cons(v2_id,  k, j, i) = rhod1 * ran_vx2*pert * cs;
          pdustfluids->df_cons(v3_id,  k, j, i) = rhod1 * (1.0 + ran_vx3*pert) * vel;

          if(NSCALARS == 1){
            pscalars->s(0,k,j,i) = pdustfluids->df_cons(4, k, j, i) * amax;
          }else if(NSCALARS==3){
            pscalars->s(0,k,j,i) = pdustfluids->df_cons(4, k, j, i) * amax;
            pscalars->s(1,k,j,i) = pdustfluids->df_cons(0, k, j, i) * (C_in + (C_out-C_in)/(1. + std::exp(-35.*(rad-R_inter))));
            pscalars->s(2,k,j,i) = pdustfluids->df_cons(4, k, j, i) * (C_in + (C_out-C_in)/(1. + std::exp(-35.*(rad-R_inter))));
          }
        }
      }
    }
  }

  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! Transform to cylindrical coordinate
void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    rad=pco->x1v(i);
    phi=pco->x2v(j);
    z = 0.0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rad=std::abs(pco->x1v(i)*std::sin(pco->x2v(j)));
    phi=pco->x3v(k);
    z=pco->x1v(i)*std::cos(pco->x2v(j));
  }
  return;
}

//----------------------------------------------------------------------------------------
//! Computes density in cylindrical coordinates (following Lynden-Bell & Pringle, 1974)
Real DenProfileCyl(const Real rad, const Real phi, const Real z) {
  Real den,cs,h;
  cs  = SspeedProfileCyl(rad, phi, z);                                        // speed of sound
  h   = cs * std::pow(rad,1.5);
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    if(!MMSN)
      den = std::pow(rad/rc, pSig) * std::exp(-std::pow(rad/rc, 2.0+pSig));   // Lynden-Bell and Pringle (1974) profile
    else
      den = std::pow(rad, pSig); // Column density profile
  }else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){
    den = std::pow(rad/rc, prho) * std::exp(-std::pow(rad/rc, 2.0+pSig))      // Lynden-Bell and Pringle (1974) profile
        * std::exp(SQR(rad/h) * (rad / std::sqrt(SQR(rad) + SQR(z)) - 1.0));  // vertical structure          
  }
  return std::max(den,dfloor);
}

//----------------------------------------------------------------------------------------
//! Computes density in cylindrical coordinates (following Lynden-Bell & Pringle, 1974)
Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z, Real St, Real eps) {
  Real den,cs,h,Hd,Sig,rhod_mid,den_g,eps_;
  cs  = SspeedProfileCyl(rad, phi, z);                                      // speed of sound
  h   = cs * std::pow(rad,1.5);                                             // pressure scale height
  Hd  = h * std::sqrt(delta_ini/(St+delta_ini)) * unit_len;
  Sig = std::pow(rad/rc, pSig) * std::exp(-std::pow(rad/rc, 2.0+pSig)) * unit_sig;
  den_g = std::pow(rad/rc, prho) * std::exp(-std::pow(rad/rc, 2.0+pSig))      // Lynden-Bell and Pringle (1974) profile
        * std::exp(SQR(rad/h) * (rad / std::sqrt(SQR(rad) + SQR(z)) - 1.0));    // vertical structure
  rhod_mid = eps*Sig / (std::sqrt(2.*PI) * Hd) / unit_rho;
  // den = rhod_mid * std::exp(SQR(rad*unit_len/Hd) * (rad / std::sqrt(SQR(rad)+SQR(z)) - 1.0)); // const St solution
  den = rhod_mid * std::exp(-St/delta_ini*(std::exp(0.5*SQR(z/h)) - 1.) - 0.5*SQR(z/h)); // constant amax solution
  eps_ = den/den_g;
  return std::sqrt(SQR(eps_floor) + SQR(eps_)) * den_g;
}

//----------------------------------------------------------------------------------------
//! Computes soundspeed
Real SspeedProfileCyl(const Real rad, const Real phi, const Real z) {
  Real cs = hr_r0 * std::pow(rad, 0.5*q); // cs in code units: H/R @code_length_cgs
  return cs;
}

//----------------------------------------------------------------------------------------
//! Computes rotational velocity in cylindrical coordinates
Real VelProfileCyl(const Real rad, const Real phi, const Real z) {
  Real cs,h, vel;
  cs  = SspeedProfileCyl(rad, phi, z); // speed of sound
  h   = cs * std::pow(rad,1.5);        // pressure scale height
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0){
    if(!MMSN){
      vel = std::sqrt(1.0/rad) * std::sqrt(1.0 + SQR(h/rad)*(pSig+q-(2.0+pSig)*std::pow(rad/rc, 2.0+pSig)));
    }
    else{
      vel = std::sqrt(1.0/rad) * std::sqrt(1.0 + (pSig+q)*SQR(h/rad));
    }
  }
  else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0){
    vel =  std::sqrt(1.0/rad)       // Keplerian velocity
        * std::sqrt((1.0+q) - q*rad/std::sqrt(SQR(rad)+SQR(z)) + (prho+q-(2.0+pSig)*std::pow(rad/rc, 2.0+pSig))*SQR(h/rad));
  }
  vel -= rad*Omega0;
  return vel;
}

//----------------------------------------------------------------------------------------
//! Computes Stokes number
Real Stokes_int(Real size, Real Sig){
  return 0.5*PI * size*rho_m/Sig;
}

Real Stokes_vol(Real size, Real rhog, Real cs, Real OmK){
  return std::sqrt(PI/8.) * size*rho_m/(cs*unit_vel * rhog*unit_rho) * OmK * unit_vel/unit_len;
}

//----------------------------------------------------------------------------------------
//! Computes the mass-averaged particle size in size interval a0 to a1
Real mean_size(Real a0, Real a1, Real qd){
  if(qd == -5.0)
      return a1*a0/(a1-a0)*std::log(a0/a1);
  else if(qd == -4.0)
      return (a1-a0)/(std::log(a1)-std::log(a0));
  else
      return (qd+4.0)/(qd+5.0) * (std::pow(a1,qd+5.0)-std::pow(a0,qd+5.0)) / (std::pow(a1,qd+4.0)-std::pow(a0,qd+4.0));
}

//----------------------------------------------------------------------------------------
//! Computes the fragmentation limit for the initial condition
Real afr_ini(const Real rad, const Real phi, const Real z){
  Real gamma    = std::fabs(prho+q-(2.0+pSig)*std::pow(rad/rc, 2.0+pSig));
  Real csiso    = SspeedProfileCyl(rad, phi, z) * unit_vel;
  Real vK       = std::pow(rad, -0.5) * unit_vel;
  Real Sigma    = DenProfileCyl(rad, phi, z) * unit_rho;

  Real afr   = 2./(3.*PI) * Sigma/(rho_m*delta_ini) * std::pow(v_frag/csiso, 2.0);
  Real adrfr = 4*v_frag*vK*Sigma/(PI*rho_m*gamma*csiso*csiso);

  return std::min(afr, adrfr);
}

//----------------------------------------------------------------------------------------
//! Computes the dust-to-gas ratio within bin n. Used for initialization and boundary.
Real eps_bin(int bin, Real amax, Real epstot, Real qd){
  Real a0, a1;
  if(bin==0){
    a0 = a_min;
    a1 = std::sqrt(a_min*amax);
  }
  else if(bin==1){
    a0 = std::sqrt(a_min*amax);
    a1 = amax;
  }
  if(qd != 4.0)
    return epstot/(std::pow(amax, qd+4.0) - std::pow(a_min, qd+4.0)) * (std::pow(a1, qd+4.0) - std::pow(a0, qd+4.0));
  else
    return epstot/(std::log(amax) - std::log(a_min)) * std::log(a1) - std::log(a0);
}

Real dv_turb(Real tau_mx, Real tau_mn, Real t0, Real v0, Real ts, Real vs, Real reynolds){
//! ***********************************************************************
//! A function that gives the velocities according to Ormel and Cuzzi (2007)
//!
//!    INPUT:   tau_1       =   stopping time 1
//!             tau_2       =   stopping time 2
//!             t0          =   large eddy turnover time (1/Omega_K)
//!             v0          =   large eddy velocity      (sqrt(alpha)*cs)
//!             ts          =   small eddy turnover time (1/(sqrt(Re)*Omega))
//!             vs          =   small eddy velocity      (vn/Re**0.25)
//!             reynolds    =   Reynolds number
//!
//!    RETURNS: v_rel_ormel =   relative velocity
//!
//! ************************************************************************
  Real st1, st2, hulp1, hulp2, eps;
  Real vg2, ya, y_star, v_rel_ormel, v_drift;

    st1 = tau_mx/t0;
    st2 = tau_mn/t0;

    vg2 = 1.5 * SQR(v0); //note the square
    ya = 1.6; // approximate solution for st*=y*st1; valid for st1 << 1.

    if (tau_mx < 0.2*ts){
        /*
        * very small regime
        */
        v_rel_ormel = 1.5 *SQR((vs/ts *(tau_mx - tau_mn)));
    }
    else if (tau_mx < ts/ya){
        v_rel_ormel = vg2 *(st1-st2)/(st1+st2)*(SQR(st1)/(st1+std::pow(reynolds,-0.5)) - SQR(st2)/(st2+std::pow(reynolds,-0.5)));
    }
    else if (tau_mx < 5.0*ts){
        /*
        * eq. 17 of oc07. the second term with st_i**2.0 is negligible (assuming !re>>1)
        * hulp1 = eq. 17; hulp2 = eq. 18
        */
        hulp1 = ( (st1-st2)/(st1+st2) * (SQR(st1)/(st1+ya*st1) - SQR(st2)/(st2+ya*st1)) ); //note the -sign
        hulp2 = 2.0*(ya*st1-std::pow(reynolds,-0.5)) + SQR(st1)/(ya*st1+st1) - SQR(st1)/(st1+std::pow(reynolds,-0.5)) + SQR(st2)/(ya*st1+st2) - SQR(st2)/(st2+std::pow(reynolds, -0.5));
        v_rel_ormel = vg2 *(hulp1 + hulp2);
    }
    else if (tau_mx < t0/5.0){
        /*
        * full intermediate regime
        */
        eps = st2/st1; // stopping time ratio
        v_rel_ormel = vg2 *( st1*(2.0*ya - (1.0+eps) + 2.0/(1.0+eps) *(1.0/(1.0+ya) + std::pow(eps,3.0)/(ya+eps) )) );
    }
    else if (tau_mx < t0){
        /*
        * now y* lies between 1.6 (st1 << 1) and 1.0 (st1>=1). the fit below fits ystar to less than 1%
        */
        Real c3 =-0.29847604;
        Real c2 = 0.32938936;
        Real c1 =-0.63119577;
        Real c0 = 1.6015125;
        Real y_star;
        y_star = c0 + c1*st1 + c2*SQR(st1) + c3*std::pow(st1,3.0);
        /*
        * we can then employ the same formula as before
        */
        eps=st2/st1; // stopping time ratio
        v_rel_ormel = vg2 *( st1*(2.0*y_star - (1.0+eps) + 2.0/(1.0+eps) *(1.0/(1.0+y_star) + std::pow(eps,3.0)/(y_star+eps) )) );
    }
    else{
        /*
        * heavy particle limit
        */
        v_rel_ormel = vg2 * (1.0/(1.0+st1) + 1.0/(1.0+st2));
    }

    return std::sqrt(v_rel_ormel);
}

Real dv_tot(Real a_0, Real a_1, Real dp, Real rhog, Real cs, Real omega, Real z){
  //! ***********************************************************************
  //! Calculates the total relative velocity of particles of sizes a_0 and a_1
  //!
  //!    INPUT:   a_0    =   size 1 (all cgs)
  //!             a_1    =   size 2
  //!             dp     =   radial pressure gradient
  //!             rhog   =   gas density
  //!             cs     =   soundspeed
  //!             omega  =   Kepler frequency
  //!
  //!    RETURNS: dv =   relative velocity
  //!
  //! ************************************************************************
  // ------------ Turbulent velocities --------------
  Real Re    = alpha_turb * 2e-15 * rhog * cs / omega / (mp * mue);
  Real vn    = std::sqrt(alpha_turb)*cs;
  Real vs    = vn * std::pow(Re,-0.25);
  Real tn    = 1/omega;
  Real ts    = tn * std::pow(Re,-0.5);
  Real tau_f = rho_m / (std::sqrt(8.0/PI)*cs * rhog);
  Real tau_0 = tau_f*a_0;
  Real tau_1 = tau_f*a_1;
  Real tau_mx, tau_mn, St_mx, St_mn, Stmx2p1, Stmn2p1;
  Real dvtr, vdrmax, dvBr, dvset, dvdr_r, dvdr_phi;
  if (tau_0 > tau_1){
      tau_mx = tau_0;
      tau_mn = tau_1;
  }
  else{
      tau_mx = tau_1;
      tau_mn = tau_0;
  }
  St_mn = tau_mn*omega;
  St_mx = tau_mx*omega;
  Stmx2p1 = St_mx*St_mx + 1.;
  Stmn2p1 = St_mn*St_mn + 1.;
  dvtr = dv_turb(tau_mx, tau_mn, tn, vn, ts, vs, Re);

  // ------------ Brownian Motion -------------
  Real m1 = 4./3. * PI * std::pow(a_1, 3.0) * rho_m;
  dvBr = std::sqrt(16.0 * cs*cs * mp * mue/(PI*m1));

  // ------------ Settling --------------------
  Real H     = cs/ (std::sqrt(gamma_gas) * omega);
  dvset = std::fabs((St_mx/(St_mx+1.) - St_mn/(St_mn+1.))*z) * omega;

  // ------------ Drift -------------
  vdrmax   = - 0.5 * H*H * omega/(rhog*cs*cs) * dp;
  dvdr_r   = std::fabs(2.*vdrmax * (St_mx/Stmx2p1 - St_mn/Stmn2p1));
  dvdr_phi = std::fabs(vdrmax * (Stmx2p1-Stmn2p1)/(Stmx2p1*Stmn2p1));

  return std::sqrt(SQR(dvtr)+SQR(dvBr)+SQR(dvset)+SQR(dvdr_r)+SQR(dvdr_phi));
}

Real deps1da(Real epstot, Real amax, Real p)
{
  Real xi   = p+4.0;
  if(xi==0.0)
  {
      return epstot * (std::log(amax*a_min)-std::log(amax)) / (amax*std::pow(std::log(amax)-std::log(a_min), 2.0));
  }
  else
  {
      return epstot*xi * (0.5*std::pow(a_min,xi)*std::pow(amax*a_min,0.5*xi) + std::pow(amax,xi)*(0.5*(std::pow(amax*a_min, 0.5*xi) - std::pow(a_min,xi)))) / (amax*std::pow(std::pow(amax,xi) - std::pow(a_min,xi),2.));
  }
}
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time) {

  Real rad,phi,z, Sig, om,  afr, amax, a_int, St0, St1, q_d, a1, a0, cs;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k);
        om = std::pow(rad, -1.5);
        cs  = std::sqrt(prim(IPR,k,j,i)/prim(IDN,k,j,i));

        amax = pmb->pscalars->r(0,k,j,i);
        a_int = std::sqrt(a_min*amax);
        q_d = std::log(prim_df(4,k,j,i)/prim_df(0,k,j,i))/std::log(amax/a_int) - 4.;
        if (q_d >= 0) q_d = std::max(q_d, 0.0);
        if (q_d  < 0) q_d = std::max(q_d, -20.0);

        a0 = mean_size(a_min, a_int, q_d);
        a1 = mean_size(a_int,  amax, q_d);
        
        St0 = Stokes_vol(a0, prim(IDN,k,j,i), cs, om);
        St1 = Stokes_vol(a1, prim(IDN,k,j,i), cs, om);

        // printf("%.3e %.3e %.3e %.3e %.3e %.3e %.3e %.3e \n", rad,phi,z,amax, a0, a1, St0, St1);

        Real &st_time_0 = stopping_time(0, k, j, i);
        st_time_0 = St0/om;

        Real &st_time_1 = stopping_time(1, k, j, i);
        st_time_1 = St1/om;
      }
    }
  }
  return;
}

void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &stopping_time, AthenaArray<Real> &nu_dust,
      AthenaArray<Real> &cs_dust, int is, int ie, int js, int je, int ks, int ke) {

  int nc1 = pmb->ncells1;
  AthenaArray<Real> rad_arr;
  rad_arr.NewAthenaArray(nc1);

  Real gamma = pmb->peos->GetGamma();
  Real rad, phi, z, afr, amax, a_int, St1, St0, q_d, a0, a1, nu_gas, cs, om, H;

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k);
        cs = std::sqrt(w(IPR,k,j,i)/w(IDN,k,j,i));
        om = std::pow(rad, -1.5);
        H  = cs/om;
        nu_gas = delta_ini * cs * H;

        amax  = pmb->pscalars->r(0,k,j,i);
        a_int = std::sqrt(a_min*amax);
        q_d   = std::log(prim_df(4,k,j,i)/prim_df(0,k,j,i))/std::log(amax/a_int) - 4.;
        if (q_d >= 0) q_d = std::max(q_d, 0.0);
        if (q_d  < 0) q_d = std::max(q_d, -20.0);
        
        a0 = mean_size(a_min, a_int, q_d);
        a1 = mean_size(a_int,  amax, q_d);

        St0 = Stokes_vol(a0, w(IDN,k,j,i), cs, om);
        St1 = Stokes_vol(a1, w(IDN,k,j,i), cs, om);

        // printf("%.3e %.3e %.3e %.3e %.3e %.3e %.3e %.3e \n", rad,phi,z,amax, a0, a1, St0, St1);

        Real &diffusivity_0 = nu_dust(0, k, j, i);
        diffusivity_0       = nu_gas / (1.0 + SQR(St0));
        Real &soundspeed_0  = cs_dust(0, k, j, i);
        soundspeed_0        = std::sqrt(diffusivity_0/om);

        Real &diffusivity_1 = nu_dust(1, k, j, i);
        diffusivity_1       = nu_gas / (1.0 + SQR(St1));
        Real &soundspeed_1  = cs_dust(1, k, j, i);
        soundspeed_1        = std::sqrt(diffusivity_1/om);
      }
    }
  }
  return;
}

void MyViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
                    const AthenaArray<Real> &bcc, int is, int ie, int js, int je,
                    int ks, int ke) {
  if (phdif->nu_iso > 0.0) {
    Real cs, om, H, rad, phi, z;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i){
          GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k);
          cs = SspeedProfileCyl(rad, phi, z);
          om = std::pow(rad, -1.5);
          H  = cs/om;
          phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = alpha_gas * cs * H;
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! Additional Sourceterms
//----------------------------------------------------------------------------------------
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {
  const int IX=0, IY=1, IZ=2;
  Real rad, phi, z, rad_i, rad_im1, z_i, z_im1, phi_im1, dr; // grid
  Real cs_iso, omega; // disk properties
  Real dPdr, cs_i, Om_i, H_i, rho, tau_f, Re, ts, vgas, // disk properties
       amax, aint, a0, a1, m0, m1, eps0, eps1, q_d, St_mx, St_mn, sig11, sig01, // current size distribution properies
       dvmax, dv11, dv01, vsmall, vinter, vtr_simp, vdrmax, dvdr_r, dvdr_phi, // relative particle velocities
       afac, s, f, // model parameters
       pfrag, pstick, pint, psmall, pdr, // transition functions
       xi_frg, xi, xi_frdr, xi_swp, // size distributione expoenents
       sm_int, Stmx2p1, Stmn2p1, vtr_vdr, // helper variables 
       deps10, deps01, epsdot_max, adot, adot_max, dm1_01, dm2_01, dm3_01, dm1_10, dm2_10, dm3_10, // TriPoD source terms
       eps1_, epsdot_, tau, adot_, depsa; // shrinkage term 
  Real tcool, cs2_old, cs2_eq, cs2_new, e_kin;
  Real th_in_b, th_out_b, f_in, f_out, f_tot, dampterm, vphi;
  int rho_id, v1_id, v2_id, v3_id;
    

  //--------------------------------------------------------------------------------------
  //! Apply source terms
  //--------------------------------------------------------------------------------------
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k);
        cs_iso  = SspeedProfileCyl(rad, phi, z);
        omega = std::pow(rad, -1.5);

        //--------------------------------------------------------------------------------------
        //! Use thermal relaxation instead
        if(isotherm){                           // equilibrium soundspeed (temperature)
          cons(IEN,k,j,i)  = SQR(cs_iso)*cons(IDN,k,j,i)/(gamma_gas - 1.0);         // constant thermal energy
          cons(IEN,k,j,i) += 0.5*(SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))
                                      + SQR(cons(IM3,k,j,i)))/cons(IDN,k,j,i);
        }
        //--------------------------------------------------------------------------------------
        //! Use thermal relaxation instead
        else if (ther_rel){
          GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k);

          //--------------------------------------------------------------------------------------
          //! Use simple beta cooling
          if(beta_cool){
            tcool = tcool_orb / omega;             
          }

          //--------------------------------------------------------------------------------------
          //! Use collisional dust cooling for a TriPoD size distribution
          else if(dust_cool){
            tcool = pmb->ruser_meshblock_data[4](j, i);
          }

          //--------------------------------------------------------------------------------------
          //! Apply cooling source term
          e_kin = .5/cons(IDN,k,j,i)*(SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))+SQR(cons(IM3,k,j,i)));
          cs2_old = (cons(IEN,k,j,i) - e_kin)*(gamma_gas-1.)/cons(IDN,k,j,i);    // current soundspeed^2 (temperature) 
          cs2_eq  = SQR(cs_iso); // equilibrium soundspeed^2 (temperature)
          cons(IEN,k,j,i) -= cons(IDN,k,j,i)/(gamma_gas-1.0) * (cs2_old-cs2_eq)*(1.-std::exp(-dt/tcool)); 
        }

          //--------------------------------------------------------------------------------------
        //! Polar Damping Zones
        if(damping){  
          th_in_b  = th_min + dsize; 
          th_out_b = th_max - dsize;

          f_in  =  std::max(0.0, (th_in_b - pmb->pcoord->x2v(j)))  / (TINY_NUMBER+dsize) / std::sqrt(t_damp);
          f_out =  std::max(0.0, (pmb->pcoord->x2v(j) - th_out_b)) / (TINY_NUMBER+dsize) / std::sqrt(t_damp);

          vphi = pmb->ruser_meshblock_data[2](j, i); //VelProfileCyl(rad,phi,z);

          f_tot = f_in + f_out;
          dampterm = std::exp(- dt * f_tot*f_tot / std::pow(rad, 1.5));
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            // damping dust
            rho_id  = 4*n;
            v1_id   = rho_id + 1;
            v2_id   = rho_id + 2;
            v3_id   = rho_id + 3;
            cons_df(v1_id,k,j,i) -= (1.-dampterm) * (cons_df(v1_id, k,j,i));
            cons_df(v2_id,k,j,i) -= (1.-dampterm) * (cons_df(v2_id, k,j,i));
            cons_df(v3_id,k,j,i) -= (1.-dampterm) * (cons_df(v3_id, k,j,i) - vphi * cons_df(rho_id,k,j,i));
          }
        }

        // --------------------------------------------------------------------------------------
        // ! Dust coagulation with the TriPoD method (Pfeil et al., 2024)
        //                 calculations done in cgs units
        // --------------------------------------------------------------------------------------
        if(NDUSTFLUIDS>0 && coag && prim_s(0,k,j,i)>a_min){
          //--------------------------------------------------------------------------------------
          //                                Coordinate grid
          //--------------------------------------------------------------------------------------
          GetCylCoord(pmb->pcoord,rad_im1,phi_im1,z_im1,i-1,j,k);
          rad_i   = rad * unit_len;
          rad_im1 *= unit_len;
          z_i   = z * unit_len;
          z_im1 *= unit_len;

          //--------------------------------------------------------------------------------------
          //              Pressure gradient for terminal velocity calculation
          //--------------------------------------------------------------------------------------
          dr     = rad_i-rad_im1;
          dPdr   = (prim(IPR,k,j,i) - prim(IPR,k,j,i-1))/dr * unit_rho*SQR(unit_vel);

          //--------------------------------------------------------------------------------------
          //                            Local gas disk properties
          //--------------------------------------------------------------------------------------
          Om_i   = std::pow(rad, -1.5) * unit_vel/unit_len;
          cs_i   = std::sqrt(gamma_gas * prim(IPR,k,j,i)/prim(IDN,k,j,i)) * unit_vel;
          H_i    = cs_i/(Om_i*std::sqrt(gamma_gas));
          rho    = prim(IDN,k,j,i) * unit_rho;
          tau_f  = rho_m / (std::sqrt(8.0/PI)*cs_i * rho);
          Re     = alpha_turb * 2e-15 * rho * cs_i / Om_i / (mp * mue);

          //--------------------------------------------------------------------------------------
          //                          Dust properties and distribution
          //--------------------------------------------------------------------------------------
          eps0  = prim_df(0,k,j,i)/prim(IDN,k,j,i); // dust-to-gas ratio of small population
          eps1  = prim_df(4,k,j,i)/prim(IDN,k,j,i); // dust-to-gas ratio of large population
          amax  = prim_s(0,k,j,i); // max. particle size
          aint  = std::sqrt(amax*a_min); // intermediate particle size (population boundary)
          q_d   = std::log(eps1/eps0)/std::log(amax/aint) - 4.; // power-law exponent
          if (q_d >= 0) q_d = std::min(q_d, 0.0);
          if (q_d  < 0) q_d = std::max(q_d, -20.0);
          a1    = mean_size(aint,  amax, q_d); // mass-averaged particle size of the large population
          a0    = mean_size(a_min, aint, q_d); // mass-averaged particle size of the small population
          m0    = 4./3.*PI*rho_m*std::pow(a0,3.0);
          m1    = 4./3.*PI*rho_m*std::pow(a1,3.0);
          // --------- Relative Grain Velocities ----------
          afac  = 0.4;
          s     = 3.0;
          dvmax = dv_tot(afac*amax, amax, dPdr, rho, cs_i, Om_i, z_i);
          dv11  = dv_tot(afac*a1, a1, dPdr, rho, cs_i, Om_i, z_i);
          dv01  = dv_tot(a0, a1, dPdr, rho, cs_i, Om_i, z_i);
          // printf("%.3e %.3e %.3e %.3e \n", r_i/au, rho, cs_i, dPdr);

          //--------------------------------------------------------------------------------------
          //               Determine the coagulation and fragmentation parameters
          //--------------------------------------------------------------------------------------
          // ----------- Fragmentation Probability -----------
          pfrag   = std::exp(-std::pow(5.*(std::min(dvmax/v_frag,1.0)-1.0),2.0));
          pstick  = 1.0 - pfrag;

          // ----------- Determine Turbulence Regime ---------
          ts      = 1/(sqrt(Re)*Om_i);
          sm_int  = 5.*ts/(amax*tau_f);
          pint    = 0.5*(-(pow(sm_int,4) - 1.)/(pow(sm_int,4) + 1.) + 1.);
          psmall  = 1-pint;

          // ------- Determine if Drift-Frag. Limited --------
          St_mx    = amax*tau_f*Om_i;
          St_mn    = 0.4*St_mx;
          Stmx2p1  = SQR(St_mx) + 1.;
          Stmn2p1  = SQR(St_mn) + 1.;
          vgas     = std::sqrt(1.5*alpha_turb)*cs_i;
          vsmall   = vgas * std::sqrt((St_mx-St_mn)/(St_mx+St_mn) * (SQR(St_mx)/(St_mx+pow(Re,-0.5)) - SQR(St_mn)/(St_mn+pow(Re,-0.5))));
          vinter   = vgas * std::sqrt(2.292*St_mx);
          vtr_simp = psmall*vsmall + pint*vinter;
          vdrmax   = - 0.5 / (rho*Om_i) * dPdr;
          dvdr_r   = std::fabs(2.*vdrmax * (St_mx/Stmx2p1 - St_mn/Stmn2p1));
          dvdr_phi = std::fabs(vdrmax * (Stmx2p1-Stmn2p1)/(Stmx2p1*Stmn2p1));
          vtr_vdr  = std::pow(0.3*vtr_simp/std::max(std::sqrt(SQR(dvdr_r) + SQR(dvdr_phi)),TINY_NUMBER), 6.0);
          pdr      = 0.5*((1-vtr_vdr)/(1+vtr_vdr)) + 0.5;

          // --------- Resulting power law exponent xi -------
          xi_frg  = -3.75*psmall - 3.5*pint;
          xi_frdr = -3.75*pdr + xi_frg*(1-pdr);
          xi_swp  = -3.0;
          xi      = pfrag*xi_frdr + pstick*xi_swp;

          //--------------------------------------------------------------------------------------
          //               Calculate the mass-exchange rate and particle growth rate
          //--------------------------------------------------------------------------------------
          sig11 = PI * SQR(afac*a1 + a1);
          sig01 = PI * SQR(a0 + a1);
          f = sig01/sig11 * dv01/dv11 * std::pow(amax/aint, -(xi+4.));
          deps10 = rho * eps1 * eps1 * sig11 * dv11 * f / m1;
          deps01 = rho * eps1 * eps0 * sig01 * dv01     / m1;

          deps01 *= unit_len/unit_vel; // unit conversion to [1/code_time]
          deps10 *= unit_len/unit_vel; // unit conversion to [1/code_time]

          epsdot_max = std::min(0.4*eps0, 0.4*eps1)/dt; // limit the rate
          deps01 = deps01 * epsdot_max / sqrt(deps01 * deps01  + epsdot_max* epsdot_max); // limit mass exchange rate
          deps10 = deps10 * epsdot_max / sqrt(deps10 * deps10  + epsdot_max* epsdot_max); // limit mass exchange rate

          adot  = prim_df(4,k,j,i)*unit_rho * dvmax / rho_m * (1.0 - 2.0 / (1.0 + pow(v_frag/dvmax,s)));
          adot *= unit_len/unit_vel; // unit conversion to [cm/code_time]
          adot_max = 0.4*amax/dt; // limit the rate
          adot     = adot * adot_max / sqrt(adot * adot  + adot_max* adot_max);

          //--------------------------------------------------------------------------------------
          //            Calculate the maximum size reduction if large dust is depleted
          //--------------------------------------------------------------------------------------
          if(eps1<(0.425*(eps1+eps0))){ // inly if there is net mass loss in the cell
            // How much mass would we have to move if we want to preserve eps1=0.425*epstot
            eps1_   = 0.425*(eps1+eps0);
            epsdot_ = -(eps1_-eps1)/dt; // mass exchange rate to restore eps1=0.425*epstot one timestep
            tau     = fabs(eps1/epsdot_); // respective timescale

            // Shrink amax on this timescale
            adot_     = std::min(0.0, amax/tau * (1-amax/1e-4));
            adot     += adot_ * adot_max / sqrt(adot_ * adot_  + adot_max* adot_max);

            // We want to keep our power law, so we move mass accordingly
            depsa    = deps1da((eps1+eps0), amax, q_d);
            epsdot_  = std::max(0.0,depsa * adot_);
            deps01  += epsdot_ * epsdot_max / sqrt(epsdot_ * epsdot_  + epsdot_max* epsdot_max);
          }

          // Momentum Exchange Rates
          dm1_01 = deps01*prim_df(1,k,j,i);
          dm2_01 = deps01*prim_df(2,k,j,i);
          dm3_01 = deps01*prim_df(3,k,j,i);

          dm1_10 = deps10*prim_df(5,k,j,i);
          dm2_10 = deps10*prim_df(6,k,j,i);
          dm3_10 = deps10*prim_df(7,k,j,i);

          //--------------------------------------------------------------------------------------
          //           Add the growth and mass-exchange rates to the source terms
          //--------------------------------------------------------------------------------------
          cons_df(0,k,j,i) += dt*prim(IDN,k,j,i) * (deps10 - deps01); // mass exchange rate
          cons_df(1,k,j,i) += dt*prim(IDN,k,j,i) * (dm1_10 - dm1_01); // mom. exchange rate dim. 1
          cons_df(2,k,j,i) += dt*prim(IDN,k,j,i) * (dm2_10 - dm2_01); // mom. exchange rate dim. 2
          cons_df(3,k,j,i) += dt*prim(IDN,k,j,i) * (dm3_10 - dm3_01); // mom. exchange rate dim. 3

          cons_df(4,k,j,i) -= dt*prim(IDN,k,j,i) * (deps10 - deps01); // mass exchange rate
          cons_df(5,k,j,i) -= dt*prim(IDN,k,j,i) * (dm1_10 - dm1_01); // mom. exchange rate dim. 1
          cons_df(6,k,j,i) -= dt*prim(IDN,k,j,i) * (dm2_10 - dm2_01); // mom. exchange rate dim. 2
          cons_df(7,k,j,i) -= dt*prim(IDN,k,j,i) * (dm3_10 - dm3_01); // mom. exchange rate dim. 3

          if(NSCALARS==1){
            cons_s(0,k,j,i)  += dt*prim(IDN,k,j,i) * (adot*eps1 - amax*(deps10 - deps01)); // particle size evolution
          }else if (NSCALARS==3){
            cons_s(0,k,j,i) += dt*prim(IDN,k,j,i) * (adot*eps1 - amax*(deps10 - deps01));
            cons_s(1,k,j,i) += dt*prim(IDN,k,j,i) * (prim_s(2,k,j,i)*deps10 - prim_s(1,k,j,i)*deps01);
            cons_s(2,k,j,i) -= dt*prim(IDN,k,j,i) * (prim_s(2,k,j,i)*deps10 - prim_s(1,k,j,i)*deps01);
          }
          // printf("%d, %d, adot=%.3e, deps10=%.3e, deps01=%.3e, amaxrho=%.3e, rho0=%.3e, rho1=%.3e \n", i,j, adot, deps10, deps01, cons_s(0,k,j,i),cons_df(0,k,j,i),cons_df(4,k,j,i));
          if(adot!=adot || deps10!=deps10 || deps01!=deps01 || cons_s(0,k,j,i)!=cons_s(0,k,j,i) || cons_df(0,k,j,i)!=cons_df(0,k,j,i) || cons_df(4,k,j,i)!=cons_df(4,k,j,i)){
            printf("%d, %d, adot=%.3e, deps10=%.3e, deps01=%.3e, amaxrho=%.3e, rho0=%.3e, rho1=%.3e \n", i,j, adot, deps10, deps01, cons_s(0,k,j,i),cons_df(0,k,j,i),cons_df(4,k,j,i));
            std::cout << "NaN Found";
            std::exit(EXIT_FAILURE);
          }
        }
      }
    }
  }
}
} //namespace

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values
//----------------------------------------------------------------------------------------
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real cs, vel, den, fr, amax, afr, eps;
  int rho_id, v1_id, v2_id, v3_id, dust_id;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          GetCylCoord(pco,rad,phi,z,il-i,j,k);
          den = pmb->ruser_meshblock_data[0](k,j,il-i);
          cs  = pmb->ruser_meshblock_data[1](k,j,il-i);
          vel = pmb->ruser_meshblock_data[2](k,j,il-i);
          // if (pmb->porb->orbital_advection_defined)
          //   vel -= vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));
          prim(IM1,k,j,il-i) = prim(IM1,k,j,il); 
          prim(IM2,k,j,il-i) = prim(IM2,k,j,il); 
          prim(IM3,k,j,il-i) = vel;
          prim(IDN,k,j,il-i) = den;
          prim(IPR,k,j,il-i) = SQR(cs)*den;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,il-i) = SQR(cs)*den;
          if (NDUSTFLUIDS > 0){
            amax = 1e-4;

            dust_id = 0;
            rho_id  = 4*dust_id;
            v1_id   = rho_id + 1;
            v2_id   = rho_id + 2;
            v3_id   = rho_id + 3;
            prim_df(rho_id, k, j, il-i) = prim_df(rho_id, k, j, il);
            prim_df(v1_id,k,j,il-i) = prim_df(v1_id,k,j,il);
            prim_df(v2_id,k,j,il-i) = prim_df(v2_id,k,j,il);
            prim_df(v3_id,k,j,il-i) = vel;

            dust_id = 1;
            rho_id  = 4*dust_id;
            v1_id   = rho_id + 1;
            v2_id   = rho_id + 2;
            v3_id   = rho_id + 3;
            prim_df(rho_id, k, j, il-i) = prim_df(rho_id, k, j, il);
            prim_df(v1_id,k,j,il-i) = prim_df(v1_id,k,j,il);
            prim_df(v2_id,k,j,il-i) = prim_df(v2_id,k,j,il);
            prim_df(v3_id,k,j,il-i) = vel;
          }
          if(NSCALARS == 1){
            pmb->pscalars->r(0,k,j,il-i) = pmb->pscalars->r(0,k,j,il);
          }else if(NSCALARS==3){
            pmb->pscalars->r(0,k,j,il-i) = amax;
            pmb->pscalars->r(1,k,j,il-i) = C_in;
            pmb->pscalars->r(2,k,j,il-i) = C_in;
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real cs, vel, den, fr, amax, afr, eps;
  int rho_id, v1_id, v2_id, v3_id, dust_id;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          GetCylCoord(pco,rad,phi,z,iu+i,j,k);
          den = pmb->ruser_meshblock_data[0](k,j,iu+i);
          cs  = pmb->ruser_meshblock_data[1](k,j,iu+i);
          vel = pmb->ruser_meshblock_data[2](k,j,iu+i);
          // if (pmb->porb->orbital_advection_defined)
          //   vel -= vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));
          prim(IM1,k,j,iu+i) = 0.;
          prim(IM2,k,j,iu+i) = 0.;
          prim(IM3,k,j,iu+i) = vel;
          prim(IDN,k,j,iu+i) = den;
          prim(IPR,k,j,iu+i) = SQR(cs)*den;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,iu+i) = SQR(cs)*den;    
          if (NDUSTFLUIDS > 0){
            amax  =  pmb->pscalars->r(0,k,j,iu); 

            dust_id = 0;
            rho_id  = 4*dust_id;
            v1_id   = rho_id + 1;
            v2_id   = rho_id + 2;
            v3_id   = rho_id + 3;
            prim_df(rho_id, k, j, iu+i) = eps_floor*den;
            prim_df(v1_id,k,j,iu+i) = 0.;
            prim_df(v2_id,k,j,iu+i) = 0.;
            prim_df(v3_id,k,j,iu+i) = vel;

            dust_id = 1;
            rho_id  = 4*dust_id;
            v1_id   = rho_id + 1;
            v2_id   = rho_id + 2;
            v3_id   = rho_id + 3;
            prim_df(rho_id, k, j, iu+i) = eps_floor*den;
            prim_df(v1_id,k,j,iu+i) = 0.;
            prim_df(v2_id,k,j,iu+i) = 0.;
            prim_df(v3_id,k,j,iu+i) = vel;

            if(NSCALARS == 1){
              pmb->pscalars->r(0,k,j,iu+i) = amax;
            }else if (NSCALARS==3){
              pmb->pscalars->r(0,k,j,iu+i) = amax;
              pmb->pscalars->r(1,k,j,iu+i) = C_out;
              pmb->pscalars->r(2,k,j,iu+i) = C_out;
            }
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
  AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real cs, vel, den, fr, amax, afr, eps;
  int rho_id, v1_id, v2_id, v3_id, dust_id;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,jl-j,k);
          den = pmb->ruser_meshblock_data[0](k,jl-j,i);
          cs  = pmb->ruser_meshblock_data[1](k,jl-j,i);
          vel = pmb->ruser_meshblock_data[2](k,jl-j,i);
          // if (pmb->porb->orbital_advection_defined)
          //   vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(jl-j), pco->x3v(k));
          prim(IM1,k,jl-j,i) = prim(IM1,k,jl+j-1,i); 
          prim(IM2,k,jl-j,i) = -prim(IM2,k,jl+j-1,i); 
          prim(IM3,k,jl-j,i) = vel;
          prim(IDN,k,jl-j,i) = prim(IDN,k,jl+j-1,i);
          prim(IPR,k,jl-j,i) = prim(IPR,k,jl+j-1,i);
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,jl-j,i) = prim(IEN,k,jl+j-1,i);
          if (NDUSTFLUIDS > 0){
            amax  = pmb->pscalars->r(0,k,jl+j-1,i); 

            dust_id = 0;
            rho_id  = 4*dust_id;
            v1_id   = rho_id + 1;
            v2_id   = rho_id + 2;
            v3_id   = rho_id + 3;
            prim_df(rho_id,k,jl-j,i) = prim_df(rho_id,k,jl+j-1,i);
            prim_df(v1_id,k,jl-j,i) = prim_df(v1_id,k,jl+j-1,i);
            prim_df(v2_id,k,jl-j,i) = -prim_df(v2_id,k,jl+j-1,i);
            prim_df(v3_id,k,jl-j,i) = vel;

            dust_id = 1;
            rho_id  = 4*dust_id;
            v1_id   = rho_id + 1;
            v2_id   = rho_id + 2;
            v3_id   = rho_id + 3;
            prim_df(rho_id,k,jl-j,i) = prim_df(rho_id,k,jl+j-1,i);
            prim_df(v1_id,k,jl-j,i) = prim_df(v1_id,k,jl+j-1,i);
            prim_df(v2_id,k,jl-j,i) = -prim_df(v2_id,k,jl+j-1,i);
            prim_df(v3_id,k,jl-j,i) = vel;

            if(NSCALARS == 1){
              pmb->pscalars->r(0,k,jl-j,i) = amax;
            }else if (NSCALARS==3){
              pmb->pscalars->r(0,k,jl-j,i) = amax;
              pmb->pscalars->r(1,k,jl-j,i) = C_out;
              pmb->pscalars->r(2,k,jl-j,i) = C_out;
            }
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values
void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
  AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real cs, vel, den, fr, amax, afr, eps;
  int rho_id, v1_id, v2_id, v3_id, dust_id;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,ju+j,k);
          den = pmb->ruser_meshblock_data[0](k,ju+j,i);
          cs  = pmb->ruser_meshblock_data[1](k,ju+j,i);
          vel = pmb->ruser_meshblock_data[2](k,ju+j,i);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(ju+j), pco->x3v(k));
          prim(IM1,k,ju+j,i) = prim(IM1,k,ju-j+1,i);
          prim(IM2,k,ju+j,i) = -prim(IM2,k,ju-j+1,i);
          prim(IM3,k,ju+j,i) = vel;
          prim(IDN,k,ju+j,i) = prim(IDN,k,ju-j+1,i); 
          prim(IPR,k,ju+j,i) = prim(IPR,k,ju-j+1,i);
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,ju+j,i) = prim(IEN,k,ju-j+1,i);
          if (NDUSTFLUIDS > 0){
            amax = pmb->pscalars->r(0,k,ju-j+1,i);

            dust_id = 0;
            rho_id  = 4*dust_id;
            v1_id   = rho_id + 1;
            v2_id   = rho_id + 2;
            v3_id   = rho_id + 3;
            prim_df(rho_id,k,ju+j,i) = prim_df(rho_id,k,ju-j+1,i);
            prim_df(v1_id,k,ju+j,i) = prim_df(v1_id,k,ju-j+1,i);
            prim_df(v2_id,k,ju+j,i) = -prim_df(v2_id,k,ju-j+1,i);
            prim_df(v3_id,k,ju+j,i) = vel;

            dust_id = 1;
            rho_id  = 4*dust_id;
            v1_id   = rho_id + 1;
            v2_id   = rho_id + 2;
            v3_id   = rho_id + 3;
            prim_df(rho_id,k,ju+j,i) = prim_df(rho_id,k,ju-j+1,i);
            prim_df(v1_id,k,ju+j,i) = prim_df(v1_id,k,ju-j+1,i);
            prim_df(v2_id,k,ju+j,i) = -prim_df(v2_id,k,ju-j+1,i);
            prim_df(v3_id,k,ju+j,i) = vel;

            if(NSCALARS == 1){
              pmb->pscalars->r(0,k,ju+j,i) = amax;
            }else if (NSCALARS==3){
              pmb->pscalars->r(0,k,ju+j,i) = amax;
              pmb->pscalars->r(1,k,ju+j,i) = C_out;
              pmb->pscalars->r(2,k,ju+j,i) = C_out;
            }
          }
        }
      }
    }
  }
}

void MeshBlock::UserWorkInLoop() {
  Real rad,phi,z,eps,q_d,amax,aint,as,sig_s,ns,tcool,rhod_tot,cs;
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        for (int k=ks; k<=ke; ++k) {
          GetCylCoord(pcoord, rad,phi,z, i,j,k);
          amax = pscalars->s(0,k,j,i)/pdustfluids->df_cons(4,k,j,i);
          aint = std::sqrt(amax*a_min);
          q_d   = std::log(pdustfluids->df_cons(4,k,j,i)/pdustfluids->df_cons(0,k,j,i))/std::log(amax/aint) - 4.; // power-law exponent
          if (q_d >= 0) q_d = std::min(q_d, 0.0);
          if (q_d  < 0) q_d = std::max(q_d, -10.0);
          
          cs    = std::sqrt(phydro->u(IPR,k,j,i)/phydro->u(IDN,k,j,i)); 
          rhod_tot = pdustfluids->df_cons(4,k,j,i) + pdustfluids->df_cons(0,k,j,i);
          as    = a_min * (q_d+3.)/(q_d+4.) * (std::pow(amax/a_min,q_d+4)-1.)/(std::pow(amax/a_min,q_d+3)-1.); // Sauter mean radius 
          sig_s = PI*SQR(as);      // Sauter mean radius collision cross section
          ns    = rhod_tot * unit_rho / (4./3.*PI*rho_m*std::pow(as, 3.0)); // Sauter mean number density
          tcool = std::min(1000., std::sqrt(PI/8.) * gamma_gas/(gamma_gas-1.) / (ns*sig_s*cs*unit_vel) / unit_time * std::pow(rad,-1.5)) / std::pow(rad,-1.5);

          ruser_meshblock_data[3](k,j,i) = q_d;
          ruser_meshblock_data[4](k,j,i) = tcool;
        }
      }
    }
  return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin ){
  Real rad,phi,z;
  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      for(int i=is; i<=ie; i++) {
        GetCylCoord(pcoord, rad,phi,z, i,j,k);
        user_out_var(0,k,j,i) = ruser_meshblock_data[3](k,j,i);
        user_out_var(1,k,j,i) = ruser_meshblock_data[4](k,j,i) * std::pow(rad,-1.5);
      }
    }
  }
}