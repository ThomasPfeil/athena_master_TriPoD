//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//! \brief Initializes stratified Keplerian accretion disk in both cylindrical and
//! spherical polar coordinates.  Initial conditions are in vertical hydrostatic eqm.

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

namespace {

void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
Real DenProfileCyl(const Real rad, const Real phi, const Real z);
Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z, const Real St, const Real eps);
Real SspeedProfileCyl(const Real rad, const Real phi, const Real z);
Real VelProfileCyl(const Real rad, const Real phi, const Real z);
Real Stokes(Real size, Real rhog, Real cs, Real OmK);
Real log_size(int n, Real amax, Real amin);
Real mean_size(int n, Real amax, Real amin, Real qd);
Real eps_bin(int n, Real amax, Real amin, Real eps_ini, Real q_d);
Real amax_growth(Real rad, Real cs, Real eps, Real time);
      
// problem parameters which are useful to make global to this file
Real gm0, ms, mdisk, r0, h_au, rc, pSig, q, prho, hr_au, hr_r0, gamma_gas, pert, tcool_orb, rho_m, delta_ini, v_frag, g, alpha_gas, eps_floor;
bool beta_cool, dust_cool, ther_rel, isotherm, growth, damping;
Real dfloor, dffloor;
Real Omega0;
Real a_min, a_max, q_dust, eps_ini;
Real unit_len, unit_vel, unit_rho, unit_time, unit_sig;
Real au, mp, M_sun, year, G;
Real th_min, th_max, dsize, t_damp;

void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &stopping_time, AthenaArray<Real> &nu_dust,
      AthenaArray<Real> &cs_dust, int is, int ie, int js, int je, int ks, int ke);

void MyViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &w,
                  const AthenaArray<Real> &bc,
                  int is, int ie, int js, int je, int ks, int ke);

void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time_array);
} // namespace


void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);

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

AthenaArray<Real> t_relax;
AthenaArray<Real> amax_arr;

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get switches for different physics
  ther_rel  = pin->GetBoolean("problem","therm_relax");
  beta_cool = pin->GetBoolean("problem","beta_cool");
  dust_cool = pin->GetBoolean("problem","dust_cool");
  isotherm  = pin->GetBoolean("problem","isotherm");
  growth    = pin->GetBoolean("problem","growth");
  damping    = pin->GetBoolean("problem","damping");

  // Constants
  au         = 1.495978707e13; // astronomical unit
  mp         = 1.67262192e-24; // proton mass in gram
  M_sun      = 1.989e33; // solar mass in gram
  year       = 3.154e7; // year in seconds
  G          = 6.67259e-8; // gravitational constant

  // Get parameters for gravitatonal potential of central point mass
  gm0       = pin->GetReal("problem","GM");
  ms        = pin->GetReal("problem","GM") * M_sun; // stellar mass
  mdisk     = pin->GetReal("problem","md_in_ms") * ms; // disk mass
  r0        = pin->GetReal("problem","r0_in_au") * au; // reference radiud
  rc        = pin->GetReal("problem","rc_in_au") * au / r0;
  pSig      = pin->GetReal("problem","beta_Sig");
  q         = pin->GetReal("problem","beta_T");
  pert      = pin->GetReal("problem","perturb");
  prho      = pSig - 0.5*(q+3.0);
  hr_au     = pin->GetReal("problem","hr_at_au");
  h_au      = pin->GetReal("problem","hr_at_au") * au; // scale height at 1 au
  hr_r0     = hr_au * std::pow(r0/au, 0.5*(q+1.0));
  tcool_orb = pin->GetReal("problem","t_cool");
  a_max     = pin->GetReal("problem","dust_amax");
  a_min     = pin->GetReal("problem","dust_amin");
  eps_ini   = pin->GetReal("problem","eps_ini");
  q_dust    = pin->GetReal("problem","dust_q");
  rho_m     = pin->GetReal("problem","dust_rho_m");
  delta_ini = pin->GetReal("problem","dust_d_ini");
  v_frag    = pin->GetReal("problem","dust_v_frag");
  eps_floor = pin->GetReal("problem","eps_floor");
  alpha_gas = pin->GetReal("problem","alpha_gas");
  g         = pin->GetReal("hydro","gamma");

  th_min = pin->GetReal("mesh","x2min");
  th_max = pin->GetReal("mesh","x2max");
  dsize  = pin->GetReal("problem","dsize");
  t_damp = pin->GetReal("problem","t_damp");

  unit_len  = r0;
  unit_sig  = mdisk*(2.+pSig) / (2.*PI*rc*rc*unit_len*unit_len);  // column density at rchar
  unit_rho  = unit_sig / (std::sqrt(2.*PI)*h_au*std::pow(rc*unit_len/au, 0.5*(q + 3.0))); // unit density (density at r0)
  unit_time = 1./std::sqrt(ms * G / std::pow(unit_len, 3.0));
  unit_vel  = unit_len/unit_time;

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    gamma_gas = pin->GetReal("hydro","gamma");
  } 
  Real float_min = std::numeric_limits<double>::min();
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(float_min)));
  dffloor=pin->GetOrAddReal("hydro","dffloor_1",(1024*(float_min)));

  Omega0 = pin->GetOrAddReal("orbital_advection","Omega0",0.0);

  // Enroll Source Terms
  EnrollUserExplicitSourceFunction(MySource);

  if (NDUSTFLUIDS > 0) {
    // Enroll user-defined dust stopping time
    EnrollUserDustStoppingTime(MyStoppingTime);
    // Enroll user-defined dust diffusivity
    EnrollDustDiffusivity(MyDustDiffusivity);
  }

  // Enroll Viscosity Function
  EnrollViscosityCoefficient(MyViscosity);

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
  AllocateUserOutputVariables(10);

  // Store initial condition and maximum particle size for internal use
  AllocateRealUserMeshBlockDataField(6+NDUSTFLUIDS); // rhog, cs, vphi, amax, rhod_i, eps_tot, trelax
  Real orb_defined;
  (porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;
  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;
  int dk = NGHOST;
  if (block_size.nx3 == 1) dk = 0;

  // Store initial condition in meshblock data -> avoid recalculation at later stages/in boundary conditions 
  // Initial condition is axisymmetric -> 2D arrays
  ruser_meshblock_data[0].NewAthenaArray(block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST); // gas density
  ruser_meshblock_data[1].NewAthenaArray(block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST); // soundspeed
  ruser_meshblock_data[2].NewAthenaArray(block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST); // azimuthal velocity
  if (NDUSTFLUIDS > 0) {
    for(int n=0; n<NDUSTFLUIDS+3; ++n){
      ruser_meshblock_data[3+n].NewAthenaArray(block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST); // amax and initial epsilon
    }
  }

  Real den, cs, vg_phi;
  Real amean, St_mid, OmK, eps, den_dust, den_mid;
  for (int j=js-NGHOST; j<=je+NGHOST; ++j) {
    Real x2 = pcoord->x2v(j);
  #pragma omp simd
    for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {
      Real x1 = pcoord->x1v(i);

      // Calculate initial condition
      Real rad, phi, z;
      GetCylCoord(pcoord,rad,phi,z,i,j,0);
      den    = DenProfileCyl(rad,phi,z);
      cs     = SspeedProfileCyl(rad, phi, z);
      vg_phi = VelProfileCyl(rad,phi,z);
      if (porb->orbital_advection_defined)
        vg_phi -= vK(porb, x1, x2, 0);

      // Assign ruser_meshblock_data 0-2
      ruser_meshblock_data[0](j, i) = den;
      ruser_meshblock_data[1](j, i) = cs;
      ruser_meshblock_data[2](j, i) = vg_phi;

      if (NDUSTFLUIDS > 0) {
        ruser_meshblock_data[3](j, i) = amax_growth(rad, ruser_meshblock_data[1](j, i)*unit_vel, eps, pmy_mesh->time); //a_max; // initialize maximum particle size

        Real a0,a1,am,as,ns,qp1,qp3,qp4,chi,xi,nd,sig_s,tcool,rhod_tot;
        Real as_num   = 0.0; // initialize numerator of Sauter mean
        Real as_denom = 0.0; // initialize denominator of Sauter mean
        rhod_tot = 0.0;
        for (int n=0; n<NDUSTFLUIDS; ++n) {
          // --------------------------------------------------------------------------------------------
          // Calculate equilibrium dust density based on Fromang & Nelson (2009), given delta_ini
          // --------------------------------------------------------------------------------------------
          amean   = mean_size(n, a_max, a_min, q_dust);        // particle size of the bin
          eps     = eps_bin(n, a_max, a_min, eps_ini, q_dust); // dust to gas ratio of the bin 
          OmK     = std::pow(rad, -1.5);                       // Keplerian frequency
          den_mid = DenProfileCyl(rad,phi,0.);                 // midplane gas density
          St_mid  = Stokes(amean, den_mid, cs, OmK);           // midplane Stokes number
          den_dust= DenProfileCyl_dust(rad,phi,z,St_mid,eps);  // dust density based on Fromang & Nelson
          rhod_tot += den_dust; // sum up dust densities
          
          int rho_id  = 4+n;
          ruser_meshblock_data[rho_id](j, i) = den_dust;

          a0   = log_size(n,   a_max, a_min);   
          a1   = log_size(n+1, a_max, a_min);
          am   = mean_size(n,  a_max, a_min, q_dust);
          nd   = den_dust*unit_rho / (4./3.*PI*rho_m*std::pow(am, 3.0));
          qp1  = q_dust+1.; 
          qp3  = q_dust+3.;
          qp4  = q_dust+4.;
          chi  = qp1/qp4 * (std::pow(a1,qp4)-std::pow(a0,qp4)) / (std::pow(a1,qp1)-std::pow(a0,qp1));
          xi   = qp1/qp3 * (std::pow(a1,qp3)-std::pow(a0,qp3)) / (std::pow(a1,qp1)-std::pow(a0,qp1));
          as_num   += chi*nd;
          as_denom += xi*nd;
        }
        as    = as_num/as_denom; // Sauter mean radius 
        sig_s = PI*SQR(as);      // Sauter mean radius collision cross section
        ns    = rhod_tot * unit_rho / (4./3.*PI*rho_m*std::pow(as, 3.0)); // Sauter mean number density
        tcool = std::min(50., std::sqrt(PI/8.) * g/(g-1.) / (ns*sig_s*cs*unit_vel) / unit_time * std::pow(rad,-1.5)) / std::pow(rad,-1.5);
        ruser_meshblock_data[3+NDUSTFLUIDS+1](j, i) = rhod_tot/ruser_meshblock_data[0](j, i); // divide by density
        ruser_meshblock_data[3+NDUSTFLUIDS+2](j, i) = tcool; 
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
  Real den, cs;
  Real amean, St_loc, St_mid, OmK, _inidelta, eps, den_dust, den_mid;
  Real ran_rho, ran_vx1, ran_vx2, ran_vx3;
  Real vg_r, vg_th, vg_phi;
  Real x1, x2, x3;
  int rho_id, v1_id, v2_id, v3_id;

  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;
  //  Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    x3 = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      x2 = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        
        // Generate random perturbations (from -1 to 1)
        ran_rho = 2.0 * (((double) rand() / RAND_MAX) - 0.5);
        ran_vx1 = 2.0 * (((double) rand() / RAND_MAX) - 0.5);
        ran_vx2 = 2.0 * (((double) rand() / RAND_MAX) - 0.5);
        ran_vx3 = 2.0 * (((double) rand() / RAND_MAX) - 0.5);

        x1 = pcoord->x1v(i);
        GetCylCoord(pcoord,rad,phi,z,i,j,k); // convert to cylindrical coordinates
        
        // compute initial conditions in cylindrical coordinates    
        den    = DenProfileCyl(rad,phi,z);
        cs     = SspeedProfileCyl(rad, phi, z);
        vg_phi = VelProfileCyl(rad,phi,z);
        if (porb->orbital_advection_defined)
          vg_phi -= vK(porb, x1, x2, x3);

        // assign initial conditions for density and pressure (perturb profile)
        phydro->u(IDN,k,j,i) = (1.0 + pert*ran_rho) * den;
        phydro->u(IPR,k,j,i) = SQR(cs) * phydro->u(IDN,k,j,i);

        // assign initial conditions for momenta (perturb profiles)
        vg_r    = ran_vx1*pert * cs;
        vg_th   = ran_vx2*pert * cs;
        vg_phi *= (1.0 + ran_vx3*pert);
        phydro->u(IM1,k,j,i) = vg_r   * phydro->u(IDN,k,j,i);
        phydro->u(IM2,k,j,i) = vg_th  * phydro->u(IDN,k,j,i);
        phydro->u(IM3,k,j,i) = vg_phi * phydro->u(IDN,k,j,i);

        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i)  = SQR(cs)*phydro->u(IDN,k,j,i)/(gamma_gas - 1.0);
          phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
                                       + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
        }
        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            // --------------------------------------------------------------------------------------------
            // Calculate equilibrium dust density based on Fromang & Nelson (2009), given delta_ini
            // --------------------------------------------------------------------------------------------
            amean   = mean_size(n, a_max, a_min, q_dust);        // particle size of the bin
            eps     = eps_bin(n, a_max, a_min, eps_ini, q_dust); // dust to gas ratio of the bin 
            OmK     = std::pow(rad, -1.5);                       // Keplerian frequency
            den_mid = DenProfileCyl(rad,phi,0.);                 // midplane gas density
            St_mid  = Stokes(amean, den_mid, cs, OmK);           // midplane Stokes number
            St_loc  = Stokes(amean, den, cs, OmK);               // local Stokes number
            den_dust= DenProfileCyl_dust(rad,phi,z,St_mid,eps);  // dust density based on Fromang & Nelson

            // --------------------------------------------------------------------------------------------
            // Calculate Nakagawa Drift velocity for the dust BASED ON LOCAL STOKES NUMBER
            // --------------------------------------------------------------------------------------------
            Real rad_ip1,phi_ip1,z_ip1;
            GetCylCoord(pcoord,rad_ip1,phi_ip1,z_ip1,i+1,j,k);

            Real den_ip1 = DenProfileCyl(rad_ip1,phi_ip1,z_ip1);
            Real cs_ip1  = SspeedProfileCyl(rad_ip1,phi_ip1,z_ip1);
            Real dr = rad_ip1 - rad;
            Real dp = (SQR(cs_ip1)*den_ip1 - SQR(cs)*den)/dr; // local pressure gradient

            Real dv_r   = St_loc*dp/(den*std::pow(rad,-1.5)) / (SQR(St_loc) + 1.);
            Real dv_phi = -0.5*SQR(St_loc)*dp/(den*std::pow(rad,-1.5)) / (SQR(St_loc) + 1.);
            // -------------  -- dv_z = 0. because of settling-mixing equilibrium --------------------------

            rho_id  = 4*n;
            v1_id   = rho_id + 1;
            v2_id   = rho_id + 2;
            v3_id   = rho_id + 3;
            pdustfluids->df_cons(rho_id,k,j,i) = den_dust; //std::sqrt(SQR(dfloor)+SQR(eps*Sig / (std::sqrt(2.*PI) * Hd) / unit_rho * std::exp(SQR(rad*unit_len/Hd) * (rad / std::sqrt(SQR(rad)+SQR(z)) - 1.0))));
            pdustfluids->df_cons(v1_id, k,j,i) = den_dust * (vg_r   + dv_r*std::sin(pcoord->x2v(j))); //projection of dv_r on spherical grid
            pdustfluids->df_cons(v2_id, k,j,i) = den_dust * (vg_th  + dv_r*std::cos(pcoord->x2v(j))); //projection of dv_r on spherical grid
            pdustfluids->df_cons(v3_id, k,j,i) = den_dust * (vg_phi + dv_phi);
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
    z=pco->x3v(k);
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
  cs  = SspeedProfileCyl(rad, phi, z);                                      // speed of sound
  h   = cs * std::pow(rad,1.5);                                             // pressure scale height
  den = std::pow(rad/rc, prho) * std::exp(-std::pow(rad/rc, 2.0+pSig))      // Lynden-Bell and Pringle (1974) profile
        * std::exp(SQR(rad/h) * (rad / std::sqrt(SQR(rad) + SQR(z)) - 1.0));    // vertical structure
  return std::max(dfloor,den);
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
  Real cs,h;
  cs  = SspeedProfileCyl(rad, phi, z); // speed of sound
  h   = cs * std::pow(rad,1.5);        // pressure scale height
  Real vel =  std::sqrt(1.0/rad)       // Keplerian velocity
              * std::sqrt((1.0+q) - q*rad/std::sqrt(SQR(rad)+SQR(z)) + (prho+q-(2.0+pSig)*std::pow(rad/rc, 2.0+pSig))*SQR(h/rad));
  // Real vel =  std::sqrt(1.0/rad) * std::sqrt((1.0+q) + (prho+q)*SQR(h/rad) - q*rad/std::sqrt(SQR(rad)+SQR(z)));
  vel -= rad*Omega0;
  return vel;
}

//----------------------------------------------------------------------------------------
//! Computes Stokes number, inputs in code units
Real Stokes(Real size, Real rhog, Real cs, Real OmK){
  Real St = std::sqrt(PI/8.) * size*rho_m/(cs*unit_vel * rhog*unit_rho) * OmK * unit_vel/unit_len;
  return St;
}

//----------------------------------------------------------------------------------------
//! Computes lower size grid interface of species n on logarithmic size grid
Real log_size(int n, Real amax, Real amin){
  return std::exp((double)n/(double)NDUSTFLUIDS * std::log(amax/amin) + std::log(amin));
}

//----------------------------------------------------------------------------------------
//! Computes the mass-averaged particle size of a size bin n for rho(a)~a^(qd+4)
Real mean_size(int n, Real amax, Real amin, Real qd){
  Real a0 = log_size(n, amax, amin);   // lower bin boundary
  Real a1 = log_size(n+1, amax, amin); // upper bin boundary
  if(qd == -5.0)
      return a1*a0/(a1-a0)*std::log(a0/a1);
  else if(qd == -4.0)
      return (a1-a0)/(std::log(a1)-std::log(a0));
  else
      return (qd+4.0)/(qd+5.0) * (std::pow(a1,qd+5.0)-std::pow(a0,qd+5.0)) / (std::pow(a1,qd+4.0)-std::pow(a0,qd+4.0));
} 

//----------------------------------------------------------------------------------------
//! Computes the dust-to-gas ratio within bin n. Used for initialization and boundary.
Real eps_bin(int n, Real amax, Real amin, Real epstot, Real qd){
  Real a0 = log_size(n, amax, amin);   // lower bin boundary
  Real a1 = log_size(n+1, amax, amin); // upper bin boundary
  if(qd != 4.0)
    return epstot/(std::pow(amax, qd+4.0) - std::pow(amin, qd+4.0)) * (std::pow(a1, qd+4.0) - std::pow(a0, qd+4.0));
  else
    return epstot/(std::log(amax) - std::log(amin)) * std::log(a1) - std::log(a0);
}

//----------------------------------------------------------------------------------------
//! Computes the particle size, growing toward the frag./drift-frag. limit.
Real amax_growth(Real rad, Real cs, Real eps, Real time){
  Real gamma, vK, Sigma, afr, adrfr, amax, rc2pSig, sqrt2pi;
  sqrt2pi  = std::sqrt(2.*PI);
  rc2pSig  = std::pow(rad/rc, 2.0+pSig);
  gamma    = std::fabs(prho+q-(2.0+pSig)*rc2pSig);
  vK       = std::pow(rad, -0.5) * unit_vel;
  Sigma    = unit_sig * std::pow(rad/rc,pSig) * std::exp(-rc2pSig);

  afr   = 2./(3.*PI) * Sigma/(rho_m*delta_ini) * std::pow(v_frag/cs, 2.0);
  adrfr = 4*v_frag*vK*Sigma/(PI*rho_m*gamma*cs*cs);
  amax  = std::min(adrfr, afr);

  // // Real tgr   = 1./(std::pow(rad,-1.5)*eps_ini);
  // Real eps, amean, St_mid, den_dust, den, den_mid, Hd, rhod_mid, h, zh2, ezh2;
  // h        = csiso/vK * rad;
  // zh2      = 0.5*SQR(z/h);
  // ezh2     = std::exp(zh2);
  // den_dust = 0.0;
  // den_mid  = Sigma/(sqrt2pi*h*unit_len);
  // den      = den_mid * std::exp(SQR(vK/csiso) * (rad / std::sqrt(SQR(rad) + SQR(z)) - 1.0));
  // for(int n=0; n<NDUSTFLUIDS; ++n){
  //   eps       = eps_bin(n, a_max, a_min, eps_ini, q_dust);
  //   amean     = mean_size(n, a_max, a_min, q_dust);
  //   St_mid    = Stokes(amean, den_mid, csiso/unit_vel, vK/unit_vel/rad);
  //   Hd        = h * std::sqrt(delta_ini/(St_mid+delta_ini)) * unit_len;
  //   rhod_mid  = eps*Sigma / (sqrt2pi * Hd) / unit_rho;
  //   den_dust += den_mid * std::exp(-St_mid/delta_ini*(ezh2- 1.) - zh2); 
  // }
  // eps = den_dust/den;
  Real tgr   = 1./(std::pow(rad,-1.5)*eps);
  
  Real a_ini = a_max;
  return a_ini*std::exp(time/tgr)/(1+a_ini/amax*(std::exp(time/tgr)-1));
}

Real tcool_dust(int i,int j,int k, const Real time, const Real amax, Hydro *hyd, Coordinates *pco, DustFluids *dst){
  int rho_id;
  Real a0,a1,am,as,ns,qp1,qp3,qp4,chi,xi,nd,sig_s,cs,rad,phi,z,rhod,tcool;
  Real as_num   = 0.0; // initialize numerator of Sauter mean
  Real as_denom = 0.0; // initialize denominator of Sauter mean
  Real rhod_tot = 0.0; // initialize total dust density
  GetCylCoord(pco,rad,phi,z,i,j,k);

  for (int n=0; n<NDUSTFLUIDS; ++n) {    // Calculate the Sauter mean radius of the distribution
    rho_id = 4*n;
    rhod = dst->df_cons(rho_id,k,j,i) + TINY_NUMBER;
    a0   = log_size(n,   amax, a_min);   
    a1   = log_size(n+1, amax, a_min);
    am   = mean_size(n,  amax, a_min, q_dust);
    nd   = rhod*unit_rho / (4./3.*PI*rho_m*std::pow(am, 3.0));
    qp1  = q_dust+1.; 
    qp3  = q_dust+3.;
    qp4  = q_dust+4.;
    chi  = qp1/qp4 * (std::pow(a1,qp4)-std::pow(a0,qp4)) / (std::pow(a1,qp1)-std::pow(a0,qp1));
    xi   = qp1/qp3 * (std::pow(a1,qp3)-std::pow(a0,qp3)) / (std::pow(a1,qp1)-std::pow(a0,qp1));

    as_num   += chi*nd;
    as_denom += xi*nd;
    rhod_tot += rhod*unit_rho;
  }
  as    = as_num/as_denom; // Sauter mean radius 
  sig_s = PI*SQR(as);      // Sauter mean radius collision cross section
  ns    = rhod_tot / (4./3.*PI*rho_m*std::pow(as, 3.0)); // Sauter mean number density
  cs    = std::sqrt(hyd->u(IPR,k,j,i)/hyd->u(IDN,k,j,i)) * unit_vel; 
  tcool = std::min(50., std::sqrt(PI/8.) * g/(g-1.) / (ns*sig_s*cs) / unit_time * std::pow(rad,-1.5))/std::pow(rad,-1.5);
  return tcool;
}

void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time) {
  Real inv_sqrt_gm0 = 1.0/std::sqrt(gm0);
  Real OmK, cs, amax, a_av, St, rho, pr,rad,phi,z;
  for (int  n=0; n< NDUSTFLUIDS; ++n) {
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
  #pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k);
          rho   = prim(IDN,k,j,i);
          pr    = prim(IPR,k,j,i);
          OmK   = std::pow(rad, -1.5);
          cs    = std::sqrt(pr/rho);
          amax  = pmb->ruser_meshblock_data[3](j, i);
          a_av  = mean_size(n, amax, a_min, q_dust);
          St    = Stokes(a_av, rho, cs, OmK);

          Real &st_time = stopping_time(n, k, j, i);
          st_time = St/OmK;
        }
      }
    }
  }
  return;
}

void MyViscosity(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
                    const AthenaArray<Real> &bcc, int is, int ie, int js, int je,
                    int ks, int ke) {
  if (phdif->nu_iso > 0.0) {
    Real cs, sig, rho, mu;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i){
          cs  = std::sqrt(prim(IPR,k,j,i)/prim(IDN,k,j,i))*unit_vel;
          sig = 1e-15;
          rho = prim(IDN,k,j,i)*unit_rho;
          mu  = 1.67e-24 * 2.33;
          phdif->nu(HydroDiffusion::DiffProcess::iso,k,j,i) = cs*mu/(rho*sig) / (unit_vel*unit_len);//phdif->nu_iso;
        }
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

    Real inv_sqrt_gm0 = 1.0/std::sqrt(gm0);
    Real gamma = pmb->peos->GetGamma();
    Real rho, pr, OmK, cs, amax, a_av, St, nu_gas,rad,phi,z;

    for (int n=0; n<NDUSTFLUIDS; n++) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k);
            rho   = w(IDN, k, j, i);
            pr    = w(IPR, k, j, i);
            OmK   = std::pow(rad, -1.5);
            cs    = std::sqrt(pr/rho);
            amax  = pmb->ruser_meshblock_data[3](j, i);
            a_av  = mean_size(n, amax, a_min, q_dust);
            St    = Stokes(a_av, rho, cs, OmK);
            
            Real nu_gas       = delta_ini*cs*cs/OmK;
            Real &diffusivity = nu_dust(n, k, j, i);
            diffusivity       = nu_gas/(1.0 + SQR(St));
            Real &soundspeed  = cs_dust(n, k, j, i);
            soundspeed        = std::sqrt(diffusivity/OmK);
          }
        }
      }
    }
  return;
}
}

//----------------------------------------------------------------------------------------
//! Additional Sourceterms
//----------------------------------------------------------------------------------------
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {
  const Real gamma = pmb->peos->GetGamma();
  Real rad, phi, z;
  Real tcool;
  int rho_id, v1_id, v2_id, v3_id;
  //--------------------------------------------------------------------------------------
  //! Thermal Relaxation or Isothermal
  //--------------------------------------------------------------------------------------
  //! Enforce isothermal EOS by hardcoding the thermal energy
  if (isotherm){
    Real cs;
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k);
          cs  = SspeedProfileCyl(rad, phi, z);                                  // equilibrium soundspeed (temperature)
          cons(IEN,k,j,i)  = SQR(cs)*cons(IDN,k,j,i)/(gamma_gas - 1.0);         // constant thermal energy
          cons(IEN,k,j,i) += 0.5*(SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))
                                       + SQR(cons(IM3,k,j,i)))/cons(IDN,k,j,i);
          tcool = 0.0;
        }
      }
    }
  }
  //--------------------------------------------------------------------------------------
  //! Use thermal relaxation instead
  else if (ther_rel){
    Real cs2_old, cs2_eq, cs2_new, omega, cs, e_kin, amax;
    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k);

          //--------------------------------------------------------------------------------------
          //! Use simple beta cooling
          if(beta_cool){
            omega = std::pow(rad, -1.5);
            tcool = tcool_orb / omega;             
          }

          //--------------------------------------------------------------------------------------
          //! Use collisional dust cooling (see Pfeil et al., 2024a)
          else if(dust_cool){
            tcool = pmb->ruser_meshblock_data[3+NDUSTFLUIDS+2](j, i);
          }

          //--------------------------------------------------------------------------------------
          //! Apply cooling source term
          e_kin = .5/cons(IDN,k,j,i)*(SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))+SQR(cons(IM3,k,j,i)));
          cs2_old = (cons(IEN,k,j,i) - e_kin)*(g-1.)/cons(IDN,k,j,i);    // current soundspeed^2 (temperature) 
          cs2_eq  = SQR(SspeedProfileCyl(rad, phi, z)); // equilibrium soundspeed^2 (temperature)
          cons(IEN,k,j,i) -= cons(IDN,k,j,i)/(g-1.0) * (cs2_old-cs2_eq)*(1.-std::exp(-dt/tcool)); 
        }
      }
    }
  }

  //--------------------------------------------------------------------------------------
  //! Polar Damping Zones
  if(damping){  
    OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;     
    Real th_in_b, th_out_b, vphi, f_in, f_out, f_tot, dampterm,
         amean, eps, OmK, den_mid, St_mid, St_loc, den, cs, den_dust,
         rad_ip1,phi_ip1,z_ip1,den_ip1,cs_ip1,dr,dp,dv_r,dv_phi,amax;
    th_in_b  = th_min + dsize; 
    th_out_b = th_max - dsize;

    for (int k = pmb->ks; k <= pmb->ke; ++k) {
      for (int j = pmb->js; j <= pmb->je; ++j) {
        for (int i = pmb->is; i <= pmb->ie; ++i) {
          GetCylCoord(pmb->pcoord,rad,phi,z,i,j,k); 
          OmK     = std::pow(rad, -1.5);    
          den_mid = DenProfileCyl(rad,phi,0.);
          den  = pmb->ruser_meshblock_data[0](j, i); //DenProfileCyl(rad,phi,z);
          cs   = pmb->ruser_meshblock_data[1](j, i); //SspeedProfileCyl(rad,phi,z);
          vphi = pmb->ruser_meshblock_data[2](j, i); //VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vphi -= vK(pmb->porb, pmb->pcoord->x1v(i), pmb-> pcoord->x2v(j), pmb->pcoord->x3v(k));

          f_in  =  std::max(0.0, (th_in_b - pmb->pcoord->x2v(j)))  / (TINY_NUMBER+dsize) / std::sqrt(t_damp);
          f_out =  std::max(0.0, (pmb->pcoord->x2v(j) - th_out_b)) / (TINY_NUMBER+dsize) / std::sqrt(t_damp);

          f_tot = f_in + f_out;
          dampterm = std::exp(- dt * f_tot*f_tot / std::pow(rad, 1.5));
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            // --------------------------------------------------------------------------------------------
            // Calculate equilibrium dust density based on Fromang & Nelson (2009), given delta_ini
            amax    = 
            amean   = mean_size(n, amax, a_min, q_dust);        // particle size of the bin
            eps     = eps_bin(n, amax, a_min, eps_ini, q_dust); // dust to gas ratio of the bin 
            St_mid  = Stokes(amean, den_mid, cs, OmK);           // midplane Stokes number
            den_dust= DenProfileCyl_dust(rad,phi,z,St_mid,eps);  // dust density based on Fromang & Nelson

            // damping dust
            rho_id  = 4*n;
            v1_id   = rho_id + 1;
            v2_id   = rho_id + 2;
            v3_id   = rho_id + 3;
            // cons_df(rho_id,k,j,i) -= (1.-dampterm) * (cons_df(rho_id,k,j,i) - den_dust);
            cons_df(v1_id,k,j,i)  -= (1.-dampterm) * (cons_df(v1_id, k,j,i));
            cons_df(v2_id,k,j,i)  -= (1.-dampterm) * (cons_df(v2_id, k,j,i));
            cons_df(v3_id,k,j,i)  -= (1.-dampterm) * (cons_df(v3_id, k,j,i) - den_dust*cons_df(rho_id,k,j,i));
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values
//----------------------------------------------------------------------------------------
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  Real cs, vel, den, fr, amax, afr, eps;
  int rho_id, v1_id, v2_id, v3_id;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          GetCylCoord(pco,rad,phi,z,il-i,j,k);
          den = pmb->ruser_meshblock_data[0](j, i); //DenProfileCyl(rad,phi,z);
          cs  = pmb->ruser_meshblock_data[1](j, i); //SspeedProfileCyl(rad,phi,z);
          vel = pmb->ruser_meshblock_data[2](j, i); //VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));
          prim(IM1,k,j,il-i) = 0.0; //- prim(IM1,k,j,il+i-1); 
          prim(IM2,k,j,il-i) = 0.0; //prim(IM2,k,j,il+i-1); 
          prim(IM3,k,j,il-i) = vel;
          prim(IDN,k,j,il-i) = den;
          prim(IPR,k,j,il-i) = SQR(cs)*den;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,il-i) = SQR(cs)*den;
          if (NDUSTFLUIDS > 0){ 
            Real amean, OmK, den_mid, St_mid, St_loc, den_dust;
            for(int n=0; n<NDUSTFLUIDS; n++){
              // // --------------------------------------------------------------------------------------------
              // // Calculate midplane Stokes numbers
              // // --------------------------------------------------------------------------------------------
              // if(growth)
              //   amax = amax_growth(rad, time);
              // else
              //   amax = a_max;
              // // --------------------------------------------------------------------------------------------
              // // Calculate equilibrium dust density based on Fromang & Nelson (2009), given delta_ini
              // // --------------------------------------------------------------------------------------------
              // amean   = mean_size(n, amax, a_min, q_dust);        // particle size of the bin
              // eps     = eps_bin(n, amax, a_min, eps_ini, q_dust); // dust to gas ratio of the bin 
              // OmK     = std::pow(rad, -1.5);                       // Keplerian frequency
              // den_mid = DenProfileCyl(rad,phi,0.);                 // midplane gas density
              // St_mid  = Stokes(amean, den_mid, cs, OmK);           // midplane Stokes number
              // St_loc  = Stokes(amean, den, cs, OmK);               // local Stokes number
              // den_dust= DenProfileCyl_dust(rad,phi,z,St_mid,eps);  // dust density based on Fromang & Nelson

              // // --------------------------------------------------------------------------------------------
              // // Calculate Pressure Gradients
              // // --------------------------------------------------------------------------------------------
              // Real dp_r   = (prim(IPR,k,j,il-i+1) - prim(IPR,k,j,il-i))/(pmb->pcoord->x1v(il-i+1) - pmb->pcoord->x1v(il-i));
              // Real den_dust = DenProfileCyl_dust(rad,phi,z, St_mid, eps_ini);

              // // --------------------------------------------------------------------------------------------
              // // Calculate Nakagawa solution
              // // --------------------------------------------------------------------------------------------
              // Real dv_r   = St_loc*dp_r/(den*std::pow(rad,-1.5)) / (SQR(St_loc) + 1.);
              // Real dv_phi = -0.5*SQR(St_loc)*dp_r/(den*std::pow(rad,-1.5)) / (SQR(St_loc) + 1.);

              rho_id  = 4*n;
              v1_id   = rho_id + 1;
              v2_id   = rho_id + 2;
              v3_id   = rho_id + 3;
              if(prim_df(v1_id, k,j,il)<0.0){
                prim_df(rho_id,k,j,il-i) = prim_df(rho_id,k,j,il);
                prim_df(v1_id, k,j,il-i) = prim_df(v1_id, k,j,il);
                prim_df(v2_id, k,j,il-i) = prim_df(v2_id, k,j,il);
                prim_df(v3_id, k,j,il-i) = vel;
              } else {
                prim_df(rho_id,k,j,il-i) = prim_df(rho_id,k,j,il+i-1);
                prim_df(v1_id, k,j,il-i) = -prim_df(v1_id, k,j,il+i-1);
                prim_df(v2_id, k,j,il-i) = prim_df(v2_id, k,j,il+i-1);
                prim_df(v3_id, k,j,il-i) = vel;
              }
            }
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
  int rho_id, v1_id, v2_id, v3_id;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          GetCylCoord(pco,rad,phi,z,iu+i,j,k);
          den = pmb->ruser_meshblock_data[0](j, i); //DenProfileCyl(rad,phi,z);
          cs  = pmb->ruser_meshblock_data[1](j, i); //SspeedProfileCyl(rad,phi,z);
          vel = pmb->ruser_meshblock_data[2](j, i); //VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));
          prim(IM1,k,j,iu+i) = 0.0; //- prim(IM1,k,j,iu-i+1);
          prim(IM2,k,j,iu+i) = 0.0; //prim(IM2,k,j,iu-i+1);
          prim(IM3,k,j,iu+i) = vel;
          prim(IDN,k,j,iu+i) = den;
          prim(IPR,k,j,iu+i) = SQR(cs)*den;
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,j,iu+i) = SQR(cs)*den;
          if (NDUSTFLUIDS > 0){ 
            Real amean, OmK, den_mid, St_mid, St_loc, den_dust;
            for(int n=0; n<NDUSTFLUIDS; n++){
              // // --------------------------------------------------------------------------------------------
              // // Calculate midplane Stokes numbers
              // // --------------------------------------------------------------------------------------------
              // if(growth)
              //   amax = amax_growth(rad, time);
              // else
              //   amax = a_max;
              // // --------------------------------------------------------------------------------------------
              // // Calculate equilibrium dust density based on Fromang & Nelson (2009), given delta_ini
              // // --------------------------------------------------------------------------------------------
              // amean   = mean_size(n, amax, a_min, q_dust);         // particle size of the bin
              // eps     = eps_bin(n, amax, a_min, eps_ini, q_dust);  // dust to gas ratio of the bin 
              // OmK     = std::pow(rad, -1.5);                       // Keplerian frequency
              // den_mid = DenProfileCyl(rad,phi,0.);                 // midplane gas density
              // St_mid  = Stokes(amean, den_mid, cs, OmK);           // midplane Stokes number
              // St_loc  = Stokes(amean, den, cs, OmK);               // local Stokes number
              // den_dust= DenProfileCyl_dust(rad,phi,z,St_mid,eps);  // dust density based on Fromang & Nelson

              // // --------------------------------------------------------------------------------------------
              // // Calculate Pressure Gradients
              // // --------------------------------------------------------------------------------------------
              // Real dp_r   = (prim(IPR,k,j,iu+i-1) - prim(IPR,k,j,iu+i))/(pmb->pcoord->x1v(iu+i-1) - pmb->pcoord->x1v(iu+i));
              // Real den_dust = DenProfileCyl_dust(rad,phi,z, St_mid, eps_ini);

              // // --------------------------------------------------------------------------------------------
              // // Calculate Nakagawa solution
              // // --------------------------------------------------------------------------------------------
              // Real dv_r   = St_loc*dp_r/(den*std::pow(rad,-1.5)) / (SQR(St_loc) + 1.);
              // Real dv_phi = -0.5*SQR(St_loc)*dp_r/(den*std::pow(rad,-1.5)) / (SQR(St_loc) + 1.);

              rho_id  = 4*n;
              v1_id   = rho_id + 1;
              v2_id   = rho_id + 2;
              v3_id   = rho_id + 3;
              prim_df(rho_id,k,j,iu+i) = 0.0;
              prim_df(v1_id, k,j,iu+i) = 0.0;
              prim_df(v2_id, k,j,iu+i) = 0.0;
              prim_df(v3_id, k,j,iu+i) = vel;
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
  int rho_id, v1_id, v2_id, v3_id;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,jl-j,k);
          den = pmb->ruser_meshblock_data[0](j, i); //DenProfileCyl(rad,phi,z);
          cs  = pmb->ruser_meshblock_data[1](j, i); //SspeedProfileCyl(rad,phi,z);
          vel = pmb->ruser_meshblock_data[2](j, i); //VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(jl-j), pco->x3v(k));
          prim(IM1,k,jl-j,i) = prim(IM1,k,jl+j-1,i); 
          prim(IM2,k,jl-j,i) = -prim(IM2,k,jl+j-1,i); 
          prim(IM3,k,jl-j,i) = vel;
          prim(IDN,k,jl-j,i) = prim(IDN,k,jl+j-1,i);
          prim(IPR,k,jl-j,i) = prim(IPR,k,jl+j-1,i);
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,jl-j,i) = prim(IEN,k,jl+j-1,i);
          if (NDUSTFLUIDS > 0){ 
            Real amean, OmK, den_mid, St_mid, St_loc, den_dust;
            for(int n=0; n<NDUSTFLUIDS; n++){
              // // --------------------------------------------------------------------------------------------
              // // Calculate midplane Stokes numbers
              // // --------------------------------------------------------------------------------------------
              // if(growth)
              //   amax = amax_growth(rad, time);
              // else
              //   amax = a_max;
              // // --------------------------------------------------------------------------------------------
              // // Calculate equilibrium dust density based on Fromang & Nelson (2009), given delta_ini
              // // --------------------------------------------------------------------------------------------
              // amean   = mean_size(n, amax, a_min, q_dust);        // particle size of the bin
              // eps     = eps_bin(n, amax, a_min, eps_ini, q_dust); // dust to gas ratio of the bin 
              // OmK     = std::pow(rad, -1.5);                       // Keplerian frequency
              // den_mid = DenProfileCyl(rad,phi,0.);                 // midplane gas density
              // St_mid  = Stokes(amean, den_mid, cs, OmK);           // midplane Stokes number
              // St_loc  = Stokes(amean, den, cs, OmK);               // local Stokes number
              // den_dust= DenProfileCyl_dust(rad,phi,z,St_mid,eps);  // dust density based on Fromang & Nelson

              // // --------------------------------------------------------------------------------------------
              // // Calculate Pressure Gradients
              // // --------------------------------------------------------------------------------------------
              // Real dp_r = (prim(IPR,k,jl-j,i+1) - prim(IPR,k,jl-j,i))/(pmb->pcoord->x1v(i+1) - pmb->pcoord->x1v(i));
              // Real den_dust = DenProfileCyl_dust(rad,phi,z, St_mid,eps_ini);

              // // --------------------------------------------------------------------------------------------
              // // Calculate Nakagawa solution
              // // --------------------------------------------------------------------------------------------
              // Real dv_r   = St_loc*dp_r/(den*std::pow(rad,-1.5)) / (SQR(St_loc) + 1.);
              // Real dv_phi = -0.5*SQR(St_loc)*dp_r/(den*std::pow(rad,-1.5)) / (SQR(St_loc) + 1.);

              rho_id  = 4*n;
              v1_id   = rho_id + 1;
              v2_id   = rho_id + 2;
              v3_id   = rho_id + 3;
              prim_df(rho_id,k,jl-j,i) = 0.0; //den_dust;
              prim_df(v1_id, k,jl-j,i) = prim_df(v1_id, k,jl+j-1,i); //den_dust>dffloor ? prim(IM1,k,jl-j,i) + dv_r * std::sin(pco->x2v(jl-j)) : 0.0;
              prim_df(v2_id, k,jl-j,i) = -prim_df(v2_id, k,jl+j-1,i); //den_dust>dffloor ? prim(IM2,k,jl-j,i) + dv_r * std::cos(pco->x2v(jl-j)) : 0.0;
              prim_df(v3_id, k,jl-j,i) = vel; //den_dust>dffloor ? prim(IM3,k,jl-j,i) + dv_phi : 0.0;
              
              // prim_df(rho_id,k,jl-j,i) = prim_df(rho_id,k,jl,i);
              // prim_df(v1_id, k,jl-j,i) = prim_df(v1_id, k,jl,i);
              // prim_df(v2_id, k,jl-j,i) = prim_df(v2_id, k,jl,i);
              // prim_df(v3_id, k,jl-j,i) = vel;
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
  int rho_id, v1_id, v2_id, v3_id;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco,rad,phi,z,i,ju+j,k);
          den = pmb->ruser_meshblock_data[0](j, i); //DenProfileCyl(rad,phi,z);
          cs  = pmb->ruser_meshblock_data[1](j, i); //SspeedProfileCyl(rad,phi,z);
          vel = pmb->ruser_meshblock_data[2](j, i); //VelProfileCyl(rad,phi,z);
          if (pmb->porb->orbital_advection_defined)
            vel -= vK(pmb->porb, pco->x1v(i), pco->x2v(ju+j), pco->x3v(k));
          prim(IM1,k,ju+j,i) =  prim(IM1,k,ju-j+1,i); 
          prim(IM2,k,ju+j,i) = -prim(IM2,k,ju-j+1,i); 
          prim(IM3,k,ju+j,i) = vel;
          prim(IDN,k,ju+j,i) =  prim(IDN,k,ju-j+1,i);
          prim(IPR,k,ju+j,i) =  prim(IPR,k,ju-j+1,i);
          if (NON_BAROTROPIC_EOS)
            prim(IEN,k,ju+j,i) = prim(IEN,k,ju-j+1,i);
          if (NDUSTFLUIDS > 0){ 
            Real amean, OmK, den_mid, St_mid, St_loc, den_dust;
            for(int n=0; n<NDUSTFLUIDS; n++){
              // // --------------------------------------------------------------------------------------------
              // // Calculate midplane Stokes numbers
              // // --------------------------------------------------------------------------------------------
              // if(growth)
              //   amax = amax_growth(rad, time);
              // else
              //   amax = a_max;
              // // --------------------------------------------------------------------------------------------
              // // Calculate equilibrium dust density based on Fromang & Nelson (2009), given delta_ini
              // // --------------------------------------------------------------------------------------------
              // amean   = mean_size(n, amax, a_min, q_dust);        // particle size of the bin
              // eps     = eps_bin(n, amax, a_min, eps_ini, q_dust); // dust to gas ratio of the bin 
              // OmK     = std::pow(rad, -1.5);                       // Keplerian frequency
              // den_mid = DenProfileCyl(rad,phi,0.);                 // midplane gas density
              // St_mid  = Stokes(amean, den_mid, cs, OmK);           // midplane Stokes number
              // St_loc  = Stokes(amean, den, cs, OmK);               // local Stokes number
              // den_dust= DenProfileCyl_dust(rad,phi,z,St_mid,eps);  // dust density based on Fromang & Nelson

              // // --------------------------------------------------------------------------------------------
              // // Calculate Pressure Gradients
              // // --------------------------------------------------------------------------------------------
              // Real dp_r = (prim(IPR,k,ju+j,i+1) - prim(IPR,k,ju+j,i))/(pmb->pcoord->x1v(i+1) - pmb->pcoord->x1v(i));
              // Real den_dust = DenProfileCyl_dust(rad,phi,z, St_mid, eps_ini);

              // // --------------------------------------------------------------------------------------------
              // // Calculate Nakagawa solution
              // // --------------------------------------------------------------------------------------------
              // Real dv_r   = St_loc*dp_r/(den*std::pow(rad,-1.5)) / (SQR(St_loc) + 1.);
              // Real dv_phi = -0.5*SQR(St_loc)*dp_r/(den*std::pow(rad,-1.5)) / (SQR(St_loc) + 1.);

              rho_id  = 4*n;
              v1_id   = rho_id + 1;
              v2_id   = rho_id + 2;
              v3_id   = rho_id + 3;
              prim_df(rho_id,k,ju+j,i) = 0.0;
              prim_df(v1_id, k,ju+j,i) = prim_df(v1_id, k,ju-j+1,i); // + dv_r * std::sin(pco->x2v(ju+j)) : 0.0;
              prim_df(v2_id, k,ju+j,i) = -prim_df(v2_id, k,ju-j+1,i); // + dv_r * std::cos(pco->x2v(ju+j)) : 0.0;
              prim_df(v3_id, k,ju+j,i) = vel; //den_dust>dffloor ? prim(IM3,k,ju+j,i) + dv_phi : 0.0;

              // prim_df(rho_id,k,ju+j,i) = prim_df(rho_id,k,ju,i);
              // prim_df(v1_id, k,ju+j,i) = prim_df(v1_id, k,ju,i);
              // prim_df(v2_id, k,ju+j,i) = prim_df(v2_id, k,ju,i);
              // prim_df(v3_id, k,ju+j,i) = vel;
            }
          }
        }
      }
    }
  }
}

void MeshBlock::UserWorkInLoop() {
  if(growth){
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real rad,phi,z,eps;
        GetCylCoord(pcoord, rad,phi,z, i,j,0); // don't need k, since axisymmetric
        pmy_mesh->time;
        eps = 0.;
        for(int n=0; n<NDUSTFLUIDS; ++n){
          eps += ruser_meshblock_data[4+n](j,i)/ruser_meshblock_data[0](j, i);
        }
        ruser_meshblock_data[3](j, i) = amax_growth(rad, ruser_meshblock_data[1](j, i)*unit_vel, eps, pmy_mesh->time);
        ruser_meshblock_data[3+NDUSTFLUIDS+2](j, i) = tcool_dust(i, j, 0, pmy_mesh->time, ruser_meshblock_data[3](j, i), phydro, pcoord, pdustfluids);
      }
    }
  }
  return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin ){
  Real rad,phi,z,trel;
  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      for(int i=is; i<=ie; i++) {
        GetCylCoord(pcoord, rad,phi,z, i,j,k);
        user_out_var(0,k,j,i) = ruser_meshblock_data[0](j, i);
        user_out_var(1,k,j,i) = ruser_meshblock_data[1](j, i);
        user_out_var(2,k,j,i) = ruser_meshblock_data[2](j, i);
        user_out_var(3,k,j,i) = ruser_meshblock_data[3](j, i);
        user_out_var(4,k,j,i) = ruser_meshblock_data[4](j, i);
        user_out_var(5,k,j,i) = ruser_meshblock_data[5](j, i);
        user_out_var(6,k,j,i) = ruser_meshblock_data[6](j, i);
        user_out_var(7,k,j,i) = ruser_meshblock_data[7](j, i);
        user_out_var(8,k,j,i) = ruser_meshblock_data[8](j, i);
        user_out_var(9,k,j,i) = ruser_meshblock_data[9](j, i) * std::pow(rad,-1.5);
      }
    }
  }
}