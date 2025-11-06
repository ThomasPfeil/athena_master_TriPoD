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

namespace {
void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &stopping_time, AthenaArray<Real> &nu_dust,
      AthenaArray<Real> &cs_dust, int is, int ie, int js, int je, int ks, int ke);
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time_array);
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);

// problem parameters which are useful to make global to this file
bool gauss_dust, gauss_gas, vx0_flag;
Real dust_diff, rho0_gas, vx0_gas;
Real Stokes_number[NDUSTFLUIDS];
Real amp_dust[NDUSTFLUIDS], sigma_dust[NDUSTFLUIDS], xc_dust[NDUSTFLUIDS], vx0_dust[NDUSTFLUIDS];
} // namespace

// User-defined boundary conditions for disk simulations
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  rho0_gas   = pin->GetOrAddReal("problem", "rho0_gas", 1.0);
  vx0_gas    = pin->GetOrAddReal("problem", "vx0_gas", 1.0);

  // Dust to gas ratio && dust stopping time
  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      Stokes_number[n] = pin->GetReal("dust", "Stokes_number_" + std::to_string(n+1));
      amp_dust[n] = pin->GetReal("dust", "amp_dust_" + std::to_string(n+1));
      sigma_dust[n] = pin->GetReal("dust", "sigma_dust_" + std::to_string(n+1));
      xc_dust[n] = pin->GetReal("dust", "xc_dust_" + std::to_string(n+1));
      vx0_dust[n] = pin->GetReal("dust", "vx0_dust_" + std::to_string(n+1));
    }
  }
  dust_diff = pin->GetOrAddReal("dust", "dust_diff", 0.0);

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
  return;
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{
  AllocateUserOutputVariables(11);
  SetUserOutputVariableName(0, "dif_flx_rho_x1");
  SetUserOutputVariableName(1, "dif_flx_vx1_x1");
  SetUserOutputVariableName(2, "dif_flx_vx2_x1");
  SetUserOutputVariableName(3, "dif_flx_vx3_x1");
  SetUserOutputVariableName(4, "dif_flx_rho_x2");
  SetUserOutputVariableName(5, "dif_flx_vx1_x2");
  SetUserOutputVariableName(6, "dif_flx_vx2_x2");
  SetUserOutputVariableName(7, "dif_flx_vx3_x2");
  SetUserOutputVariableName(8, "dif_mom_x1");
  SetUserOutputVariableName(9, "dif_mom_x2");
  SetUserOutputVariableName(10,"dif_mom_x3");
  return;
}


//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  srand(221094);
  Real rad(0.0), phi(0.0), z(0.0);
  Real den, vel, cs, eps, depsdx;
  Real x1, x2, x3;

  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;
  //  Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    x3 = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      x2 = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {

        x1 = pcoord->x1v(i);

        den = rho0_gas;
        phydro->u(IDN,k,j,i) = den;

        // assign initial conditions for momenta
        phydro->u(IM1,k,j,i) = den * vx0_gas;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;

        if (NDUSTFLUIDS > 0) {
            for(int n=0; n<NDUSTFLUIDS; ++n){
              // assign initial conditions for dust fluid
              int rho_id  = 4*n;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;
              pdustfluids->df_cons(rho_id, k, j, i) = phydro->u(IDN,k,j,i) * (1. + amp_dust[n] * std::exp(-0.5*SQR((x1-xc_dust[n])/sigma_dust[n])));

              pdustfluids->df_cons(v1_id,  k, j, i) = vx0_dust[n] * pdustfluids->df_cons(rho_id, k, j, i);
              pdustfluids->df_cons(v2_id,  k, j, i) = 0.0;
              pdustfluids->df_cons(v3_id,  k, j, i) = 0.0;
            }
          }
        }
      }
    } 
  return;
}

namespace {

void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time) {
  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real &st_time = stopping_time(dust_id, k, j, i);
          //Constant Stokes number in disk problems
          st_time = Stokes_number[dust_id];
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
    for (int n=0; n<NDUSTFLUIDS; n++) {
      int dust_id = n;
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            Real &diffusivity = nu_dust(dust_id, k, j, i);
            diffusivity       = dust_diff;

            Real &soundspeed  = cs_dust(dust_id, k, j, i);
            soundspeed        = std::sqrt(diffusivity);
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
}
} //namespace

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values
//----------------------------------------------------------------------------------------
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  int dust_id, rho_id, v1_id, v2_id, v3_id;
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        prim(IM1,k,j,il-i) = prim(IM1,k,j,il);
        prim(IM2,k,j,il-i) = prim(IM2,k,j,il);
        prim(IM3,k,j,il-i) = prim(IM3,k,j,il);
        prim(IDN,k,j,il-i) = prim(IDN,k,j,il);
        prim(IPR,k,j,il-i) = prim(IPR,k,j,il);

        if (NDUSTFLUIDS > 0){
          for(int n=0; n<NDUSTFLUIDS; ++n){
            rho_id  = 4*n;
            v1_id   = rho_id + 1;
            v2_id   = rho_id + 2;
            v3_id   = rho_id + 3;
            prim_df(rho_id,k,j,il-i) = prim_df(rho_id,k,j,il);
            prim_df(v1_id,k,j,il-i)  = prim_df(v1_id,k,j,il);
            prim_df(v2_id,k,j,il-i)  = prim_df(v2_id,k,j,il);
            prim_df(v3_id,k,j,il-i)  = prim_df(v3_id,k,j,il);
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
    int rho_id, v1_id, v2_id, v3_id;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,iu+i) = prim(IDN,k,j,iu);
        prim(IPR,k,j,iu+i) = prim(IPR,k,j,iu);
        prim(IM1,k,j,iu+i) = prim(IM1,k,j,iu);
        prim(IM2,k,j,iu+i) = prim(IM2,k,j,iu);
        prim(IM3,k,j,iu+i) = prim(IM3,k,j,iu);
        
        for(int n=0; n<NDUSTFLUIDS; ++n){
          rho_id  = 4*n;
          v1_id   = rho_id + 1;
          v2_id   = rho_id + 2;
          v3_id   = rho_id + 3;
          prim_df(rho_id,k,j,iu+i) = prim_df(rho_id,k,j,iu);
          prim_df(v1_id,k,j,iu+i)  = prim_df(v1_id,k,j,iu);
          prim_df(v2_id,k,j,iu+i)  = prim_df(v2_id,k,j,iu);
          prim_df(v3_id,k,j,iu+i)  = prim_df(v3_id,k,j,iu);
        }
      }
    }
  }
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin ){
  Real rad,phi,z;
  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      for(int i=is; i<=ie; i++) {
        // Assign full dust diffusion flux in x1
        Real &dif_flux_rho_x1   = user_out_var(0, k, j, i);
        dif_flux_rho_x1         = pdustfluids->dfdif.dustfluids_diffusion_flux[X1DIR](0, k, j, i);
        Real &dif_flux_vx1_x1   = user_out_var(1, k, j, i);
        dif_flux_vx1_x1         = pdustfluids->dfdif.dustfluids_diffusion_flux[X1DIR](1, k, j, i);
        Real &dif_flux_vx2_x1   = user_out_var(2, k, j, i);
        dif_flux_vx2_x1         = pdustfluids->dfdif.dustfluids_diffusion_flux[X1DIR](2, k, j, i);
        Real &dif_flux_vx3_x1   = user_out_var(3, k, j, i);
        dif_flux_vx3_x1         = pdustfluids->dfdif.dustfluids_diffusion_flux[X1DIR](3, k, j, i);

        // Assign full dust diffusion flux in x2
        Real &dif_flux_rho_x2   = user_out_var(4, k, j, i);
        dif_flux_rho_x2         = pdustfluids->dfdif.dustfluids_diffusion_flux[X2DIR](0, k, j, i);
        Real &dif_flux_vx1_x2   = user_out_var(5, k, j, i);
        dif_flux_vx1_x2         = pdustfluids->dfdif.dustfluids_diffusion_flux[X2DIR](1, k, j, i);
        Real &dif_flux_vx2_x2   = user_out_var(6, k, j, i);
        dif_flux_vx2_x2         = pdustfluids->dfdif.dustfluids_diffusion_flux[X2DIR](2, k, j, i);
        Real &dif_flux_vx3_x2   = user_out_var(7, k, j, i);
        dif_flux_vx3_x2         = pdustfluids->dfdif.dustfluids_diffusion_flux[X2DIR](3, k, j, i);
        
        // Assign cel centered diffusive momenta
        Real &dif_ccflux_x1 = user_out_var(8, k, j, i);
        dif_ccflux_x1       = pdustfluids->dfccdif.diff_mom_cc(1, k, j, i);
        Real &dif_ccflux_x2 = user_out_var(9, k, j, i);
        dif_ccflux_x2       = pdustfluids->dfccdif.diff_mom_cc(2, k, j, i);
        Real &dif_ccflux_x3 = user_out_var(10, k, j, i);
        dif_ccflux_x3       = pdustfluids->dfccdif.diff_mom_cc(3, k, j, i);
      }
    }
  }
}