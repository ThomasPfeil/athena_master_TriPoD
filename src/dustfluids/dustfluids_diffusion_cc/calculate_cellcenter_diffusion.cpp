//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file calculate_cellcenter_diffusion.cpp
//! \brief Compute dust fluids cell centered diffusive fluxes corresponding to concentration diffusion.

// C headers

// C++ headers
#include <algorithm>   // min,max
#include <limits>

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../defs.hpp"
#include "../../mesh/mesh.hpp"
#include "../../parameter_input.hpp"
#include "../dustfluids.hpp"
#include "../dustfluids_diffusion/dustfluids_diffusion.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif


void DustFluidsCellCenterDiffusion::CalculateDiffusiveMomentum(const AthenaArray<Real> &prim_df,
            const AthenaArray<Real> &w) {
  DustFluids *pdf = pmb_->pdustfluids;
  const bool f2 = pmb_->pmy_mesh->f2;
  const bool f3 = pmb_->pmy_mesh->f3;

  int il, iu, jl, ju, kl, ku;
  int is = pmb_->is; int js = pmb_->js; int ks = pmb_->ks;
  int ie = pmb_->ie; int je = pmb_->je; int ke = pmb_->ke;

  // i-direction
  const AthenaArray<Real> &x1flux = pdf->dfdif.dustfluids_diffusion_flux[X1DIR];

  jl = js, ju = je, kl = ks, ku = ke;
  if (f2) {
    if (!f3) // 2D
      jl = js - 1, ju = je + 1, kl = ks, ku = ke;
    else // 3D
      jl = js - 1, ju = je + 1, kl = ks - 1, ku = ke + 1;
  }

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    int rho_id  = 4*dust_id;
    int v1_id   = rho_id + 1;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          diff_mom_cc(v1_id, k, j, i) = 0.5*(x1flux(rho_id, k, j, i) + x1flux(rho_id, k, j, i+1));
        }
      }
    }
  }

  // j-direction
  il = is, iu = ie, kl = ks, ku = ke;
  if (!f3) // 2D
    il = is - 1, iu = ie + 1, kl = ks, ku = ke;
  else // 3D
    il = is - 1, iu = ie + 1, kl = ks - 1, ku = ke + 1;

  if (f2) { // 2D or 3D
    const AthenaArray<Real> &x2flux = pdf->dfdif.dustfluids_diffusion_flux[X2DIR];
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      int dust_id = n;
      int rho_id  = 4*dust_id;
      int v2_id   = rho_id + 2;
      for (int k=kl; k<=ku; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            diff_mom_cc(v2_id, k, j, i) = 0.5*(x2flux(rho_id, k, j, i) + x2flux(rho_id, k, j+1, i));
          }
        }
      }
    } // zero flux for 1D
  }

  // k-direction
  il = is, iu = ie, jl = js, ju = je;
  if (f2) // 2D or 3D
    il = is - 1, iu = ie + 1, jl = js - 1, ju = je + 1;
  else // 1D
    il = is - 1, iu = ie + 1;

  if (f3) { // 3D
    const AthenaArray<Real> &x3flux = pdf->dfdif.dustfluids_diffusion_flux[X3DIR];
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      int dust_id = n;
      int rho_id  = 4*dust_id;
      int v3_id   = rho_id + 3;
      for (int k=ks; k<=ke; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            diff_mom_cc(v3_id, k, j, i) = 0.5*(x3flux(rho_id, k, j, i) + x3flux(rho_id, k+1, j, i));
          }
        }
      }
    } // zero flux for 1D/2D
  }
  return;
}

void DustFluidsCellCenterDiffusion::CalculateDiffusiveMomentum_New(const AthenaArray<Real> &cons_df,
  const AthenaArray<Real> &w) {
    DustFluids *pdf = pmb_->pdustfluids;
    const bool f2   = pmb_->pmy_mesh->f2;
    const bool f3   = pmb_->pmy_mesh->f3;
  
    int il, iu, jl, ju, kl, ku;
    int is = pmb_->is; int js = pmb_->js; int ks = pmb_->ks;
    int ie = pmb_->ie; int je = pmb_->je; int ke = pmb_->ke;
    AthenaArray<Real> &nu_dust = pdf->nu_dustfluids_array;
  
    // i-direction
    jl = js, ju = je, kl = ks, ku = ke;
    if (f2) {
      if (!f3) // 2D
        jl = js - 1, ju = je + 1, kl = ks, ku = ke;
      else // 3D
        jl = js - 1, ju = je + 1, kl = ks - 1, ku = ke + 1;
    }
  
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      int dust_id = n;
      int rho_id  = 4*dust_id;
      int v1_id   = rho_id + 1;
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
  #pragma omp simd
          for (int i=is; i<=ie+1; ++i) {
            Real nu      = nu_dust(dust_id,k,j,i);
            Real gas_rho = w(IDN,k,j,i);
  
            Real dcons_df_dx1 = (cons_df(rho_id,k,j,i+1)/w(IDN,k,j,i+1) - cons_df(rho_id,k,j,i-1)/w(IDN,k,j,i-1))
                                        /(pco_->x1v(i+1)-pco_->x1v(i-1));
            diff_mom_cc(v1_id, k, j, i) = - nu*gas_rho*dcons_df_dx1;
          }
        }
      }
    }
  
    // j-direction
    il = is, iu = ie, kl = ks, ku = ke;
    if (!f3) // 2D
      il = is - 1, iu = ie + 1, kl = ks, ku = ke;
    else // 3D
      il = is - 1, iu = ie + 1, kl = ks - 1, ku = ke + 1;
  
    if (f2) { // 2D or 3D
      for (int n=0; n<NDUSTFLUIDS; ++n) {
        int dust_id = n;
        int rho_id  = 4*dust_id;
        int v2_id   = rho_id + 2;
        for (int k=kl; k<=ku; ++k) {
          for (int j=js; j<=je+1; ++j) {
  #pragma omp simd
            for (int i=il; i<=iu; ++i) {
              Real nu      = nu_dust(dust_id,k,j,i);
              Real gas_rho = w(IDN,k,j,i);
  
              Real dcons_df_dx2 = (cons_df(rho_id,k,j+1,i)/w(IDN,k,j+1,i) - cons_df(rho_id,k,j-1,i)/w(IDN,k,j-1,i))
                                        /pco_->h2v(i)/(pco_->x2v(j+1)-pco_->x2v(j-1));
              diff_mom_cc(v2_id, k, j, i) = - nu*gas_rho*dcons_df_dx2;
            }
          }
        }
      } // zero flux for 1D
    }
  
    // k-direction
    il = is, iu = ie, jl = js, ju = je;
    if (f2) // 2D or 3D
      il = is - 1, iu = ie + 1, jl = js - 1, ju = je + 1;
    else // 1D
      il = is - 1, iu = ie + 1;
  
    if (f3) { // 3D
      for (int n=0; n<NDUSTFLUIDS; ++n) {
        int dust_id = n;
        int rho_id  = 4*dust_id;
        int v3_id   = rho_id + 3;
        for (int k=ks; k<=ke+1; ++k) {
          for (int j=jl; j<=ju; ++j) {
  #pragma omp simd
            for (int i=il; i<=iu; ++i) {
              Real nu      = nu_dust(dust_id,k,j,i);
              Real gas_rho = w(IDN,k,j,i);
  
              Real dcons_df_dx3 = (cons_df(rho_id,k+1,j,i)/w(IDN,k+1,j,i) - cons_df(rho_id,k-1,j,i)/w(IDN,k-1,j,i))
                                        /((pco_->x3v(k+1)-pco_->x3v(k-1))/pco_->h31v(i)/pco_->h32v(j));
              diff_mom_cc(v3_id, k, j, i) = - nu*gas_rho*dcons_df_dx3;
            }
          }
        }
      } // zero flux for 1D/2D
    }
    return;
}