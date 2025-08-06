//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file momentum_diffusion.cpp
//! \brief Compute the correction of momentum diffusion due to the artificial turubulence concentration diffusion.

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
#include "dustfluids_diffusion.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

// Van Leer flux limiter
Real DustFluidsDiffusion::VanLeerLimiter(const Real a, const Real b) {
  Real c = a * b;
  return (c > 0.0) ? (2.0*c)/(a+b) : 0.0;
}


void DustFluidsDiffusion::DustFluidsMomentumDiffusiveFlux(const AthenaArray<Real> &prim_df,
            const AthenaArray<Real> &w, AthenaArray<Real> *df_diff_flux) {

  DustFluids *pdf = pmb_->pdustfluids;
  const bool f2   = pmb_->pmy_mesh->f2;
  const bool f3   = pmb_->pmy_mesh->f3;

  // rho_id: concentration diffusive flux; v1_id, v2_id, v3_id: momentum diffusive flux
  AthenaArray<Real> &x1flux = df_diff_flux[X1DIR];
  AthenaArray<Real> &x2flux = df_diff_flux[X2DIR];
  AthenaArray<Real> &x3flux = df_diff_flux[X3DIR];

  AthenaArray<Real> &x1flux_adv = pdf->df_flux[X1DIR];
  AthenaArray<Real> &x2flux_adv = pdf->df_flux[X2DIR];
  AthenaArray<Real> &x3flux_adv = pdf->df_flux[X3DIR];

  int il, iu, jl, ju, kl, ku;
  int is = pmb_->is; int js = pmb_->js; int ks = pmb_->ks;
  int ie = pmb_->ie; int je = pmb_->je; int ke = pmb_->ke;

  // i-direction
  jl = js, ju = je, kl = ks, ku = ke;
  if (f2) {
    if (!f3) // 2D
      jl = js - 1, ju = je + 1, kl = ks, ku = ke;
    else // 3D
      jl = js - 1, ju = je + 1, kl = ks - 1, ku = ke + 1;
  }

  pdf->dfccdif.diff_mom_cc.ZeroClear();
  pdf->dfccdif.CalculateDiffusiveMomentum_New(prim_df, w);

  // i-direction loop
  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    int rho_id  = 4*dust_id;
    int v1_id   = rho_id + 1;
    int v2_id   = rho_id + 2;
    int v3_id   = rho_id + 3;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=is; i<=ie+1; ++i) {
          int di; Real same_sign;

          // Diffusion flux transports advective momenta
          (x1flux(rho_id,k,j,i) >= 0.0) ? di = 1 : di = 0;
          x1flux(v1_id,k,j,i) += x1flux(rho_id,k,j,i)*prim_df(v1_id,k,j,i-di);
          x1flux(v2_id,k,j,i) += x1flux(rho_id,k,j,i)*prim_df(v2_id,k,j,i-di);
          x1flux(v3_id,k,j,i) += x1flux(rho_id,k,j,i)*prim_df(v3_id,k,j,i-di);
          
          // Advective flux transports diffusive momenta (calculate diffusive velocity at upwind cell center)
          (prim_df(v1_id,k,j,i-1) * prim_df(v1_id,k,j,i) >= 0.0) ? same_sign = 1.0 : same_sign = 0.0;
          //((prim_df(v1_id,k,j,i-1) >= 0.0) && (prim_df(v1_id,k,j,i) >= 0.0)) ? di = 1 : di = 0;
          ((prim_df(v1_id,k,j,i-1) + prim_df(v1_id,k,j,i)) >= 0.0) ? di = 1 : di = 0;

          x1flux(v1_id,k,j,i) += same_sign*prim_df(v1_id,k,j,i)*x1flux(rho_id,k,j,i-di);
          x1flux(v2_id,k,j,i) += same_sign*prim_df(v1_id,k,j,i)*VanLeerLimiter(x2flux(rho_id,k,j+1,i-di), x2flux(rho_id,k,j,i-di));
          x1flux(v3_id,k,j,i) += same_sign*prim_df(v1_id,k,j,i)*VanLeerLimiter(x3flux(rho_id,k+1,j,i-di), x3flux(rho_id,k,j,i-di));

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
      int v1_id   = rho_id + 1;
      int v2_id   = rho_id + 2;
      int v3_id   = rho_id + 3;
      for (int k=kl; k<=ku; ++k) {
        for (int j=js; j<=je+1; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            int dj; Real same_sign;

            (x2flux(rho_id,k,j,i) >= 0.0) ? dj = 1 : dj = 0;
            x2flux(v1_id,k,j,i) += x2flux(rho_id,k,j,i)*prim_df(v1_id,k,j-dj,i);
            x2flux(v2_id,k,j,i) += x2flux(rho_id,k,j,i)*prim_df(v2_id,k,j-dj,i);
            x2flux(v3_id,k,j,i) += x2flux(rho_id,k,j,i)*prim_df(v3_id,k,j-dj,i);

            Real v2_int = 0.5*(prim_df(v2_id,k,j-1,i) + prim_df(v2_id,k,j,i));
            (v2_int >= 0.0) ? dj = 1 : dj = 0;
            same_sign = (prim_df(v2_id,k,j-1,i) * prim_df(v2_id,k,j,i) > 0.0) ? 1 : 0;
            Real mom1_diff = pdf->dfccdif.diff_mom_cc(v1_id, k,j-dj,i); //VanLeerLimiter(x1flux(rho_id,k,j-dj,i+1), x1flux(rho_id,k,j,i)) / prim_df(rho_id,k,j-dj,i);
            Real mom2_diff = pdf->dfccdif.diff_mom_cc(v2_id, k,j-dj,i); //VanLeerLimiter(x2flux(rho_id,k,j-dj+1,i), x2flux(rho_id,k,j,i)) / prim_df(rho_id,k,j-dj,i);
            Real mom3_diff = pdf->dfccdif.diff_mom_cc(v3_id, k,j-dj,i); //VanLeerLimiter(x3flux(rho_id,k+1,j-dj,i), x3flux(rho_id,k,j,i)) / prim_df(rho_id,k,j-dj,i);

            x2flux(v1_id,k,j,i) += same_sign * v2_int * mom1_diff;
            x2flux(v2_id,k,j,i) += same_sign * v2_int * mom2_diff;
            x2flux(v3_id,k,j,i) += same_sign * v2_int * mom3_diff;
          }
        }
      }
    }
  } // zero flux for 1D

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
      int v1_id   = rho_id + 1;
      int v2_id   = rho_id + 2;
      int v3_id   = rho_id + 3;
      for (int k=ks; k<=ke+1; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            int dk; Real same_sign;

            (x3flux(rho_id,k,j,i) >= 0.0) ? dk = 1 : dk = 0;
            x3flux(v1_id,k,j,i) += x3flux(rho_id,k,j,i)*prim_df(v1_id,k-dk,j,i);
            x3flux(v2_id,k,j,i) += x3flux(rho_id,k,j,i)*prim_df(v2_id,k-dk,j,i);
            x3flux(v3_id,k,j,i) += x3flux(rho_id,k,j,i)*prim_df(v3_id,k-dk,j,i);

            // Real v3_int = 0.5*(prim_df(v3_id,k-1,j,i) + prim_df(v3_id,k,j,i));
            // (v3_int >= 0.0) ? dk = 1 : dk = 0;
            // same_sign = (prim_df(v3_id,k-1,j,i) * prim_df(v3_id,k,j,i) > 0.0) ? 1 : 0;
            // Real mom1_diff = pdf->dfccdif.diff_mom_cc(v1_id, k-dk,j,i); //VanLeerLimiter(x1flux(rho_id,k-dk,j,i+1), x1flux(rho_id,k,j,i)) / prim_df(rho_id,k-dk,j,i);
            // Real mom2_diff = pdf->dfccdif.diff_mom_cc(v2_id, k-dk,j,i); //VanLeerLimiter(x2flux(rho_id,k-dk,j+1,i), x2flux(rho_id,k,j,i)) / prim_df(rho_id,k-dk,j,i);
            // Real mom3_diff = pdf->dfccdif.diff_mom_cc(v3_id, k-dk,j,i); //VanLeerLimiter(x3flux(rho_id,k-dk+1,j,i), x3flux(rho_id,k,j,i)) / prim_df(rho_id,k-dk,j,i);

            // x3flux(v1_id,k,j,i) += same_sign * v3_int * mom1_diff;
            // x3flux(v2_id,k,j,i) += same_sign * v3_int * mom2_diff;
            // x3flux(v3_id,k,j,i) += same_sign * v3_int * mom3_diff;
          }
        }
      }
    }
  } // zero flux for 1D/2D
  return;
}
