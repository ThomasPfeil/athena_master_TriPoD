//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file scalar_diffusion.cpp
//! \brief Compute passive scalar fluxes corresponding to diffusion processes.

// C headers

// C++ headers
#include <algorithm>   // min,max
#include <limits>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "scalars.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn void PassiveScalars::AddDiffusionFluxes
//! \brief
//!
//! \note
//! Currently, no need to have 2x sets of wrapper fns like:
//! Hydro::AddDiffusionFluxes()
//! +
//! 2x HydroDiffusion::AddDiffusion*Flux(), FieldDiffusion::AddPoyntingFlux

void PassiveScalars::AddDiffusionFluxes() {
  if (scalar_diffusion_defined) {
    // if (nu_scalar_iso > 0.0 || nu_scalar_aniso > 0.0)
    // AddDiffusionFlux(diffusion_flx, flux);

    // TODO(felker): copied wholesale from HydroDiffusion::AddDiffusionFlux, see notes
    int size1 = s_flux[X1DIR].GetSize();
#pragma omp simd
    for (int i=0; i<size1; ++i)
      s_flux[X1DIR](i) += diffusion_flx[X1DIR](i);

    if (pmy_block->pmy_mesh->f2) {
      int size2 = s_flux[X2DIR].GetSize();
#pragma omp simd
      for (int i=0; i<size2; ++i)
        s_flux[X2DIR](i) += diffusion_flx[X2DIR](i);
    }
    if (pmy_block->pmy_mesh->f3) {
      int size3 = s_flux[X3DIR].GetSize();
#pragma omp simd
      for (int i=0; i<size3; ++i)
        s_flux[X3DIR](i) += diffusion_flx[X3DIR](i);
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PassiveScalars::DiffusiveFluxIso
//! \brief

void PassiveScalars::DiffusiveFluxIso(const AthenaArray<Real> &prim_r,
                                      const AthenaArray<Real> &w,
                                      AthenaArray<Real> *flx_out) {
  MeshBlock *pmb = pmy_block;
  Coordinates *pco = pmb->pcoord;
  const bool f2 = pmb->pmy_mesh->f2;
  const bool f3 = pmb->pmy_mesh->f3;
  AthenaArray<Real> &x1flux = flx_out[X1DIR];
  int il, iu, jl, ju, kl, ku;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  Real nu_face, rho_face, rhod_face, pr_face, inv_OmK, dprim_r_dx, dprim_r_dy, dprim_r_dz, St1_face, rad_face, scalar_face, F, Fmax, lam, Chi, nu_face_;

  // i-direction
  jl = js, ju = je, kl = ks, ku = ke;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (f2) {
      if (!f3) // 2D
        jl = js - 1, ju = je + 1, kl = ks, ku = ke;
      else // 3D
        jl = js - 1, ju = je + 1, kl = ks - 1, ku = ke + 1;
    }
  }
  for (int n=0; n<NSCALARS; n++) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=is; i<=ie+1; ++i) {
          // nu_face = nu_scalar_iso[n];
          // rho_face = 0.5*(w(IDN,k,j,i) + w(IDN,k,j,i-1));
          // int idf = (n==1) ? 0 : 4;
          // rhod_face = 0.5*(pmb->pdustfluids->df_prim(idf,k,j,i) + pmb->pdustfluids->df_prim(idf,k,j,i-1));
          // pr_face = 0.5*(w(IPR,k,j,i) + w(IPR,k,j,i-1));
          // inv_OmK = std::pow(pco->x1f(i), 1.5);
          // nu_face *= pr_face/rho_face * inv_OmK; 
          // dprim_r_dx = (prim_r(n,k,j,i) - prim_r(n,k,j,i-1))/pco->dx1v(i-1);
          // x1flux(n,k,j,i) -= nu_face*rhod_face*dprim_r_dx;

          int idf   = (n==1) ? 0 : 4;
          int nu_id = (n==1) ? 0 : 1;
          nu_face   = nu_scalar_iso[n];

          rho_face    = 0.5*(w(IDN,k,j,i) + w(IDN,k,j,i-1));
          rhod_face   = 0.5*(pmb->pdustfluids->df_prim(idf,k,j,i) + pmb->pdustfluids->df_prim(idf,k,j,i-1));
          pr_face     = 0.5*(w(IPR,k,j,i) + w(IPR,k,j,i-1));
          scalar_face = 0.5*(prim_r(n,k,j,i) + prim_r(n,k,j,i-1));
          nu_face     = 0.5*(pmb->pdustfluids->nu_dustfluids_array(nu_id,k,j,i) + pmb->pdustfluids->nu_dustfluids_array(nu_id,k,j,i-1));
          dprim_r_dx  = (prim_r(n,k,j,i) - prim_r(n,k,j,i-1))/pco->dx1v(i-1);

          Fmax = std::sqrt(nu_scalar_iso[n]*pr_face/rho_face) * rhod_face*scalar_face;
          F    = nu_face*rhod_face*dprim_r_dx;
          Chi  = std::fabs(F/(Fmax+TINY_NUMBER));
          lam  = (1+Chi)/(1+Chi+SQR(Chi));

          x1flux(n,k,j,i) -= lam*F;
        }
      }
    }
  }

  // j-direction
  il = is, iu = ie, kl = ks, ku = ke;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (!f3) // 2D
      il = is - 1, iu = ie + 1, kl = ks, ku = ke;
    else // 3D
      il = is - 1, iu = ie + 1, kl = ks - 1, ku = ke + 1;
  }
  if (f2) { // 2D or 3D
    AthenaArray<Real> &x2flux = flx_out[X2DIR];
    for (int n=0; n<NSCALARS; n++) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=js; j<=je+1; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            // nu_face = nu_scalar_iso[n];
            // rho_face = 0.5*(w(IDN,k,j,i) + w(IDN,k,j-1,i));
            // int idf = (n==1) ? 0 : 4;
            // rhod_face = 0.5*(pmb->pdustfluids->df_prim(idf,k,j,i) + pmb->pdustfluids->df_prim(idf,k,j-1,i));
            // pr_face = 0.5*(w(IPR,k,j,i) + w(IPR,k,j-1,i));
            // inv_OmK = std::pow(pco->x1v(i), 1.5);
            // nu_face *= pr_face/rho_face * inv_OmK; 
            // dprim_r_dy = (prim_r(n,k,j,i) - prim_r(n,k,j-1,i))/pco->h2v(i)/pco->dx2v(j-1);
            // x2flux(n,k,j,i) -= nu_face*rhod_face*dprim_r_dy;

            int idf   = (n==1) ? 0 : 4;
            int nu_id = (n==1) ? 0 : 1;
            nu_face   = nu_scalar_iso[n];

            rho_face    = 0.5*(w(IDN,k,j,i) + w(IDN,k,j-1,i));
            rhod_face   = 0.5*(pmb->pdustfluids->df_prim(idf,k,j,i) + pmb->pdustfluids->df_prim(idf,k,j-1,i));
            pr_face     = 0.5*(w(IPR,k,j,i) + w(IPR,k,j-1,i));
            scalar_face = 0.5*(prim_r(n,k,j,i) + prim_r(n,k,j-1,i));
            nu_face     = 0.5*(pmb->pdustfluids->nu_dustfluids_array(nu_id,k,j,i) + pmb->pdustfluids->nu_dustfluids_array(nu_id,k,j-1,i));
            dprim_r_dy  = (prim_r(n,k,j,i) - prim_r(n,k,j-1,i))/pco->h2v(i)/pco->dx2v(j-1);

            Fmax = std::sqrt(nu_scalar_iso[n]*pr_face/rho_face) * rhod_face*scalar_face;
            F    = nu_face*rhod_face*dprim_r_dy;
            Chi  = std::fabs(F/(Fmax+TINY_NUMBER));
            lam  = (1+Chi)/(1+Chi+SQR(Chi));

            x2flux(n,k,j,i) -= lam*F; // flux = D*rhod * grad(c)
          }
        }
      }
    } // zero flux for 1D
  }

  // k-direction
  il = is, iu = ie, jl = js, ju = je;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (f2) // 2D or 3D
      il = is - 1, iu = ie + 1, jl = js - 1, ju = je + 1;
    else // 1D
      il = is - 1, iu = ie + 1;
  }
  if (f3) { // 3D
    AthenaArray<Real> &x3flux = flx_out[X3DIR];
    for (int n=0; n<NSCALARS; n++) {
      for (int k=ks; k<=ke+1; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            // nu_face = nu_scalar_iso[n];
            // rho_face = 0.5*(w(IDN,k,j,i) + w(IDN,k-1,j,i));
            // int idf = (n==1) ? 0 : 4;
            // rhod_face = 0.5*(pmb->pdustfluids->df_prim(idf,k,j,i) + pmb->pdustfluids->df_prim(idf,k-1,j,i));
            // pr_face = 0.5*(w(IPR,k,j,i) + w(IPR,k-1,j,i));
            // inv_OmK = std::pow(pco->x1v(i), 1.5);
            // nu_face *= pr_face/rho_face * inv_OmK; 
            // dprim_r_dz = (prim_r(n,k,j,i) - prim_r(n,k-1,j,i))/pco->dx3v(k-1)/pco->h31v(i)
            //              /pco->h32v(j);
            // x3flux(n,k,j,i) -= nu_face*rhod_face*dprim_r_dz;
            
            int idf   = (n==1) ? 0 : 4;
            int nu_id = (n==1) ? 0 : 1;
            nu_face = nu_scalar_iso[n];

            rho_face  = 0.5*(w(IDN,k,j,i) + w(IDN,k-1,j,i));
            rhod_face = 0.5*(pmb->pdustfluids->df_prim(idf,k,j,i) + pmb->pdustfluids->df_prim(idf,k-1,j,i));
            pr_face   = 0.5*(w(IPR,k,j,i) + w(IPR,k-1,j,i));
            scalar_face = 0.5*(prim_r(n,k,j,i) + prim_r(n,k-1,j,i));
            nu_face   = 0.5*(pmb->pdustfluids->nu_dustfluids_array(nu_id,k,j,i) + pmb->pdustfluids->nu_dustfluids_array(nu_id,k-1,j,i));
            dprim_r_dz = (prim_r(n,k,j,i) - prim_r(n,k-1,j,i))/pco->dx3v(k-1)/pco->h31v(i)/pco->h32v(j);

            Fmax = std::sqrt(nu_scalar_iso[n]*pr_face/rho_face) * rhod_face*scalar_face;
            F    = nu_face*rhod_face*dprim_r_dz;
            Chi  = std::fabs(F/(Fmax+TINY_NUMBER));
            lam  = (1+Chi)/(1+Chi+SQR(Chi));

            x3flux(n,k,j,i) -= lam*F;
          }
        }
      }
    } // zero flux for 1D/2D
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PassiveScalars::NewDiffusionDt
//! \brief

Real PassiveScalars::NewDiffusionDt() {
  Real real_max = std::numeric_limits<Real>::max();
  MeshBlock *pmb = pmy_block;
  const bool f2 = pmb->pmy_mesh->f2;
  const bool f3 = pmb->pmy_mesh->f3;
  AthenaArray<Real> &w = pmb->phydro->w;
  int il = pmb->is - NGHOST; int jl = pmb->js; int kl = pmb->ks;
  int iu = pmb->ie + NGHOST; int ju = pmb->je; int ku = pmb->ke;
  Real fac;
  if (f3)
    fac = 1.0/6.0;
  else if (f2)
    fac = 0.25;
  else
    fac = 0.5;

  Real dt_diff = real_max;
  // Commented-out future extensions: local diffusion coefficients, anisotropic diffusion
  // for passive scalars:
  // AthenaArray<Real> &nu_scalar_t = nu_scalar_tot_;
  AthenaArray<Real> &len = dx1_, &dx2 = dx2_, &dx3 = dx3_;
  Real rho_face, pr_face, inv_OmK, nu_face;

  for (int n=0; n<NSCALARS; ++n) {
		for (int k=kl; k<=ku; ++k) {
			for (int j=jl; j<=ju; ++j) {
				// #pragma omp simd
	//       for (int i=il; i<=iu; ++i) {
	//         nu_scalar_t(i) = 0.0;
	//       }
	//       if (nu_scalar_iso > 0.0) {
	// #pragma omp simd
	//         for (int i=il; i<=iu; ++i) nu_scalar_t(i) += nu(DiffProcess::iso,k,j,i);
	//       }
	//       if (nu_scalar_aniso > 0.0) {
	// #pragma omp simd
	//         for (int i=il; i<=iu; ++i) nu_scalar_t(i) += nu(DiffProcess::aniso,k,j,i);
	//       }
				pmb->pcoord->CenterWidth1(k, j, il, iu, len);
				pmb->pcoord->CenterWidth2(k, j, il, iu, dx2);
				pmb->pcoord->CenterWidth3(k, j, il, iu, dx3);
#pragma omp simd
				for (int i=il; i<=iu; ++i) {
					len(i) = (f2) ? std::min(len(i), dx2(i)) : len(i);
					len(i) = (f3) ? std::min(len(i), dx3(i)) : len(i);
				}
				if (nu_scalar_iso[n] > 0.0) { // || (nu_scalar_aniso > 0.0)) {
					for (int i=il; i<=iu; ++i) {
            nu_face  = nu_scalar_iso[n];
            rho_face = 0.5*(w(IDN,k,j,i) + w(IDN,k,j,i-1));
            pr_face  = 0.5*(w(IPR,k,j,i) + w(IPR,k,j,i-1));
            inv_OmK  = std::pow(pmb->pcoord->x1f(i), 1.5);
            nu_face  *= pr_face/rho_face * inv_OmK; 
						dt_diff  = std::min(dt_diff, static_cast<Real>(
								SQR(len(i))*fac/(nu_face + TINY_NUMBER)));
								// /(nu_scalar_t(i) + TINY_NUMBER)));
					}
				}
			}
		}
	}
  return dt_diff;
}
