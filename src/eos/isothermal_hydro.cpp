//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file isothermal_hydro.cpp
//! \brief implements functions in class EquationOfState for isothermal hydrodynamics`

// C headers

// C++ headers
#include <cmath>   // sqrt()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "eos.hpp"

// EquationOfState constructor

EquationOfState::EquationOfState(MeshBlock *pmb, ParameterInput *pin) :
    pmy_block_(pmb),
    neighbor_flooring_{pin->GetOrAddBoolean("hydro", "neighbor_flooring", false)},
    iso_sound_speed_{pin->GetReal("hydro", "iso_sound_speed")},  // error if missing!
    density_floor_{pin->GetOrAddReal("hydro", "dfloor", std::sqrt(1024*float_min) )},
    scalar_floor_{pin->GetOrAddReal("hydro", "sfloor", std::sqrt(1024*float_min))} {


    for (int n=0; n<NDUSTFLUIDS; ++n)
      dustfluids_floor_[n] = pin->GetOrAddReal("dust", "dffloor_"
                           + std::to_string(n+1), std::sqrt(1024*float_min));


      if (pmb->phydro->fofc_enabled)
        fofc_.NewAthenaArray(pmb->ncells3, pmb->ncells2, pmb->ncells1);
      if (neighbor_flooring_)
        nbavg_d_.NewAthenaArray(pmb->ncells3, pmb->ncells2, pmb->ncells1);

    }

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::ConservedToPrimitive(AthenaArray<Real> &cons,
//!           const AthenaArray<Real> &prim_old, const FaceField &b,
//!           AthenaArray<Real> &prim, AthenaArray<Real> &bcc, Coordinates *pco,
//!           int il, int iu, int jl, int ju, int kl, int ku)
void EquationOfState::ConservedToPrimitive(
    AthenaArray<Real> &cons, const AthenaArray<Real> &prim_old, const FaceField &b,
    AthenaArray<Real> &prim, AthenaArray<Real> &bcc,
    Coordinates *pco, int il, int iu, int jl, int ju, int kl, int ku) {
  int nbad_d = 0;
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=il; i<=iu; ++i) {
        Real u_d  = cons(IDN,k,j,i);
        Real u_m1 = cons(IM1,k,j,i);
        Real u_m2 = cons(IM2,k,j,i);
        Real u_m3 = cons(IM3,k,j,i);

        Real w_d, w_vx, w_vy, w_vz;
        bool dfloor_used = false, pfloor_used = false;
        Real u_e, w_p, dp; // garbage
        SingleConservativeToPrimitiveHydro(u_d, u_m1, u_m2, u_m3, u_e,
                                           w_d, w_vx, w_vy, w_vz, w_p,
                                           dp, dfloor_used, pfloor_used);

        // update counter
        if (neighbor_flooring_) nbavg_d_(k,j,i) = dfloor_used;

        if (dfloor_used) {
          cons(IDN,k,j,i) = u_d;
          nbad_d++;
        }

        // update primitives
        prim(IDN,k,j,i) = w_d;
        prim(IVX,k,j,i) = w_vx;
        prim(IVY,k,j,i) = w_vy;
        prim(IVZ,k,j,i) = w_vz;
      }
    }
  }

  // apply neighbor averaging
  if (neighbor_flooring_) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          if (nbavg_d_(k,j,i)) {
            AthenaArray<Real> cons_avg(NHYDRO), prim_avg(NHYDRO);
            NeighborAveragingConserved(cons,bcc,cons_avg,prim_avg,
                                       k,j,i,il,iu,jl,ju,kl,ku);
            for (int n=0; n<NHYDRO; ++n) {
              cons(n,k,j,i) = cons_avg(n);
              prim(n,k,j,i) = prim_avg(n);
            }
          }
        }
      }
    }
  }

  // updated number of bad cells in the mesh block
  // to be used elsewhere for diagnosing purposes
  pmy_block_->nbad_d = nbad_d;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::PrimitiveToConserved(const AthenaArray<Real> &prim,
//!           const AthenaArray<Real> &bc, AthenaArray<Real> &cons, Coordinates *pco,
//!           int il, int iu, int jl, int ju, int kl, int ku);
//! \brief Converts primitive variables into conservative variables

void EquationOfState::PrimitiveToConserved(
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bc,
    AthenaArray<Real> &cons, Coordinates *pco,
    int il, int iu, int jl, int ju, int kl, int ku) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real& u_d  = cons(IDN,k,j,i);
        Real& u_m1 = cons(IM1,k,j,i);
        Real& u_m2 = cons(IM2,k,j,i);
        Real& u_m3 = cons(IM3,k,j,i);

        const Real& w_d  = prim(IDN,k,j,i);
        const Real& w_vx = prim(IVX,k,j,i);
        const Real& w_vy = prim(IVY,k,j,i);
        const Real& w_vz = prim(IVZ,k,j,i);

        u_d = w_d;
        u_m1 = w_vx*w_d;
        u_m2 = w_vy*w_d;
        u_m3 = w_vz*w_d;
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::SoundSpeed(Real dummy_arg[NHYDRO])
//! \brief returns isothermal sound speed

Real EquationOfState::SoundSpeed(const Real dummy_arg[NHYDRO]) {
  return iso_sound_speed_;
}

//---------------------------------------------------------------------------------------
//! \fn void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim, int k, int j,
//!                                                 int i)
//! \brief Apply density floor to reconstructed L/R cell interface states

void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim, int k, int j, int i) {
  Real& w_d  = prim(IDN,i);

  // apply density floor
  w_d = (w_d > density_floor_) ?  w_d : density_floor_;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::ApplyPrimitiveConservedFloors(AthenaArray<Real> &prim,
//!           AthenaArray<Real> &cons, FaceField &b, int k, int j, int i) {
//! \brief Apply pressure (prim) floor and correct energy (cons) (typically after W(U))

void EquationOfState::ApplyPrimitiveConservedFloors(
    AthenaArray<Real> &prim, AthenaArray<Real> &cons, AthenaArray<Real> &bcc,
    int k, int j, int i) {
  Real& w_d  = prim(IDN,k,j,i);
  Real& u_d  = cons(IDN,k,j,i);

  // apply (prim) density floor, without changing momentum or energy
  w_d = (w_d > density_floor_) ?  w_d : density_floor_;
  // ensure cons density matches
  u_d = w_d;

  return;
}
