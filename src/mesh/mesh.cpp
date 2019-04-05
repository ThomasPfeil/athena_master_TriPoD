//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mesh.cpp
//  \brief implementation of functions in Mesh class

// C headers
// pre-C11: needed before including inttypes.h, else won't define int64_t for C++ code
// #define __STDC_FORMAT_MACROS

// C++ headers
#include <algorithm>  // std::sort()
#include <cinttypes>  // format macro "PRId64" for fixed-width integer type std::int64_t
#include <cmath>      // std::abs(), std::pow()
#include <cstdint>    // std::int64_t fixed-wdith integer type alias
#include <cstdlib>
#include <cstring>    // std::memcpy()
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../fft/athena_fft.hpp"
#include "../fft/turbulence.hpp"
#include "../field/field.hpp"
#include "../field/field_diffusion/field_diffusion.hpp"
#include "../globals.hpp"
#include "../gravity/fft_gravity.hpp"
#include "../gravity/gravity.hpp"
#include "../gravity/mg_gravity.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../multigrid/multigrid.hpp"
#include "../outputs/io_wrapper.hpp"
#include "../parameter_input.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "../utils/buffer_utils.hpp"
#include "mesh.hpp"
#include "mesh_refinement.hpp"
#include "meshblock_tree.hpp"

// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

//----------------------------------------------------------------------------------------
// Mesh constructor, builds mesh at start of calculation using parameters in input file

Mesh::Mesh(ParameterInput *pin, int mesh_test) {
  std::stringstream msg;
  RegionSize block_size;
  MeshBlock *pfirst{};
  BoundaryFlag block_bcs[6];
  std::int64_t nbmax;
  int dim;

  // mesh test
  if (mesh_test > 0) Globals::nranks = mesh_test;

  // read time and cycle limits from input file
  start_time = pin->GetOrAddReal("time", "start_time", 0.0);
  tlim       = pin->GetReal("time", "tlim");
  cfl_number = pin->GetReal("time", "cfl_number");
  ncycle_out = pin->GetOrAddInteger("time", "ncycle_out", 1);
  time = start_time;
  Real real_max = std::numeric_limits<Real>::max();
  dt = dt_diff = (real_max);
  muj = 0.0;
  nuj = 0.0;
  muj_tilde = 0.0;
  nbnew = 0; nbdel = 0;

  four_pi_G_ = 0.0, grav_eps_ = -1.0, grav_mean_rho_ = -1.0;

  turb_flag = 0;

  nlim = pin->GetOrAddInteger("time", "nlim", -1);
  ncycle = 0;
  nint_user_mesh_data_ = 0;
  nreal_user_mesh_data_ = 0;
  nuser_history_output_ = 0;

  // read number of OpenMP threads for mesh
  num_mesh_threads_ = pin->GetOrAddInteger("mesh", "num_threads", 1);
  if (num_mesh_threads_ < 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Number of OpenMP threads must be >= 1, but num_threads="
        << num_mesh_threads_ << std::endl;
    ATHENA_ERROR(msg);
  }

  // read number of grid cells in root level of mesh from input file.
  mesh_size.nx1 = pin->GetInteger("mesh","nx1");
  if (mesh_size.nx1 < 4) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "In mesh block in input file nx1 must be >= 4, but nx1="
        << mesh_size.nx1 << std::endl;
    ATHENA_ERROR(msg);
  }

  mesh_size.nx2 = pin->GetInteger("mesh","nx2");
  if (mesh_size.nx2 < 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "In mesh block in input file nx2 must be >= 1, but nx2="
        << mesh_size.nx2 << std::endl;
    ATHENA_ERROR(msg);
  }

  mesh_size.nx3 = pin->GetInteger("mesh","nx3");
  if (mesh_size.nx3 < 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "In mesh block in input file nx3 must be >= 1, but nx3="
        << mesh_size.nx3 << std::endl;
    ATHENA_ERROR(msg);
  }
  if (mesh_size.nx2 == 1 && mesh_size.nx3 > 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "In mesh block in input file: nx2=1, nx3=" << mesh_size.nx3
        << ", 2D problems in x1-x3 plane not supported" << std::endl;
    ATHENA_ERROR(msg);
  }

  // setup convenience variables involving Mesh dimensionality
  dim = 1;
  if (mesh_size.nx2 > 1) dim = 2;
  if (mesh_size.nx3 > 1) dim = 3;
  f2_ = (mesh_size.nx2 > 1) ? 1 : 0;
  f3_ = (mesh_size.nx3 > 1) ? 1 : 0;

  // read physical size of mesh (root level) from input file.
  mesh_size.x1min = pin->GetReal("mesh","x1min");
  mesh_size.x2min = pin->GetReal("mesh","x2min");
  mesh_size.x3min = pin->GetReal("mesh","x3min");

  mesh_size.x1max = pin->GetReal("mesh","x1max");
  mesh_size.x2max = pin->GetReal("mesh","x2max");
  mesh_size.x3max = pin->GetReal("mesh","x3max");

  if (mesh_size.x1max <= mesh_size.x1min) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Input x1max must be larger than x1min: x1min=" << mesh_size.x1min
        << " x1max=" << mesh_size.x1max << std::endl;
    ATHENA_ERROR(msg);
  }
  if (mesh_size.x2max <= mesh_size.x2min) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Input x2max must be larger than x2min: x2min=" << mesh_size.x2min
        << " x2max=" << mesh_size.x2max << std::endl;
    ATHENA_ERROR(msg);
  }
  if (mesh_size.x3max <= mesh_size.x3min) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Input x3max must be larger than x3min: x3min=" << mesh_size.x3min
        << " x3max=" << mesh_size.x3max << std::endl;
    ATHENA_ERROR(msg);
  }

  // read ratios of grid cell size in each direction
  block_size.x1rat = mesh_size.x1rat = pin->GetOrAddReal("mesh", "x1rat", 1.0);
  block_size.x2rat = mesh_size.x2rat = pin->GetOrAddReal("mesh", "x2rat", 1.0);
  block_size.x3rat = mesh_size.x3rat = pin->GetOrAddReal("mesh", "x3rat", 1.0);

  // read BC flags for each of the 6 boundaries in turn.
  mesh_bcs[BoundaryFace::inner_x1] =
      GetBoundaryFlag(pin->GetOrAddString("mesh", "ix1_bc", "none"));
  mesh_bcs[BoundaryFace::outer_x1] =
      GetBoundaryFlag(pin->GetOrAddString("mesh", "ox1_bc", "none"));
  mesh_bcs[BoundaryFace::inner_x2] =
      GetBoundaryFlag(pin->GetOrAddString("mesh", "ix2_bc", "none"));
  mesh_bcs[BoundaryFace::outer_x2] =
      GetBoundaryFlag(pin->GetOrAddString("mesh", "ox2_bc", "none"));
  mesh_bcs[BoundaryFace::inner_x3] =
      GetBoundaryFlag(pin->GetOrAddString("mesh", "ix3_bc", "none"));
  mesh_bcs[BoundaryFace::outer_x3] =
      GetBoundaryFlag(pin->GetOrAddString("mesh", "ox3_bc", "none"));

  // read MeshBlock parameters
  block_size.nx1 = pin->GetOrAddInteger("meshblock", "nx1", mesh_size.nx1);
  if (dim >= 2)
    block_size.nx2 = pin->GetOrAddInteger("meshblock", "nx2", mesh_size.nx2);
  else
    block_size.nx2 = mesh_size.nx2;
  if (dim == 3)
    block_size.nx3 = pin->GetOrAddInteger("meshblock", "nx3", mesh_size.nx3);
  else
    block_size.nx3 = mesh_size.nx3;

  // check consistency of the block and mesh
  if (mesh_size.nx1 % block_size.nx1 != 0
      || mesh_size.nx2 % block_size.nx2 != 0
      || mesh_size.nx3 % block_size.nx3 != 0) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "the Mesh must be evenly divisible by the MeshBlock" << std::endl;
    ATHENA_ERROR(msg);
  }
  if (block_size.nx1 < 4 || (block_size.nx2 < 4 && dim >= 2)
      || (block_size.nx3 < 4 && dim == 3)) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "block_size must be larger than or equal to 4 cells." << std::endl;
    ATHENA_ERROR(msg);
  }

  // calculate the number of the blocks
  nrbx1 = mesh_size.nx1/block_size.nx1;
  nrbx2 = mesh_size.nx2/block_size.nx2;
  nrbx3 = mesh_size.nx3/block_size.nx3;
  nbmax = (nrbx1 > nrbx2) ? nrbx1:nrbx2;
  nbmax = (nbmax > nrbx3) ? nbmax:nrbx3;

  // initialize user-enrollable functions
  if (mesh_size.x1rat != 1.0) {
    use_uniform_meshgen_fn_[X1DIR] = false;
    MeshGenerator_[X1DIR] = DefaultMeshGeneratorX1;
  } else {
    use_uniform_meshgen_fn_[X1DIR] = true;
    MeshGenerator_[X1DIR] = UniformMeshGeneratorX1;
  }
  if (mesh_size.x2rat != 1.0) {
    use_uniform_meshgen_fn_[X2DIR] = false;
    MeshGenerator_[X2DIR] = DefaultMeshGeneratorX2;
  } else {
    use_uniform_meshgen_fn_[X2DIR] = true;
    MeshGenerator_[X2DIR] = UniformMeshGeneratorX2;
  }
  if (mesh_size.x3rat != 1.0) {
    use_uniform_meshgen_fn_[X3DIR] = false;
    MeshGenerator_[X3DIR] = DefaultMeshGeneratorX3;
  } else {
    use_uniform_meshgen_fn_[X3DIR] = true;
    MeshGenerator_[X3DIR] = UniformMeshGeneratorX3;
  }

  for (int dir=0; dir<6; dir++)
    BoundaryFunction_[dir] = nullptr;
  AMRFlag_ = nullptr;
  UserSourceTerm_ = nullptr;
  UserTimeStep_ = nullptr;
  ViscosityCoeff_ = nullptr;
  ConductionCoeff_ = nullptr;
  FieldDiffusivity_ = nullptr;
  MGBoundaryFunction_[BoundaryFace::inner_x1] = MGPeriodicInnerX1;
  MGBoundaryFunction_[BoundaryFace::outer_x1] = MGPeriodicOuterX1;
  MGBoundaryFunction_[BoundaryFace::inner_x2] = MGPeriodicInnerX2;
  MGBoundaryFunction_[BoundaryFace::outer_x2] = MGPeriodicOuterX2;
  MGBoundaryFunction_[BoundaryFace::inner_x3] = MGPeriodicInnerX3;
  MGBoundaryFunction_[BoundaryFace::outer_x3] = MGPeriodicOuterX3;


  // calculate the logical root level and maximum level
  for (root_level=0; (1<<root_level)<nbmax; root_level++) {}
  current_level = root_level;

  // create the root grid
  tree.CreateRootGrid(nrbx1, nrbx2, nrbx3, root_level);

  // SMR / AMR: create finer grids here
  multilevel = false;
  adaptive = false;
  if (pin->GetOrAddString("mesh", "refinement", "none") == "adaptive")
    adaptive = true, multilevel = true;
  else if (pin->GetOrAddString("mesh", "refinement", "none") == "static")
    multilevel = true;
  if (adaptive == true) {
    max_level = pin->GetOrAddInteger("mesh", "numlevel", 1) + root_level - 1;
    if (max_level > 63) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The number of the refinement level must be smaller than "
          << 63-root_level+1 << "." << std::endl;
      ATHENA_ERROR(msg);
    }
  } else {
    max_level = 63;
  }

  if (EOS_TABLE_ENABLED) peos_table = new EosTable(pin);
  InitUserMeshData(pin);

  if (multilevel == true) {
    if (block_size.nx1 % 2 == 1 || (block_size.nx2 % 2 == 1 && block_size.nx2>1)
        || (block_size.nx3 % 2 == 1 && block_size.nx3>1)) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The size of MeshBlock must be divisible by 2 in order to use SMR or AMR."
          << std::endl;
      ATHENA_ERROR(msg);
    }

    InputBlock *pib = pin->pfirst_block;
    while (pib != nullptr) {
      if (pib->block_name.compare(0, 10, "refinement") == 0) {
        RegionSize ref_size;
        ref_size.x1min = pin->GetReal(pib->block_name, "x1min");
        ref_size.x1max = pin->GetReal(pib->block_name, "x1max");
        if (dim>=2) {
          ref_size.x2min = pin->GetReal(pib->block_name, "x2min");
          ref_size.x2max = pin->GetReal(pib->block_name, "x2max");
        } else {
          ref_size.x2min=mesh_size.x2min;
          ref_size.x2max=mesh_size.x2max;
        }
        if (dim>=3) {
          ref_size.x3min = pin->GetReal(pib->block_name, "x3min");
          ref_size.x3max = pin->GetReal(pib->block_name, "x3max");
        } else {
          ref_size.x3min=mesh_size.x3min;
          ref_size.x3max=mesh_size.x3max;
        }
        int ref_lev = pin->GetInteger(pib->block_name, "level");
        int lrlev=ref_lev+root_level;
        if (lrlev>current_level) current_level=lrlev;
        // range check
        if (ref_lev<1) {
          msg << "### FATAL ERROR in Mesh constructor" << std::endl
              << "Refinement level must be larger than 0 (root level = 0)" << std::endl;
          ATHENA_ERROR(msg);
        }
        if (lrlev > max_level) {
          msg << "### FATAL ERROR in Mesh constructor" << std::endl
              << "Refinement level exceeds the maximum level (specify"
              << "maxlevel in <mesh> if adaptive)."
              << std::endl;
          ATHENA_ERROR(msg);
        }
        if (ref_size.x1min > ref_size.x1max || ref_size.x2min > ref_size.x2max
            || ref_size.x3min > ref_size.x3max)  {
          msg << "### FATAL ERROR in Mesh constructor" << std::endl
              << "Invalid refinement region is specified."<<  std::endl;
          ATHENA_ERROR(msg);
        }
        if (ref_size.x1min < mesh_size.x1min || ref_size.x1max > mesh_size.x1max
            || ref_size.x2min < mesh_size.x2min || ref_size.x2max > mesh_size.x2max
            || ref_size.x3min < mesh_size.x3min || ref_size.x3max > mesh_size.x3max) {
          msg << "### FATAL ERROR in Mesh constructor" << std::endl
              << "Refinement region must be smaller than the whole mesh." << std::endl;
          ATHENA_ERROR(msg);
        }
        // find the logical range in the ref_level
        // note: if this is too slow, this should be replaced with bi-section search.
        std::int64_t lx1min=0, lx1max=0, lx2min=0, lx2max=0, lx3min=0, lx3max=0;
        std::int64_t lxmax=nrbx1*(1LL<<ref_lev);
        for (lx1min=0; lx1min<lxmax; lx1min++) {
          Real rx=ComputeMeshGeneratorX(lx1min+1, lxmax, use_uniform_meshgen_fn_[X1DIR]);
          if (MeshGenerator_[X1DIR](rx, mesh_size) > ref_size.x1min)
            break;
        }
        for (lx1max=lx1min; lx1max<lxmax; lx1max++) {
          Real rx=ComputeMeshGeneratorX(lx1max+1, lxmax, use_uniform_meshgen_fn_[X1DIR]);
          if (MeshGenerator_[X1DIR](rx, mesh_size) >= ref_size.x1max)
            break;
        }
        if (lx1min % 2 == 1) lx1min--;
        if (lx1max % 2 == 0) lx1max++;
        if (dim>=2) { // 2D or 3D
          lxmax=nrbx2*(1LL<<ref_lev);
          for (lx2min=0; lx2min<lxmax; lx2min++) {
            Real rx=ComputeMeshGeneratorX(lx2min+1,lxmax,use_uniform_meshgen_fn_[X2DIR]);
            if (MeshGenerator_[X2DIR](rx, mesh_size) > ref_size.x2min)
              break;
          }
          for (lx2max=lx2min; lx2max<lxmax; lx2max++) {
            Real rx=ComputeMeshGeneratorX(lx2max+1,lxmax,use_uniform_meshgen_fn_[X2DIR]);
            if (MeshGenerator_[X2DIR](rx, mesh_size) >= ref_size.x2max)
              break;
          }
          if (lx2min % 2 == 1) lx2min--;
          if (lx2max % 2 == 0) lx2max++;
        }
        if (dim == 3) { // 3D
          lxmax=nrbx3*(1LL<<ref_lev);
          for (lx3min=0; lx3min<lxmax; lx3min++) {
            Real rx=ComputeMeshGeneratorX(lx3min+1,lxmax,use_uniform_meshgen_fn_[X3DIR]);
            if (MeshGenerator_[X3DIR](rx, mesh_size) > ref_size.x3min)
              break;
          }
          for (lx3max=lx3min; lx3max<lxmax; lx3max++) {
            Real rx=ComputeMeshGeneratorX(lx3max+1,lxmax,use_uniform_meshgen_fn_[X3DIR]);
            if (MeshGenerator_[X3DIR](rx, mesh_size) >= ref_size.x3max)
              break;
          }
          if (lx3min % 2 == 1) lx3min--;
          if (lx3max % 2 == 0) lx3max++;
        }
        // create the finest level
        if (dim == 1) {
          for (std::int64_t i=lx1min; i<lx1max; i+=2) {
            LogicalLocation nloc;
            nloc.level=lrlev, nloc.lx1=i, nloc.lx2=0, nloc.lx3=0;
            int nnew;
            tree.AddMeshBlock(tree, nloc, dim, mesh_bcs, nrbx1, nrbx2, nrbx3, root_level,
                              nnew);
          }
        }
        if (dim == 2) {
          for (std::int64_t j=lx2min; j<lx2max; j+=2) {
            for (std::int64_t i=lx1min; i<lx1max; i+=2) {
              LogicalLocation nloc;
              nloc.level=lrlev, nloc.lx1=i, nloc.lx2=j, nloc.lx3=0;
              int nnew;
              tree.AddMeshBlock(tree, nloc, dim, mesh_bcs, nrbx1, nrbx2, nrbx3,
                                root_level, nnew);
            }
          }
        }
        if (dim == 3) {
          for (std::int64_t k=lx3min; k<lx3max; k+=2) {
            for (std::int64_t j=lx2min; j<lx2max; j+=2) {
              for (std::int64_t i=lx1min; i<lx1max; i+=2) {
                LogicalLocation nloc;
                nloc.level=lrlev, nloc.lx1=i, nloc.lx2=j, nloc.lx3=k;
                int nnew;
                tree.AddMeshBlock(tree, nloc, dim, mesh_bcs, nrbx1, nrbx2, nrbx3,
                                  root_level, nnew);
              }
            }
          }
        }
      }
      pib = pib->pnext;
    }
  }

  // initial mesh hierarchy construction is completed here

  tree.CountMeshBlock(nbtotal);
  loclist=new LogicalLocation[nbtotal];
  tree.GetMeshBlockList(loclist,nullptr,nbtotal);

#ifdef MPI_PARALLEL
  // check if there are sufficient blocks
  if (nbtotal < Globals::nranks) {
    if (mesh_test == 0) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "Too few mesh blocks: nbtotal ("<< nbtotal <<") < nranks ("
          << Globals::nranks << ")" << std::endl;
      ATHENA_ERROR(msg);
    } else { // test
      std::cout << "### Warning in Mesh constructor" << std::endl
                << "Too few mesh blocks: nbtotal ("<< nbtotal <<") < nranks ("
                << Globals::nranks << ")" << std::endl;
    }
  }
#endif

  ranklist=new int[nbtotal];
  nslist=new int[Globals::nranks];
  nblist=new int[Globals::nranks];
  costlist=new Real[nbtotal];
  if (adaptive == true) { // allocate arrays for AMR
    nref = new int[Globals::nranks];
    nderef = new int[Globals::nranks];
    rdisp = new int[Globals::nranks];
    ddisp = new int[Globals::nranks];
    bnref = new int[Globals::nranks];
    bnderef = new int[Globals::nranks];
    brdisp = new int[Globals::nranks];
    bddisp = new int[Globals::nranks];
  }

  // initialize cost array with the simplest estimate; all the blocks are equal
  for (int i=0; i<nbtotal; i++) costlist[i]=1.0;

  LoadBalance(costlist, ranklist, nslist, nblist, nbtotal);

  // Output some diagnostic information to terminal

  // Output MeshBlock list and quit (mesh test only); do not create meshes
  if (mesh_test>0) {
    if (Globals::my_rank == 0) OutputMeshStructure(dim);
    return;
  }

  // set gravity flag
  gflag=0;
  if (SELF_GRAVITY_ENABLED) gflag=1;
  //  if (SELF_GRAVITY_ENABLED == 2 && ...) // independent allocation
  //    gflag=2;

  // create MeshBlock list for this process
  int nbs=nslist[Globals::my_rank];
  int nbe=nbs+nblist[Globals::my_rank]-1;
  // create MeshBlock list for this process
  for (int i=nbs; i<=nbe; i++) {
    SetBlockSizeAndBoundaries(loclist[i], block_size, block_bcs);
    // create a block and add into the link list
    if (i == nbs) {
      pblock = new MeshBlock(i, i-nbs, loclist[i], block_size, block_bcs, this,
                             pin, gflag);
      pfirst = pblock;
    } else {
      pblock->next = new MeshBlock(i, i-nbs, loclist[i], block_size, block_bcs,
                                   this, pin, gflag);
      pblock->next->prev = pblock;
      pblock = pblock->next;
    }
    pblock->pbval->SearchAndSetNeighbors(tree, ranklist, nslist);
  }
  pblock = pfirst;

  if (SELF_GRAVITY_ENABLED == 1)
    pfgrd = new FFTGravityDriver(this, pin);
  else if (SELF_GRAVITY_ENABLED == 2)
    pmgrd = new MGGravityDriver(this, MGBoundaryFunction_, pin);

  if (turb_flag > 0)
    ptrbd = new TurbulenceDriver(this, pin);
}

//----------------------------------------------------------------------------------------
// Mesh constructor for restarts. Load the restart file

Mesh::Mesh(ParameterInput *pin, IOWrapper& resfile, int mesh_test) {
  std::stringstream msg;
  RegionSize block_size;
  BoundaryFlag block_bcs[6];
  MeshBlock *pfirst{};
  IOWrapperSizeT *offset{};
  IOWrapperSizeT datasize, listsize, headeroffset;

  // mesh test
  if (mesh_test>0) Globals::nranks=mesh_test;

  // read time and cycle limits from input file
  start_time = pin->GetOrAddReal("time","start_time",0.0);
  tlim       = pin->GetReal("time","tlim");
  ncycle_out = pin->GetOrAddInteger("time","ncycle_out",1);
  nlim = pin->GetOrAddInteger("time","nlim",-1);
  nint_user_mesh_data_=0;
  nreal_user_mesh_data_=0;
  nuser_history_output_=0;

  four_pi_G_=0.0, grav_eps_=-1.0, grav_mean_rho_=-1.0;

  turb_flag = 0;

  nbnew=0; nbdel=0;

  // read number of OpenMP threads for mesh
  num_mesh_threads_ = pin->GetOrAddInteger("mesh","num_threads",1);
  if (num_mesh_threads_ < 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Number of OpenMP threads must be >= 1, but num_threads="
        << num_mesh_threads_ << std::endl;
    ATHENA_ERROR(msg);
  }

  // read BC flags for each of the 6 boundaries
  mesh_bcs[BoundaryFace::inner_x1] =
      GetBoundaryFlag(pin->GetOrAddString("mesh", "ix1_bc", "none"));
  mesh_bcs[BoundaryFace::outer_x1] =
      GetBoundaryFlag(pin->GetOrAddString("mesh", "ox1_bc", "none"));
  mesh_bcs[BoundaryFace::inner_x2] =
      GetBoundaryFlag(pin->GetOrAddString("mesh", "ix2_bc", "none"));
  mesh_bcs[BoundaryFace::outer_x2] =
      GetBoundaryFlag(pin->GetOrAddString("mesh", "ox2_bc", "none"));
  mesh_bcs[BoundaryFace::inner_x3] =
      GetBoundaryFlag(pin->GetOrAddString("mesh", "ix3_bc", "none"));
  mesh_bcs[BoundaryFace::outer_x3] =
      GetBoundaryFlag(pin->GetOrAddString("mesh", "ox3_bc", "none"));

  // get the end of the header
  headeroffset=resfile.GetPosition();
  // read the restart file
  // the file is already open and the pointer is set to after <par_end>
  IOWrapperSizeT headersize = sizeof(int)*3+sizeof(Real)*2
                              + sizeof(RegionSize)+sizeof(IOWrapperSizeT);
  char *headerdata = new char[headersize];
  if (Globals::my_rank == 0) { // the master process reads the header data
    if (resfile.Read(headerdata, 1, headersize) != headersize) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The restart file is broken." << std::endl;
      ATHENA_ERROR(msg);
    }
  }
#ifdef MPI_PARALLEL
  // then broadcast the header data
  MPI_Bcast(headerdata, headersize, MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
  IOWrapperSizeT hdos = 0;
  std::memcpy(&nbtotal, &(headerdata[hdos]), sizeof(int));
  hdos += sizeof(int);
  std::memcpy(&root_level, &(headerdata[hdos]), sizeof(int));
  hdos += sizeof(int);
  current_level=root_level;
  std::memcpy(&mesh_size, &(headerdata[hdos]), sizeof(RegionSize));
  hdos += sizeof(RegionSize);
  std::memcpy(&time, &(headerdata[hdos]), sizeof(Real));
  hdos += sizeof(Real);
  std::memcpy(&dt, &(headerdata[hdos]), sizeof(Real));
  hdos += sizeof(Real);
  std::memcpy(&ncycle, &(headerdata[hdos]), sizeof(int));
  hdos += sizeof(int);
  std::memcpy(&datasize, &(headerdata[hdos]), sizeof(IOWrapperSizeT));
  hdos += sizeof(IOWrapperSizeT);   // (this updated value is never used)

  delete [] headerdata;

  int dim = 1;
  if (mesh_size.nx2 > 1) dim = 2;
  if (mesh_size.nx3 > 1) dim = 3;

  // initialize
  loclist = new LogicalLocation[nbtotal];
  offset = new IOWrapperSizeT[nbtotal];
  costlist = new Real[nbtotal];
  ranklist = new int[nbtotal];
  nslist = new int[Globals::nranks];
  nblist = new int[Globals::nranks];

  block_size.nx1 = pin->GetOrAddInteger("meshblock","nx1",mesh_size.nx1);
  block_size.nx2 = pin->GetOrAddInteger("meshblock","nx2",mesh_size.nx2);
  block_size.nx3 = pin->GetOrAddInteger("meshblock","nx3",mesh_size.nx3);

  // calculate the number of the blocks
  nrbx1 = mesh_size.nx1/block_size.nx1;
  nrbx2 = mesh_size.nx2/block_size.nx2;
  nrbx3 = mesh_size.nx3/block_size.nx3;

  // initialize user-enrollable functions
  if (mesh_size.x1rat != 1.0) {
    use_uniform_meshgen_fn_[X1DIR] = false;
    MeshGenerator_[X1DIR] = DefaultMeshGeneratorX1;
  } else {
    use_uniform_meshgen_fn_[X1DIR] = true;
    MeshGenerator_[X1DIR] = UniformMeshGeneratorX1;
  }
  if (mesh_size.x2rat != 1.0) {
    use_uniform_meshgen_fn_[X2DIR] = false;
    MeshGenerator_[X2DIR] = DefaultMeshGeneratorX2;
  } else {
    use_uniform_meshgen_fn_[X2DIR] = true;
    MeshGenerator_[X2DIR] = UniformMeshGeneratorX2;
  }
  if (mesh_size.x3rat != 1.0) {
    use_uniform_meshgen_fn_[X3DIR] = false;
    MeshGenerator_[X3DIR] = DefaultMeshGeneratorX3;
  } else {
    use_uniform_meshgen_fn_[X3DIR] = true;
    MeshGenerator_[X3DIR] = UniformMeshGeneratorX3;
  }

  for (int dir=0; dir<6; dir++)
    BoundaryFunction_[dir] = nullptr;
  AMRFlag_ = nullptr;
  UserSourceTerm_ = nullptr;
  UserTimeStep_ = nullptr;
  ViscosityCoeff_ = nullptr;
  ConductionCoeff_ = nullptr;
  FieldDiffusivity_ = nullptr;
  MGBoundaryFunction_[BoundaryFace::inner_x1] = MGPeriodicInnerX1;
  MGBoundaryFunction_[BoundaryFace::outer_x1] = MGPeriodicOuterX1;
  MGBoundaryFunction_[BoundaryFace::inner_x2] = MGPeriodicInnerX2;
  MGBoundaryFunction_[BoundaryFace::outer_x2] = MGPeriodicOuterX2;
  MGBoundaryFunction_[BoundaryFace::inner_x3] = MGPeriodicInnerX3;
  MGBoundaryFunction_[BoundaryFace::outer_x3] = MGPeriodicOuterX3;

  multilevel = false;
  adaptive = false;
  if (pin->GetOrAddString("mesh","refinement","none") == "adaptive")
    adaptive = true, multilevel = true;
  else if (pin->GetOrAddString("mesh","refinement","none") == "static")
    multilevel = true;
  if (adaptive == true) {
    max_level = pin->GetOrAddInteger("mesh","numlevel",1)+root_level-1;
    if (max_level > 63) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The number of the refinement level must be smaller than "
          << 63-root_level+1 << "." << std::endl;
      ATHENA_ERROR(msg);
    }
  } else {
    max_level = 63;
  }

  if (EOS_TABLE_ENABLED) peos_table = new EosTable(pin);
  InitUserMeshData(pin);

  // read user Mesh data
  IOWrapperSizeT udsize = 0;
  for (int n=0; n<nint_user_mesh_data_; n++)
    udsize += iuser_mesh_data[n].GetSizeInBytes();
  for (int n=0; n<nreal_user_mesh_data_; n++)
    udsize += ruser_mesh_data[n].GetSizeInBytes();
  if (udsize != 0) {
    char *userdata = new char[udsize];
    if (Globals::my_rank == 0) { // only the master process reads the ID list
      if (resfile.Read(userdata,1,udsize) != udsize) {
        msg << "### FATAL ERROR in Mesh constructor" << std::endl
            << "The restart file is broken." << std::endl;
        ATHENA_ERROR(msg);
      }
    }
#ifdef MPI_PARALLEL
    // then broadcast the ID list
    MPI_Bcast(userdata, udsize, MPI_BYTE, 0, MPI_COMM_WORLD);
#endif

    IOWrapperSizeT udoffset=0;
    for (int n=0; n<nint_user_mesh_data_; n++) {
      std::memcpy(iuser_mesh_data[n].data(), &(userdata[udoffset]),
                  iuser_mesh_data[n].GetSizeInBytes());
      udoffset += iuser_mesh_data[n].GetSizeInBytes();
    }
    for (int n=0; n<nreal_user_mesh_data_; n++) {
      std::memcpy(ruser_mesh_data[n].data(), &(userdata[udoffset]),
                  ruser_mesh_data[n].GetSizeInBytes());
      udoffset += ruser_mesh_data[n].GetSizeInBytes();
    }
    delete [] userdata;
  }

  // read the ID list
  listsize = sizeof(LogicalLocation)+sizeof(Real);
  //allocate the idlist buffer
  char *idlist = new char[listsize*nbtotal];
  if (Globals::my_rank == 0) { // only the master process reads the ID list
    if (resfile.Read(idlist,listsize,nbtotal) != static_cast<unsigned int>(nbtotal)) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The restart file is broken." << std::endl;
      ATHENA_ERROR(msg);
    }
  }
#ifdef MPI_PARALLEL
  // then broadcast the ID list
  MPI_Bcast(idlist, listsize*nbtotal, MPI_BYTE, 0, MPI_COMM_WORLD);
#endif

  int os = 0;
  for (int i=0; i<nbtotal; i++) {
    std::memcpy(&(loclist[i]), &(idlist[os]), sizeof(LogicalLocation));
    os += sizeof(LogicalLocation);
    std::memcpy(&(costlist[i]), &(idlist[os]), sizeof(Real));
    os += sizeof(Real);
    if (loclist[i].level > current_level) current_level = loclist[i].level;
  }
  delete [] idlist;

  // calculate the header offset and seek
  headeroffset += headersize+udsize+listsize*nbtotal;
  if (Globals::my_rank != 0)
    resfile.Seek(headeroffset);

  // rebuild the Block Tree
  for (int i=0; i<nbtotal; i++)
    tree.AddMeshBlockWithoutRefine(loclist[i],nrbx1,nrbx2,nrbx3,root_level);
  int nnb;
  // check the tree structure, and assign GID
  tree.GetMeshBlockList(loclist, nullptr, nnb);
  if (nnb != nbtotal) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Tree reconstruction failed. The total numbers of the blocks do not match. ("
        << nbtotal << " != " << nnb << ")" << std::endl;
    ATHENA_ERROR(msg);
  }

#ifdef MPI_PARALLEL
  if (nbtotal < Globals::nranks) {
    if (mesh_test == 0) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "Too few mesh blocks: nbtotal ("<< nbtotal <<") < nranks ("
          << Globals::nranks << ")" << std::endl;
      ATHENA_ERROR(msg);
    } else { // test
      std::cout << "### Warning in Mesh constructor" << std::endl
                << "Too few mesh blocks: nbtotal ("<< nbtotal <<") < nranks ("
                << Globals::nranks << ")" << std::endl;
      delete [] offset;
      return;
    }
  }
#endif

  if (adaptive == true) { // allocate arrays for AMR
    nref = new int[Globals::nranks];
    nderef = new int[Globals::nranks];
    rdisp = new int[Globals::nranks];
    ddisp = new int[Globals::nranks];
    bnref = new int[Globals::nranks];
    bnderef = new int[Globals::nranks];
    brdisp = new int[Globals::nranks];
    bddisp = new int[Globals::nranks];
  }

  LoadBalance(costlist, ranklist, nslist, nblist, nbtotal);

  // Output MeshBlock list and quit (mesh test only); do not create meshes
  if (mesh_test > 0) {
    if (Globals::my_rank == 0) OutputMeshStructure(dim);
    delete [] offset;
    return;
  }

  // set gravity flag
  gflag = 0;
  if (SELF_GRAVITY_ENABLED) gflag = 1;
  //  if (SELF_GRAVITY_ENABLED == 2 && ...) // independent allocation
  //    gflag=2;

  // allocate data buffer
  int nb = nblist[Globals::my_rank];
  int nbs = nslist[Globals::my_rank];
  int nbe = nbs+nb-1;
  char *mbdata = new char[datasize*nb];
  // load MeshBlocks (parallel)
  if (resfile.Read_at_all(mbdata, datasize, nb, headeroffset+nbs*datasize) !=
      static_cast<unsigned int>(nb)) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "The restart file is broken or input parameters are inconsistent."
        << std::endl;
    ATHENA_ERROR(msg);
  }
  for (int i=nbs; i<=nbe; i++) {
    // Match fixed-width integer precision of IOWrapperSizeT datasize
    std::uint64_t buff_os = datasize * (i-nbs);
    SetBlockSizeAndBoundaries(loclist[i], block_size, block_bcs);
    // create a block and add into the link list
    if (i == nbs) {
      pblock = new MeshBlock(i, i-nbs, this, pin, loclist[i], block_size,
                             block_bcs, costlist[i], mbdata+buff_os, gflag);
      pfirst = pblock;
    } else {
      pblock->next = new MeshBlock(i, i-nbs, this, pin, loclist[i], block_size,
                                   block_bcs, costlist[i], mbdata+buff_os, gflag);
      pblock->next->prev = pblock;
      pblock = pblock->next;
    }
    pblock->pbval->SearchAndSetNeighbors(tree, ranklist, nslist);
  }
  pblock = pfirst;
  delete [] mbdata;
  // check consistency
  if (datasize != pblock->GetBlockSizeInBytes()) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "The restart file is broken or input parameters are inconsistent."
        << std::endl;
    ATHENA_ERROR(msg);
  }

  // clean up
  delete [] offset;

  if (SELF_GRAVITY_ENABLED == 1)
    pfgrd = new FFTGravityDriver(this, pin);
  else if (SELF_GRAVITY_ENABLED == 2)
    pmgrd = new MGGravityDriver(this, MGBoundaryFunction_, pin);

  if (turb_flag > 0)
    ptrbd = new TurbulenceDriver(this, pin);
}

//----------------------------------------------------------------------------------------
// destructor

Mesh::~Mesh() {
  while (pblock->prev != nullptr) // should not be true
    delete pblock->prev;
  while (pblock->next != nullptr)
    delete pblock->next;
  delete pblock;
  delete [] nslist;
  delete [] nblist;
  delete [] ranklist;
  delete [] costlist;
  delete [] loclist;
  if (SELF_GRAVITY_ENABLED == 1) delete pfgrd;
  else if (SELF_GRAVITY_ENABLED == 2) delete pmgrd;
  if (turb_flag > 0) delete ptrbd;
  if (adaptive == true) { // deallocate arrays for AMR
    delete [] nref;
    delete [] nderef;
    delete [] rdisp;
    delete [] ddisp;
    delete [] bnref;
    delete [] bnderef;
    delete [] brdisp;
    delete [] bddisp;
  }
  // delete user Mesh data
  for (int n=0; n<nreal_user_mesh_data_; n++)
    ruser_mesh_data[n].DeleteAthenaArray();
  if (nreal_user_mesh_data_>0) delete [] ruser_mesh_data;
  for (int n=0; n<nint_user_mesh_data_; n++)
    iuser_mesh_data[n].DeleteAthenaArray();
  if (nint_user_mesh_data_>0) delete [] iuser_mesh_data;
  if (EOS_TABLE_ENABLED) delete peos_table;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::OutputMeshStructure(int dim)
//  \brief print the mesh structure information

void Mesh::OutputMeshStructure(int dim) {
  RegionSize block_size;
  BoundaryFlag block_bcs[6];
  FILE *fp = nullptr;

  // open 'mesh_structure.dat' file
  if (dim >= 2) {
    if ((fp = std::fopen("mesh_structure.dat","wb")) == nullptr) {
      std::cout << "### ERROR in function Mesh::OutputMeshStructure" << std::endl
                << "Cannot open mesh_structure.dat" << std::endl;
      return;
    }
  }

  // Write overall Mesh structure to stdout and file
  std::cout << std::endl;
  std::cout << "Root grid = " << nrbx1 << " x " << nrbx2 << " x " << nrbx3
            << " MeshBlocks" << std::endl;
  std::cout << "Total number of MeshBlocks = " << nbtotal << std::endl;
  std::cout << "Number of physical refinement levels = "
            << (current_level - root_level) << std::endl;
  std::cout << "Number of logical  refinement levels = " << current_level << std::endl;

  // compute/output number of blocks per level, and cost per level
  int *nb_per_plevel = new int[max_level];
  int *cost_per_plevel = new int[max_level];
  for (int i=0; i<=max_level; ++i) {
    nb_per_plevel[i]=0;
    cost_per_plevel[i]=0;
  }
  for (int i=0; i<nbtotal; i++) {
    nb_per_plevel[(loclist[i].level - root_level)]++;
    cost_per_plevel[(loclist[i].level - root_level)] += costlist[i];
  }
  for (int i=root_level; i<=max_level; i++) {
    if (nb_per_plevel[i-root_level] != 0) {
      std::cout << "  Physical level = " << i-root_level << " (logical level = " << i
                << "): " << nb_per_plevel[i-root_level] << " MeshBlocks, cost = "
                << cost_per_plevel[i-root_level] <<  std::endl;
    }
  }

  // compute/output number of blocks per rank, and cost per rank
  std::cout << "Number of parallel ranks = " << Globals::nranks << std::endl;
  int *nb_per_rank = new int[Globals::nranks];
  int *cost_per_rank = new int[Globals::nranks];
  for (int i=0; i<Globals::nranks; ++i) {
    nb_per_rank[i] = 0;
    cost_per_rank[i] = 0;
  }
  for (int i=0; i<nbtotal; i++) {
    nb_per_rank[ranklist[i]]++;
    cost_per_rank[ranklist[i]] += costlist[i];
  }
  for (int i=0; i<Globals::nranks; ++i) {
    std::cout << "  Rank = " << i << ": " << nb_per_rank[i] <<" MeshBlocks, cost = "
              << cost_per_rank[i] << std::endl;
  }

  // output relative size/locations of meshblock to file, for plotting
  Real real_max = std::numeric_limits<Real>::max();
  Real mincost = real_max, maxcost = 0.0, totalcost = 0.0;
  for (int i=root_level; i<=max_level; i++) {
    for (int j=0; j<nbtotal; j++) {
      if (loclist[j].level == i) {
        SetBlockSizeAndBoundaries(loclist[j], block_size, block_bcs);
        std::int64_t &lx1 = loclist[j].lx1;
        std::int64_t &lx2 = loclist[j].lx2;
        std::int64_t &lx3 = loclist[j].lx3;
        int &ll = loclist[j].level;
        mincost = std::min(mincost,costlist[i]);
        maxcost = std::max(maxcost,costlist[i]);
        totalcost += costlist[i];
        std::fprintf(fp,"#MeshBlock %d on rank=%d with cost=%g\n", j, ranklist[j],
                     costlist[j]);
        std::fprintf(
            fp, "#  Logical level %d, location = (%" PRId64 " %" PRId64 " %" PRId64")\n",
            ll, lx1, lx2, lx3);
        if (dim == 2) {
          std::fprintf(fp, "%g %g\n", block_size.x1min, block_size.x2min);
          std::fprintf(fp, "%g %g\n", block_size.x1max, block_size.x2min);
          std::fprintf(fp, "%g %g\n", block_size.x1max, block_size.x2max);
          std::fprintf(fp, "%g %g\n", block_size.x1min, block_size.x2max);
          std::fprintf(fp, "%g %g\n", block_size.x1min, block_size.x2min);
          std::fprintf(fp, "\n\n");
        }
        if (dim == 3) {
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2min,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2min,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2max,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2max,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2min,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2min,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2min,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2min,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2min,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2max,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2max,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1max, block_size.x2max,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2max,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2max,
                       block_size.x3min);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2max,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2min,
                       block_size.x3max);
          std::fprintf(fp, "%g %g %g\n", block_size.x1min, block_size.x2min,
                       block_size.x3min);
          std::fprintf(fp, "\n\n");
        }
      }
    }
  }

  // close file, final outputs
  if (dim>=2) std::fclose(fp);
  std::cout << "Load Balancing:" << std::endl;
  std::cout << "  Minimum cost = " << mincost << ", Maximum cost = " << maxcost
            << ", Average cost = " << totalcost/nbtotal << std::endl << std::endl;
  std::cout << "See the 'mesh_structure.dat' file for a complete list"
            << " of MeshBlocks." << std::endl;
  std::cout << "Use 'python ../vis/python/plot_mesh.py' or gnuplot"
            << " to visualize mesh structure." << std::endl << std::endl;

  delete [] nb_per_plevel;
  delete [] cost_per_plevel;
  delete [] nb_per_rank;
  delete [] cost_per_rank;

  return;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::NewTimeStep()
// \brief function that loops over all MeshBlocks and find new timestep
//        this assumes that phydro->NewBlockTimeStep is already called

void Mesh::NewTimeStep() {
  MeshBlock *pmb = pblock;

  dt_diff=dt=static_cast<Real>(2.0)*dt;

  while (pmb != nullptr)  {
    dt = std::min(dt,pmb->new_block_dt_);
    dt_diff  = std::min(dt_diff, pmb->new_block_dt_diff_);
    pmb = pmb->next;
  }

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE,&dt,1,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
  if (STS_ENABLED)
    MPI_Allreduce(MPI_IN_PLACE,&dt_diff,1,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
#endif

  if (time < tlim && tlim-time < dt) // timestep would take us past desired endpoint
    dt = tlim-time;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserBoundaryFunction(BoundaryFace dir, BValHydro my_bc)
//  \brief Enroll a user-defined boundary function

void Mesh::EnrollUserBoundaryFunction(BoundaryFace dir, BValFunc my_bc) {
  std::stringstream msg;
  if (dir<0 || dir>5) {
    msg << "### FATAL ERROR in EnrollBoundaryCondition function" << std::endl
        << "dirName = " << dir << " not valid" << std::endl;
    ATHENA_ERROR(msg);
  }
  if (mesh_bcs[dir] != BoundaryFlag::user) {
    msg << "### FATAL ERROR in EnrollUserBoundaryFunction" << std::endl
        << "The boundary condition flag must be set to the string 'user' in the "
        << " <mesh> block in the input file to use user-enrolled BCs" << std::endl;
    ATHENA_ERROR(msg);
  }
  BoundaryFunction_[static_cast<int>(dir)]=my_bc;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserMGBoundaryFunction(BoundaryFace dir
//                                              MGBoundaryFunc my_bc)
//  \brief Enroll a user-defined Multigrid boundary function

void Mesh::EnrollUserMGBoundaryFunction(BoundaryFace dir, MGBoundaryFunc my_bc) {
  std::stringstream msg;
  if (dir<0 || dir>5) {
    msg << "### FATAL ERROR in EnrollBoundaryCondition function" << std::endl
        << "dirName = " << dir << " not valid" << std::endl;
    ATHENA_ERROR(msg);
  }
  MGBoundaryFunction_[static_cast<int>(dir)]=my_bc;
  return;
}

// DEPRECATED(felker): provide trivial overloads for old-style BoundaryFace enum argument
void Mesh::EnrollUserBoundaryFunction(int dir, BValFunc my_bc) {
  EnrollUserBoundaryFunction(static_cast<BoundaryFace>(dir), my_bc);
  return;
}

void Mesh::EnrollUserMGBoundaryFunction(int dir, MGBoundaryFunc my_bc) {
  EnrollUserMGBoundaryFunction(static_cast<BoundaryFace>(dir), my_bc);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserRefinementCondition(AMRFlagFunc amrflag)
//  \brief Enroll a user-defined function for checking refinement criteria

void Mesh::EnrollUserRefinementCondition(AMRFlagFunc amrflag) {
  if (adaptive == true)
    AMRFlag_=amrflag;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserMeshGenerator(CoordinateDirection,MeshGenFunc my_mg)
//  \brief Enroll a user-defined function for Mesh generation

void Mesh::EnrollUserMeshGenerator(CoordinateDirection dir, MeshGenFunc my_mg) {
  std::stringstream msg;
  if (dir<0 || dir>=3) {
    msg << "### FATAL ERROR in EnrollUserMeshGenerator function" << std::endl
        << "dirName = " << dir << " not valid" << std::endl;
    ATHENA_ERROR(msg);
  }
  if (dir == X1DIR && mesh_size.x1rat > 0.0) {
    msg << "### FATAL ERROR in EnrollUserMeshGenerator function" << std::endl
        << "x1rat = " << mesh_size.x1rat <<
        " must be negative for user-defined mesh generator in X1DIR " << std::endl;
    ATHENA_ERROR(msg);
  }
  if (dir == X2DIR && mesh_size.x2rat > 0.0) {
    msg << "### FATAL ERROR in EnrollUserMeshGenerator function" << std::endl
        << "x2rat = " << mesh_size.x2rat <<
        " must be negative for user-defined mesh generator in X2DIR " << std::endl;
    ATHENA_ERROR(msg);
  }
  if (dir == X3DIR && mesh_size.x3rat > 0.0) {
    msg << "### FATAL ERROR in EnrollUserMeshGenerator function" << std::endl
        << "x3rat = " << mesh_size.x3rat <<
        " must be negative for user-defined mesh generator in X3DIR " << std::endl;
    ATHENA_ERROR(msg);
  }
  use_uniform_meshgen_fn_[dir]=false;
  MeshGenerator_[dir]=my_mg;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserExplicitSourceFunction(SrcTermFunc my_func)
//  \brief Enroll a user-defined source function

void Mesh::EnrollUserExplicitSourceFunction(SrcTermFunc my_func) {
  UserSourceTerm_ = my_func;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserTimeStepFunction(TimeStepFunc my_func)
//  \brief Enroll a user-defined time step function

void Mesh::EnrollUserTimeStepFunction(TimeStepFunc my_func) {
  UserTimeStep_ = my_func;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::AllocateUserHistoryOutput(int n)
//  \brief set the number of user-defined history outputs

void Mesh::AllocateUserHistoryOutput(int n) {
  nuser_history_output_ = n;
  user_history_output_names_ = new std::string[n];
  user_history_func_ = new HistoryOutputFunc[n];
  for (int i=0; i<n; i++) user_history_func_[i] = nullptr;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserHistoryOutput(int i, HistoryOutputFunc my_func,
//                                         const char *name)
//  \brief Enroll a user-defined history output function and set its name

void Mesh::EnrollUserHistoryOutput(int i, HistoryOutputFunc my_func, const char *name) {
  std::stringstream msg;
  if (i>=nuser_history_output_) {
    msg << "### FATAL ERROR in EnrollUserHistoryOutput function" << std::endl
        << "The number of the user-defined history output (" << i << ") "
        << "exceeds the declared number (" << nuser_history_output_ << ")." << std::endl;
    ATHENA_ERROR(msg);
  }
  user_history_output_names_[i] = name;
  user_history_func_[i] = my_func;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollUserMetric(MetricFunc my_func)
//  \brief Enroll a user-defined metric for arbitrary GR coordinates

void Mesh::EnrollUserMetric(MetricFunc my_func) {
  UserMetric_ = my_func;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollViscosityCoefficient(ViscosityCoeff my_func)
//  \brief Enroll a user-defined magnetic field diffusivity function

void Mesh::EnrollViscosityCoefficient(ViscosityCoeffFunc my_func) {
  ViscosityCoeff_ = my_func;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollConductionCoefficient(ConductionCoeff my_func)
//  \brief Enroll a user-defined thermal conduction function

void Mesh::EnrollConductionCoefficient(ConductionCoeffFunc my_func) {
  ConductionCoeff_ = my_func;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::EnrollFieldDiffusivity(FieldDiffusionCoeff my_func)
//  \brief Enroll a user-defined magnetic field diffusivity function

void Mesh::EnrollFieldDiffusivity(FieldDiffusionCoeffFunc my_func) {
  FieldDiffusivity_ = my_func;
  return;
}
//----------------------------------------------------------------------------------------
//! \fn void Mesh::AllocateRealUserMeshDataField(int n)
//  \brief Allocate Real AthenaArrays for user-defned data in Mesh

void Mesh::AllocateRealUserMeshDataField(int n) {
  if (nreal_user_mesh_data_ != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::AllocateRealUserMeshDataField"
        << std::endl << "User Mesh data arrays are already allocated" << std::endl;
    ATHENA_ERROR(msg);
  }
  nreal_user_mesh_data_=n;
  ruser_mesh_data = new AthenaArray<Real>[n];
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Mesh::AllocateIntUserMeshDataField(int n)
//  \brief Allocate integer AthenaArrays for user-defned data in Mesh

void Mesh::AllocateIntUserMeshDataField(int n) {
  if (nint_user_mesh_data_ != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::AllocateIntUserMeshDataField"
        << std::endl << "User Mesh data arrays are already allocated" << std::endl;
    ATHENA_ERROR(msg);
  }
  nint_user_mesh_data_=n;
  iuser_mesh_data = new AthenaArray<int>[n];
  return;
}


//----------------------------------------------------------------------------------------
// \!fn void Mesh::ApplyUserWorkBeforeOutput(ParameterInput *pin)
// \brief Apply MeshBlock::UserWorkBeforeOutput

void Mesh::ApplyUserWorkBeforeOutput(ParameterInput *pin) {
  MeshBlock *pmb = pblock;
  while (pmb != nullptr)  {
    pmb->UserWorkBeforeOutput(pin);
    pmb = pmb->next;
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::Initialize(int res_flag, ParameterInput *pin)
// \brief  initialization before the main loop

void Mesh::Initialize(int res_flag, ParameterInput *pin) {
  bool iflag = true;
  int inb = nbtotal;
  int nthreads = GetNumMeshThreads();
  int nmb = GetNumMeshBlocksThisRank(Globals::my_rank);
  std::vector<MeshBlock*> pmb_array(nmb);

  do {
    // initialize a vector of MeshBlock pointers
    nmb = GetNumMeshBlocksThisRank(Globals::my_rank);
    if (static_cast<unsigned int>(nmb) != pmb_array.size()) pmb_array.resize(nmb);
    MeshBlock *pmbl = pblock;
    for (int i=0; i<nmb; ++i) {
      pmb_array[i] = pmbl;
      pmbl = pmbl->next;
    }

    if (res_flag == 0) {
#pragma omp parallel for num_threads(nthreads)
      for (int i=0; i<nmb; ++i) {
        MeshBlock *pmb = pmb_array[i];
        pmb->ProblemGenerator(pin);
        pmb->pbval->CheckBoundary();
      }
    }

    // add initial perturbation for decaying or impulsive turbulence
    if (((turb_flag == 1) || (turb_flag == 2)) && (res_flag == 0))
      ptrbd->Driving();

    // Create send/recv MPI_Requests for all BoundaryData objects
#pragma omp parallel for num_threads(nthreads)
    for (int i=0; i<nmb; ++i) {
      MeshBlock *pmb = pmb_array[i];
      // BoundaryVariable objects evolved in main TimeIntegratorTaskList:
      pmb->pbval->SetupPersistentMPI();
      // other BoundaryVariable objects:
      if (SELF_GRAVITY_ENABLED == 1)
        pmb->pgrav->pgbval->SetupPersistentMPI();
    }

    // solve gravity for the first time
    if (SELF_GRAVITY_ENABLED == 1)
      pfgrd->Solve(1, 0);
    else if (SELF_GRAVITY_ENABLED == 2)
      pmgrd->Solve(1);

#pragma omp parallel num_threads(nthreads)
    {
      MeshBlock *pmb;
      Hydro *phydro;
      Field *pfield;
      BoundaryValues *pbval;

      // prepare to receive conserved variables
#pragma omp for private(pmb,pbval)
      for (int i=0; i<nmb; ++i) {
        pmb = pmb_array[i]; pbval = pmb->pbval;
        pbval->StartReceiving(BoundaryCommSubset::mesh_init);
      }

      // send conserved variables
#pragma omp for private(pmb,pbval)
      for (int i=0; i<nmb; ++i) {
        pmb = pmb_array[i]; pbval = pmb->pbval;
        pmb->phydro->phbval->SwapHydroQuantity(pmb->phydro->u,
                                               HydroBoundaryQuantity::cons);
        pmb->phydro->phbval->SendBoundaryBuffers();
        if (MAGNETIC_FIELDS_ENABLED)
          pmb->pfield->pfbval->SendBoundaryBuffers();
      }

      // wait to receive conserved variables
#pragma omp for private(pmb,pbval)
      for (int i=0; i<nmb; ++i) {
        pmb = pmb_array[i]; pbval = pmb->pbval;
        pmb->phydro->phbval->SwapHydroQuantity(pmb->phydro->u,
                                               HydroBoundaryQuantity::cons);
        pmb->phydro->phbval->ReceiveAndSetBoundariesWithWait();
        if (MAGNETIC_FIELDS_ENABLED)
          pmb->pfield->pfbval->ReceiveAndSetBoundariesWithWait();
        // KGF: disable shearing box bvals/ calls
        // send and receive shearingbox boundary conditions
        // if (SHEARING_BOX)
        //   pmb->phydro->phbval->
        //   SendHydroShearingboxBoundaryBuffersForInit(pmb->phydro->u, true);
        pbval->ClearBoundary(BoundaryCommSubset::mesh_init);
      }

      // With AMR/SMR GR send primitives to enable cons->prim before prolongation
      if (GENERAL_RELATIVITY && multilevel) {
        // prepare to receive primitives
#pragma omp for private(pmb,pbval)
        for (int i=0; i<nmb; ++i) {
          pmb = pmb_array[i]; pbval = pmb->pbval;
          pbval->StartReceiving(BoundaryCommSubset::gr_amr);
        }

        // send primitives
#pragma omp for private(pmb,pbval)
        for (int i=0; i<nmb; ++i) {
          pmb = pmb_array[i]; pbval = pmb->pbval;
          pmb->phydro->phbval->SwapHydroQuantity(pmb->phydro->w,
                                                 HydroBoundaryQuantity::prim);
          pmb->phydro->phbval->SendBoundaryBuffers();
        }

        // wait to receive AMR/SMR GR primitives
#pragma omp for private(pmb,pbval)
        for (int i=0; i<nmb; ++i) {
          pmb = pmb_array[i]; pbval = pmb->pbval;
          pmb->phydro->phbval->SwapHydroQuantity(pmb->phydro->w,
                                                 HydroBoundaryQuantity::prim);
          pmb->phydro->phbval->ReceiveAndSetBoundariesWithWait();
          pbval->ClearBoundary(BoundaryCommSubset::gr_amr);
          pmb->phydro->phbval->SwapHydroQuantity(pmb->phydro->u,
                                                 HydroBoundaryQuantity::cons);
        }
      } // multilevel

      // perform fourth-order correction of midpoint initial condition:
      // (correct IC on all MeshBlocks or none; switch cannot be toggled independently)
      bool correct_ic = pmb_array[0]->precon->correct_ic;
      if (correct_ic == true)
        CorrectMidpointInitialCondition(pmb_array, nmb);

      // Now do prolongation, compute primitives, apply BCs
#pragma omp for private(pmb,pbval,phydro,pfield)
      for (int i=0; i<nmb; ++i) {
        pmb = pmb_array[i];
        pbval = pmb->pbval, phydro = pmb->phydro, pfield = pmb->pfield;
        if (multilevel == true)
          pbval->ProlongateBoundaries(time, 0.0);

        int il = pmb->is, iu = pmb->ie,
            jl = pmb->js, ju = pmb->je,
            kl = pmb->ks, ku = pmb->ke;
        if (pbval->nblevel[1][1][0] != -1) il -= NGHOST;
        if (pbval->nblevel[1][1][2] != -1) iu += NGHOST;
        if (pmb->block_size.nx2 > 1) {
          if (pbval->nblevel[1][0][1] != -1) jl -= NGHOST;
          if (pbval->nblevel[1][2][1] != -1) ju += NGHOST;
        }
        if (pmb->block_size.nx3 > 1) {
          if (pbval->nblevel[0][1][1] != -1) kl -= NGHOST;
          if (pbval->nblevel[2][1][1] != -1) ku += NGHOST;
        }
        pmb->peos->ConservedToPrimitive(phydro->u, phydro->w1, pfield->b,
                                        phydro->w, pfield->bcc, pmb->pcoord,
                                        il, iu, jl, ju, kl, ku);
        // --------------------------
        int order = pmb->precon->xorder;
        if (order == 4) {
          // fourth-order EOS:
          // for hydro, shrink buffer by 1 on all sides
          if (pbval->nblevel[1][1][0] != -1) il += 1;
          if (pbval->nblevel[1][1][2] != -1) iu -= 1;
          if (pbval->nblevel[1][0][1] != -1) jl += 1;
          if (pbval->nblevel[1][2][1] != -1) ju -= 1;
          if (pbval->nblevel[0][1][1] != -1) kl += 1;
          if (pbval->nblevel[2][1][1] != -1) ku -= 1;
          // for MHD, shrink buffer by 3
          // TODO(felker): add MHD loop limit calculation for 4th order W(U)
          pmb->peos->ConservedToPrimitiveCellAverage(phydro->u, phydro->w1, pfield->b,
                                                     phydro->w, pfield->bcc, pmb->pcoord,
                                                     il, iu, jl, ju, kl, ku);
        }
        // --------------------------
        // end fourth-order EOS
        pmb->phydro->phbval->SwapHydroQuantity(pmb->phydro->w,
                                               HydroBoundaryQuantity::prim);
        pbval->ApplyPhysicalBoundaries(time, 0.0);
      }

      // Calc initial diffusion coefficients
#pragma omp for private(pmb,phydro,pfield)
      for (int i=0; i<nmb; ++i) {
        pmb = pmb_array[i]; phydro = pmb->phydro, pfield = pmb->pfield;
        if (phydro->phdif->hydro_diffusion_defined)
          phydro->phdif->SetHydroDiffusivity(phydro->w, pfield->bcc);
        if (MAGNETIC_FIELDS_ENABLED) {
          if (pfield->pfdif->field_diffusion_defined)
            pfield->pfdif->SetFieldDiffusivity(phydro->w, pfield->bcc);
        }
      }

      if ((res_flag == 0) && (adaptive == true)) {
#pragma omp for
        for (int i=0; i<nmb; ++i) {
          pmb_array[i]->pmr->CheckRefinementCondition();
        }
      }
    } // omp parallel

    if ((res_flag == 0) && (adaptive == true)) {
      iflag = false;
      int onb = nbtotal;
      AdaptiveMeshRefinement(pin);
      if (nbtotal == onb) {
        iflag = true;
      } else if (nbtotal < onb && Globals::my_rank == 0) {
        std::cout << "### Warning in Mesh::Initialize" << std::endl
                  << "The number of MeshBlocks decreased during AMR grid initialization."
                  << std::endl
                  << "Possibly the refinement criteria have a problem." << std::endl;
      }
      if (nbtotal > 2*inb && Globals::my_rank == 0) {
        std::cout
            << "### Warning in Mesh::Initialize" << std::endl
            << "The number of MeshBlocks increased more than twice during initialization."
            << std::endl
            << "More computing power than you expected may be required." << std::endl;
      }
    }
  } while (iflag == false);

  // calculate the first time step
#pragma omp parallel for num_threads(nthreads)
  for (int i=0; i<nmb; ++i) {
    pmb_array[i]->phydro->NewBlockTimeStep();
  }

  NewTimeStep();
  return;
}

//----------------------------------------------------------------------------------------
//! \fn MeshBlock* Mesh::FindMeshBlock(int tgid)
//  \brief return the MeshBlock whose gid is tgid

MeshBlock* Mesh::FindMeshBlock(int tgid) {
  MeshBlock *pbl = pblock;
  while (pbl != nullptr) {
    if (pbl->gid == tgid)
      break;
    pbl = pbl->next;
  }
  return pbl;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::LoadBalance(Real *clist, int *rlist, int *slist, int *nlist, int nb)
// \brief Calculate distribution of MeshBlocks based on the cost list

void Mesh::LoadBalance(Real *clist, int *rlist, int *slist, int *nlist, int nb) {
  std::stringstream msg;
  Real real_max  =  std::numeric_limits<Real>::max();
  Real totalcost = 0, maxcost = 0.0, mincost = (real_max);

  for (int i=0; i<nb; i++) {
    totalcost += clist[i];
    mincost = std::min(mincost,clist[i]);
    maxcost = std::max(maxcost,clist[i]);
  }
  int j = (Globals::nranks) - 1;
  Real targetcost = totalcost/Globals::nranks;
  Real mycost = 0.0;
  // create rank list from the end: the master MPI rank should have less load
  for (int i=nb-1; i>=0; i--) {
    if (targetcost == 0.0) {
      msg << "### FATAL ERROR in LoadBalance" << std::endl
          << "There is at least one process which has no MeshBlock" << std::endl
          << "Decrease the number of processes or use smaller MeshBlocks." << std::endl;
      ATHENA_ERROR(msg);
    }
    mycost += clist[i];
    rlist[i] = j;
    if (mycost >= targetcost && j>0) {
      j--;
      totalcost -= mycost;
      mycost = 0.0;
      targetcost = totalcost/(j + 1);
    }
  }
  slist[0] = 0;
  j = 0;
  for (int i=1; i<nb; i++) { // make the list of nbstart and nblocks
    if (rlist[i] != rlist[i-1]) {
      nlist[j] = i-nslist[j];
      slist[++j] = i;
    }
  }
  nlist[j] = nb - slist[j];

#ifdef MPI_PARALLEL
  if (nb % (Globals::nranks * num_mesh_threads_) != 0 && adaptive == false
      && maxcost == mincost && Globals::my_rank == 0) {
    std::cout << "### Warning in LoadBalance" << std::endl
              << "The number of MeshBlocks cannot be divided evenly. "
              << "This will result in poor load balancing." << std::endl;
  }
#endif
  if ((Globals::nranks)*(num_mesh_threads_) > nb) {
    msg << "### FATAL ERROR in LoadBalance" << std::endl
        << "There are fewer MeshBlocks than OpenMP threads on each MPI rank" << std::endl
        << "Decrease the number of threads or use more MeshBlocks." << std::endl;
    ATHENA_ERROR(msg);
  }

  return;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::SetBlockSizeAndBoundaries(LogicalLocation loc,
//                 RegionSize &block_size, BundaryFlag *block_bcs)
// \brief Set the physical part of a block_size structure and block boundary conditions

void Mesh::SetBlockSizeAndBoundaries(LogicalLocation loc, RegionSize &block_size,
                                     BoundaryFlag *block_bcs) {
  std::int64_t &lx1 = loc.lx1;
  int &ll = loc.level;
  std::int64_t nrbx_ll = nrbx1 << (ll - root_level);

  // calculate physical block size, x1
  if (lx1 == 0) {
    block_size.x1min = mesh_size.x1min;
    block_bcs[BoundaryFace::inner_x1] = mesh_bcs[BoundaryFace::inner_x1];
  } else {
    Real rx = ComputeMeshGeneratorX(lx1, nrbx_ll, use_uniform_meshgen_fn_[X1DIR]);
    block_size.x1min = MeshGenerator_[X1DIR](rx, mesh_size);
    block_bcs[BoundaryFace::inner_x1] = BoundaryFlag::block;
  }
  if (lx1 == nrbx_ll - 1) {
    block_size.x1max = mesh_size.x1max;
    block_bcs[BoundaryFace::outer_x1] = mesh_bcs[BoundaryFace::outer_x1];
  } else {
    Real rx = ComputeMeshGeneratorX(lx1+1, nrbx_ll, use_uniform_meshgen_fn_[X1DIR]);
    block_size.x1max = MeshGenerator_[X1DIR](rx, mesh_size);
    block_bcs[BoundaryFace::outer_x1] = BoundaryFlag::block;
  }

  // calculate physical block size, x2
  if (mesh_size.nx2 == 1) {
    block_size.x2min = mesh_size.x2min;
    block_size.x2max = mesh_size.x2max;
    block_bcs[BoundaryFace::inner_x2] = mesh_bcs[BoundaryFace::inner_x2];
    block_bcs[BoundaryFace::outer_x2] = mesh_bcs[BoundaryFace::outer_x2];
  } else {
    std::int64_t &lx2 = loc.lx2;
    nrbx_ll = nrbx2 << (ll - root_level);
    if (lx2 == 0) {
      block_size.x2min = mesh_size.x2min;
      block_bcs[BoundaryFace::inner_x2] = mesh_bcs[BoundaryFace::inner_x2];
    } else {
      Real rx = ComputeMeshGeneratorX(lx2, nrbx_ll, use_uniform_meshgen_fn_[X2DIR]);
      block_size.x2min = MeshGenerator_[X2DIR](rx, mesh_size);
      block_bcs[BoundaryFace::inner_x2] = BoundaryFlag::block;
    }
    if (lx2 == (nrbx_ll) - 1) {
      block_size.x2max = mesh_size.x2max;
      block_bcs[BoundaryFace::outer_x2] = mesh_bcs[BoundaryFace::outer_x2];
    } else {
      Real rx = ComputeMeshGeneratorX(lx2+1, nrbx_ll, use_uniform_meshgen_fn_[X2DIR]);
      block_size.x2max = MeshGenerator_[X2DIR](rx, mesh_size);
      block_bcs[BoundaryFace::outer_x2] = BoundaryFlag::block;
    }
  }

  // calculate physical block size, x3
  if (mesh_size.nx3 == 1) {
    block_size.x3min = mesh_size.x3min;
    block_size.x3max = mesh_size.x3max;
    block_bcs[BoundaryFace::inner_x3] = mesh_bcs[BoundaryFace::inner_x3];
    block_bcs[BoundaryFace::outer_x3] = mesh_bcs[BoundaryFace::outer_x3];
  } else {
    std::int64_t &lx3 = loc.lx3;
    nrbx_ll = nrbx3 << (ll - root_level);
    if (lx3 == 0) {
      block_size.x3min = mesh_size.x3min;
      block_bcs[BoundaryFace::inner_x3] = mesh_bcs[BoundaryFace::inner_x3];
    } else {
      Real rx = ComputeMeshGeneratorX(lx3, nrbx_ll, use_uniform_meshgen_fn_[X3DIR]);
      block_size.x3min = MeshGenerator_[X3DIR](rx, mesh_size);
      block_bcs[BoundaryFace::inner_x3] = BoundaryFlag::block;
    }
    if (lx3 == (nrbx_ll) - 1) {
      block_size.x3max = mesh_size.x3max;
      block_bcs[BoundaryFace::outer_x3] = mesh_bcs[BoundaryFace::outer_x3];
    } else {
      Real rx = ComputeMeshGeneratorX(lx3+1, nrbx_ll, use_uniform_meshgen_fn_[X3DIR]);
      block_size.x3max = MeshGenerator_[X3DIR](rx, mesh_size);
      block_bcs[BoundaryFace::outer_x3] = BoundaryFlag::block;
    }
  }

  block_size.x1rat = mesh_size.x1rat;
  block_size.x2rat = mesh_size.x2rat;
  block_size.x3rat = mesh_size.x3rat;

  return;
}


void Mesh::CorrectMidpointInitialCondition(std::vector<MeshBlock*> &pmb_array, int nmb) {
  MeshBlock *pmb;
  Hydro *phydro;
#pragma omp for private(pmb, phydro)
  for (int nb=0; nb<nmb; ++nb) {
    pmb = pmb_array[nb];
    phydro = pmb->phydro;

    // Assume cell-centered analytic value is computed at all real cells, and ghost
    // cells with the cell-centered U have been exchanged
    int il = pmb->is, iu = pmb->ie, jl = pmb->js, ju = pmb->je,
        kl = pmb->ks, ku = pmb->ke;

    // Laplacian of cell-averaged conserved variables
    AthenaArray<Real> delta_cons_;

    // Allocate memory for 4D Laplacian
    int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
    int ncells2 = 1, ncells3 = 1;
    if (pmb->block_size.nx2 > 1) ncells2 = pmb->block_size.nx2 + 2*(NGHOST);
    if (pmb->block_size.nx3 > 1) ncells3 = pmb->block_size.nx3 + 2*(NGHOST);
    int ncells4 = NHYDRO;
    int nl = 0;
    int nu = ncells4-1;
    delta_cons_.NewAthenaArray(ncells4, ncells3, ncells2, ncells1);

    // Compute and store Laplacian of cell-averaged conserved variables
    pmb->pcoord->Laplacian(phydro->u, delta_cons_, il, iu, jl, ju, kl, ku, nl, nu);
    // TODO(felker): assuming uniform mesh with dx1f=dx2f=dx3f, so this factors out
    // TODO(felker): also, this may need to be dx1v, since Laplacian is cell-center
    Real h = pmb->pcoord->dx1f(il);  // pco->dx1f(i); inside loop
    Real C = (h*h)/24.0;

    // Compute fourth-order approximation to cell-centered conserved variables
    for (int n=nl; n<=nu; ++n) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
          for (int i=il; i<=iu; ++i) {
            // We do not actually need to store all cell-centered cons. variables,
            // but the ConservedToPrimitivePointwise() implementation operates on 4D
            phydro->u(n,k,j,i) = phydro->u(n,k,j,i) + C*delta_cons_(n,k,j,i);
          }
        }
      }
    }
    delta_cons_.DeleteAthenaArray();
  }

  // begin second exchange of ghost cells with corrected cell-averaged <U>
  // -----------------  (mostly copied from above section in Mesh::Initialize())
  BoundaryValues *pbval;
  // prepare to receive conserved variables
#pragma omp for private(pmb,pbval)
  for (int i=0; i<nmb; ++i) {
    pmb = pmb_array[i]; pbval = pmb->pbval;
    // no need to re-SetupPersistentMPI() the MPI requests for boundary values
    pbval->StartReceiving(BoundaryCommSubset::mesh_init);
  }

#pragma omp for private(pmb,pbval)
  for (int i=0; i<nmb; ++i) {
    pmb = pmb_array[i];
    pmb->phydro->phbval->SwapHydroQuantity(pmb->phydro->u,
                                           HydroBoundaryQuantity::cons);
    pmb->phydro->phbval->SendBoundaryBuffers();
    if (MAGNETIC_FIELDS_ENABLED)
      pmb->pfield->pfbval->SendBoundaryBuffers();
  }

  // wait to receive conserved variables
#pragma omp for private(pmb,pbval)
  for (int i=0; i<nmb; ++i) {
    pmb = pmb_array[i]; pbval = pmb->pbval;
    pmb->phydro->phbval->SwapHydroQuantity(pmb->phydro->u,
                                           HydroBoundaryQuantity::cons);
    pmb->phydro->phbval->ReceiveAndSetBoundariesWithWait();
    if (MAGNETIC_FIELDS_ENABLED)
      pmb->pfield->pfbval->ReceiveAndSetBoundariesWithWait();
    // KGF: disable shearing box bvals/ calls
    // send and receive shearingbox boundary conditions
    // if (SHEARING_BOX)
    //   pmb->phydro->phbval->
    //   SendHydroShearingboxBoundaryBuffersForInit(pmb->phydro->u, true);
    pbval->ClearBoundary(BoundaryCommSubset::mesh_init);
  } // end second exchange of ghost cells
  return;
}
