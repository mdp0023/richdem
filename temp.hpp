/**
  @file
  @brief TODO

  Richard Barnes (rijard.barnes@gmail.com), 2024
*/
#pragma once

#include <richdem/common/logger.hpp>
#include <richdem/common/Array2D.hpp>
#include <richdem/common/grid_cell.hpp>
#include <richdem/common/disjoint_dense_int_set.hpp>
#include <richdem/flowmet/d8_flowdirs.hpp>

#ifdef _OPENMP
#include <omp.h>
#else
#define get_omp_num_threads() 1
#define get_omp_thread_num() 0
#endif

#include <functional>
#include <random>

namespace richdem {

/**
  @brief  Fills all pits and removes all digital dams from a DEM, but faster
  @author Richard Barnes (rijard.barnes@gmail.com)

    Union-Flood processes all cells of the DEM in parallel and, from this,
    extracts the level to which cells should be filled.

  @param[in,out]  &elevations   A grid of cell elevations

  @pre
    1. **elevations** contains the elevations of every cell or a value _NoData_
       for cells not part of the DEM. Note that the _NoData_ value is assumed to
       be a negative number less than any actual data value.

  @post
    1. **elevations** contains the elevations of every cell or a value _NoData_
       for cells not part of the DEM.
    2. **elevations** contains no landscape depressions or digital dams.

  @correctness
    The correctness of this command is determined by testing against other
    depression-filling algorithms.
*/
template <Topology topo, class elev_t>
void UnionFlood_Barnes2024(Array2D<elev_t> &elevations){
  RDLOG_ALG_NAME << "Union-Flood";
  RDLOG_CITATION << "TODO";
  RDLOG_CONFIG   <<"topology = "<<TopologyName(topo);

  static_assert(topo==Topology::D8 || topo==Topology::D4);
  constexpr auto dx = get_dx_for_topology<topo>();
  constexpr auto dy = get_dy_for_topology<topo>();
  constexpr auto nmax = get_nmax_for_topology<topo>();


  RDLOG_PROGRESS<<"Setting up disjoint set...";
  DisjointDenseIntSet dis(elevations.size());

  #pragma omp parallel
  {
    const auto nthreads = get_omp_num_threads();
    const auto tid = get_omp_thread_num();
    const auto yrange = elevations.height();
    const auto ystep = yrange / nthreads;
    const auto ystart = tid * ystep + (tid>0);
    const auto yend = (tid==nthreads-1) ? yrange : (tid+1)*ystep - 1;

    const auto process_row = [&](const auto y, std::functional<void(uint32_t, uint32_t>) union_func){
        for(int x = 0; x<elevations.width();x++){
            const auto ci = elevations.xyToI(x,y);
            const auto e = elevations(ci);
            if(e == elevations.noData()){
                continue;
            }

            // Get neighbor with lowest elevation
            uint32_t lowest_ni = ci;
            elev_t lowest_elev = elevations(ci);
            for(int n=1;n<=nmax;n++){
                const auto nx=c.x+dx[n];
                const auto ny=c.y+dy[n];
                if(!elevations.inGrid(nx,ny)) continue;
                const auto ne=elevations(nx,ny);
                if(ne < lowest_elev){
                    lowest_elev = ne;
                    lowest_ni = elevations.xyToI(nx,ny);
                }
            }

            const auto lowest_ni = get_lowest_ni(ci);
            if(lowest_ni != ci){
                // Not thread-safe, but maintains tree-depth warranty
                union_func(ci, get_lowest_ni);
            }
        }
    };

    // Head
    if(tid > 0){
        process_row(ystart-1, [&](uint32_t ci, uint32_t lowest_ni){
            dis.set_parent(ci, lowest_ni);
        });
    }

    // Body
    for(auto y = ystart; y < yend ; y++){
        process_row(y, [&](uint32_t ci, uint32_t lowest_ni){
            dis.unionSet(ci, lowest_ni);
        });
    }

    // Foot
    if(tid < nthreads-1){
        process_row(yend+1, [&](uint32_t ci, uint32_t lowest_ni){
            dis.set_parent(ci, lowest_ni);
        });
    }
  }



  RDLOG_TIME_USE<<"Succeeded in "<<std::fixed<<std::setprecision(1)<<progress.stop()<<" s";
  RDLOG_MISC    <<"Cells processed = "<<processed_cells;
  RDLOG_MISC    <<"Cells in pits = "  <<pitc;
}


}
