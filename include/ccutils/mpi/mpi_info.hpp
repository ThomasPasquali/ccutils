#ifndef CCUTILS_MPI_INFO_HPP
#define CCUTILS_MPI_INFO_HPP

#include <mpi.h>
#include <algorithm>
#include <string>
#include "../machine.hpp"

namespace ccutils {

/* Supply MPI facts to the MPI-free machine stream before its first record. */
inline void capture_mpi_facts() {
  static bool captured = false;
  if (captured) return;
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) return;

  char library[MPI_MAX_LIBRARY_VERSION_STRING];
  int library_length = 0;
  MPI_Get_library_version(library, &library_length);
  library_length = std::max(0, std::min(library_length, MPI_MAX_LIBRARY_VERSION_STRING));
  int world_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int attribute_present = 0;
  int* wtime_is_global = nullptr;
  MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_WTIME_IS_GLOBAL, &wtime_is_global,
                    &attribute_present);
  const bool global = attribute_present && wtime_is_global != nullptr && *wtime_is_global != 0;
  machine::set_mpi_facts(std::string(library, static_cast<std::size_t>(library_length)),
                         world_size, global, MPI_Wtick());
  captured = true;
}

}  // namespace ccutils

#endif
