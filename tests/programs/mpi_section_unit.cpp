#include <mpi.h>
#include <cstdio>
#include <stdexcept>
#include <ccutils/ccutils.hpp>
#include <ccutils/mpi/mpi_macros.hpp>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  CCUTILS_MPI_INIT;
  ccutils::Config cfg;
  cfg.human_out = "none";
  cfg.machine_out = "none";
  ccutils::init(cfg);

  if (ccutils::mpi_rank() == 1) {
    try {
      ccutils::MpiSection section("asymmetric", "Asymmetric exception");
      throw std::runtime_error("rank-local failure");
    } catch (const std::runtime_error&) {}
  } else {
    ccutils::MpiSection section("asymmetric", "Asymmetric exception");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  {
    ccutils::MpiSection section("collective", "Collective close");
    section.put("rank", ccutils::mpi_rank());
    section.end();
  }
  if (ccutils::mpi_rank() == 0) std::puts("mpi_section_unit: ok");
  MPI_Finalize();
  return 0;
}
