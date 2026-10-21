Real outputs of the multi-file test programs, recorded with `CCUTILS_MACHINE_OUT=inline`
so each file holds the human log AND the machine records of the same run (the parser
tests parse one file in several formats and compare the models).

    g++ -std=c++17 -Iinclude tests/programs/multifile/*.cpp -o mf
    CCUTILS_MACHINE_OUT=inline ./mf > tests/fixtures/multifile.inline.txt

    mpicxx -std=c++17 -Iinclude -DCCUTILS_ENABLE_MPI tests/programs/multifile_mpi/*.cpp -o mfm
    CCUTILS_MACHINE_OUT=inline mpirun -n 2 ./mfm > tests/fixtures/multifile_mpi.inline.txt
