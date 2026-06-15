#include <cstdio>

#include <ccutils/ccutilsfilepath.h>
#include <ccutils/csvlogger.h>

int main() {
    // --- MyFilePath: build path and create directories ---
    CcutilsFilePath fp("results.csv", "output", "run01", "data", NULL);

    printf("directory : %s\n", fp.get_directory());
    printf("full path : %s\n", fp.get_fullpath());

    // --- MyLogFile: write a CSV with mixed types ---
    CcutilsLogFile log(fp.get_fullpath());

    log.set_header("label", "category", "rank", "iter", "value", NULL);
    log.set_types(5, MYSTRING, MYCHAR, MYINT, MYUINT, MYFLOAT);
    log.set_writing(5, MYSTRING_S, MYCHAR_C, MYINT_D, MYUINT_U, MYFLOAT_E);

    log.write_line(5, "alpha",   'a', 0, 0u, 1.0e-3);
    log.write_line(5, "beta",    'b', 0, 1u, 2.5e-3);
    log.write_line(5, "gamma",   'c', 0, 2u, 3.7e-3);

    log.finalize();
    fp.finalize();

    printf("Done. Open %s to inspect the output.\n", "output/run01/data/results.csv");
    return 0;
}
