#pragma once

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Enums                                                               */
/* ------------------------------------------------------------------ */

enum CcutilsTypesOptions   { MYINT, MYUINT, MYFLOAT, MYCHAR, MYSTRING };
enum CcutilsWritingOptions { MYINT_D, MYUINT_U, MYFLOAT_F, MYFLOAT_LF, MYFLOAT_E, MYCHAR_C, MYSTRING_S };

/* ------------------------------------------------------------------ */
/*  MyLogFile                                                           */
/* ------------------------------------------------------------------ */

struct CcutilsLogFile {
    FILE *log_file;

  private:
    size_t nfields;
    char  *header;
    char  *filename;
    int    is_ready; /* 1 only after set_types + set_writing both called */

    enum CcutilsTypesOptions   *types;
    enum CcutilsWritingOptions *writing;

    static int writing_matches_type(enum CcutilsTypesOptions t, enum CcutilsWritingOptions w) {
        switch (t) {
            case MYINT:    return w == MYINT_D;
            case MYUINT:   return w == MYUINT_U;
            case MYFLOAT:  return w == MYFLOAT_F || w == MYFLOAT_LF || w == MYFLOAT_E;
            case MYCHAR:   return w == MYCHAR_C;
            case MYSTRING: return w == MYSTRING_S;
            default:       return 0;
        }
    }

    void open_file(void) {
        FILE *probe = fopen(filename, "r");

        if (probe == NULL) {
            log_file = fopen(filename, "w");
            if (log_file == NULL) {
                fprintf(stderr, "Error opening log file %s for writing: ", filename);
                perror("");
                exit(EXIT_FAILURE);
            }
            fputs(header, log_file);
            return;
        }

        char existing[512] = {0};
        if (fgets(existing, sizeof(existing), probe) == NULL) {
            fclose(probe);
            log_file = fopen(filename, "w");
            if (log_file == NULL) {
                fprintf(stderr, "Error opening log file %s for writing: ", filename);
                perror("");
                exit(EXIT_FAILURE);
            }
            fputs(header, log_file);
            return;
        }
        fclose(probe);

        if (strcmp(existing, header) != 0) {
            fprintf(stderr,
                    "Error: log file %s header mismatch.\n"
                    "  existing: %s"
                    "  expected: %s",
                    filename, existing, header);
            exit(EXIT_FAILURE);
        }

        log_file = fopen(filename, "a");
        if (log_file == NULL) {
            fprintf(stderr, "Error opening log file %s for append: ", filename);
            perror("");
            exit(EXIT_FAILURE);
        }
    }

  public:
    CcutilsLogFile(const char *in_filename) {
        log_file = NULL;
        header   = NULL;
        types    = NULL;
        writing  = NULL;
        nfields  = 0;
        is_ready = 0;
        filename = strdup(in_filename);
    }

    /* set_header("rank", "time", NULL)
       Builds "rank,time\n", sets nfields, opens/validates the file. */
    void set_header(const char *first, ...) {
        nfields = 0;
        size_t total_chars = 0;

        va_list args;
        va_start(args, first);
        const char *s = first;
        while (s != NULL) {
            total_chars += strlen(s);
            nfields++;
            s = va_arg(args, const char *);
        }
        va_end(args);

        size_t buf_len = total_chars + (nfields > 0 ? nfields - 1 : 0) + 2; /* commas + '\n' + '\0' */

        size_t *disp = (size_t *)malloc(sizeof(size_t) * nfields);
        va_start(args, first);
        s = first;
        disp[0] = 0;
        for (size_t i = 0; i < nfields - 1; i++) {
            disp[i + 1] = disp[i] + strlen(s) + 1;
            s = va_arg(args, const char *);
        }
        va_end(args);

        char *tmp = (char *)malloc(buf_len);
        va_start(args, first);
        s = first;
        for (size_t i = 0; i < nfields; i++) {
            size_t len = strlen(s);
            memcpy(tmp + disp[i], s, len);
            if (i < nfields - 1)
                tmp[disp[i] + len] = ',';
            s = va_arg(args, const char *);
        }
        va_end(args);
        tmp[buf_len - 2] = '\n';
        tmp[buf_len - 1] = '\0';

        free(disp);
        free(header);
        header = tmp;

        free(types);
        free(writing);
        types   = (enum CcutilsTypesOptions *)  malloc(sizeof(enum CcutilsTypesOptions)   * nfields);
        writing = (enum CcutilsWritingOptions *) malloc(sizeof(enum CcutilsWritingOptions) * nfields);
        memset(types,   0, sizeof(enum CcutilsTypesOptions)   * nfields);
        memset(writing, 0, sizeof(enum CcutilsWritingOptions) * nfields);
        is_ready = 0;

        open_file();
    }

    /* set_types(2, MYINT, MYFLOAT)
       count must equal nfields. */
    void set_types(int count, ...) {
        if ((size_t)count != nfields) {
            fprintf(stderr, "set_types: got %d types but header has %zu fields\n",
                    count, nfields);
            exit(EXIT_FAILURE);
        }
        va_list args;
        va_start(args, count);
        for (size_t i = 0; i < nfields; i++)
            types[i] = (enum CcutilsTypesOptions)va_arg(args, int);
        va_end(args);
        is_ready = 0; /* require set_writing to be (re)called after changing types */
    }

    /* set_writing(2, MYINT_D, MYFLOAT_LF)
       count must equal nfields; each option must match the corresponding type. */
    void set_writing(int count, ...) {
        if ((size_t)count != nfields) {
            fprintf(stderr, "set_writing: got %d options but header has %zu fields\n",
                    count, nfields);
            exit(EXIT_FAILURE);
        }
        va_list args;
        va_start(args, count);
        for (size_t i = 0; i < nfields; i++) {
            enum CcutilsWritingOptions w = (enum CcutilsWritingOptions)va_arg(args, int);
            if (!writing_matches_type(types[i], w)) {
                fprintf(stderr,
                    "set_writing: field %zu — option %d incompatible with type %d\n",
                    i, w, types[i]);
                va_end(args);
                exit(EXIT_FAILURE);
            }
            writing[i] = w;
        }
        va_end(args);
        is_ready = 1;
    }

    /* write_line(2, 42, 3.14)
       Writes one CSV row. count must equal nfields.
       Caller must pass values with the C types implied by the writing options:
         MYINT_D            -> int
         MYUINT_U           -> unsigned int
         MYFLOAT_F / LF / E -> double  (float is promoted in variadic calls)
         MYCHAR_C           -> int     (char is promoted in variadic calls)
         MYSTRING_S         -> const char*                                   */
    void write_line(int count, ...) {
        if (!is_ready) {
            fprintf(stderr, "write_line: call set_types and set_writing before writing\n");
            exit(EXIT_FAILURE);
        }
        if ((size_t)count != nfields) {
            fprintf(stderr, "write_line: got %d values but header has %zu fields\n",
                    count, nfields);
            exit(EXIT_FAILURE);
        }

        va_list args;
        va_start(args, count);
        for (size_t i = 0; i < nfields; i++) {
            switch (writing[i]) {
                case MYINT_D:    fprintf(log_file, "%d",  va_arg(args, int));          break;
                case MYUINT_U:   fprintf(log_file, "%u",  va_arg(args, unsigned int)); break;
                case MYFLOAT_F:  fprintf(log_file, "%f",  va_arg(args, double));       break;
                case MYFLOAT_LF: fprintf(log_file, "%lf", va_arg(args, double));       break;
                case MYFLOAT_E:  fprintf(log_file, "%e",  va_arg(args, double));       break;
                case MYCHAR_C:   fprintf(log_file, "%c",  va_arg(args, int));          break;
                case MYSTRING_S: fprintf(log_file, "%s",  va_arg(args, const char *)); break;
            }
            if (i < nfields - 1)
                fputc(',', log_file);
        }
        va_end(args);
        fputc('\n', log_file);
    }

    void finalize(void) {
        if (log_file != NULL) {
            fclose(log_file);
            log_file = NULL;
        }
        free(header);
        free(types);
        free(writing);
        free(filename);
        header   = NULL;
        types    = NULL;
        writing  = NULL;
        filename = NULL;
    }
};
