#pragma once

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  include <direct.h>
#  define MKDIR(p) _mkdir(p)
#  define GETCWD(buf, sz) _getcwd(buf, sz)
#  define PATH_SEP "\\"
#else
#  include <sys/stat.h>
#  include <unistd.h>
#  define MKDIR(p) mkdir(p, 0755)
#  define GETCWD(buf, sz) getcwd(buf, sz)
#  define PATH_SEP "/"
#endif

/* ------------------------------------------------------------------ */
/*  MyFilePath                                                          */
/*                                                                      */
/*  MyFilePath fp("test.csv", "first", "second", "third", NULL);       */
/*  fp.get_fullpath()  ->  /cwd/first/second/third/test.csv            */
/* ------------------------------------------------------------------ */

struct MyFilePath {
  private:
    char *directory; /* absolute path to the leaf folder, trailing separator */
    char *fullpath;  /* directory + filename                                  */

    /* Creates every component of path that does not exist yet. */
    static void mkdir_recursive(const char *path) {
        char *tmp = strdup(path);
        size_t len = strlen(tmp);

        /* strip trailing separator so the loop always processes the last component */
        if (len > 0 && (tmp[len - 1] == '/' || tmp[len - 1] == '\\'))
            tmp[--len] = '\0';

        for (size_t i = 1; i <= len; i++) {
            if (tmp[i] == '/' || tmp[i] == '\\' || tmp[i] == '\0') {
                char saved = tmp[i];
                tmp[i] = '\0';
                MKDIR(tmp); /* ignore error — directory may already exist */
                tmp[i] = saved;
            }
        }
        free(tmp);
    }

  public:
    /* MyFilePath("test.csv", "first", "second", "third", NULL)
       Resolves the current working directory, appends each folder in order,
       creates the full tree if needed, and stores the resulting paths. */
    MyFilePath(const char *filename, ...) {
        /* --- get current working directory --- */
        char cwd[4096];
        if (GETCWD(cwd, sizeof(cwd)) == NULL) {
            perror("MyFilePath: getcwd failed");
            exit(EXIT_FAILURE);
        }

        /* --- count folders and measure total length --- */
        size_t nfolders    = 0;
        size_t folder_len  = 0;

        va_list args;
        va_start(args, filename);
        const char *s = va_arg(args, const char *);
        while (s != NULL) {
            folder_len += strlen(s) + 1; /* +1 for separator */
            nfolders++;
            s = va_arg(args, const char *);
        }
        va_end(args);

        /* --- build directory string: cwd + sep + folders + trailing sep --- */
        size_t cwd_len = strlen(cwd);
        size_t dir_len = cwd_len + 1 + folder_len + 1; /* sep after cwd + trailing sep + '\0' */
        directory = (char *)malloc(dir_len);

        memcpy(directory, cwd, cwd_len);
        directory[cwd_len] = PATH_SEP[0];
        size_t pos = cwd_len + 1;

        va_start(args, filename);
        s = va_arg(args, const char *);
        while (s != NULL) {
            size_t slen = strlen(s);
            memcpy(directory + pos, s, slen);
            pos += slen;
            directory[pos++] = PATH_SEP[0];
            s = va_arg(args, const char *);
        }
        va_end(args);
        directory[pos] = '\0';

        /* --- create the directory tree --- */
        mkdir_recursive(directory);

        /* --- build fullpath: directory + filename --- */
        size_t fname_len = strlen(filename);
        fullpath = (char *)malloc(pos + fname_len + 1);
        memcpy(fullpath, directory, pos);
        memcpy(fullpath + pos, filename, fname_len + 1); /* +1 for '\0' */
    }

    const char *get_fullpath(void) const {
        return fullpath;
    }

    const char *get_directory(void) const {
        return directory;
    }

    void finalize(void) {
        free(directory);
        free(fullpath);
        directory = NULL;
        fullpath  = NULL;
    }
};
