# Structured output

The `prints` example demonstrates named sections, nested JSON values, and the optional MPI
section macros. In CCUTILS 1.0, the default human renderer shows section paths and JSON
values as readable key/value lines. To keep the older banners and delimited JSON blocks,
run it with `CCUTILS_FORMAT=legacy`.

Build and install CCUTILS, then build the examples against that installation:

```sh
cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build build
cmake --install build
cmake -S examples -B build/examples -DCMAKE_PREFIX_PATH="$HOME/.local"
cmake --build build/examples --target prints
./build/examples/prints
```

To collect machine-readable records alongside the human output:

```sh
CCUTILS_MACHINE_OUT=file:prints.jsonl ./build/examples/prints
```

The Python parser can read either output. See [parser usage](PARSER.md) and the
[record format](../FORMAT.md). Define `CCUTILS_NO_JSON` when compiling to keep section
markers while disabling JSON storage and output.
