# Structured output

The `prints` example demonstrates named sections and JSON fields. Install CCUTILS first, then build from the repository root (replace the prefix if you installed elsewhere):

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --install build
cmake -S examples -B build/examples -DCMAKE_PREFIX_PATH="$HOME/.local"
cmake --build build/examples --target prints
./build/examples/prints
```

The output separates human-readable section markers from a delimited JSON object. The parser can read the markers and object; see [parser usage](PARSER.md). Define `CCUTILS_NO_JSON` to keep section markers while compiling out JSON storage and output.
