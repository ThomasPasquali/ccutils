# Timers

The `timers` example records repeated CPU work, excludes warm-up samples, prints individual
samples, and combines timer results. CPU and CUDA elapsed times are reported in
milliseconds. Timer print calls produce no output until a timer has samples.

Build the examples against an installed CCUTILS package:

```sh
cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build build
cmake --install build
cmake -S examples -B build/examples -DCMAKE_PREFIX_PATH="$HOME/.local"
cmake --build build/examples --target timers
./build/examples/timers
```

With MPI enabled in the CCUTILS install and examples configuration:

```sh
cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$HOME/.local" -DCCUTILS_ENABLE_MPI=ON
cmake --build build
cmake --install build
cmake -S examples -B build/examples -DCMAKE_PREFIX_PATH="$HOME/.local" -DWITH_MPI=ON
cmake --build build/examples --target timers
mpirun -n 2 ./build/examples/timers
```

The v2 human renderer is the default and displays timer summaries in tables. Set
`CCUTILS_FORMAT=legacy` to use the previous `CCUTILS_TIMER` key/value text. For analysis,
write the separate JSON Lines stream and load it with `ccutils`:

```sh
CCUTILS_MACHINE_OUT=file:timers.jsonl ./build/examples/timers
ccutils timers timers.jsonl --csv > timers.csv
```

`CCUTILS_COLOR=auto` is the default, so color is used only for terminal sinks. Set
`CCUTILS_COLOR=never` for plain output. See the [core timer API](../API.md#cpu-timers),
[output settings](../OUTPUT.md), and [parser guide](PARSER.md).
