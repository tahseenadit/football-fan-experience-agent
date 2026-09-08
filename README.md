## Dependencies

- CMake 3.16+
- C++17 compiler
- [SFML](https://www.sfml-dev.org/) 3 (`brew install sfml`)

## Build

```bash
cmake -S . -B build
cmake --build build
./build/football-fan-ex
```

After configuring, `compile_commands.json` is generated under `build/` (and symlinked at the project root) so the editor can resolve includes like `<SFML/Graphics.hpp>`.