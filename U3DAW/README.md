# U3DAW - Universal 3D Audio Workstation

## Development Status
**Phase 1: TEP Engine** (Week 1 of 12)

## Current Sprint (This Week)
- Task 1.1: Linkwitz-Riley Filters (SearchAgent + MathAgent + CodeAgent)
- Task 1.2: cuFFT Wrapper (CodeAgent)
- Task 1.3: Overlap-Save Algorithm (CodeAgent)
- Task 1.4: Unit Tests Setup (TestAgent)

## Build System
```bash
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/bin/nvcc.exe"
cmake --build . --config Release
```

## Run Tests
```bash
cd build
ctest --output-on-failure
```

## Documentation
- Full specification: `U3DAW_MASTER_SPEC.md`
- Research notes: `docs/research/`
- Learning log: `learning_logs/learning_log.json`

## Success Metrics (Phase 1)
- Latency: <2.7ms for TEP stage
- GPU Usage: <25%
- Tests: 100% pass rate
- Transients preserved (listening tests)
