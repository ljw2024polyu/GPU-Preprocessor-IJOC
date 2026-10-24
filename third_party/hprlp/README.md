# HPR-LP experiment implementation

This directory contains the HPR-LP-C 0.1.3-based implementation from the Section 4 experiment project, including its reduced-matrix solver paths, timing fields, and preprocessing adapters. The upstream reference is [HPR-LP-C v0.1.3](https://github.com/PolyU-IOR/HPR-LP-C/tree/v0.1.3). This experiment implementation includes local developments and is not an unmodified copy of that release. The original [MIT license](LICENSE) is retained.

For this repository, the GPU adapter was adjusted to the shared `cpp/` library's runtime-options API. Both `pslp` and `pslp0011` now select the bundled PSLP 0.0.11, replacing the former dual-version arrangement. CLI help and integration tests were updated accordingly. The `numThreads` launch constant was changed from a macro to `constexpr` to avoid a CUDA 12.9 header collision, without changing its value. Build rules are maintained in `integrations/hprlp/`; no second GPU-Preprocessor copy is bundled here.

See [solver integration](../../docs/SOLVER_INTEGRATION.md) for build and experiment commands.
