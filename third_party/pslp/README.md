# PSLP 0.0.11

The source and headers in this directory are the PSLP 0.0.11 copy used by the experiment implementation. The upstream reference is [PSLP](https://github.com/dance858/PSLP/tree/ae2318909b2f3096a656a642b3d706abafd7e167). The original [Apache License 2.0](LICENSE) is retained. The PSLP source is unmodified; the IJOC build configuration is maintained in `solver_interface/`.

The external HPR-LP adapter runs CPU presolve in a worker process, solves the reduced LP with the official HPR-LP v0.1.3 release on the GPU, and uses PSLP postsolve to recover the original solution. Both `--presolver pslp` and the recorded experiment spelling `--presolver pslp0011` use this version. Compile-time symbol prefixes isolate this copy from the official HPR-LP release's native PSLP 0.0.8 dependency, without changing either copy's source.
