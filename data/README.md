# Experiment inputs

[section_4_input_inventory.csv](section_4_input_inventory.csv) lists the dataset, filename, and size in bytes of each input model used in Section 4. The benchmark MPS files are not included in this repository.

| Group | Instances | Source and construction |
| --- | --- | --- |
| Mittelmann | 49 | LP models from [Mittelmann's benchmark collection](https://plato.asu.edu/bench.html); use the instance names in the inventory. |
| MIPLIB Large | 18 | Root-node LP relaxations of [MIPLIB 2017](https://miplib.zib.de/) instances, with more than 10 million matrix nonzeros. The inventory identifies the models used. |
| QAP20 | 20 | Structured LPs generated from [QAPLIB instances](https://coral.ise.lehigh.edu/data-sets/qaplib/qaplib-problem-instances-and-solutions/). |
| QAP scalability | 5 | LP relaxations derived from `wil100` at dimensions 20, 40, 60, 80, and 100 using [Hinder's large-scale LP generator](https://github.com/ohinder/large-scale-LP-test-problems). |

For the scalability experiment, the paper uses the leading N-by-N principal submatrices of the `wil100` flow and distance matrices and the Adams–Johnson LP relaxation, with the same formulation and column-scaling rule across sizes. The exact experiment export and generation scripts remain to be incorporated; the source collections alone do not specify the exported LPs.

The small [examples/model.mps](../examples/model.mps) distributed with GPU-Preprocessor is a usage example, not a benchmark input. It is covered by the repository's MIT license. The external benchmark collections retain their own distribution terms.

## References

- H. D. Mittelmann (1999). Benchmarking interior point LP/QP solvers. *Optimization Methods and Software*, 11(1–4), 655–670. [DOI](https://doi.org/10.1080/10556789908805767).
- A. Gleixner et al. (2021). MIPLIB 2017: Data-Driven Compilation of the 6th Mixed-Integer Programming Library. *Mathematical Programming Computation*, 13, 443–490. [DOI](https://doi.org/10.1007/s12532-020-00194-3).
- R. E. Burkard, S. E. Karisch, and F. Rendl (1997). QAPLIB—A Quadratic Assignment Problem Library. *Journal of Global Optimization*, 10, 391–403. [DOI](https://doi.org/10.1023/A:1008293323270).
- O. Hinder (2024). *Large-Scale LP Test Problems*. [Repository](https://github.com/ohinder/large-scale-LP-test-problems).
