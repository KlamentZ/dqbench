# DQBench

> Parameterized benchmark suite for quantum circuit simulation, built on [DeepQuantum](https://github.com/TuringQ/deepquantum).

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-early%20development-orange.svg)](#roadmap)

DQBench is a benchmark generator for quantum circuit simulators and compilers. While [DeepQuantum](https://github.com/TuringQ/deepquantum) provides reference implementations of individual quantum algorithms (QAOA, VQE, GBS, ...), DQBench builds *on top* of it by providing:

- **Parameterized circuit families** — sweep over graph types, qubit counts, ansatz depths, and noise levels with one call.
- **Standardized metrics** — gate count, depth, expressivity, entanglement entropy, noise resilience.
- **ML-native pipelines** — PyTorch-native training loops for variational algorithms, with differentiable noise models.
- **Reproducible benchmark reports** — JSON outputs + Jupyter notebooks that any reviewer can re-run.

## Status

🚧 **Early development (v0.1).** The 12-week roadmap is below. Current focus: qubit-based generators (Phase 1).

## Roadmap

| Phase | Weeks  | Focus               | Deliverables                                                                       |
|-------|--------|---------------------|------------------------------------------------------------------------------------|
| 1     | 1–4    | Qubit generators    | QAOA Max-Cut, VQE (H₂/LiH), random circuits, metrics module                        |
| 2     | 5–6    | ML × benchmark      | PyTorch training loops, noise-aware runner, noise resilience score                 |
| 3     | 7–9    | Scale + QEC         | Tensor-network simulation (30–50 qubits via MPS), surface-code decoder (PyMatching)|
| 4     | 10–12  | Photonic + polish   | Gaussian Boson Sampling benchmarks, PyPI release, technical report                 |

## Planned API

The following examples illustrate the target API for Phase 1. **Implementation in progress.**

```python
from dqbench.generators import qaoa_maxcut
from dqbench.benchmarks import run_benchmark

# Generate a QAOA Max-Cut instance family
instances = qaoa_maxcut.generate(
    n_qubits=[6, 8, 10],
    p_layers=[1, 2, 4],
    graph_type='erdos_renyi',
    graph_params={'p_edge': 0.5},
    seed=42,
)

# Run a noise-aware benchmark sweep
report = run_benchmark(
    instances,
    noise_model='depolarizing',
    noise_levels=[0.0, 0.001, 0.01],
    metrics=['gate_count', 'depth', 'entanglement_entropy', 'expectation_error'],
)

report.summary()      # print stats
report.to_json('report.json')
```

## Installation

> Not yet released to PyPI. Install from source:

```bash
git clone https://github.com/KlamentZ/dqbench.git
cd dqbench
pip install -e .
```

Requires Python 3.10+, PyTorch 2.x, and [DeepQuantum](https://pypi.org/project/deepquantum/) ≥ 4.5.0.

## Repository Structure

```
dqbench/
├── generators/         # Circuit family generators (QAOA, VQE, random, photonic GBS)
├── benchmarks/         # Metrics, noise-aware runner, tensor-network comparison
├── ml_pipelines/       # Training loops and decoder demos
├── notebooks/          # Tutorial and walkthrough notebooks
└── docs/               # Technical report and design notes
```

## Built on

- [DeepQuantum](https://github.com/TuringQ/deepquantum) — PyTorch-based quantum framework (qubit, photonic, MBQC)
- [PyMatching](https://github.com/oscarhiggott/PyMatching) — Surface-code decoder (Phase 3)

## Citation

If DQBench is useful in your work, please also cite the underlying DeepQuantum paper:

```bibtex
@article{he2025deepquantum,
  title={DeepQuantum: A PyTorch-based Software Platform for Quantum Machine Learning and Photonic Quantum Computing},
  author={He, Jun-Jie and Hu, Ke-Ming and Zhu, Yu-Ze and Yan, Guan-Ju and Liang, Shu-Yi and Zhao, Xiang and Wang, Ding and Guo, Fei-Xiang and Lan, Ze-Feng and Shang, Xiao-Wen and others},
  journal={arXiv preprint arXiv:2512.18995},
  year={2025}
}
```

A DQBench-specific citation will be added upon technical report release.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Contact

Shukai (Kevin) Zhao — [github.com/KlamentZ](https://github.com/KlamentZ)

University of Rochester
