# vLLM RBLN Plugin
<div align="center">
<picture>
  <source srcset="assets/vllm-rbln-white.png" media="(prefers-color-scheme: dark)">
  <source srcset="assets/vllm-rbln-black.png" media="(prefers-color-scheme: light)">
  <img src="assets/vllm-rbln-black.png" alt="main-logo" width=90%>
</picture>

[![PyPI version](https://badge.fury.io/py/vllm-rbln.svg)](https://badge.fury.io/py/vllm-rbln)
[![License](https://img.shields.io/github/license/rbln-sw/vllm-rbln)](https://github.com/rbln-sw/vllm-rbln/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://docs.rbln.ai/software/model_serving/vllm_support/vllm-rbln.html)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](./CODE_OF_CONDUCT.md)
</div>

This repository provides the hardware plugin that enables vLLM on RBLN NPUs, including [ATOM](https://rebellions.ai/rebellions-product/rbln-ca25/) and [REBEL](https://rebellions.ai/rebellions-product/rebel-quad/).

Built on top of [vLLM’s Plugin System](https://docs.vllm.ai/en/latest/design/plugin_system.html), it allows seamless integration with the vLLM ecosystem and provides high-throughput, low-latency LLM serving on RBLN hardware. Our plugin supports a wide range of popular LLMs and continues to expand to support all features enabled in vLLM, including advanced attention mechanisms.

## 🚀 Getting Started

### 📋 Prerequisites

- `rebel-compiler`
- `optimum-rbln`

### ⚙️ Installation

You can install this project using `pip` or from source.

Install commands below use [`requirements-cpu.txt`](./requirements-cpu.txt) —
a pinned dep list exported from `uv.lock` — with `pip install --no-deps` so
pip skips transitive resolution and installs exactly the set `uv sync`
would. This is what excludes `intel-openmp`: the file is generated with
`--no-emit-project` from a lock that has `intel-openmp` overridden away
(see [`pyproject.toml`](./pyproject.toml) `[tool.uv].override-dependencies`),
and `--no-deps` prevents pip from re-adding it from `vllm==0.22.0+cpu`'s
`Requires-Dist`. Plain `pip install vllm-rbln` (without the two flags below)
would pull `intel-openmp` — Intel proprietary, unused on RBLN NPU.

#### Install via PyPI

##### Using uv
```bash
uv pip install --no-deps \
  -r https://raw.githubusercontent.com/RBLN-SW/vllm-rbln/main/requirements-cpu.txt \
  vllm-rbln \
  --extra-index-url https://wheels.vllm.ai/0.22.0/cpu \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

##### Using pip
```bash
pip install --no-deps \
  -r https://raw.githubusercontent.com/RBLN-SW/vllm-rbln/main/requirements-cpu.txt \
  vllm-rbln \
  --extra-index-url https://wheels.vllm.ai/0.22.0/cpu \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

#### Or from source

##### Using uv
```bash
git clone https://github.com/rbln-sw/vllm-rbln.git
cd vllm-rbln
uv pip install --no-deps -r requirements-cpu.txt -e .
```

##### Using pip
```bash
git clone https://github.com/rbln-sw/vllm-rbln.git
cd vllm-rbln
pip install --no-deps -r requirements-cpu.txt -e . \
  --extra-index-url https://wheels.vllm.ai/0.22.0/cpu \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

### 🛠️ Development setup

Building from source? See **[DEVELOPMENT.md](./DEVELOPMENT.md)** for the `uv` dev workflow.

### 📚 Documentation

- [Overview & Supported Models](https://docs.rbln.ai/software/model_serving/vllm_support/vllm-rbln.html)
- [API Tutorial](https://docs.rbln.ai/software/model_serving/vllm_support/tutorial/vllm_llama3-8b.html)


## 🤝 Contributing

We welcome all contributions! Whether it's reporting issues, proposing enhancements, or improving docs—your input helps make the project better.

See our [CONTRIBUTING.md](./CONTRIBUTING.md) for more information.

## 📄 License

This project is licensed under the Apache License 2.0.

See the [LICENSE](./LICENSE) file for more information.

## 📧 Contact

- Join discussions and get answers in our [Developer Community](https://discuss.rebellions.ai/)
- Contact maintainers at [support@rebellions.ai](mailto:support@rebellions.ai)
