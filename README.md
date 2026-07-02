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

#### Install via PyPI

##### Using uv
```bash
uv pip install vllm-rbln --extra-index-url https://wheels.vllm.ai/0.22.0/cpu --torch-backend cpu
```

##### Using pip
```bash
pip install vllm-rbln --extra-index-url https://wheels.vllm.ai/0.22.0/cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

#### Or from source

##### Using uv
```bash
git clone https://github.com/rbln-sw/vllm-rbln.git
cd vllm-rbln
uv pip install -e .
```

##### Using pip
```bash
git clone https://github.com/rbln-sw/vllm-rbln.git
cd vllm-rbln
pip install -e . --extra-index-url https://wheels.vllm.ai/0.22.0/cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

### 🛠️ Development setup (uv)

Requirements:
- Linux x86_64 with access to the internal network (Nexus / internal PyPI)
- Python **3.12** for this dev workflow (`rebel-compiler` nightly wheels are currently cp312-only; the package itself targets 3.10-3.13)

Internal indexes require your **LDAP account** credentials (set once, e.g. in your shell profile):

```bash
export UV_INDEX_RBLN_NEXUS_NIGHTLY_USERNAME=<ldap-username>
export UV_INDEX_RBLN_NEXUS_NIGHTLY_PASSWORD=<ldap-password>
export UV_INDEX_REBELLIONS_USERNAME=<ldap-username>
export UV_INDEX_REBELLIONS_PASSWORD=<ldap-password>
export UV_INDEX_RBLN_RELEASE_USERNAME=<ldap-username>
export UV_INDEX_RBLN_RELEASE_PASSWORD=<ldap-password>
```

Then install the locked, team-identical environment with a single command:

```bash
uv sync
```

To bump `rebel-compiler` to the latest nightly (do not edit `pyproject.toml`):

```bash
uv lock --upgrade-package rebel-compiler
# or pin a specific version:
uv lock --upgrade-package rebel-compiler==0.11.1.dev200
# commit the updated uv.lock
```

Available versions can be checked by browsing the index directly
(e.g. <https://nexus.mgmt.rbln.in/repository/pypi-group-nightly/simple/rebel-compiler/>,
LDAP login required), or:

```bash
curl -s -u "$UV_INDEX_RBLN_NEXUS_NIGHTLY_USERNAME:$UV_INDEX_RBLN_NEXUS_NIGHTLY_PASSWORD" \
  https://nexus.mgmt.rbln.in/repository/pypi-group-nightly/simple/rebel-compiler/ \
  | grep -oE 'rebel_compiler-[0-9][A-Za-z0-9.+]*' | sort -uV | tail -10
```

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
