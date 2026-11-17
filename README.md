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

The vLLM CPU wheel depends on `triton` through a direct URL, which `uv` only accepts when the URL is also given as a constraint:

1. Create `constraints.txt` with the Triton CPU wheel URLs. Copy all three entries. uv automatically selects the one matching your Python version.

   ```bash
   cat > constraints.txt <<'EOF'
   triton @ https://github.com/triton-lang/triton-cpu/releases/download/v3.8.0/triton-3.8.0+cpu-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl ; python_version == '3.10'
   triton @ https://github.com/triton-lang/triton-cpu/releases/download/v3.8.0/triton-3.8.0+cpu-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl ; python_version == '3.11'
   triton @ https://github.com/triton-lang/triton-cpu/releases/download/v3.8.0/triton-3.8.0+cpu-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl ; python_version >= '3.12'
   EOF
   ```

2. Install `vllm-rbln` using the constraints file.

   ```bash
   uv pip install vllm-rbln \
     -c constraints.txt \
     --extra-index-url https://wheels.vllm.ai/0.31.0/cpu \
     --torch-backend cpu
   ```

##### Using pip
```bash
pip install vllm-rbln --extra-index-url https://wheels.vllm.ai/0.31.0/cpu --extra-index-url https://download.pytorch.org/whl/cpu
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
pip install -e . --extra-index-url https://wheels.vllm.ai/0.31.0/cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

#### ⚠️ Installing on Ubuntu 22.04 or Enterprise Linux 9

If installation fails with an error like the following, add the option below to your installation command:

```text
vllm==0.31.0+cpu has no wheels with a matching platform tag
(e.g., `manylinux_2_35_x86_64`)

hint: Wheels are available for `vllm` (v0.31.0+cpu) on the following platforms:
`manylinux_2_39_aarch64`, `manylinux_2_39_x86_64`
```

This error means that the upstream vLLM CPU wheel requires a newer glibc version than your system provides.
The wheel requires glibc 2.39 or later and libstdc++ from GCC 14 or later (`CXXABI_1.3.15`). Ubuntu 22.04 (glibc 2.35, GCC 12) and Enterprise Linux 9 (RHEL, AlmaLinux, and Rocky Linux 9.x; glibc 2.34, GCC 11) do not meet these requirements.

The `--find-links` option makes compatible vLLM wheels from the [vllm-rbln v0.12.0 release assets](https://github.com/RBLN-SW/vllm-rbln/releases/tag/v0.12.0) available to the installer:

```text
--find-links https://github.com/RBLN-SW/vllm-rbln/releases/expanded_assets/v0.12.0
```

When using `uv`, also add `--no-sources-package vllm` to allow installation of the compatible wheel.

### 🛠️ Development setup

Building from source? See **[DEVELOPMENT.md](./DEVELOPMENT.md)** for the `uv` dev workflow.

### 📚 Documentation

- [Overview & Supported Models](https://docs.rbln.ai/latest/software/model_serving/vllm_support/index.html)
- [Recipes](https://docs.rbln.ai/latest/software/model_serving/vllm_support/recipes/index.html)


## 🤝 Contributing

We welcome all contributions! Whether it's reporting issues, proposing enhancements, or improving docs—your input helps make the project better.

See our [CONTRIBUTING.md](./CONTRIBUTING.md) for more information.

## 📄 License

This project is licensed under the Apache License 2.0.

See the [LICENSE](./LICENSE) file for more information.

## 📧 Contact

- Join discussions and get answers in our [Developer Community](https://discuss.rebellions.ai/)
- Contact maintainers at [support@rebellions.ai](mailto:support@rebellions.ai)
