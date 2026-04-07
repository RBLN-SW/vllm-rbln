# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# install_prerequisites.py
import argparse
import glob
import json
import os
import subprocess
import sys
import urllib.request

# --- Configuration ---
WHEELS_CACHE_HOME = os.environ.get("WHEELS_CACHE_HOME", "/tmp/wheels_cache")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UCX_DIR = os.path.join("/tmp", "ucx_source")
NIXL_DIR = os.path.join("/tmp", "nixl_source")
UCX_INSTALL_DIR = os.path.join("/tmp", "ucx_install")
UCX_REPO_URL = "https://github.com/openucx/ucx.git"
NIXL_REPO_URL = "https://github.com/ai-dynamo/nixl.git"


# --- Helper Functions ---
def get_latest_nixl_version():
    """Helper function to get latest release version of NIXL"""
    try:
        nixl_release_url = "https://api.github.com/repos/ai-dynamo/nixl/releases/latest"
        with urllib.request.urlopen(nixl_release_url) as response:
            data = json.load(response)
            return data.get("tag_name", "0.7.0")
    except Exception:
        return "0.7.0"


NIXL_VERSION = os.environ.get("NIXL_VERSION", get_latest_nixl_version())


def run_command(command, cwd=".", env=None):
    """Helper function to run a shell command and check for errors."""
    print(f"--> Running command: {' '.join(command)} in '{cwd}'", flush=True)
    subprocess.check_call(command, cwd=cwd, env=env)


def is_pip_package_installed(package_name):
    """Checks if a package is installed via pip without raising an exception."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", package_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def find_nixl_wheel_in_cache(cache_dir):
    """Finds a nixl wheel file in the specified cache directory."""
    # The repaired wheel will have a 'manylinux' tag, but this glob still works.
    search_pattern = os.path.join(cache_dir, f"nixl*{NIXL_VERSION}*.whl")
    wheels = glob.glob(search_pattern)
    if wheels:
        # Sort to get the most recent/highest version if multiple exist
        wheels.sort()
        return wheels[-1]
    return None


def get_site_packages_dir():
    """Returns the site-packages directory for the current Python."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import site; print(site.getsitepackages()[0])",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def install_nixl_shim_for_meta_compat(site_packages_dir):
    """
    Installs a minimal 'nixl' shim so that `import nixl` and
    `from nixl._api import nixl_agent` work like with the PyPI meta package.
    The source-built wheel installs as 'nixl-cu12' (import name: nixl_cu12);
    this shim re-exposes it under the 'nixl' namespace.
    """
    nixl_dir = os.path.join(site_packages_dir, "nixl")
    os.makedirs(nixl_dir, exist_ok=True)
    init_py = os.path.join(nixl_dir, "__init__.py")
    shim_content = """# Shim so that source-installed nixl-cu12 is usable as "nixl" (meta-package style).
# See install_nixl_from_source_ubuntu.py
import sys
import nixl_cu12
sys.modules["nixl._api"] = nixl_cu12._api
sys.modules["nixl._bindings"] = nixl_cu12._bindings
"""
    with open(init_py, "w") as f:
        f.write(shim_content)
    print(
        "--> Installed 'nixl' shim for meta-package-style import (import nixl, from nixl._api import ...)",
        flush=True,
    )


def install_system_dependencies():
    """Installs required system packages using apt-get if run as root."""
    if os.geteuid() != 0:
        print("\n---", flush=True)
        print(
            "WARNING: Not running as root. \
            Skipping system dependency installation.",
            flush=True,
        )
        print(
            "Please ensure the listed packages are installed on your system:",
            flush=True,
        )
        print(
            "  patchelf build-essential git cmake ninja-build \
            autotools-dev automake meson libtool libtool-bin",
            flush=True,
        )
        print("---\n", flush=True)
        return

    print("--- Running as root. Installing system dependencies... ---", flush=True)
    apt_packages = [
        "patchelf",  # <-- Add patchelf here
        "build-essential",
        "git",
        "cmake",
        "ninja-build",
        "autotools-dev",
        "automake",
        "meson",
        "libtool",
        "libtool-bin",
        "pkg-config",
    ]
    run_command(["apt-get", "update"])
    run_command(["apt-get", "install", "-y"] + apt_packages)
    print("--- System dependencies installed successfully. ---\n", flush=True)


def build_and_install_prerequisites(args):
    """Builds UCX and NIXL from source, creating a self-contained wheel."""

    if not args.force_reinstall and is_pip_package_installed("nixl"):
        print("--> NIXL is already installed. Nothing to do.", flush=True)
        return

    cached_wheel = find_nixl_wheel_in_cache(WHEELS_CACHE_HOME)
    if not args.force_reinstall and cached_wheel:
        print(
            f"\n--> Found self-contained wheel: \
                {os.path.basename(cached_wheel)}.",
            flush=True,
        )
        print("--> Installing from cache, skipping all source builds.", flush=True)
        install_command = [sys.executable, "-m", "pip", "install", cached_wheel]
        run_command(install_command)
        install_nixl_shim_for_meta_compat(get_site_packages_dir())
        print("\n--- Installation from cache complete. ---", flush=True)
        return

    print(
        "\n--> No installed package or cached wheel found. \
         Starting full build process...",
        flush=True,
    )
    print("\n--> Installing auditwheel...", flush=True)
    run_command([sys.executable, "-m", "pip", "install", "auditwheel"])
    install_system_dependencies()
    ucx_install_path = os.path.abspath(UCX_INSTALL_DIR)
    print(f"--> Using wheel cache directory: {WHEELS_CACHE_HOME}", flush=True)
    os.makedirs(WHEELS_CACHE_HOME, exist_ok=True)

    # -- Step 1: Build UCX from source --
    print("\n[1/3] Configuring and building UCX from source...", flush=True)
    if not os.path.exists(UCX_DIR):
        run_command(["git", "clone", UCX_REPO_URL, UCX_DIR])
    ucx_source_path = os.path.abspath(UCX_DIR)
    run_command(["git", "checkout", "v1.19.x"], cwd=ucx_source_path)
    run_command(["./autogen.sh"], cwd=ucx_source_path)
    configure_command = [
        "./configure",
        f"--prefix={ucx_install_path}",
        "--enable-shared",
        "--disable-static",
        "--disable-doxygen-doc",
        "--enable-optimizations",
        "--enable-cma",
        "--enable-devel-headers",
        "--with-verbs",
        "--enable-mt",
        "--with-ze=no",
    ]
    run_command(configure_command, cwd=ucx_source_path)
    run_command(["make", "-j", str(os.cpu_count() or 1)], cwd=ucx_source_path)
    run_command(["make", "install"], cwd=ucx_source_path)
    print("--- UCX build and install complete ---", flush=True)

    # -- Step 2: Build NIXL wheel from source --
    print("\n[2/3] Building NIXL wheel from source...", flush=True)
    if not os.path.exists(NIXL_DIR):
        run_command(["git", "clone", NIXL_REPO_URL, NIXL_DIR])
    else:
        run_command(["git", "fetch", "--tags"], cwd=NIXL_DIR)
    run_command(["git", "checkout", NIXL_VERSION], cwd=NIXL_DIR)
    print(f"--> Checked out NIXL version: {NIXL_VERSION}", flush=True)

    build_env = os.environ.copy()
    build_env["PKG_CONFIG_PATH"] = os.path.join(ucx_install_path, "lib", "pkgconfig")
    ucx_lib_path = os.path.join(ucx_install_path, "lib")
    ucx_plugin_path = os.path.join(ucx_lib_path, "ucx")
    existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    build_env["LD_LIBRARY_PATH"] = (
        f"{ucx_lib_path}:{ucx_plugin_path}:{existing_ld_path}".strip(":")
    )
    build_env["LDFLAGS"] = "-Wl,-rpath,$ORIGIN"
    print(f"--> Using LD_LIBRARY_PATH: {build_env['LD_LIBRARY_PATH']}", flush=True)

    temp_wheel_dir = os.path.join(ROOT_DIR, "temp_wheelhouse")
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            f"--wheel-dir={temp_wheel_dir}",
        ],
        cwd=os.path.abspath(NIXL_DIR),
        env=build_env,
    )

    # -- Step 3: Repair the wheel by copying UCX libraries --
    print("\n[3/3] Repairing NIXL wheel to include UCX libraries...", flush=True)
    unrepaired_wheel = find_nixl_wheel_in_cache(temp_wheel_dir)
    if not unrepaired_wheel:
        raise RuntimeError("Failed to find the NIXL wheel after building it.")

    # We tell auditwheel to ignore the plugin that mesonpy already handled.
    auditwheel_command = [
        "auditwheel",
        "repair",
        "--exclude",
        "libplugin_UCX.so",  # <-- Exclude because mesonpy already includes it
        unrepaired_wheel,
        f"--wheel-dir={WHEELS_CACHE_HOME}",
    ]
    run_command(auditwheel_command, env=build_env)

    # --- CLEANUP ---
    # No more temporary files to remove, just the temp wheelhouse
    run_command(["rm", "-rf", temp_wheel_dir])
    # --- END CLEANUP ---

    newly_built_wheel = find_nixl_wheel_in_cache(WHEELS_CACHE_HOME)
    if not newly_built_wheel:
        raise RuntimeError("Failed to find the repaired NIXL wheel.")

    print(
        f"--> Successfully built self-contained wheel: \
            {os.path.basename(newly_built_wheel)}. Now installing...",
        flush=True,
    )
    install_command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",  # w/o "no-deps", it will install cuda-torch
        newly_built_wheel,
    ]
    if args.force_reinstall:
        install_command.insert(-1, "--force-reinstall")

    run_command(install_command)
    install_nixl_shim_for_meta_compat(get_site_packages_dir())
    print("--- NIXL installation complete ---", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and install UCX and NIXL dependencies."
    )
    parser.add_argument(
        "--force-reinstall",
        action="store_true",
        help="Force rebuild and reinstall of UCX and NIXL \
        even if they are already installed.",
    )
    args = parser.parse_args()
    build_and_install_prerequisites(args)
