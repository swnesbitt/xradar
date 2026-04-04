# Installation

## Stable release

To install xradar, run this command in your terminal:

```bash
$ pip install xradar
```

This is the preferred method to install xradar, as it will always install the most recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide
you through the process.

## From conda-forge

To install xradar into your conda environment, run this command:

```bash
(my-nifty-env) $ conda install xradar -c conda-forge
```

You might also use the `conda` drop-in `mamba`. Of course you can omit `-c conda-forge` if you are already on that channel.

## From sources

The sources for xradar can be downloaded from the [Github repo](https://github.com/openradar/xradar).

You can either clone the public repository (recommended):

```bash
$ git clone git://github.com/openradar/xradar
```

Or download the [tarball](https://github.com/openradar/xradar/tarball/master):

```bash
$ curl -OJL https://github.com/openradar/xradar/tarball/master
```

```{warning}
The github tarballs have no notion of the xradar version. xradar will thus be installed as version 999.
```

Once you have a copy of the source, you can install it with:

```bash
$ python -m pip install .
```

## From github commit, branch or tag

You might also install directly from a specific commit, branch or tag:

```bash
$ python -m pip install git+https://github.com/openradar/xradar.git@92e2e4
$ python -m pip install git+https://github.com/openradar/xradar.git@main
$ python -m pip install git+https://github.com/openradar/xradar.git@0.0.5
```

## Rust NEXRAD extension (optional)

xradar includes an optional Rust-based NEXRAD Level2 parser that provides
3-6x faster ingest compared to the pure-Python parser. The extension uses
[PyO3](https://pyo3.rs/) and is built with [maturin](https://www.maturin.rs/).

When the Rust extension is not installed, xradar falls back to the pure-Python
parser automatically. No code changes are needed.

### Prerequisites

- [Rust toolchain](https://rustup.rs/) (stable, 1.70+)
- Python 3.11+

Install the Rust toolchain if you don't have it:

```bash
$ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
$ source "$HOME/.cargo/env"
```

### Building the extension

First, install xradar from source:

```bash
$ git clone https://github.com/openradar/xradar.git
$ cd xradar
$ pip install -e . --no-deps
```

Then build and install the Rust extension:

```bash
$ pip install maturin
$ maturin build --manifest-path rust/Cargo.toml --release --interpreter python
```

Install the built wheel (adjust the path for your platform/Python version):

```bash
$ pip install rust/target/wheels/xradar-*.whl --force-reinstall --no-deps
```

### Verifying the installation

```python
>>> from xradar.io.backends.nexrad_level2 import _HAS_RUST
>>> print(_HAS_RUST)
True
```

If `_HAS_RUST` is `False`, the Rust extension is not installed and xradar
will use the pure-Python parser.

### Development workflow

For iterative development on the Rust code, `maturin develop` builds and
installs the extension in one step (requires a virtualenv or conda env):

```bash
$ maturin develop --manifest-path rust/Cargo.toml --release
```

Run the Rust unit tests with:

```bash
$ cd rust && cargo test
```
