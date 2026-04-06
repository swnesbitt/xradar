#!/usr/bin/env python
# Copyright (c) 2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Robustness tests for the Argentine BUFR backend against a large collection
of real-world radar volumes from multiple RMA and AR-series sites.

Data root: /Volumes/bigssd1/rma_data/
Structure: <DATA_ROOT>/<SITE>/<HHMMSS>/<files>.BUFR[.gz]

These tests are skipped automatically when the external drive is absent.
Run explicitly with:
    pytest tests/io/test_argentina_bufr_robustness.py -v
"""

import importlib
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

# Skip whole module if Rust extension is unavailable
if importlib.util.find_spec("xradar.io.backends._nexrad_rust") is None:
    pytest.skip("Rust extension not available", allow_module_level=True)

from xradar.io.backends._nexrad_rust import BufrRustFile
from xradar.io.backends.argentina_bufr import open_argentina_bufr_datatree

DATA_ROOT = Path("/Volumes/bigssd1/rma_data")


# ---------------------------------------------------------------------------
# Autouse fixture: skip every test when the drive is not mounted
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def require_data_root():
    if not DATA_ROOT.exists():
        pytest.skip("External drive /Volumes/bigssd1/rma_data not mounted")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_volume_dirs():
    """Return every volume directory (site/HHMMSS) that contains BUFR files."""
    if not DATA_ROOT.exists():
        return []
    dirs = []
    for site_dir in sorted(DATA_ROOT.iterdir()):
        if not site_dir.is_dir():
            continue
        for vol_dir in sorted(site_dir.iterdir()):
            if not vol_dir.is_dir():
                continue
            has_bufr = (
                any(vol_dir.glob("*.BUFR.gz"))
                or any(vol_dir.glob("*.BUFR"))
            )
            if has_bufr:
                dirs.append(vol_dir)
    return dirs


def _volume_ids(dirs):
    return [f"{d.parent.name}/{d.name}" for d in dirs]


def _one_dbzh_file(vol_dir):
    """Return the DBZH moment file from a volume directory (gz or plain)."""
    for pattern in ("*_DBZH_*.BUFR.gz", "*_DBZH_*.BUFR"):
        f = next(vol_dir.glob(pattern), None)
        if f is not None:
            return f
    return None


ALL_VOLUMES = _all_volume_dirs()
ALL_IDS = _volume_ids(ALL_VOLUMES)
SITES = sorted({d.parent.name for d in ALL_VOLUMES})


# ---------------------------------------------------------------------------
# Per-volume parametrised tests
# ---------------------------------------------------------------------------

class TestEveryVolume:
    """Open every volume as a DataTree and perform basic sanity checks."""

    @pytest.mark.parametrize("vol_dir", ALL_VOLUMES, ids=ALL_IDS)
    def test_datatree_opens(self, vol_dir):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dtree = open_argentina_bufr_datatree(vol_dir)

        sweep_keys = [k for k in dtree.children if k.startswith("sweep_")]
        assert len(sweep_keys) >= 1, f"{vol_dir}: no sweeps found"

    @pytest.mark.parametrize("vol_dir", ALL_VOLUMES, ids=ALL_IDS)
    def test_has_dbzh(self, vol_dir):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dtree = open_argentina_bufr_datatree(vol_dir, moments=["DBZH"])

        ds = dtree["sweep_0"].ds
        assert "DBZH" in ds, f"{vol_dir}: DBZH missing from sweep_0"

    @pytest.mark.parametrize("vol_dir", ALL_VOLUMES, ids=ALL_IDS)
    def test_sweep0_coords(self, vol_dir):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dtree = open_argentina_bufr_datatree(vol_dir, moments=["DBZH"])

        ds = dtree["sweep_0"].ds
        for coord in ("azimuth", "elevation", "range", "time", "latitude", "longitude"):
            assert coord in ds.coords, f"{vol_dir}: missing coord '{coord}'"

    @pytest.mark.parametrize("vol_dir", ALL_VOLUMES, ids=ALL_IDS)
    def test_elevations_positive_and_ascending(self, vol_dir):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dtree = open_argentina_bufr_datatree(vol_dir, moments=["DBZH"])

        sweep_keys = sorted(
            (k for k in dtree.children if k.startswith("sweep_")),
            key=lambda k: int(k.split("_")[1]),
        )
        elevs = [dtree[k].ds.attrs["fixed_angle"] for k in sweep_keys]

        assert all(e > 0 for e in elevs), f"{vol_dir}: non-positive elevation: {elevs}"
        assert elevs == sorted(elevs), f"{vol_dir}: elevations not ascending: {elevs}"

    @pytest.mark.parametrize("vol_dir", ALL_VOLUMES, ids=ALL_IDS)
    def test_dbzh_physical_range(self, vol_dir):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dtree = open_argentina_bufr_datatree(vol_dir, moments=["DBZH"])

        dbzh = dtree["sweep_0"].ds["DBZH"].values
        valid = dbzh[np.isfinite(dbzh)]
        if len(valid) == 0:
            pytest.skip(f"{vol_dir}: sweep_0 DBZH has no finite values")

        assert valid.min() > -50.0, f"{vol_dir}: DBZH min {valid.min():.1f} < -50"
        assert valid.max() < 80.0,  f"{vol_dir}: DBZH max {valid.max():.1f} > 80"

    @pytest.mark.parametrize("vol_dir", ALL_VOLUMES, ids=ALL_IDS)
    def test_azimuths_monotonic(self, vol_dir):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dtree = open_argentina_bufr_datatree(vol_dir, moments=["DBZH"])

        az = dtree["sweep_0"].ds["azimuth"].values
        assert (np.diff(az) > 0).all(), (
            f"{vol_dir}: sweep_0 azimuths not monotonically increasing"
        )

    @pytest.mark.parametrize("vol_dir", ALL_VOLUMES, ids=ALL_IDS)
    def test_georeference(self, vol_dir):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dtree = open_argentina_bufr_datatree(vol_dir, moments=["DBZH"])
            dtree = dtree.xradar.georeference()

        ds = dtree["sweep_0"].ds
        for coord in ("x", "y", "z"):
            assert coord in ds.coords, f"{vol_dir}: georeference missing '{coord}'"
        x = ds["x"].values
        assert np.any(np.isfinite(x)), f"{vol_dir}: all x coords NaN after georeference"


# ---------------------------------------------------------------------------
# Per-site consistency tests
# ---------------------------------------------------------------------------

class TestPerSite:
    """Cross-volume consistency checks per radar site."""

    @pytest.mark.parametrize("site", SITES)
    def test_consistent_station_id(self, site):
        """station_id should be the same across all volumes for a given site."""
        site_dir = DATA_ROOT / site
        station_ids = set()
        for vol_dir in sorted(site_dir.iterdir()):
            if not vol_dir.is_dir():
                continue
            f = _one_dbzh_file(vol_dir)
            if f is None:
                continue
            with open(f, "rb") as fh:
                bufr = BufrRustFile(fh.read())
            station_ids.add(bufr.station_id.upper())

        assert len(station_ids) == 1, (
            f"{site}: inconsistent station_id across volumes: {station_ids}"
        )

    @pytest.mark.parametrize("site", SITES)
    def test_consistent_location(self, site):
        """Lat/lon should not vary by more than 0.01° across volumes."""
        site_dir = DATA_ROOT / site
        lats, lons = [], []
        for vol_dir in sorted(site_dir.iterdir()):
            if not vol_dir.is_dir():
                continue
            f = _one_dbzh_file(vol_dir)
            if f is None:
                continue
            with open(f, "rb") as fh:
                bufr = BufrRustFile(fh.read())
            lats.append(bufr.latitude)
            lons.append(bufr.longitude)

        if len(lats) < 2:
            pytest.skip(f"{site}: only one volume, nothing to compare")

        assert max(lats) - min(lats) < 0.01, (
            f"{site}: lat varies {max(lats) - min(lats):.4f}°"
        )
        assert max(lons) - min(lons) < 0.01, (
            f"{site}: lon varies {max(lons) - min(lons):.4f}°"
        )


# ---------------------------------------------------------------------------
# Uncompressed .BUFR format
# ---------------------------------------------------------------------------

class TestUncompressedFormat:
    """Verify that plain .BUFR files (no gzip) open correctly."""

    @pytest.fixture(scope="class")
    def uncompressed_vol(self):
        for vol_dir in ALL_VOLUMES:
            if any(vol_dir.glob("*.BUFR")) and not any(vol_dir.glob("*.BUFR.gz")):
                return vol_dir
        pytest.skip("No uncompressed .BUFR volumes found")

    def test_datatree_opens(self, uncompressed_vol):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dtree = open_argentina_bufr_datatree(uncompressed_vol)
        assert len([k for k in dtree.children if k.startswith("sweep_")]) >= 1

    def test_xarray_engine(self, uncompressed_vol):
        f = _one_dbzh_file(uncompressed_vol)
        if f is None:
            pytest.skip("No DBZH file in volume")
        ds = xr.open_dataset(f, engine="argentina_bufr", group="sweep_0")
        assert "DBZH" in ds
        ds.close()


# ---------------------------------------------------------------------------
# Pyart interop smoke test
# ---------------------------------------------------------------------------

class TestPyartInterop:
    @pytest.fixture(scope="class")
    def pyart(self):
        pyart = pytest.importorskip("pyart")
        return pyart

    @pytest.fixture(scope="class")
    def multi_sweep_vol(self):
        """Pick any volume with at least 3 BUFR files (enough for a pseudo-RHI)."""
        for vol_dir in ALL_VOLUMES:
            files = list(vol_dir.glob("*.BUFR*"))
            if len(files) >= 3:
                return vol_dir
        pytest.skip("No multi-sweep volume available")

    def test_to_radar(self, pyart, multi_sweep_vol):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dtree = open_argentina_bufr_datatree(multi_sweep_vol, moments=["DBZH"])

        radar = dtree.pyart.to_radar()
        assert radar.nsweeps >= 1
        assert "DBZH" in radar.fields

    def test_plot_azimuth_to_rhi(self, pyart, multi_sweep_vol):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dtree = open_argentina_bufr_datatree(multi_sweep_vol, moments=["DBZH"])

        radar = dtree.pyart.to_radar()
        display = pyart.graph.RadarDisplay(radar)
        fig, ax = plt.subplots()
        display.plot_azimuth_to_rhi("DBZH", 235, ax=ax)
        plt.close(fig)
