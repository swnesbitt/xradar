#!/usr/bin/env python
# Copyright (c) 2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for Argentine BUFR radar backend."""

import pytest
import numpy as np
from pathlib import Path

pytest.importorskip("xradar.io.backends._nexrad_rust", reason="Rust extension not available")

from xradar.io.backends._nexrad_rust import BufrRustFile
from xradar.io.backends.argentina_bufr import open_argentina_bufr_datatree

# Test data directory — scan_02 volume (3 sweeps, 12 moments)
BUFR_DIR = Path("/Users/snesbitt/data/bufr_relampago/00/0457")
DBZH_FILE = next(BUFR_DIR.glob("*_DBZH_*.BUFR.gz"), None)
VRADH_FILE = next(BUFR_DIR.glob("*_VRAD_*.BUFR.gz"), None)

pytestmark = pytest.mark.skipif(
    not BUFR_DIR.exists(),
    reason="RELAMPAGO test data not available at expected path",
)


@pytest.fixture(scope="module")
def dbzh_bufr():
    """BufrRustFile for the DBZH moment file."""
    with open(DBZH_FILE, "rb") as f:
        data = f.read()
    return BufrRustFile(data)


@pytest.fixture(scope="module")
def vradh_bufr():
    """BufrRustFile for the VRAD moment file."""
    with open(VRADH_FILE, "rb") as f:
        data = f.read()
    return BufrRustFile(data)


class TestBufrRustFileMetadata:
    def test_station_id(self, dbzh_bufr):
        assert dbzh_bufr.station_id == "RMA1"

    def test_station_type(self, dbzh_bufr):
        assert dbzh_bufr.station_type == "RMA"

    def test_latitude(self, dbzh_bufr):
        assert abs(dbzh_bufr.latitude - (-31.4413)) < 0.01

    def test_longitude(self, dbzh_bufr):
        assert abs(dbzh_bufr.longitude - (-64.1919)) < 0.01

    def test_height(self, dbzh_bufr):
        # RMA1 height ~484 m
        assert 400 < dbzh_bufr.height_m < 600

    def test_num_sweeps(self, dbzh_bufr):
        # scan_02 has 3 sweeps
        assert dbzh_bufr.num_sweeps == 3

    def test_nominal_time(self, dbzh_bufr):
        t = dbzh_bufr.nominal_time
        assert t.startswith("2018-11-11T")
        assert t.endswith("Z")

    def test_moment_names(self, dbzh_bufr):
        names = dbzh_bufr.moment_names
        assert "DBZH" in names


class TestBufrRustFileSweepGeometry:
    def test_elevation_sweep0(self, dbzh_bufr):
        elev = dbzh_bufr.get_elevation(0)
        # First sweep is low: typically ~0.5°
        assert 0.0 < elev < 5.0

    def test_n_bins(self, dbzh_bufr):
        assert dbzh_bufr.get_n_bins(0) == 956

    def test_bin_size(self, dbzh_bufr):
        assert dbzh_bufr.get_bin_size_m(0) == 120.0

    def test_bin_offset(self, dbzh_bufr):
        # First bin center at 2100 m
        assert abs(dbzh_bufr.get_bin_offset_m(0) - 2100.0) < 1.0

    def test_n_azimuths(self, dbzh_bufr):
        assert dbzh_bufr.get_n_azimuths(0) == 360

    def test_sweep_times(self, dbzh_bufr):
        start = dbzh_bufr.get_sweep_start_time(0)
        end = dbzh_bufr.get_sweep_end_time(0)
        assert start.startswith("2018-11-11T")
        assert end >= start

    def test_all_sweeps_have_different_elevations(self, dbzh_bufr):
        elevs = [dbzh_bufr.get_elevation(i) for i in range(dbzh_bufr.num_sweeps)]
        # All elevations should be distinct
        assert len(set(round(e, 2) for e in elevs)) == len(elevs)


class TestBufrRustFileMomentData:
    def test_moment_data_shape(self, dbzh_bufr):
        arr = dbzh_bufr.get_moment_data(0, "DBZH")
        assert arr.shape == (360, 956)

    def test_moment_data_dtype(self, dbzh_bufr):
        arr = dbzh_bufr.get_moment_data(0, "DBZH")
        assert arr.dtype == np.float64

    def test_moment_data_physical_range(self, dbzh_bufr):
        arr = dbzh_bufr.get_moment_data(0, "DBZH")
        # DBZH values should be in plausible physical range
        valid = arr[np.isfinite(arr)]
        if len(valid) > 0:
            assert valid.min() > -50.0
            assert valid.max() < 80.0

    def test_moment_not_found_raises(self, dbzh_bufr):
        with pytest.raises(Exception):
            dbzh_bufr.get_moment_data(0, "NONEXISTENT")

    def test_sweep_out_of_range_raises(self, dbzh_bufr):
        with pytest.raises(Exception):
            dbzh_bufr.get_elevation(99)


class TestElevationMatching:
    def test_dbzh_vrad_same_elevations(self, dbzh_bufr, vradh_bufr):
        """DBZH and VRAD files should have matching sweep elevations."""
        dbzh_elevs = [round(dbzh_bufr.get_elevation(i), 1) for i in range(dbzh_bufr.num_sweeps)]
        vrad_elevs = [round(vradh_bufr.get_elevation(i), 1) for i in range(vradh_bufr.num_sweeps)]
        assert dbzh_elevs == vrad_elevs

    def test_dbzh_vrad_same_geometry(self, dbzh_bufr, vradh_bufr):
        """Geometry should match between moment files for the same sweep."""
        assert dbzh_bufr.get_n_bins(0) == vradh_bufr.get_n_bins(0)
        assert dbzh_bufr.get_n_azimuths(0) == vradh_bufr.get_n_azimuths(0)


class TestOpenArgentinaBufrDatatree:
    @pytest.fixture(scope="class")
    def dtree(self):
        return open_argentina_bufr_datatree(BUFR_DIR)

    def test_num_sweeps(self, dtree):
        sweep_keys = [k for k in dtree.children if k.startswith("sweep_")]
        assert len(sweep_keys) == 3

    def test_sweep_keys(self, dtree):
        assert "sweep_0" in dtree.children
        assert "sweep_1" in dtree.children
        assert "sweep_2" in dtree.children

    def test_root_attrs(self, dtree):
        assert dtree.attrs.get("instrument_name") == "RMA1"
        assert abs(dtree.attrs.get("latitude", 0) - (-31.4413)) < 0.01

    def test_root_coords(self, dtree):
        assert "latitude" in dtree.coords
        assert "longitude" in dtree.coords
        assert "altitude" in dtree.coords

    def test_sweep_has_dbzh(self, dtree):
        ds = dtree["sweep_0"].ds
        assert "DBZH" in ds

    def test_sweep_has_vradh(self, dtree):
        ds = dtree["sweep_0"].ds
        assert "VRADH" in ds

    def test_sweep_coords(self, dtree):
        ds = dtree["sweep_0"].ds
        assert "azimuth" in ds.coords
        assert "elevation" in ds.coords
        assert "range" in ds.coords

    def test_sweep_dbzh_shape(self, dtree):
        ds = dtree["sweep_0"].ds
        dbzh = ds["DBZH"]
        assert dbzh.dims == ("azimuth", "range")
        assert dbzh.shape[0] == 360
        assert dbzh.shape[1] == 956

    def test_sweep_elevation_values(self, dtree):
        elevs = [dtree[f"sweep_{i}"].ds.attrs.get("fixed_angle") for i in range(3)]
        # Should be 3 distinct elevations in ascending order
        assert sorted(elevs) == elevs
        assert len(set(round(e, 1) for e in elevs)) == 3

    def test_range_coordinate(self, dtree):
        ds = dtree["sweep_0"].ds
        r = ds["range"].values
        assert r[0] == pytest.approx(2100.0, abs=1.0)
        assert (r[1] - r[0]) == pytest.approx(120.0, abs=0.1)


class TestOpenArgentinaBufrDatatreeMomentSubset:
    def test_moment_filter(self):
        dtree = open_argentina_bufr_datatree(BUFR_DIR, moments=["DBZH"])
        ds = dtree["sweep_0"].ds
        assert "DBZH" in ds
        assert "VRADH" not in ds

    def test_sweep_filter(self):
        dtree = open_argentina_bufr_datatree(BUFR_DIR, sweep=[0])
        assert "sweep_0" in dtree.children
        assert "sweep_1" not in dtree.children


class TestXarrayEngine:
    def test_open_dataset_engine(self):
        import xarray as xr
        ds = xr.open_dataset(DBZH_FILE, engine="argentina_bufr", group="sweep_0")
        assert "DBZH" in ds
        assert "azimuth" in ds.coords
        assert "range" in ds.coords
        ds.close()

    def test_open_dataset_sweep1(self):
        import xarray as xr
        ds0 = xr.open_dataset(DBZH_FILE, engine="argentina_bufr", group="sweep_0")
        ds1 = xr.open_dataset(DBZH_FILE, engine="argentina_bufr", group="sweep_1")
        elev0 = float(ds0["elevation"].mean())
        elev1 = float(ds1["elevation"].mean())
        assert elev1 > elev0
        ds0.close()
        ds1.close()
