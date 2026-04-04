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

# Second volume — scan_01 (surveillance, 15 sweeps, 450 m bins)
BUFR_DIR2 = Path("/Users/snesbitt/data/bufr_relampago/00/0620")

pytestmark_vol2 = pytest.mark.skipif(
    not BUFR_DIR2.exists(),
    reason="RELAMPAGO vol2 data not available",
)

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

    def test_open_dataset_file_like(self):
        """Backend entrypoint should accept file-like objects."""
        import io
        import xarray as xr
        with open(DBZH_FILE, "rb") as f:
            buf = io.BytesIO(f.read())
        ds = xr.open_dataset(buf, engine="argentina_bufr", group="sweep_0")
        assert "DBZH" in ds
        ds.close()

    def test_open_dataset_invalid_group_raises(self):
        import xarray as xr
        with pytest.raises(ValueError):
            xr.open_dataset(DBZH_FILE, engine="argentina_bufr", group="bad_group")

    def test_open_dataset_group_out_of_range_raises(self):
        import xarray as xr
        with pytest.raises(ValueError):
            xr.open_dataset(DBZH_FILE, engine="argentina_bufr", group="sweep_99")


class TestSweepGeometryFields:
    """Validate geometry fields not otherwise checked."""

    @pytest.fixture(scope="class")
    def dbzh_bufr(self):
        with open(DBZH_FILE, "rb") as f:
            return BufrRustFile(f.read())

    def test_rotation_clockwise(self, dbzh_bufr):
        g = dbzh_bufr.get_sweep_geometry(0)
        assert g["rotation_clockwise"] is True

    def test_start_azimuth_sweep0(self, dbzh_bufr):
        az = dbzh_bufr.get_start_azimuth(0)
        assert 0.0 <= az < 360.0
        # Should match geometry dict
        g = dbzh_bufr.get_sweep_geometry(0)
        assert az == pytest.approx(g["start_azimuth_deg"], abs=0.1)

    def test_geometry_all_sweeps(self, dbzh_bufr):
        for i in range(dbzh_bufr.num_sweeps):
            g = dbzh_bufr.get_sweep_geometry(i)
            assert g["n_bins"] > 0
            assert g["bin_size_m"] > 0
            assert g["n_azimuths"] == 360
            assert 0.0 < g["elevation_deg"] < 90.0


class TestDualPolMoments:
    """Verify dual-polarization moment files are readable.

    Physical range checks are applied only to moments tightly bounded by
    physics (DBZV, RHOHV, PHIDP, CM).  ZDR and KDP are stored as
    pre-computed float64 values without QC, so noise-driven outliers are
    expected; only shape and dtype are verified for those moments.
    """

    # (moment, glob, vmin, vmax) — vmin/vmax=None means skip range check
    MOMENTS = [
        ("DBZV",  "*_DBZV_*.BUFR.gz",  -50.0, 80.0),
        ("ZDR",   "*_ZDR_*.BUFR.gz",   None,  None),
        ("RHOHV", "*_RHOHV_*.BUFR.gz",  0.0,  1.05),
        ("KDP",   "*_KDP_*.BUFR.gz",   None,  None),
        ("PHIDP", "*_PHIDP_*.BUFR.gz",  0.0, 360.0),
        ("CM",    "*_CM_*.BUFR.gz",     0.0,  16.0),
    ]

    @pytest.mark.parametrize("moment,glob,vmin,vmax", MOMENTS)
    def test_moment_readable(self, moment, glob, vmin, vmax):
        file = next(BUFR_DIR.glob(glob), None)
        if file is None:
            pytest.skip(f"No file matching {glob}")
        with open(file, "rb") as f:
            bufr = BufrRustFile(f.read())
        names = bufr.get_sweep_moment_names(0)
        assert moment in names
        arr = bufr.get_moment_data(0, moment)
        assert arr.shape[0] == 360
        assert arr.dtype == np.float64
        assert np.any(np.isfinite(arr)), f"{moment} has no finite values"
        if vmin is not None:
            valid = arr[np.isfinite(arr)]
            assert valid.min() >= vmin
            assert valid.max() <= vmax

    def test_all_moments_same_n_azimuths(self):
        """All moment files for the same sweep must have the same number of azimuths."""
        n_az_values = set()
        for fp in sorted(BUFR_DIR.glob("*.BUFR.gz")):
            with open(fp, "rb") as f:
                bufr = BufrRustFile(f.read())
            n_az_values.add(bufr.get_n_azimuths(0))
        assert len(n_az_values) == 1, f"Inconsistent n_azimuths across moment files: {n_az_values}"


class TestOpenArgentinaBufrDatatreeOptions:
    """Test optional parameters of open_argentina_bufr_datatree."""

    def test_site_coords_false(self):
        dtree = open_argentina_bufr_datatree(BUFR_DIR, site_coords=False)
        assert "latitude" not in dtree.coords
        assert "longitude" not in dtree.coords

    def test_reindex_angle(self):
        dtree = open_argentina_bufr_datatree(BUFR_DIR, reindex_angle=True)
        ds = dtree["sweep_0"].ds
        az = ds["azimuth"].values
        # Should be regularly spaced at 1.0° resolution
        diffs = np.diff(az)
        assert np.allclose(diffs, 1.0, atol=1e-3)
        assert len(az) == 360

    def test_list_of_paths_input(self):
        """Explicit file list should produce same result as directory."""
        file_list = sorted(BUFR_DIR.glob("*.BUFR.gz"))
        dtree = open_argentina_bufr_datatree(file_list)
        sweep_keys = [k for k in dtree.children if k.startswith("sweep_")]
        assert len(sweep_keys) == 3

    def test_sweep_list_multiple(self):
        dtree = open_argentina_bufr_datatree(BUFR_DIR, sweep=[0, 2])
        assert "sweep_0" in dtree.children
        assert "sweep_1" in dtree.children  # renumbered from original sweep_2
        assert "sweep_2" not in dtree.children

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, RuntimeError)):
            open_argentina_bufr_datatree(tmp_path)


@pytestmark_vol2
class TestVol2SurveillanceScan:
    """Second volume: scan_01, 15 sweeps, 450 m bin spacing."""

    @pytest.fixture(scope="class")
    def dtree2(self):
        return open_argentina_bufr_datatree(BUFR_DIR2)

    def test_num_sweeps(self, dtree2):
        sweep_keys = [k for k in dtree2.children if k.startswith("sweep_")]
        assert len(sweep_keys) == 15

    def test_bin_size_450m(self, dtree2):
        ds = dtree2["sweep_0"].ds
        r = ds["range"].values
        assert (r[1] - r[0]) == pytest.approx(450.0, abs=0.5)

    def test_elevations_ascending(self, dtree2):
        elevs = [dtree2[f"sweep_{i}"].ds.attrs["fixed_angle"] for i in range(15)]
        assert sorted(elevs) == elevs

    def test_high_elevation_sweep(self, dtree2):
        # Last sweep should be near 30°
        elev = dtree2["sweep_14"].ds.attrs["fixed_angle"]
        assert 25.0 < elev < 35.0

    def test_same_site_metadata(self, dtree2):
        assert dtree2.attrs["instrument_name"] == "RMA1"
        assert abs(dtree2.attrs["latitude"] - (-31.4413)) < 0.01
