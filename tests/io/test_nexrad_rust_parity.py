#!/usr/bin/env python
# Copyright (c) 2024-2025, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Parity tests: verify Rust and Python NEXRAD backends produce identical output."""

import numpy as np
import pytest
import xarray as xr

from xradar.io.backends.nexrad_level2 import _HAS_RUST

pytestmark = pytest.mark.skipif(not _HAS_RUST, reason="Rust NEXRAD extension not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_datatree(filename_or_obj, use_rust):
    """Open a NEXRAD file with a specific backend forced."""
    import xradar.io.backends.nexrad_level2 as mod

    saved = mod._HAS_RUST
    mod._HAS_RUST = use_rust
    try:
        dtree = mod.open_nexradlevel2_datatree(
            filename_or_obj, reindex_angle=False, site_coords=True
        )
    finally:
        mod._HAS_RUST = saved
    return dtree


def _open_dataset(filename, group, use_rust):
    """Open a single sweep with a specific backend forced."""
    import xradar.io.backends.nexrad_level2 as mod

    saved = mod._HAS_RUST
    mod._HAS_RUST = use_rust
    try:
        ds = xr.open_dataset(
            filename,
            engine="nexradlevel2",
            group=group,
            first_dim="auto",
        )
    finally:
        mod._HAS_RUST = saved
    return ds


# ---------------------------------------------------------------------------
# DataTree structure parity
# ---------------------------------------------------------------------------

class TestDataTreeParity:
    """Rust and Python DataTrees have the same structure and attributes."""

    @pytest.mark.parametrize(
        "fixture_name",
        ["nexradlevel2_file", "nexradlevel2_bzfile", "nexradlevel2_msg1_file"],
    )
    def test_same_sweep_keys(self, fixture_name, request):
        filename = request.getfixturevalue(fixture_name)
        dt_rust = _open_datatree(filename, use_rust=True)
        dt_py = _open_datatree(filename, use_rust=False)

        rust_keys = sorted(dt_rust.match("sweep_*").keys())
        py_keys = sorted(dt_py.match("sweep_*").keys())
        assert rust_keys == py_keys, f"Sweep keys differ: {rust_keys} vs {py_keys}"

    @pytest.mark.parametrize(
        "fixture_name",
        ["nexradlevel2_file", "nexradlevel2_bzfile", "nexradlevel2_msg1_file"],
    )
    def test_root_attrs_match(self, fixture_name, request):
        filename = request.getfixturevalue(fixture_name)
        dt_rust = _open_datatree(filename, use_rust=True)
        dt_py = _open_datatree(filename, use_rust=False)

        # Compare key root attributes
        for attr in [
            "instrument_name",
            "scan_name",
            "number_elevation_cuts",
            "actual_elevation_cuts",
        ]:
            rust_val = dt_rust.ds.attrs.get(attr)
            py_val = dt_py.ds.attrs.get(attr)
            assert rust_val == py_val, f"Root attr '{attr}': {rust_val} != {py_val}"

    @pytest.mark.parametrize(
        "fixture_name",
        ["nexradlevel2_file", "nexradlevel2_bzfile", "nexradlevel2_msg1_file"],
    )
    def test_root_coords_match(self, fixture_name, request):
        filename = request.getfixturevalue(fixture_name)
        dt_rust = _open_datatree(filename, use_rust=True)
        dt_py = _open_datatree(filename, use_rust=False)

        for coord in ["latitude", "longitude", "altitude"]:
            if coord in dt_py.ds.coords and coord in dt_rust.ds.coords:
                np.testing.assert_allclose(
                    float(dt_rust.ds[coord]),
                    float(dt_py.ds[coord]),
                    atol=1e-3,
                    err_msg=f"Root coord '{coord}' mismatch",
                )

    def test_chunk_files_same_sweeps(self, nexrad_chunks_klot):
        dt_rust = _open_datatree(nexrad_chunks_klot, use_rust=True)
        dt_py = _open_datatree(nexrad_chunks_klot, use_rust=False)

        rust_keys = sorted(dt_rust.match("sweep_*").keys())
        py_keys = sorted(dt_py.match("sweep_*").keys())
        assert rust_keys == py_keys


# ---------------------------------------------------------------------------
# Per-sweep coordinate parity
# ---------------------------------------------------------------------------

class TestCoordinateParity:
    """Coordinates match between Rust and Python for each sweep."""

    @pytest.mark.parametrize(
        "fixture_name",
        ["nexradlevel2_file", "nexradlevel2_bzfile"],
    )
    def test_azimuth_match(self, fixture_name, request):
        filename = request.getfixturevalue(fixture_name)
        for group in ["sweep_0", "sweep_1"]:
            ds_rust = _open_dataset(filename, group, use_rust=True)
            ds_py = _open_dataset(filename, group, use_rust=False)
            np.testing.assert_allclose(
                ds_rust.azimuth.values,
                ds_py.azimuth.values,
                atol=1e-4,
                err_msg=f"{fixture_name} {group} azimuth mismatch",
            )

    @pytest.mark.parametrize(
        "fixture_name",
        ["nexradlevel2_file", "nexradlevel2_bzfile"],
    )
    def test_elevation_match(self, fixture_name, request):
        filename = request.getfixturevalue(fixture_name)
        for group in ["sweep_0", "sweep_1"]:
            ds_rust = _open_dataset(filename, group, use_rust=True)
            ds_py = _open_dataset(filename, group, use_rust=False)
            np.testing.assert_allclose(
                ds_rust.elevation.values,
                ds_py.elevation.values,
                atol=1e-4,
                err_msg=f"{fixture_name} {group} elevation mismatch",
            )

    @pytest.mark.parametrize(
        "fixture_name",
        ["nexradlevel2_file", "nexradlevel2_bzfile"],
    )
    def test_time_match(self, fixture_name, request):
        filename = request.getfixturevalue(fixture_name)
        for group in ["sweep_0", "sweep_1"]:
            ds_rust = _open_dataset(filename, group, use_rust=True)
            ds_py = _open_dataset(filename, group, use_rust=False)
            np.testing.assert_array_equal(
                ds_rust.time.values,
                ds_py.time.values,
                err_msg=f"{fixture_name} {group} time mismatch",
            )

    @pytest.mark.parametrize(
        "fixture_name",
        ["nexradlevel2_file", "nexradlevel2_bzfile"],
    )
    def test_range_match(self, fixture_name, request):
        filename = request.getfixturevalue(fixture_name)
        for group in ["sweep_0", "sweep_1"]:
            ds_rust = _open_dataset(filename, group, use_rust=True)
            ds_py = _open_dataset(filename, group, use_rust=False)
            np.testing.assert_allclose(
                ds_rust.range.values,
                ds_py.range.values,
                atol=0.1,
                err_msg=f"{fixture_name} {group} range mismatch",
            )

    @pytest.mark.parametrize(
        "fixture_name",
        ["nexradlevel2_file", "nexradlevel2_bzfile"],
    )
    def test_sweep_fixed_angle_match(self, fixture_name, request):
        filename = request.getfixturevalue(fixture_name)
        for group in ["sweep_0", "sweep_1"]:
            ds_rust = _open_dataset(filename, group, use_rust=True)
            ds_py = _open_dataset(filename, group, use_rust=False)
            np.testing.assert_allclose(
                ds_rust.sweep_fixed_angle.values,
                ds_py.sweep_fixed_angle.values,
                atol=1e-6,
                err_msg=f"{fixture_name} {group} sweep_fixed_angle mismatch",
            )


# ---------------------------------------------------------------------------
# Per-sweep variable parity
# ---------------------------------------------------------------------------

class TestVariableParity:
    """Data variables match between Rust and Python for each sweep."""

    @pytest.mark.parametrize(
        "fixture_name",
        ["nexradlevel2_file", "nexradlevel2_bzfile"],
    )
    def test_same_data_vars(self, fixture_name, request):
        """Both backends produce the same set of moment variable names."""
        filename = request.getfixturevalue(fixture_name)
        for group in ["sweep_0", "sweep_1"]:
            ds_rust = _open_dataset(filename, group, use_rust=True)
            ds_py = _open_dataset(filename, group, use_rust=False)

            # Filter to actual radar moments (not sweep metadata)
            meta_vars = {"sweep_mode", "sweep_number", "prt_mode", "follow_mode", "sweep_fixed_angle"}
            rust_vars = sorted(set(ds_rust.data_vars) - meta_vars)
            py_vars = sorted(set(ds_py.data_vars) - meta_vars)
            assert rust_vars == py_vars, (
                f"{fixture_name} {group} data vars differ: {rust_vars} vs {py_vars}"
            )

    @pytest.mark.parametrize(
        "fixture_name",
        ["nexradlevel2_file", "nexradlevel2_bzfile"],
    )
    def test_data_shapes_match(self, fixture_name, request):
        """All moment arrays have the same shape."""
        filename = request.getfixturevalue(fixture_name)
        for group in ["sweep_0", "sweep_1"]:
            ds_rust = _open_dataset(filename, group, use_rust=True)
            ds_py = _open_dataset(filename, group, use_rust=False)

            meta_vars = {"sweep_mode", "sweep_number", "prt_mode", "follow_mode", "sweep_fixed_angle"}
            for var in set(ds_py.data_vars) - meta_vars:
                assert ds_rust[var].shape == ds_py[var].shape, (
                    f"{fixture_name} {group} {var} shape: "
                    f"{ds_rust[var].shape} != {ds_py[var].shape}"
                )

    @pytest.mark.parametrize(
        "fixture_name",
        ["nexradlevel2_file", "nexradlevel2_bzfile"],
    )
    def test_dbzh_values_match(self, fixture_name, request):
        """DBZH (reflectivity) values are identical."""
        filename = request.getfixturevalue(fixture_name)
        for group in ["sweep_0", "sweep_1"]:
            ds_rust = _open_dataset(filename, group, use_rust=True)
            ds_py = _open_dataset(filename, group, use_rust=False)

            if "DBZH" not in ds_rust.data_vars or "DBZH" not in ds_py.data_vars:
                continue

            rust_vals = ds_rust["DBZH"].values
            py_vals = ds_py["DBZH"].values

            # Both should have the same NaN pattern
            np.testing.assert_array_equal(
                np.isnan(rust_vals),
                np.isnan(py_vals),
                err_msg=f"{fixture_name} {group} DBZH NaN pattern differs",
            )

            # Non-NaN values should be close
            valid = ~np.isnan(py_vals)
            if valid.any():
                np.testing.assert_allclose(
                    rust_vals[valid],
                    py_vals[valid],
                    atol=0.1,
                    err_msg=f"{fixture_name} {group} DBZH values differ",
                )

    @pytest.mark.parametrize(
        "fixture_name",
        ["nexradlevel2_file", "nexradlevel2_bzfile"],
    )
    def test_all_moments_values_match(self, fixture_name, request):
        """All moment variables have matching values."""
        filename = request.getfixturevalue(fixture_name)
        ds_rust = _open_dataset(filename, "sweep_0", use_rust=True)
        ds_py = _open_dataset(filename, "sweep_0", use_rust=False)

        meta_vars = {"sweep_mode", "sweep_number", "prt_mode", "follow_mode", "sweep_fixed_angle"}
        for var in set(ds_py.data_vars) - meta_vars:
            if var not in ds_rust.data_vars:
                continue

            rust_vals = ds_rust[var].values
            py_vals = ds_py[var].values

            valid_rust = ~np.isnan(rust_vals)
            valid_py = ~np.isnan(py_vals)
            np.testing.assert_array_equal(
                valid_rust,
                valid_py,
                err_msg=f"{fixture_name} sweep_0 {var} NaN pattern differs",
            )

            valid = valid_py
            if valid.any():
                np.testing.assert_allclose(
                    rust_vals[valid],
                    py_vals[valid],
                    atol=0.1,
                    err_msg=f"{fixture_name} sweep_0 {var} values differ",
                )

    def test_scale_offset_attrs_match(self, nexradlevel2_file):
        """scale_factor and add_offset attributes match for all moments."""
        ds_rust = _open_dataset(nexradlevel2_file, "sweep_0", use_rust=True)
        ds_py = _open_dataset(nexradlevel2_file, "sweep_0", use_rust=False)

        meta_vars = {"sweep_mode", "sweep_number", "prt_mode", "follow_mode", "sweep_fixed_angle"}
        for var in set(ds_py.data_vars) - meta_vars:
            if var not in ds_rust.data_vars:
                continue
            for attr in ["scale_factor", "add_offset"]:
                rust_a = ds_rust[var].encoding.get(attr, ds_rust[var].attrs.get(attr))
                py_a = ds_py[var].encoding.get(attr, ds_py[var].attrs.get(attr))
                if py_a is not None and rust_a is not None:
                    np.testing.assert_allclose(
                        float(rust_a),
                        float(py_a),
                        atol=1e-6,
                        err_msg=f"sweep_0 {var} {attr} mismatch",
                    )


# ---------------------------------------------------------------------------
# Legacy MSG1 parity
# ---------------------------------------------------------------------------

class TestLegacyParity:
    """Legacy MSG1 files produce identical output from both backends."""

    def test_legacy_sweep_keys(self, nexradlevel2_msg1_file):
        dt_rust = _open_datatree(nexradlevel2_msg1_file, use_rust=True)
        dt_py = _open_datatree(nexradlevel2_msg1_file, use_rust=False)
        assert sorted(dt_rust.match("sweep_*").keys()) == sorted(
            dt_py.match("sweep_*").keys()
        )

    def test_legacy_data_vars(self, nexradlevel2_msg1_file):
        ds_rust = _open_dataset(nexradlevel2_msg1_file, "sweep_0", use_rust=True)
        ds_py = _open_dataset(nexradlevel2_msg1_file, "sweep_0", use_rust=False)
        meta_vars = {"sweep_mode", "sweep_number", "prt_mode", "follow_mode", "sweep_fixed_angle"}
        assert sorted(set(ds_rust.data_vars) - meta_vars) == sorted(
            set(ds_py.data_vars) - meta_vars
        )

    def test_legacy_dbzh_values(self, nexradlevel2_msg1_file):
        ds_rust = _open_dataset(nexradlevel2_msg1_file, "sweep_0", use_rust=True)
        ds_py = _open_dataset(nexradlevel2_msg1_file, "sweep_0", use_rust=False)

        if "DBZH" not in ds_rust.data_vars or "DBZH" not in ds_py.data_vars:
            pytest.skip("DBZH not in sweep_0")

        rust_vals = ds_rust["DBZH"].values
        py_vals = ds_py["DBZH"].values

        np.testing.assert_array_equal(
            np.isnan(rust_vals), np.isnan(py_vals),
            err_msg="Legacy DBZH NaN pattern differs",
        )
        valid = ~np.isnan(py_vals)
        if valid.any():
            np.testing.assert_allclose(
                rust_vals[valid], py_vals[valid], atol=0.1,
                err_msg="Legacy DBZH values differ",
            )


# ---------------------------------------------------------------------------
# Chunk file parity
# ---------------------------------------------------------------------------

class TestChunkParity:
    """Chunk-reassembled volumes produce identical output from both backends."""

    def test_chunk_data_vars(self, nexrad_chunks_klot):
        dt_rust = _open_datatree(nexrad_chunks_klot, use_rust=True)
        dt_py = _open_datatree(nexrad_chunks_klot, use_rust=False)

        meta_vars = {"sweep_mode", "sweep_number", "prt_mode", "follow_mode", "sweep_fixed_angle"}
        for key in sorted(dt_py.match("sweep_*").keys())[:3]:
            rust_vars = sorted(set(dt_rust[key].ds.data_vars) - meta_vars)
            py_vars = sorted(set(dt_py[key].ds.data_vars) - meta_vars)
            assert rust_vars == py_vars, f"{key} data vars differ"

    def test_chunk_dbzh_values(self, nexrad_chunks_klot):
        dt_rust = _open_datatree(nexrad_chunks_klot, use_rust=True)
        dt_py = _open_datatree(nexrad_chunks_klot, use_rust=False)

        ds_rust = dt_rust["sweep_0"].ds
        ds_py = dt_py["sweep_0"].ds

        if "DBZH" not in ds_rust.data_vars or "DBZH" not in ds_py.data_vars:
            pytest.skip("DBZH not in sweep_0")

        rust_vals = ds_rust["DBZH"].values
        py_vals = ds_py["DBZH"].values

        np.testing.assert_array_equal(
            np.isnan(rust_vals), np.isnan(py_vals),
            err_msg="Chunk DBZH NaN pattern differs",
        )
        valid = ~np.isnan(py_vals)
        if valid.any():
            np.testing.assert_allclose(
                rust_vals[valid], py_vals[valid], atol=0.1,
                err_msg="Chunk DBZH values differ",
            )
