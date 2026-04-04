#!/usr/bin/env python
# Copyright (c) 2024, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Argentine BUFR Radar Backend
=============================

Backend for reading Argentine RMA network radar data in BUFR Edition 4 format.

Each volume scan is stored as a directory of ``.BUFR.gz`` files, one per moment.
Use :func:`open_argentina_bufr_datatree` to open an entire volume.

Example::

    import xradar as xr
    dtree = xr.io.open_argentina_bufr_datatree("/path/to/volume/dir/")
    # or a single file
    ds = xr.open_dataset("RMA1_..._DBZH_...BUFR.gz", engine="argentina_bufr", group="sweep_0")

"""

import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from xarray.backends import BackendEntrypoint
from xarray.backends.common import AbstractDataStore, BackendArray
from xarray.core import indexing

from xradar.model import get_altitude_attrs, get_latitude_attrs, get_longitude_attrs

try:
    from xradar.io.backends._nexrad_rust import BufrRustFile

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
    BufrRustFile = None

__all__ = [
    "ArgentinaBufrBackendEntrypoint",
    "open_argentina_bufr_datatree",
]


class ArgentinaBufrArrayWrapper(BackendArray):
    """Holds a pre-loaded numpy array for lazy xarray indexing.

    PyO3 objects cannot be pickled, so we eagerly load the data into numpy
    and store it here, mirroring the NEXRAD Rust pattern.
    """

    def __init__(self, data: np.ndarray):
        self._data = data
        self.dtype = data.dtype
        self.shape = data.shape

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._getitem
        )

    def _getitem(self, key):
        return self._data[key]


class ArgentinaBufrSweepStore(AbstractDataStore):
    """xarray DataStore for one sweep from the merged multi-moment volume.

    Args:
        geometry: dict of sweep geometry from BufrRustFile.get_sweep_geometry()
        moment_arrays: dict of {cfradial_name: np.ndarray (n_azimuths × n_bins)}
        site_attrs: dict with latitude, longitude, height_m, station_id
        sweep_idx: integer sweep index
    """

    def __init__(self, geometry, moment_arrays, site_attrs, sweep_idx):
        self._geometry = geometry
        self._moment_arrays = moment_arrays
        self._site_attrs = site_attrs
        self._sweep_idx = sweep_idx

    def get_variables(self):
        g = self._geometry
        n_az = g["n_azimuths"]

        # Use the maximum n_bins across all moments as the canonical range dimension.
        # Moments with fewer bins are padded with NaN on the far end.
        max_n_bins = g["n_bins"]
        for data in self._moment_arrays.values():
            if data.shape[1] > max_n_bins:
                max_n_bins = data.shape[1]

        # The Argentine processing pipeline stores data in a fixed physical order
        # (0°→360°) regardless of where the antenna started rotating.
        # start_azimuth_deg is metadata only; do not use it to reorder rows.
        az_step = 360.0 / n_az
        azimuths = np.arange(n_az) * az_step

        # Build range coordinate using canonical geometry bins
        ranges = g["bin_offset_m"] + np.arange(max_n_bins) * g["bin_size_m"]

        # Elevation: uniform for PPI
        elevations = np.full(n_az, g["elevation_deg"], dtype=np.float32)

        # Per-ray times: linearly interpolated between sweep start and end
        t0 = np.datetime64(g["start_time"].replace("Z", ""), "ns")
        t1 = np.datetime64(g["end_time"].replace("Z", ""), "ns")
        ray_times = np.linspace(0, 1, n_az) * (t1 - t0) + t0

        variables = {
            "azimuth": xr.Variable(
                ("azimuth",),
                azimuths.astype(np.float32),
                attrs={"units": "degrees", "standard_name": "ray_azimuth_angle"},
            ),
            "elevation": xr.Variable(
                ("azimuth",),
                elevations,
                attrs={"units": "degrees", "standard_name": "ray_elevation_angle"},
            ),
            "range": xr.Variable(
                ("range",),
                ranges.astype(np.float32),
                attrs={
                    "units": "meters",
                    "standard_name": "projection_range_coordinate",
                    "meters_to_center_of_first_gate": float(g["bin_offset_m"]),
                    "meters_between_gates": float(g["bin_size_m"]),
                },
            ),
            "time": xr.Variable(
                ("azimuth",),
                ray_times,
                attrs={"standard_name": "time"},
            ),
            "sweep_number": xr.Variable(
                (),
                np.int32(self._sweep_idx),
                attrs={"units": "1", "standard_name": "sweep_number"},
            ),
            "sweep_fixed_angle": xr.Variable(
                (),
                np.float32(g["elevation_deg"]),
                attrs={"units": "degrees", "standard_name": "beam_elevation_angle"},
            ),
            "sweep_mode": xr.Variable(
                (),
                "azimuth_surveillance",
                attrs={"standard_name": "sweep_mode"},
            ),
            "latitude": xr.Variable(
                (),
                self._site_attrs.get("latitude", np.nan),
                attrs=get_latitude_attrs(),
            ),
            "longitude": xr.Variable(
                (),
                self._site_attrs.get("longitude", np.nan),
                attrs=get_longitude_attrs(),
            ),
            "altitude": xr.Variable(
                (),
                self._site_attrs.get("altitude", np.nan),
                attrs=get_altitude_attrs(),
            ),
        }

        for moment_name, data in self._moment_arrays.items():
            # Pad to max_n_bins if needed
            if data.shape[1] < max_n_bins:
                pad = np.full((n_az, max_n_bins - data.shape[1]), np.nan, dtype=np.float32)
                data = np.concatenate([data.astype(np.float32), pad], axis=1)
            else:
                data = data.astype(np.float32)
            wrapper = ArgentinaBufrArrayWrapper(data)
            variables[moment_name] = xr.Variable(
                ("azimuth", "range"),
                indexing.LazilyIndexedArray(wrapper),
                attrs={
                    "units": _moment_units(moment_name),
                    "long_name": moment_name,
                },
            )

        return variables

    def get_attrs(self):
        g = self._geometry
        return {
            "sweep_number": self._sweep_idx,
            "sweep_mode": "azimuth_surveillance",
            "fixed_angle": float(g["elevation_deg"]),
            "sweep_start_time": g["start_time"],
            "sweep_end_time": g["end_time"],
            "instrument_name": self._site_attrs.get("instrument_name", ""),
            "latitude": self._site_attrs.get("latitude", np.nan),
            "longitude": self._site_attrs.get("longitude", np.nan),
            "altitude": self._site_attrs.get("altitude", np.nan),
        }

    def get_dimensions(self):
        g = self._geometry
        max_n_bins = max(
            (data.shape[1] for data in self._moment_arrays.values()),
            default=g["n_bins"],
        )
        max_n_bins = max(max_n_bins, g["n_bins"])
        return {"azimuth": g["n_azimuths"], "range": max_n_bins}

    def get_encoding(self):
        return {}


def _moment_units(name):
    units_map = {
        "DBZH": "dBZ",
        "DBZV": "dBZ",
        "DBTH": "dBZ",
        "DBTV": "dBZ",
        "VRADH": "m/s",
        "WRADH": "m/s",
        "ZDR": "dB",
        "KDP": "deg/km",
        "PHIDP": "degrees",
        "RHOHV": "1",
        "CM": "1",
        "TDR": "dB",
    }
    return units_map.get(name, "")


class ArgentinaBufrBackendEntrypoint(BackendEntrypoint):
    """xarray BackendEntrypoint for Argentine BUFR radar files.

    Opens a single ``.BUFR.gz`` file (one moment, all sweeps).
    Use ``group='sweep_N'`` to select a sweep.

    For full multi-moment volumes, use :func:`open_argentina_bufr_datatree`.
    """

    description = "Open Argentine RMA BUFR radar files (.BUFR.gz)"
    open_dataset_parameters = [
        "filename_or_obj",
        "mask_and_scale",
        "decode_times",
        "concat_characters",
        "decode_coords",
        "drop_variables",
        "group",
    ]

    def open_dataset(
        self,
        filename_or_obj,
        *,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables=None,
        group=None,
    ):
        if not _HAS_RUST:
            raise ImportError(
                "The Rust extension (_nexrad_rust) is required for the argentina_bufr backend. "
                "Install xradar with Rust support: pip install xradar"
            )

        # Read raw bytes
        if isinstance(filename_or_obj, (str, Path)):
            with open(filename_or_obj, "rb") as f:
                data = f.read()
        else:
            data = filename_or_obj.read()

        bufr = BufrRustFile(data)

        # Parse group -> sweep index
        sweep_idx = _parse_group(group, bufr.num_sweeps)

        geometry = bufr.get_sweep_geometry(sweep_idx)
        moment_names = bufr.get_sweep_moment_names(sweep_idx)
        moment_arrays = {
            name: bufr.get_moment_data(sweep_idx, name)
            for name in moment_names
        }

        site_attrs = {
            "instrument_name": bufr.station_id,
            "latitude": bufr.latitude,
            "longitude": bufr.longitude,
            "altitude": bufr.height_m,
        }

        store = ArgentinaBufrSweepStore(geometry, moment_arrays, site_attrs, sweep_idx)
        store_entrypoint = xr.backends.store.StoreBackendEntrypoint()

        ds = store_entrypoint.open_dataset(
            store,
            mask_and_scale=mask_and_scale,
            decode_times=False,  # we handle times as strings
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
        )
        ds = ds.set_coords(["azimuth", "elevation", "range", "time", "latitude", "longitude", "altitude"])
        return ds


def _parse_group(group, num_sweeps):
    """Parse 'sweep_N' group string to integer sweep index."""
    if group is None:
        return 0
    if isinstance(group, int):
        return group
    if isinstance(group, str) and group.startswith("sweep_"):
        idx = int(group.split("_")[1])
        if idx >= num_sweeps:
            raise ValueError(f"Group '{group}' out of range (only {num_sweeps} sweeps)")
        return idx
    raise ValueError(f"Invalid group: '{group}'. Use 'sweep_N' format.")


def open_argentina_bufr_datatree(
    path,
    moments=None,
    sweep=None,
    reindex_angle=False,
    site_coords=True,
    **kwargs,
):
    """Open an Argentine RMA BUFR radar volume as an xarray DataTree.

    A volume is a directory of ``.BUFR.gz`` files — one per moment.
    This function opens all moment files, matches sweeps by elevation angle,
    and assembles them into a single DataTree with one node per sweep.

    Parameters
    ----------
    path : str or Path or list of str/Path
        Path to a directory containing ``.BUFR.gz`` files, or an explicit
        list of ``.BUFR.gz`` file paths for one volume.
    moments : list of str, optional
        Subset of CFRadial moment names to load (e.g. ``['DBZH', 'VRADH']``).
        Default is ``None`` (load all available moments).
    sweep : int or list of int, optional
        Sweep indices to include. Default is ``None`` (all sweeps).
    reindex_angle : bool, optional
        If True, reindex azimuths to a regular grid. Default is False.
    site_coords : bool, optional
        If True, add latitude/longitude/altitude as root-level coordinates.
    **kwargs
        Additional keyword arguments (reserved for future use).

    Returns
    -------
    xarray.DataTree
        Root node contains site metadata; child nodes are ``sweep_0``,
        ``sweep_1``, etc., each containing all moments for that elevation.

    Examples
    --------
    >>> dtree = open_argentina_bufr_datatree("/data/bufr/0457/")
    >>> dtree["sweep_0"]["DBZH"]
    """
    if not _HAS_RUST:
        raise ImportError(
            "The Rust extension (_nexrad_rust) is required. "
            "Install xradar with Rust support."
        )

    # Collect file paths
    file_paths = _collect_bufr_files(path)
    if not file_paths:
        raise FileNotFoundError(f"No .BUFR.gz files found in {path}")

    # Parse each file: read bytes, create BufrRustFile
    bufr_files = []
    for fp in file_paths:
        with open(fp, "rb") as f:
            raw = f.read()
        try:
            bufr = BufrRustFile(raw)
            bufr_files.append((fp, bufr))
        except Exception as e:
            warnings.warn(f"Failed to parse {fp}: {e}")

    if not bufr_files:
        raise RuntimeError("No BUFR files could be parsed.")

    # Use first file for site metadata
    _first_path, first_bufr = bufr_files[0]
    site_attrs = {
        "instrument_name": first_bufr.station_id,
        "latitude": first_bufr.latitude,
        "longitude": first_bufr.longitude,
        "altitude": first_bufr.height_m,
        "time_coverage_start": first_bufr.nominal_time,
    }

    # Build elevation index: elevation_deg → {moment_name: (BufrRustFile, sweep_idx)}
    elev_index = _build_elevation_index(bufr_files, moments)

    # Sort elevations
    sorted_elevs = sorted(elev_index.keys())

    # Filter by requested sweeps
    if sweep is not None:
        if isinstance(sweep, int):
            sweep = [sweep]
        sorted_elevs = [sorted_elevs[i] for i in sweep if i < len(sorted_elevs)]

    # Build DataTree nodes
    tree_dict = {}

    for sweep_idx, elev in enumerate(sorted_elevs):
        moment_map = elev_index[elev]  # {moment_name: (bufr_file, file_sweep_idx)}

        # Get geometry from first available moment
        first_moment = next(iter(moment_map.values()))
        first_bufr_file, first_sweep_idx = first_moment
        geometry = first_bufr_file.get_sweep_geometry(first_sweep_idx)

        # Load all moment arrays
        moment_arrays = {}
        for moment_name, (bufr_obj, file_sweep_idx) in moment_map.items():
            try:
                arr = bufr_obj.get_moment_data(file_sweep_idx, moment_name)
                moment_arrays[moment_name] = arr
            except Exception as e:
                warnings.warn(f"Failed to load {moment_name} at elevation {elev:.2f}°: {e}")

        store = ArgentinaBufrSweepStore(geometry, moment_arrays, site_attrs, sweep_idx)
        store_entrypoint = xr.backends.store.StoreBackendEntrypoint()
        ds = store_entrypoint.open_dataset(
            store,
            mask_and_scale=True,
            decode_times=False,
            concat_characters=True,
            decode_coords=True,
            drop_variables=None,
        )
        ds = ds.set_coords(["azimuth", "elevation", "range", "time", "latitude", "longitude", "altitude"])

        if reindex_angle:
            ds = _reindex_azimuth(ds)

        tree_dict[f"sweep_{sweep_idx}"] = ds

    # Build root dataset with site metadata
    root_ds = xr.Dataset(attrs=site_attrs)
    if site_coords:
        root_ds = root_ds.assign_coords(
            latitude=xr.Variable((), site_attrs["latitude"], attrs={"units": "degrees_north"}),
            longitude=xr.Variable((), site_attrs["longitude"], attrs={"units": "degrees_east"}),
            altitude=xr.Variable((), site_attrs["altitude"], attrs={"units": "meters"}),
        )

    tree_dict["/"] = root_ds
    return xr.DataTree.from_dict(tree_dict)


def _collect_bufr_files(path):
    """Return sorted list of .BUFR.gz file paths from a directory or file list."""
    if isinstance(path, (list, tuple)):
        return [Path(p) for p in path]
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.BUFR.gz"))
        if not files:
            files = sorted(path.glob("*.bufr.gz"))
        return files
    # Single file
    return [path]


def _build_elevation_index(bufr_files, moments_filter=None):
    """Match sweeps across files by elevation angle.

    Returns:
        dict: {elevation_deg (rounded): {moment_name: (BufrRustFile, sweep_idx)}}
    """
    ELEV_TOL = 0.1  # degrees

    # Collect all (elevation, moment_name, bufr_file, sweep_idx)
    all_sweeps = []
    for _fp, bufr in bufr_files:
        moment_names_in_file = bufr.moment_names
        for si in range(bufr.num_sweeps):
            elev = bufr.get_elevation(si)
            sweep_moments = bufr.get_sweep_moment_names(si)
            for m in sweep_moments:
                if moments_filter is not None and m not in moments_filter:
                    continue
                all_sweeps.append((elev, m, bufr, si))

    # Group by elevation using tolerance matching
    elev_index = {}  # representative_elev -> {moment -> (bufr, sweep_idx)}

    def find_matching_elev(elev):
        for existing in elev_index:
            if abs(existing - elev) <= ELEV_TOL:
                return existing
        return None

    for elev, moment_name, bufr, sweep_idx in all_sweeps:
        key = find_matching_elev(elev)
        if key is None:
            key = elev
            elev_index[key] = {}
        if moment_name in elev_index[key]:
            warnings.warn(
                f"Duplicate moment '{moment_name}' at elevation {elev:.2f}° — "
                "keeping first occurrence."
            )
        else:
            elev_index[key][moment_name] = (bufr, sweep_idx)

    return elev_index


def _reindex_azimuth(ds, resolution=1.0):
    """Reindex azimuths to a regular grid at given resolution (degrees).

    The raw azimuth coordinate may start at an arbitrary angle and wrap around
    360°, making it non-monotonic.  We sort by azimuth first so that xarray's
    ``reindex`` (which requires a monotonic index) works correctly.
    """
    ds = ds.sortby("azimuth")
    n = int(round(360.0 / resolution))
    target_az = np.arange(n) * resolution
    return ds.reindex(
        azimuth=target_az,
        method="nearest",
        tolerance=resolution / 2,
    )
