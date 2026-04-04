/// PyO3 bindings for the Argentine BUFR parser.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::errors::BufrError;
use super::parser::BufrFile;

/// Python-accessible Argentine BUFR file parser.
///
/// Accepts either raw BUFR bytes or gzip-compressed .BUFR.gz bytes.
/// One BufrRustFile instance = one .BUFR.gz file = one moment for all sweeps.
#[pyclass]
pub struct BufrRustFile {
    inner: BufrFile,
}

#[pymethods]
impl BufrRustFile {
    /// Create a new BufrRustFile from raw bytes (BUFR or gzip-compressed).
    #[new]
    fn new(data: &[u8]) -> PyResult<Self> {
        let inner = BufrFile::parse(data)?;
        Ok(BufrRustFile { inner })
    }

    /// Full station identifier, e.g. "RMA1" (type + id concatenated)
    #[getter]
    fn station_id(&self) -> String {
        format!("{}{}", self.inner.station_type, self.inner.station_id)
    }

    /// Station type, e.g. "RMA"
    #[getter]
    fn station_type(&self) -> &str {
        &self.inner.station_type
    }

    /// Latitude in degrees (negative = South)
    #[getter]
    fn latitude(&self) -> f64 {
        self.inner.latitude
    }

    /// Longitude in degrees (negative = West)
    #[getter]
    fn longitude(&self) -> f64 {
        self.inner.longitude
    }

    /// Station height above MSL in meters
    #[getter]
    fn height_m(&self) -> f64 {
        self.inner.height_m
    }

    /// Nominal observation time as ISO8601 string from Section 1
    #[getter]
    fn nominal_time(&self) -> String {
        let s1 = &self.inner.section1;
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
            s1.year, s1.month, s1.day, s1.hour, s1.minute, 0
        )
    }

    /// Number of sweeps (elevation cuts) in this file
    #[getter]
    fn num_sweeps(&self) -> usize {
        self.inner.sweeps.len()
    }

    /// CFRadial moment names present across all sweeps
    #[getter]
    fn moment_names(&self) -> Vec<String> {
        let mut names: std::collections::HashSet<String> = std::collections::HashSet::new();
        for sweep in &self.inner.sweeps {
            for k in sweep.moments.keys() {
                names.insert(k.clone());
            }
        }
        let mut v: Vec<String> = names.into_iter().collect();
        v.sort();
        v
    }

    /// Elevation angle of a sweep in degrees
    fn get_elevation(&self, sweep: usize) -> PyResult<f64> {
        Ok(self.get_sweep(sweep)?.geometry.elevation_deg)
    }

    /// Number of range bins in a sweep
    fn get_n_bins(&self, sweep: usize) -> PyResult<usize> {
        Ok(self.get_sweep(sweep)?.geometry.n_bins)
    }

    /// Number of azimuths in a sweep
    fn get_n_azimuths(&self, sweep: usize) -> PyResult<usize> {
        Ok(self.get_sweep(sweep)?.geometry.n_azimuths)
    }

    /// Range bin size in meters
    fn get_bin_size_m(&self, sweep: usize) -> PyResult<f64> {
        Ok(self.get_sweep(sweep)?.geometry.bin_size_m)
    }

    /// Distance to first bin center in meters
    fn get_bin_offset_m(&self, sweep: usize) -> PyResult<f64> {
        Ok(self.get_sweep(sweep)?.geometry.bin_offset_m)
    }

    /// Start azimuth of the sweep in degrees
    fn get_start_azimuth(&self, sweep: usize) -> PyResult<f64> {
        Ok(self.get_sweep(sweep)?.geometry.start_azimuth_deg)
    }

    /// Sweep start time as ISO8601 string
    fn get_sweep_start_time(&self, sweep: usize) -> PyResult<String> {
        Ok(self.get_sweep(sweep)?.geometry.start_time.clone())
    }

    /// Sweep end time as ISO8601 string
    fn get_sweep_end_time(&self, sweep: usize) -> PyResult<String> {
        Ok(self.get_sweep(sweep)?.geometry.end_time.clone())
    }

    /// Sweep geometry as a Python dict
    fn get_sweep_geometry<'py>(&self, py: Python<'py>, sweep: usize) -> PyResult<Bound<'py, PyDict>> {
        let sw = self.get_sweep(sweep)?;
        let g = &sw.geometry;
        let dict = PyDict::new(py);
        dict.set_item("elevation_deg", g.elevation_deg)?;
        dict.set_item("n_bins", g.n_bins)?;
        dict.set_item("bin_size_m", g.bin_size_m)?;
        dict.set_item("bin_offset_m", g.bin_offset_m)?;
        dict.set_item("n_azimuths", g.n_azimuths)?;
        dict.set_item("start_azimuth_deg", g.start_azimuth_deg)?;
        dict.set_item("rotation_clockwise", g.rotation_clockwise)?;
        dict.set_item("start_time", &g.start_time)?;
        dict.set_item("end_time", &g.end_time)?;
        Ok(dict)
    }

    /// Moment data for a sweep as a 2D numpy array (n_azimuths × n_bins, f64).
    ///
    /// Values are in physical units (dBZ, m/s, etc.) — no scale/offset needed.
    fn get_moment_data<'py>(
        &self,
        py: Python<'py>,
        sweep: usize,
        moment: &str,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let sw = self.get_sweep(sweep)?;
        let data = sw.moments.get(moment).ok_or_else(|| {
            BufrError::MomentNotFound(format!(
                "Moment '{}' not found in sweep {} (available: {:?})",
                moment,
                sweep,
                sw.moments.keys().collect::<Vec<_>>()
            ))
        })?;

        let n_azimuths = sw.geometry.n_azimuths;
        let n_bins = sw.geometry.n_bins;

        // Replace BUFR missing value indicators with NaN.
        // Missing values in the decompressed f64 stream appear as very large
        // negative numbers (effectively -f64::MAX or similar sentinel values).
        let clean: Vec<f64> = data.iter().map(|&v| {
            if v < -1e30 || v > 1e30 { f64::NAN } else { v }
        }).collect();

        let arr = PyArray1::from_vec(py, clean);
        Ok(arr.reshape([n_azimuths, n_bins])?)
    }

    /// CFRadial moment names available in a specific sweep
    fn get_sweep_moment_names(&self, sweep: usize) -> PyResult<Vec<String>> {
        let sw = self.get_sweep(sweep)?;
        let mut names: Vec<String> = sw.moments.keys().cloned().collect();
        names.sort();
        Ok(names)
    }
}

impl BufrRustFile {
    fn get_sweep(&self, sweep: usize) -> Result<&super::sweep::SweepData, BufrError> {
        self.inner.sweeps.get(sweep).ok_or_else(|| {
            BufrError::SweepOutOfRange(sweep)
        })
    }
}
