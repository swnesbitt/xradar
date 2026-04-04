use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BufrError {
    #[error("Invalid BUFR magic: expected 'BUFR', got {0:?}")]
    InvalidMagic(Vec<u8>),

    #[error("Unsupported BUFR edition: {0}")]
    UnsupportedEdition(u8),

    #[error("Invalid section: {0}")]
    InvalidSection(String),

    #[error("Bit read error: {0}")]
    BitReadError(String),

    #[error("Decompression error: {0}")]
    DecompressionError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Sweep index out of range: {0}")]
    SweepOutOfRange(usize),

    #[error("Moment not found: {0}")]
    MomentNotFound(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

impl From<BufrError> for PyErr {
    fn from(err: BufrError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}
