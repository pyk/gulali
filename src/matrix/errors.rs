// Copyright (c) 2019, Bayu Aldi Yansyah <bayualdiyansyah@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Matrix errors.
//!
//! An error which can be returned when creating, operating, loading and
//! indexing matrices.

use crate::vector::errors::{VectorBuilderError, VectorBuilderErrorKind};
use csv;
use std::convert;
use std::fmt;
use std::io;

/// Enum to store the various types of errors that can cause creating a matrix
/// to fail.
pub enum MatrixBuilderErrorKind {
    /// Invalid range for matrix input.
    ///
    /// Among other causes, this variant will be constructed when creating
    /// new matrix with invalid range, for example `low >= high`.
    InvalidRange,

    /// Any error not part of this list.
    Other,
}

/// An error which can be returned when creating new matrices.
///
/// # Potential causes
/// Among other causes, `MatrixBuilderError` can be thrown because of the
/// input when creating new matrix is invalid.
pub struct MatrixBuilderError {
    pub(crate) kind: MatrixBuilderErrorKind,
    pub(crate) message: String,
}

impl MatrixBuilderError {
    /// Creates a new `MatrixBuilderError` from a known kind of error as well as an
    /// error message.
    pub fn new(kind: MatrixBuilderErrorKind, message: String) -> Self {
        MatrixBuilderError { kind, message }
    }

    fn description(&self) -> String {
        match self.kind {
            MatrixBuilderErrorKind::InvalidRange => {
                format!("Matrix builder invalid range: {}", self.message)
            }
            MatrixBuilderErrorKind::Other => {
                format!("Matrix builder error: {}", self.message)
            }
        }
    }
}

/// Convert `VectorBuilderError` to `MatrixBuilderError`
impl convert::From<VectorBuilderError> for MatrixBuilderError {
    fn from(error: VectorBuilderError) -> Self {
        match error.kind {
            VectorBuilderErrorKind::InvalidRange => MatrixBuilderError::new(
                MatrixBuilderErrorKind::InvalidRange,
                error.message,
            ),
            VectorBuilderErrorKind::InvalidStepValue => {
                MatrixBuilderError::new(
                    MatrixBuilderErrorKind::Other,
                    error.message,
                )
            }
        }
    }
}

impl fmt::Debug for MatrixBuilderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

impl fmt::Display for MatrixBuilderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// Enum to store the various types of errors that can cause loading a matrix to fail.
pub enum MatrixLoadErrorKind {
    /// I/O Error
    ///
    /// Among other causes, this variant will be constructed when failed loading a file
    /// due to I/O problem.
    IOError,
    /// CSV Error
    ///
    /// Among other causes, this variant will be constructed when failed loading a CSV file.
    CSVError,
    /// File being loaded is empty.
    ///
    /// Among other causes, this variant will be constructed when loading an empty file.
    Empty,
    /// Contains an invalid element.
    ///
    /// Among other causes, this variant will be constructed when parsing a string that
    /// contains non-numeric letter.
    InvalidElement,
}

/// An error which can be returned when loading matrix from a file.
///
/// # Potential causes
/// Among other causes, `LoadError` can be thrown because of loaded file is not exists.
pub struct MatrixLoadError {
    pub(crate) kind: MatrixLoadErrorKind,
    pub(crate) message: String,
}

impl MatrixLoadError {
    /// Creates a new `LoadError` from a known kind of error as well as an error message.
    pub fn new(
        kind: MatrixLoadErrorKind,
        message: String,
    ) -> MatrixLoadError {
        MatrixLoadError { kind, message }
    }

    /// Outputs the detailed cause of loading file failing.
    pub fn kind(&self) -> &MatrixLoadErrorKind {
        &self.kind
    }

    fn description(&self) -> String {
        match self.kind {
            MatrixLoadErrorKind::IOError => format!(
                "Cannot load Matrix from file due to: {}",
                self.message
            ),
            MatrixLoadErrorKind::CSVError => {
                format!("Cannot load Matrix, {}", self.message)
            }
            MatrixLoadErrorKind::Empty => {
                format!("Cannot load Matrix from empty file")
            }
            MatrixLoadErrorKind::InvalidElement => format!(
                "Cannot load Matrix, invalid element: {}",
                self.message
            ),
        }
    }
}

/// Convert `io::Error` to `matrix::LoadError`
impl convert::From<io::Error> for MatrixLoadError {
    fn from(error: io::Error) -> Self {
        MatrixLoadError {
            kind: MatrixLoadErrorKind::IOError,
            message: format!("{}", error),
        }
    }
}

/// Convert `csv::Error` to `matrix::LoadError`
impl convert::From<csv::Error> for MatrixLoadError {
    fn from(error: csv::Error) -> Self {
        MatrixLoadError {
            kind: MatrixLoadErrorKind::CSVError,
            message: format!("{}", error),
        }
    }
}

impl fmt::Debug for MatrixLoadError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

impl fmt::Display for MatrixLoadError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}
