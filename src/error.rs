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

use csv;
use std::convert;
use std::fmt;
use std::io;

/// Enum to store the various types of errors that can cause loading a numeric vector
/// or matrix to fail.
pub enum LoadErrorKind {
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

/// An error which can be returned when loading numeric vector or matrix from a file.
///
/// # Potential causes
/// Among other causes, `LoadError` can be thrown because of loaded file is not exists.
pub struct LoadError {
    kind: LoadErrorKind,
    message: String,
}

impl LoadError {
    /// Creates a new `LoadError` from a known kind of error as well as an error message.
    pub fn new(kind: LoadErrorKind, message: String) -> LoadError {
        LoadError { kind, message }
    }

    /// Outputs the detailed cause of loading file failing.
    pub fn kind(&self) -> &LoadErrorKind {
        &self.kind
    }

    fn description(&self) -> String {
        match self.kind {
            LoadErrorKind::IOError => format!(
                "Cannot load Matrix from file due to: {}",
                self.message
            ),
            LoadErrorKind::CSVError => {
                format!("Cannot load Matrix, {}", self.message)
            }
            LoadErrorKind::Empty => {
                format!("Cannot load Matrix from empty file")
            }
            LoadErrorKind::InvalidElement => format!(
                "Cannot load Matrix, invalid element: {}",
                self.message
            ),
        }
    }
}

/// Convert `io::Error` to `matrix::LoadError`
impl convert::From<io::Error> for LoadError {
    fn from(error: io::Error) -> Self {
        LoadError {
            kind: LoadErrorKind::IOError,
            message: format!("{}", error),
        }
    }
}

/// Convert `csv::Error` to `matrix::LoadError`
impl convert::From<csv::Error> for LoadError {
    fn from(error: csv::Error) -> Self {
        LoadError {
            kind: LoadErrorKind::CSVError,
            message: format!("{}", error),
        }
    }
}

impl fmt::Debug for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}
