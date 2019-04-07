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

//! Numeric vector errors.
//!
//! An error which can be returned when creating, operating, loading and
//! indexing numeric vectors.
use std::fmt;

/// Enum to store the various types of errors that can cause creating a numeric
/// vector to fail.
pub enum VectorBuilderErrorKind {
    /// Invalid step value for numeric vectors range.
    ///
    /// Among other causes, this variant will be constructed when creating
    /// new numeric vector range and step value is invalid.
    InvalidStepValue,

    /// Invalid range for numeric vectors input.
    ///
    /// Among other causes, this variant will be constructed when creating
    /// new vector with invalid range, for example `low >= high`.
    InvalidRange,

    /// Standard deviation value should not be negative.
    ///
    /// Among other causes, this variant will be constructed when creating
    /// new random numeric vector using normal distribution with `std_dev < 0`.
    NegativeStandardDeviation,
}

/// An error which can be returned when creating new numeric vectors.
///
/// # Potential causes
/// Among other causes, `VectorBuilderError` can be thrown because of the
/// input when creating new numeric vector is invalid.
pub struct VectorBuilderError {
    pub(crate) kind: VectorBuilderErrorKind,
    pub(crate) message: String,
}

impl VectorBuilderError {
    /// Creates a new `VectorBuilderError` from a known kind of error as well as an
    /// error message.
    pub fn new(kind: VectorBuilderErrorKind, message: String) -> Self {
        VectorBuilderError { kind, message }
    }

    fn description(&self) -> String {
        match self.kind {
            VectorBuilderErrorKind::InvalidRange => {
                format!("Vector builder invalid range: {}", self.message)
            }
            VectorBuilderErrorKind::InvalidStepValue => {
                format!("Vector builder invalid step value: {}", self.message)
            }
            VectorBuilderErrorKind::NegativeStandardDeviation => format!(
                "Random vector builder standard deviation should not \
                 be negative: {}",
                self.message
            ),
        }
    }
}

impl fmt::Debug for VectorBuilderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

impl fmt::Display for VectorBuilderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}
