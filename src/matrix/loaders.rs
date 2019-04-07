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

// TODO(pyk): Add docs about how to load matrix from external file here

use crate::matrix::errors::{MatrixLoadError, MatrixLoadErrorKind};
use crate::matrix::Matrix;
use crate::utils;
use csv;
use num::{FromPrimitive, Num};
use std::fs::File;
use std::marker::PhantomData;
use std::path::Path;

/// Matrix loader for CSV formatted file.
///
/// See also: [`Matrix::from_csv`].
///
/// [`Matrix::from_csv`]: struct.Matrix.html#method.from_csv
#[derive(Debug)]
pub struct MatrixLoaderForCSV<T, P>
where
    P: AsRef<Path>,
{
    file_path: P,
    has_headers: bool,
    // We use this to make compiler happy
    phantom: PhantomData<T>,
}

impl<T, P> MatrixLoaderForCSV<T, P>
where
    P: AsRef<Path>,
{
    /// Set to true to treat the first row as a special header row. By default, it is set
    /// to false.
    ///
    /// # Examples
    ///
    /// ```
    /// use crabsformer::prelude::*;
    ///
    /// let dataset: Matrix<f32> = Matrix::from_csv("tests/data/dataset.csv")
    ///     .has_headers(true)
    ///     .load()
    ///     .unwrap();
    /// ```
    pub fn has_headers(self, yes: bool) -> MatrixLoaderForCSV<T, P> {
        MatrixLoaderForCSV {
            file_path: self.file_path,
            has_headers: yes,
            phantom: PhantomData,
        }
    }

    /// Load Matrix from CSV file. You need to explicitly annotate the numeric type.
    ///
    /// # Examples
    /// ```
    /// use crabsformer::prelude::*;
    ///
    /// let dataset: Matrix<f32> = Matrix::from_csv("tests/data/weight.csv").load().unwrap();
    /// ```
    pub fn load(self) -> Result<Matrix<T>, MatrixLoadError>
    where
        T: FromPrimitive + Num + Copy + utils::TypeName,
    {
        // Open CSV file
        let file = File::open(self.file_path)?;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(self.has_headers)
            .from_reader(file);

        // Collect each row
        let mut elements = Vec::new();
        for result in rdr.records() {
            // Convert each row in the CSV file to RowMatrix
            let record = result?;
            let mut rows = Vec::with_capacity(record.len());
            for value in record.iter() {
                // It will return error if any
                let element = match T::from_str_radix(value.trim(), 10) {
                    Ok(value) => value,
                    Err(_err) => {
                        // Return error early
                        return Err(MatrixLoadError::new(
                            MatrixLoadErrorKind::InvalidElement,
                            format!(
                                "{:?} is not valid {}",
                                value,
                                T::type_name()
                            ),
                        ));
                    }
                };
                rows.push(element);
            }
            elements.push(rows);
        }
        if elements.len() == 0 {
            return Err(MatrixLoadError::new(
                MatrixLoadErrorKind::Empty,
                String::from("Cannot load empty file"),
            ));
        }
        Ok(Matrix::from(elements))
    }
}

impl<T> Matrix<T>
where
    T: Num + Copy,
{
    /// Load Matrix from CSV file. You need to explicitly annotate the numeric type.
    ///
    /// # Examples
    ///
    /// ```
    /// use crabsformer::prelude::*;
    ///
    /// let dataset: Matrix<f32> = Matrix::from_csv("tests/data/weight.csv").load().unwrap();
    /// ```
    ///
    pub fn from_csv<P>(file_path: P) -> MatrixLoaderForCSV<T, P>
    where
        P: AsRef<Path>,
    {
        MatrixLoaderForCSV {
            file_path,
            has_headers: false,
            phantom: PhantomData,
        }
    }
}
