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

//! A list of function to build a new Matrix.
//!
//! TODO(pyk): Add docs here
//!
//!

use crate::matrix::errors::MatrixBuilderError;
use crate::matrix::Matrix;
use crate::vector::Vector;
use num::{FromPrimitive, Num};
use rand::distributions::uniform::SampleUniform;
use std::fmt;

/// Creates a [matrix] containing the arguments.
///
/// `matrix!` allows matrix to be defined with
/// the same syntax as array expressions.
///
/// There are two forms of this macro:
///
/// 1. Create a matrix containing a given list of elements:
///
/// ```
/// # use crabsformer::prelude::*;
/// let w = matrix![
///     3, 1, 4;
///     1, 5, 9;
/// ];
/// assert_eq!(*w.at(0,0), 3);
/// assert_eq!(*w.at(0,1), 1);
/// assert_eq!(*w.at(0,2), 4);
/// assert_eq!(*w.at(1,0), 1);
/// assert_eq!(*w.at(1,1), 5);
/// assert_eq!(*w.at(1,2), 9);
/// ```
///
/// 2. Create a matrix from a given shape and element:
///
/// ```
/// # use crabsformer::prelude::*;
/// let w = matrix![[3, 3] => 1];
/// assert_eq!(w, matrix![
///     1, 1, 1;
///     1, 1, 1;
///     1, 1, 1;
/// ]);
/// ```
///
/// [matrix]: struct.Matrix.html
#[macro_export]
macro_rules! matrix {
    // NOTE: the order of the rules is very important

    // Samples: matrix![[3, 3] => 0]
    ($shape:expr => $elem:expr) => {{
        let nrows = $shape[0];
        let ncols = $shape[1];
        let elements = vec![vec![$elem; ncols]; nrows];
        $crate::matrix::Matrix::from(elements)
    }};

    // Samples: matrix![1, 3, 4]
    ($($x:expr),*) => {{
        let elements = vec![vec![$($x),*]];
        $crate::matrix::Matrix::from(elements)
    }};

    // Samples: matrix![1, 2, 3, 4,]
    ($($x:expr,)*) => {{
        let elements = vec![vec![$($x),*]];
        $crate::matrix::Matrix::from(elements)
    }};

    // Samples: matrix![2.0, 1.0, 4.0; 2.0, 4.0, 2.0;]
    ($($($x:expr),*;)*) => {{
        let elements = vec![$(vec![$($x),*]),*];
        $crate::matrix::Matrix::from(elements)
    }};

    // Samples: matrix![2.0, 1.0, 4.0; 2.0, 4.0, 2.0]
    ($($($x:expr),*);*) => {{
        let elements = vec![$(vec![$($x),*]),*];
        $crate::matrix::Matrix::from(elements)
    }};
}

impl<T> Matrix<T>
where
    T: Num + Copy,
{
    /// Create a new matrix of given shape `shape` and type `T`,
    /// filled with `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::prelude::*;
    /// let w = Matrix::full([5, 5], 2.5);
    /// ```
    pub fn full(shape: [usize; 2], value: T) -> Matrix<T>
    where
        T: FromPrimitive,
    {
        matrix![shape => value]
    }

    /// Create a new matrix that have the same shape and type
    /// as matrix `m`, filled with `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::prelude::*;
    /// let w1 = matrix![
    ///     3.0, 1.0;
    ///     4.0, 1.0;
    /// ];
    /// let w2 = Matrix::full_like(&w1, 3.1415);
    /// ```
    pub fn full_like(m: &Matrix<T>, value: T) -> Matrix<T>
    where
        T: FromPrimitive,
    {
        matrix![m.shape() => value]
    }

    /// Create a new matrix of given shape `shape` and type `T`,
    /// filled with zeros. You need to explicitly annotate the
    /// numeric type.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::prelude::*;
    /// let w: Matrix<i32> = Matrix::zeros([5, 5]);
    /// ```
    pub fn zeros(shape: [usize; 2]) -> Matrix<T>
    where
        T: FromPrimitive,
    {
        matrix![shape => T::from_i32(0).unwrap()]
    }

    /// Create a new matrix that have the same shape and type
    /// as matrix `m`, filled with zeros.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::prelude::*;
    /// let w1 = matrix![3.0, 1.0; 4.0, 1.0];
    /// let w2 = Matrix::zeros_like(&w1);
    /// ```
    pub fn zeros_like(m: &Matrix<T>) -> Matrix<T>
    where
        T: FromPrimitive,
    {
        matrix![m.shape() => T::from_i32(0).unwrap()]
    }

    /// Create a new matrix of given shaoe `shape` and type `T`,
    /// filled with ones. You need to explicitly annotate the
    /// numeric type.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::prelude::*;
    /// let w: Matrix<i32> = Matrix::ones([3, 5]);
    /// ```
    pub fn ones(shape: [usize; 2]) -> Matrix<T>
    where
        T: FromPrimitive,
    {
        matrix![shape => T::from_i32(1).unwrap()]
    }

    /// Create a new matrix that have the same shape and type
    /// as matrix `m`, filled with ones.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::prelude::*;
    /// let w1 = matrix![3, 1; 4, 1; 5, 9];
    /// let w2 = Matrix::ones_like(&w1);
    /// ```
    pub fn ones_like(m: &Matrix<T>) -> Matrix<T>
    where
        T: FromPrimitive,
    {
        matrix![m.shape() => T::from_i32(1).unwrap()]
    }
}

impl<T> Matrix<T>
where
    T: Num + Copy,
{
    /// Create a new matrix of the given shape `shape` and populate it with
    /// random samples from a uniform distribution over the half-open
    /// interval `[low, high)` (includes `low`, but excludes `high`).
    ///
    /// **Note that**: If `low >= high` it will returns an error.
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::prelude::*;
    /// let w = Matrix::uniform([5, 5], 0.0, 1.0);
    /// ```
    pub fn uniform(
        shape: [usize; 2],
        low: T,
        high: T,
    ) -> Result<Matrix<T>, MatrixBuilderError>
    where
        T: SampleUniform + PartialOrd + fmt::Display,
    {
        let total_elements = shape.iter().product();
        let vec = Vector::uniform(total_elements, low, high)?;

        Ok(Matrix {
            nrows: shape[0],
            ncols: shape[1],
            vec,
        })
    }
}

impl Matrix<f64> {
    /// Create a new matrix of the given shape `shape` and
    /// populate it with random samples from a normal distribution
    /// `N(mean, std_dev**2)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::prelude::*;
    /// let w = Matrix::normal([5, 5], 0.0, 1.0); // Gaussian mean=0.0 std_dev=1.0
    /// ```
    pub fn normal(shape: [usize; 2], mean: f64, std_dev: f64) -> Matrix<f64> {
        let total_elements = shape.iter().product();
        let vec = Vector::normal(total_elements, mean, std_dev);
        Matrix {
            nrows: shape[0],
            ncols: shape[1],
            vec,
        }
    }
}
