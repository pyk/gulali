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

use crate::error::{LoadError, LoadErrorKind};
use crate::vector;
use crate::utils;
use csv;
use num::{FromPrimitive, Num};
use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Distribution, Normal, Uniform};
use std::fs::File;
use std::ops;
use std::path::Path;

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
/// # use crabsformer::*;
/// let w = matrix![
///     3, 1, 4;
///     1, 5, 9;
/// ];
/// assert_eq!(w[0][0], 3);
/// assert_eq!(w[0][1], 1);
/// assert_eq!(w[0][2], 4);
/// assert_eq!(w[1][0], 1);
/// assert_eq!(w[1][1], 5);
/// assert_eq!(w[1][2], 9);
/// ```
///
/// 2. Create a matrix from a given element and shape:
///
/// ```
/// # use crabsformer::*;
/// let w = matrix![1; [3, 3]];
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

    // Samples: matrix![0; [3, 3]]
    ($elem:expr; $shape:expr) => {{
        let nrows = $shape[0];
        let ncols = $shape[1];
        let elements = vec![vec![$elem; ncols]; nrows];
        $crate::Matrix::from(elements)
    }};

    // Samples: matrix![1, 3, 4]
    ($($x:expr),*) => {{
        let elements = vec![vec![$($x),*]];
        $crate::Matrix::from(elements)
    }};

    // Samples: matrix![1, 2, 3, 4,]
    ($($x:expr,)*) => {{
        let elements = vec![vec![$($x),*]];
        Matrix::from(elements)
    }};

    // Samples: matrix![2.0, 1.0, 4.0; 2.0, 4.0, 2.0;]
    ($($($x:expr),*;)*) => {{
        let elements = vec![$(vec![$($x),*]),*];
        Matrix::from(elements)
    }};

    // Samples: matrix![2.0, 1.0, 4.0; 2.0, 4.0, 2.0]
    ($($($x:expr),*);*) => {{
        let elements = vec![$(vec![$($x),*]),*];
        Matrix::from(elements)
    }};
}

/// Row & Column Matrix
/// https://en.wikipedia.org/wiki/Row_and_column_vectors
type RowMatrix<T> = vector::Vector<T>;
// type ColMatrix<T> = vector::Vector<T>;

/// Matrix
///
/// TODO: add overview about matrix here.
/// 1. how to create a matrix
/// 2. Matrix operation
/// 3. Indexing, etc.
#[derive(Debug)]
pub struct Matrix<T> {
    /// Matrix size
    nrows: usize,
    ncols: usize,
    elements: Vec<RowMatrix<T>>,
}

impl<T> Matrix<T> {
    /// The shape of the matrix `[nrows, ncols]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let W = matrix![
    ///     3.0, 1.0;
    ///     4.0, 1.0;
    ///     5.0, 9.0;
    /// ];
    /// assert_eq!(W.shape(), [3, 2]);
    /// ```
    pub fn shape(&self) -> [usize; 2] {
        [self.nrows, self.ncols]
    }

    /// Create a new matrix of given shape `shape` and type `T`,
    /// filled with `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let W = Matrix::full([5, 5], 2.5);
    /// ```
    pub fn full(shape: [usize; 2], value: T) -> Matrix<T>
    where
        T: FromPrimitive + Num + Copy,
    {
        // Initialize and populate the matrix with specified value
        let nrows = shape[0];
        let ncols = shape[1];
        let elements = vec![vector![value; ncols]; nrows];
        Matrix {
            nrows,
            ncols,
            elements,
        }
    }

    /// Create a new matrix that have the same shape and type
    /// as matrix `m`, filled with `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let w1 = matrix![
    ///     3.0, 1.0;
    ///     4.0, 1.0;
    /// ];
    /// let w2 = Matrix::full_like(&w1, 3.1415);
    /// ```
    pub fn full_like(m: &Matrix<T>, value: T) -> Matrix<T>
    where
        T: FromPrimitive + Num + Copy,
    {
        // Initialize and populate the matrix with specified value
        let nrows = m.nrows;
        let ncols = m.ncols;
        let elements = vec![vector![value; ncols]; nrows];
        Matrix {
            nrows,
            ncols,
            elements,
        }
    }

    /// Create a new matrix of given shape `shape` and type `T`,
    /// filled with zeros. You need to explicitly annotate the
    /// numeric type.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let W: Matrix<i32> = Matrix::zeros([5, 5]);
    /// ```
    pub fn zeros(shape: [usize; 2]) -> Matrix<T>
    where
        T: FromPrimitive + Num + Copy,
    {
        Self::full(shape, T::from_i32(0).unwrap())
    }

    /// Create a new matrix that have the same shape and type
    /// as matrix `m`, filled with zeros.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let W1 = matrix![3.0, 1.0; 4.0, 1.0];
    /// let W2 = Matrix::zeros_like(&W1);
    /// ```
    pub fn zeros_like(m: &Matrix<T>) -> Matrix<T>
    where
        T: FromPrimitive + Num + Copy,
    {
        Self::full([m.nrows, m.ncols], T::from_i32(0).unwrap())
    }

    /// Create a new matrix of given shaoe `shape` and type `T`,
    /// filled with ones. You need to explicitly annotate the
    /// numeric type.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let W: Matrix<i32> = Matrix::ones([3, 5]);
    /// ```
    pub fn ones(shape: [usize; 2]) -> Matrix<T>
    where
        T: FromPrimitive + Num + Copy,
    {
        Self::full(shape, T::from_i32(1).unwrap())
    }

    /// Create a new matrix that have the same shape and type
    /// as matrix `m`, filled with ones.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let W1 = matrix![3, 1; 4, 1; 5, 9];
    /// let W2 = Matrix::ones_like(&W1);
    /// ```
    pub fn ones_like(m: &Matrix<T>) -> Matrix<T>
    where
        T: FromPrimitive + Num + Copy,
    {
        Self::full([m.nrows, m.ncols], T::from_i32(1).unwrap())
    }

    /// Raises each elements of matrix to the power of `exp`,
    /// using exponentiation by squaring.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let W1 = matrix![3, 1, 4; 1, 5, 9];
    /// let W2 = W1.power(2);
    /// assert_eq!(W2, matrix![9, 1, 16; 1, 25, 81]);
    /// ```
    pub fn power(&self, exp: usize) -> Matrix<T>
    where
        T: FromPrimitive + Num + Copy,
    {
        let elements =
            self.elements.iter().map(|row| row.power(exp)).collect();
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            elements,
        }
    }

    /// Create a new matrix of the given shape `shape` and
    /// populate it with random samples from a uniform distribution
    /// over the half-open interval `[low, high)` (includes `low`,
    /// but excludes `high`).
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let W = Matrix::uniform([5, 5], 0.0, 1.0);
    /// ```
    pub fn uniform(shape: [usize; 2], low: T, high: T) -> Matrix<T>
    where
        T: Num + SampleUniform + Copy,
    {
        // Get the shape of the matrix
        let nrows = shape[0];
        let ncols = shape[1];

        let mut elements = Vec::with_capacity(nrows);
        let uniform_distribution = Uniform::new(low, high);
        // Populate the Matrix with the default value
        let mut rng = rand::thread_rng();
        for _ in 0..nrows {
            let mut cols = Vec::with_capacity(ncols);
            for _ in 0..ncols {
                cols.push(uniform_distribution.sample(&mut rng));
            }
            elements.push(RowMatrix::from(cols));
        }

        Matrix {
            nrows,
            ncols,
            elements,
        }
    }

    // Load Matrix from CSV file
    pub fn from_csv<P>(
        file_path: P,
        has_headers: bool,
    ) -> Result<Matrix<T>, LoadError>
    where
        P: AsRef<Path>,
        T: FromPrimitive + Num + Copy + utils::TypeName,
    {
        // Open CSV file
        let file = File::open(file_path)?;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(has_headers)
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
                        return Err(LoadError::new(
                            LoadErrorKind::InvalidElement,
                            format!("{:?} is not valid {}", value, T::type_name()),
                        ));
                    }
                };
                rows.push(element);
            }
            elements.push(rows);
        }
        if elements.len() == 0 {
            return Err(LoadError::new(
                LoadErrorKind::Empty,
                String::from("Cannot load empty file"),
            ));
        }
        Ok(Matrix::from(elements))
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
    /// # use crabsformer::*;
    /// let W = Matrix::normal([5, 5], 0.0, 1.0); // Gaussian mean=0.0 std_dev=1.0
    /// ```
    pub fn normal(shape: [usize; 2], mean: f64, std_dev: f64) -> Matrix<f64> {
        // Get the shape of the matrix
        let nrows = shape[0];
        let ncols = shape[1];

        let mut elements = Vec::with_capacity(nrows);
        let normal_distribution = Normal::new(mean, std_dev);
        // Populate the Matrix with the default value
        let mut rng = rand::thread_rng();
        for _ in 0..nrows {
            let mut cols = Vec::with_capacity(ncols);
            for _ in 0..ncols {
                cols.push(normal_distribution.sample(&mut rng));
            }
            elements.push(RowMatrix::from(cols));
        }

        Matrix {
            nrows,
            ncols,
            elements,
        }
    }
}

// Conversion from Vec<Vec<T>>
impl<T> From<Vec<Vec<T>>> for Matrix<T>
where
    T: Num + Copy,
{
    fn from(source: Vec<Vec<T>>) -> Self {
        let nrows = source.len();
        let ncols = source[0].len();
        // Raise panic if number of columns on each row is inconsistent
        let ncols_inconsistent = source.iter().any(|v| v.len() != ncols);
        if ncols_inconsistent {
            panic!("Invalid matrix: the number of columns is inconsistent")
        }
        // Convert each row to RowMatrix
        let elements = source
            .iter()
            .map(|v| {
                // We cannot directly convert &Vec<T>
                // to RowMatrix<T> because we cannot
                // move out borrowed content
                let mut row = Vec::new();
                v.iter().for_each(|x| row.push(*x));
                RowMatrix::from(row)
            })
            .collect();

        Matrix {
            nrows,
            ncols,
            elements,
        }
    }
}

// Matrix comparison
impl<T> PartialEq for Matrix<T>
where
    T: Num + Copy,
{
    fn eq(&self, other: &Matrix<T>) -> bool {
        if self.elements != other.elements {
            return false;
        }
        true
    }
    fn ne(&self, other: &Matrix<T>) -> bool {
        if self.elements == other.elements {
            return false;
        }
        true
    }
}

// Implement matrix indexing
impl<T> ops::Index<usize> for Matrix<T> {
    type Output = RowMatrix<T>;

    fn index(&self, i: usize) -> &RowMatrix<T> {
        &self.elements[i]
    }
}

// Implement iterator for matrix
impl<T> IntoIterator for Matrix<T> {
    type Item = RowMatrix<T>;
    type IntoIter = ::std::vec::IntoIter<RowMatrix<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}

// This trait is implemented to support for matrix addition operator
impl<T> ops::Add<Matrix<T>> for Matrix<T>
where
    T: Num + Copy,
{
    type Output = Matrix<T>;

    fn add(self, other: Matrix<T>) -> Matrix<T> {
        if self.shape() != other.shape() {
            panic!(
                "Matrix addition with invalid shape: {:?} != {:?}",
                self.shape(),
                other.shape()
            );
        }

        // Add the matrix
        let elements = self
            .elements
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.elements
                    .iter()
                    .enumerate()
                    .map(|(j, value)| *value + other[i][j])
                    .collect()
            })
            .collect();
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            elements,
        }
    }
}

// This trait is implemented to support for matrix addition
// operator with scalar on the right side,
// for example:
//
// let a = matrix![5, 5; 5, 5] + 6;
//
impl<T> ops::Add<T> for Matrix<T>
where
    T: Num + Copy,
{
    type Output = Matrix<T>;

    fn add(self, value: T) -> Matrix<T> {
        // Add the matrix
        let elements = self
            .elements
            .iter()
            .map(|row| {
                row.elements.iter().map(|elem| *elem + value).collect()
            })
            .collect();
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            elements,
        }
    }
}

// This macro is to generate support for matrix addition
// operator with scalar on the left side,
// for example:
//
// let a = 6 + matrix![5, 5; 5, 5];
//
macro_rules! impl_add_matrix_for_type {
    ($t: ty) => {
        impl ops::Add<Matrix<$t>> for $t {
            type Output = Matrix<$t>;

            fn add(self, m: Matrix<$t>) -> Matrix<$t> {
                // Add the matrix
                let elements = m
                    .elements
                    .iter()
                    .map(|row| {
                        row.elements.iter().map(|elem| elem + self).collect()
                    })
                    .collect();
                Matrix {
                    nrows: m.nrows,
                    ncols: m.ncols,
                    elements,
                }
            }
        }
    };
}

impl_add_matrix_for_type!(usize);
impl_add_matrix_for_type!(i8);
impl_add_matrix_for_type!(i16);
impl_add_matrix_for_type!(i32);
impl_add_matrix_for_type!(i64);
impl_add_matrix_for_type!(i128);
impl_add_matrix_for_type!(u8);
impl_add_matrix_for_type!(u16);
impl_add_matrix_for_type!(u32);
impl_add_matrix_for_type!(u64);
impl_add_matrix_for_type!(u128);
impl_add_matrix_for_type!(f32);
impl_add_matrix_for_type!(f64);

// This trait is implemented to support for matrix addition
// and assignment operator (+=)
impl<T> ops::AddAssign<Matrix<T>> for Matrix<T>
where
    T: Num + Copy + ops::AddAssign,
{
    fn add_assign(&mut self, other: Matrix<T>) {
        if self.shape() != other.shape() {
            panic!(
                "Matrix addition with invalid length: {:?} != {:?}",
                self.shape(),
                other.shape()
            );
        }

        self.elements.iter_mut().enumerate().for_each(|(i, row)| {
            row.elements
                .iter_mut()
                .enumerate()
                .for_each(|(j, value)| *value += other[i][j])
        })
    }
}

// This trait is implemented to support for matrix addition
// assignment operator (+=) with scalar on the right side,
// for example:
//
// let a = matrix![5, 5; 5, 5];
// a += 6;
//
impl<T> ops::AddAssign<T> for Matrix<T>
where
    T: Num + Copy + ops::AddAssign,
{
    fn add_assign(&mut self, value: T) {
        self.elements.iter_mut().for_each(|row| {
            row.elements.iter_mut().for_each(|elem| *elem += value)
        })
    }
}

// This trait is implemented to support for matrix
// substraction operator
impl<T> ops::Sub<Matrix<T>> for Matrix<T>
where
    T: Num + Copy,
{
    type Output = Matrix<T>;

    fn sub(self, other: Matrix<T>) -> Matrix<T> {
        if self.shape() != other.shape() {
            panic!(
                "Matrix substraction with invalid shape: {:?} != {:?}",
                self.shape(),
                other.shape()
            );
        }

        // Substract the matrix
        let elements = self
            .elements
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.elements
                    .iter()
                    .enumerate()
                    .map(|(j, value)| *value - other[i][j])
                    .collect()
            })
            .collect();
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            elements,
        }
    }
}

// This trait is implemented to support for matrix substraction
// operator with scalar on the right side,
// for example:
//
// let a = matrix![5, 5; 5, 5] - 6;
//
impl<T> ops::Sub<T> for Matrix<T>
where
    T: Num + Copy,
{
    type Output = Matrix<T>;

    fn sub(self, value: T) -> Matrix<T> {
        // Substract the matrix
        let elements = self
            .elements
            .iter()
            .map(|row| {
                row.elements.iter().map(|elem| *elem - value).collect()
            })
            .collect();
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            elements,
        }
    }
}

// This macro is to generate support for matrix substraction
// operator with scalar on the left side,
// for example:
//
// let a = 6 - matrix![5, 5; 5, 5];
//
macro_rules! impl_sub_matrix_for_type {
    ($t: ty) => {
        impl ops::Sub<Matrix<$t>> for $t {
            type Output = Matrix<$t>;

            fn sub(self, m: Matrix<$t>) -> Matrix<$t> {
                // Substract the matrix
                let elements = m
                    .elements
                    .iter()
                    .map(|row| {
                        row.elements.iter().map(|elem| self - elem).collect()
                    })
                    .collect();
                Matrix {
                    nrows: m.nrows,
                    ncols: m.ncols,
                    elements,
                }
            }
        }
    };
}

impl_sub_matrix_for_type!(usize);
impl_sub_matrix_for_type!(i8);
impl_sub_matrix_for_type!(i16);
impl_sub_matrix_for_type!(i32);
impl_sub_matrix_for_type!(i64);
impl_sub_matrix_for_type!(i128);
impl_sub_matrix_for_type!(u8);
impl_sub_matrix_for_type!(u16);
impl_sub_matrix_for_type!(u32);
impl_sub_matrix_for_type!(u64);
impl_sub_matrix_for_type!(u128);
impl_sub_matrix_for_type!(f32);
impl_sub_matrix_for_type!(f64);

// This trait is implemented to support for matrix substraction
// and assignment operator (-=)
impl<T> ops::SubAssign<Matrix<T>> for Matrix<T>
where
    T: Num + Copy + ops::SubAssign,
{
    fn sub_assign(&mut self, other: Matrix<T>) {
        if self.shape() != other.shape() {
            panic!(
                "Matrix substraction with invalid length: {:?} != {:?}",
                self.shape(),
                other.shape()
            );
        }

        self.elements.iter_mut().enumerate().for_each(|(i, row)| {
            row.elements
                .iter_mut()
                .enumerate()
                .for_each(|(j, value)| *value -= other[i][j])
        })
    }
}

// This trait is implemented to support for matrix substraction
// assignment operator (-=) with scalar on the right side,
// for example:
//
// let a = matrix![5, 5; 5, 5];
// a -= 6;
//
impl<T> ops::SubAssign<T> for Matrix<T>
where
    T: Num + Copy + ops::SubAssign,
{
    fn sub_assign(&mut self, value: T) {
        self.elements.iter_mut().for_each(|row| {
            row.elements.iter_mut().for_each(|elem| *elem -= value)
        })
    }
}

// This trait is implemented to support for matrix
// multiplication operator
impl<T> ops::Mul<Matrix<T>> for Matrix<T>
where
    T: Num + Copy,
{
    type Output = Matrix<T>;

    fn mul(self, other: Matrix<T>) -> Matrix<T> {
        if self.shape() != other.shape() {
            panic!(
                "Matrix multiplication with invalid shape: {:?} != {:?}",
                self.shape(),
                other.shape()
            );
        }

        // Multiply the matrix
        let elements = self
            .elements
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.elements
                    .iter()
                    .enumerate()
                    .map(|(j, value)| *value * other[i][j])
                    .collect()
            })
            .collect();
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            elements,
        }
    }
}

// This trait is implemented to support for matrix multiplication
// operator with scalar on the right side,
// for example:
//
// let a = matrix![5, 5; 5, 5] * 6;
//
impl<T> ops::Mul<T> for Matrix<T>
where
    T: Num + Copy,
{
    type Output = Matrix<T>;

    fn mul(self, value: T) -> Matrix<T> {
        // Multiply the matrix
        let elements = self
            .elements
            .iter()
            .map(|row| {
                row.elements.iter().map(|elem| *elem * value).collect()
            })
            .collect();
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            elements,
        }
    }
}

// This macro is to generate support for matrix multiplication
// operator with scalar on the left side,
// for example:
//
// let a = 6 * matrix![5, 5; 5, 5];
//
macro_rules! impl_sub_matrix_for_type {
    ($t: ty) => {
        impl ops::Mul<Matrix<$t>> for $t {
            type Output = Matrix<$t>;

            fn mul(self, m: Matrix<$t>) -> Matrix<$t> {
                // Multiply the matrix
                let elements = m
                    .elements
                    .iter()
                    .map(|row| {
                        row.elements.iter().map(|elem| self * elem).collect()
                    })
                    .collect();
                Matrix {
                    nrows: m.nrows,
                    ncols: m.ncols,
                    elements,
                }
            }
        }
    };
}

impl_sub_matrix_for_type!(usize);
impl_sub_matrix_for_type!(i8);
impl_sub_matrix_for_type!(i16);
impl_sub_matrix_for_type!(i32);
impl_sub_matrix_for_type!(i64);
impl_sub_matrix_for_type!(i128);
impl_sub_matrix_for_type!(u8);
impl_sub_matrix_for_type!(u16);
impl_sub_matrix_for_type!(u32);
impl_sub_matrix_for_type!(u64);
impl_sub_matrix_for_type!(u128);
impl_sub_matrix_for_type!(f32);
impl_sub_matrix_for_type!(f64);

// This trait is implemented to support for matrix substraction
// and assignment operator (-=)
impl<T> ops::MulAssign<Matrix<T>> for Matrix<T>
where
    T: Num + Copy + ops::MulAssign,
{
    fn mul_assign(&mut self, other: Matrix<T>) {
        if self.shape() != other.shape() {
            panic!(
                "Matrix multiplication with invalid length: {:?} != {:?}",
                self.shape(),
                other.shape()
            );
        }

        self.elements.iter_mut().enumerate().for_each(|(i, row)| {
            row.elements
                .iter_mut()
                .enumerate()
                .for_each(|(j, value)| *value *= other[i][j])
        })
    }
}

// This trait is implemented to support for matrix multiplication
// assignment operator (*=) with scalar on the right side,
// for example:
//
// let a = matrix![5, 5; 5, 5];
// a *= 6;
//
impl<T> ops::MulAssign<T> for Matrix<T>
where
    T: Num + Copy + ops::MulAssign,
{
    fn mul_assign(&mut self, value: T) {
        self.elements.iter_mut().for_each(|row| {
            row.elements.iter_mut().for_each(|elem| *elem *= value)
        })
    }
}
