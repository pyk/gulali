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

use crate::vector;
use num::{FromPrimitive, Num};
use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Distribution, Normal, Uniform};
use std::ops;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector;

    #[test]
    fn test_macro() {
        let a = matrix![0; [12, 10]];
        assert_eq!(a.shape(), [12, 10]);
        assert_eq!(a, Matrix::full([12, 10], 0));

        let b = matrix![
            2.0, 1.0, 4.0;
            2.0, 4.0, 2.0
        ];
        assert_eq!(b.shape(), [2, 3]);

        let c = matrix![
            2.0, 1.0, 4.0;
            2.0, 4.0, 2.0;
        ];
        assert_eq!(c.shape(), [2, 3]);

        let d = matrix![2.0, 1.0, 4.0];
        assert_eq!(d.shape(), [1, 3]);

        let e = matrix![2.0, 1.0, 4.0;];
        assert_eq!(e.shape(), [1, 3]);

        let f = matrix![2.0, 1.0, 4.0,];
        assert_eq!(f.shape(), [1, 3]);
    }

    #[test]
    #[should_panic]
    fn test_invalid_macro() {
        let _x = matrix![2, 2; 2, 2, 2];
    }

    #[test]
    fn test_full() {
        let a = Matrix::full([2, 2], 5.0);
        assert_eq!(a, matrix![5.0, 5.0; 5.0, 5.0]);

        let b = Matrix::full([2, 2], 2);
        assert_eq!(b, matrix![2, 2; 2, 2]);
    }

    #[test]
    fn test_full_like() {
        let a = matrix![3.0, 4.0; 4.0, 5.0];
        let b = Matrix::full_like(&a, 5.0);
        assert_eq!(b, matrix![5.0, 5.0; 5.0, 5.0]);
    }

    #[test]
    fn test_zeros() {
        let vf1: Matrix<f64> = Matrix::zeros([2, 2]);
        assert_eq!(vf1, matrix![0.0, 0.0; 0.0, 0.0]);

        let vf2: Matrix<f32> = Matrix::zeros([2, 2]);
        assert_eq!(vf2, matrix![0.0, 0.0; 0.0, 0.0]);

        let ms1: Matrix<usize> = Matrix::zeros([2, 2]);
        assert_eq!(ms1, matrix![0, 0; 0, 0]);

        let mu1: Matrix<u8> = Matrix::zeros([2, 2]);
        assert_eq!(mu1, matrix![0, 0; 0, 0]);

        let mu2: Matrix<u16> = Matrix::zeros([2, 2]);
        assert_eq!(mu2, matrix![0, 0; 0, 0]);

        let mu3: Matrix<u32> = Matrix::zeros([2, 2]);
        assert_eq!(mu3, matrix![0, 0; 0, 0]);

        let mu4: Matrix<u64> = Matrix::zeros([2, 2]);
        assert_eq!(mu4, matrix![0, 0; 0, 0]);

        let mu5: Matrix<u128> = Matrix::zeros([2, 2]);
        assert_eq!(mu5, matrix![0, 0; 0, 0]);

        let vi1: Matrix<i8> = Matrix::zeros([2, 2]);
        assert_eq!(vi1, matrix![0, 0; 0, 0]);

        let vi2: Matrix<i16> = Matrix::zeros([2, 2]);
        assert_eq!(vi2, matrix![0, 0; 0, 0]);

        let vi3: Matrix<i32> = Matrix::zeros([2, 2]);
        assert_eq!(vi3, matrix![0, 0; 0, 0]);

        let vi4: Matrix<i64> = Matrix::zeros([2, 2]);
        assert_eq!(vi4, matrix![0, 0; 0, 0]);

        let vi5: Matrix<i128> = Matrix::zeros([2, 2]);
        assert_eq!(vi5, matrix![0, 0; 0, 0]);
    }

    #[test]
    fn test_zeros_like() {
        let mi1: Matrix<i32> = Matrix::ones([2, 2]);
        let mi2 = Matrix::zeros_like(&mi1);
        assert_eq!(mi2, matrix![0, 0; 0, 0]);
    }

    #[test]
    fn test_ones() {
        let mf1: Matrix<f64> = Matrix::ones([2, 2]);
        assert_eq!(mf1, matrix![1.0, 1.0; 1.0, 1.0]);

        let mf2: Matrix<f32> = Matrix::ones([2, 2]);
        assert_eq!(mf2, matrix![1.0, 1.0; 1.0, 1.0]);

        let ms1: Matrix<usize> = Matrix::ones([2, 2]);
        assert_eq!(ms1, matrix![1, 1; 1, 1]);

        let mu1: Matrix<u8> = Matrix::ones([2, 2]);
        assert_eq!(mu1, matrix![1, 1; 1, 1]);

        let mu2: Matrix<u16> = Matrix::ones([2, 2]);
        assert_eq!(mu2, matrix![1, 1; 1, 1]);

        let mu3: Matrix<u32> = Matrix::ones([2, 2]);
        assert_eq!(mu3, matrix![1, 1; 1, 1]);

        let mu4: Matrix<u64> = Matrix::ones([2, 2]);
        assert_eq!(mu4, matrix![1, 1; 1, 1]);

        let mu5: Matrix<u128> = Matrix::ones([2, 2]);
        assert_eq!(mu5, matrix![1, 1; 1, 1]);

        let mi1: Matrix<i8> = Matrix::ones([2, 2]);
        assert_eq!(mi1, matrix![1, 1; 1, 1]);

        let mi2: Matrix<i16> = Matrix::ones([2, 2]);
        assert_eq!(mi2, matrix![1, 1; 1, 1]);

        let mi3: Matrix<i32> = Matrix::ones([2, 2]);
        assert_eq!(mi3, matrix![1, 1; 1, 1]);

        let mi4: Matrix<i64> = Matrix::ones([2, 2]);
        assert_eq!(mi4, matrix![1, 1; 1, 1]);

        let mi5: Matrix<i128> = Matrix::ones([2, 2]);
        assert_eq!(mi5, matrix![1, 1; 1, 1]);
    }

    #[test]
    fn test_ones_like() {
        let mi1: Matrix<i32> = Matrix::ones([2, 2]);
        let mi2 = Matrix::ones_like(&mi1);
        assert_eq!(mi2, matrix![1, 1; 1, 1]);
    }

    #[test]
    fn test_power() {
        let w1 = matrix![3, 1; 4, 1];
        let w2 = w1.power(2);
        assert_eq!(w2, matrix![9, 1; 16, 1]);
    }

    #[test]
    fn test_uniform() {
        let mf1: Matrix<f32> = Matrix::uniform([5, 5], 0.0, 1.0);
        for cols in mf1.into_iter() {
            for value in cols.into_iter() {
                assert!((0.0 <= value) && (value < 1.0));
            }
        }

        let mf2: Matrix<f64> = Matrix::uniform([5, 5], 0.0, 1.0);
        for cols in mf2.into_iter() {
            for value in cols.into_iter() {
                assert!((0.0 <= value) && (value < 1.0));
            }
        }

        let ms1: Matrix<usize> = Matrix::uniform([5, 5], 1, 10);
        for cols in ms1.into_iter() {
            for value in cols.into_iter() {
                assert!((1 <= value) && (value < 10));
            }
        }

        let mu1: Matrix<u8> = Matrix::uniform([5, 5], 1, 10);
        for cols in mu1.into_iter() {
            for value in cols.into_iter() {
                assert!((1 <= value) && (value < 10));
            }
        }

        let mu2: Matrix<u16> = Matrix::uniform([5, 5], 1, 10);
        for cols in mu2.into_iter() {
            for value in cols.into_iter() {
                assert!((1 <= value) && (value < 10));
            }
        }

        let mu3: Matrix<u32> = Matrix::uniform([5, 5], 1, 10);
        for cols in mu3.into_iter() {
            for value in cols.into_iter() {
                assert!((1 <= value) && (value < 10));
            }
        }

        let mu4: Matrix<u64> = Matrix::uniform([5, 5], 1, 10);
        for cols in mu4.into_iter() {
            for value in cols.into_iter() {
                assert!((1 <= value) && (value < 10));
            }
        }

        let mu5: Matrix<u128> = Matrix::uniform([5, 5], 1, 10);
        for cols in mu5.into_iter() {
            for value in cols.into_iter() {
                assert!((1 <= value) && (value < 10));
            }
        }

        let mi1: Matrix<i8> = Matrix::uniform([5, 5], -10, 10);
        for cols in mi1.into_iter() {
            for value in cols.into_iter() {
                assert!((-10 <= value) && (value < 10));
            }
        }

        let mi2: Matrix<i16> = Matrix::uniform([5, 5], -10, 10);
        for cols in mi2.into_iter() {
            for value in cols.into_iter() {
                assert!((-10 <= value) && (value < 10));
            }
        }

        let mi3: Matrix<i32> = Matrix::uniform([5, 5], -10, 10);
        for cols in mi3.into_iter() {
            for value in cols.into_iter() {
                assert!((-10 <= value) && (value < 10));
            }
        }

        let mi4: Matrix<i64> = Matrix::uniform([5, 5], -10, 10);
        for cols in mi4.into_iter() {
            for value in cols.into_iter() {
                assert!((-10 <= value) && (value < 10));
            }
        }

        let mi5: Matrix<i128> = Matrix::uniform([5, 5], -10, 10);
        for cols in mi5.into_iter() {
            for value in cols.into_iter() {
                assert!((-10 <= value) && (value < 10));
            }
        }
    }

    #[test]
    fn test_normal() {
        let a = Matrix::normal([5, 5], 2.0, 4.0);
        let b = Matrix::normal([5, 5], 2.0, 4.0);
        assert_eq!(a.nrows, b.nrows);
        assert_eq!(a.ncols, b.ncols);
        assert_ne!(a.elements, b.elements);
    }

    #[test]
    fn test_indexing() {
        let w = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        assert_eq!(w[0][0], 3);
        assert_eq!(w[0][1], 1);
        assert_eq!(w[0][2], 4);
        assert_eq!(w[1][0], 1);
        assert_eq!(w[1][1], 5);
        assert_eq!(w[1][2], 9);

        assert_eq!(w[0], vector![3, 1, 4]);
        assert_eq!(w[1], vector![1, 5, 9]);
    }
}
