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
use crate::utils;
use crate::vector::*;
use csv;
use num::{FromPrimitive, Num};
use rand::distributions::uniform::SampleUniform;
use std::fmt;
use std::fs::File;
use std::marker::PhantomData;
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
/// # use crabsformer::*;
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

/// Matrix.
///
/// TODO: add overview about matrix here.
/// 1. how to create a matrix
/// 2. Matrix operation
/// 3. Indexing, etc.
#[derive(Debug)]
pub struct Matrix<T>
where
    T: Num + Copy,
{
    /// Matrix size
    nrows: usize,
    ncols: usize,
    vec: Vector<T>,
}

impl<T> Matrix<T>
where
    T: Num + Copy,
{
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

    // Bound check
    fn bound_check(&self, i: Option<usize>, j: Option<usize>) {
        if i.is_some() && i.unwrap() >= self.nrows {
            panic!(
                "index {} out of range for matrix of number of rows {}",
                i.unwrap(),
                self.nrows
            )
        }
        if j.is_some() && j.unwrap() >= self.ncols {
            panic!(
                "index {} out of range for matrix of number of columns {}",
                j.unwrap(),
                self.ncols
            )
        }
    }

    /// Get element of the matrix at row `i` and column `j`.
    ///
    /// # Examples
    /// ```
    /// let w = matrix![
    ///     3, 1, 4;
    ///     1, 5, 9;
    /// ];
    ///
    /// assert_eq!(w.at(0, 0), 3);
    /// assert_eq!(w.at(0, 1), 1);
    /// assert_eq!(w.at(0, 2), 4);
    /// assert_eq!(w.at(1, 0), 1);
    /// assert_eq!(w.at(1, 1), 1);
    /// assert_eq!(w.at(1, 2), 1);
    /// ```
    ///
    /// # Panics
    /// Panics if `i >= nrows` and `j >= ncols`.
    pub fn at(&self, i: usize, j: usize) -> &T {
        self.bound_check(Some(i), Some(j));
        &self.vec[(self.ncols * i) + j]
    }

    /// Get the row of the matrix. It will returns a reference to a row
    /// of the matrix.
    ///
    /// Because we store the the matrix element in row-major order,
    /// this operation is `O(1)`.
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::*;
    /// let W = matrix![
    ///     3.0, 1.0;
    ///     4.0, 1.0;
    ///     5.0, 9.0;
    /// ];
    /// assert_eq!(W.row(0), [3.0, 1.0]);
    /// assert_eq!(W.row(1), [4.0, 1.0]);
    /// assert_eq!(W.row(1), [5.0, 9.0]);
    /// ```
    ///
    /// # Panics
    /// Panics if `i >= n` where `n` is number of rows.
    pub fn row<'a>(&'a self, i: usize) -> RowMatrix<'a, T> {
        self.bound_check(Some(i), None);
        let begin = self.ncols * i;
        let end = (self.ncols * i) + self.ncols;
        let subvec = self.vec.slice(begin..end);
        RowMatrix {
            matrix: self,
            vec: Vector::from(subvec),
        }
    }

    /// Iterates over rows of the matrix.
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::*;
    /// let w = matrix![3, 1, 4; 1, 5, 9; 2, 6, 5];
    /// let mut rows = w.rows();
    ///
    /// assert_eq!(rows.next(), Some([3, 1, 4]));
    /// assert_eq!(rows.next(), Some([1, 5, 9]));
    /// assert_eq!(rows.next(), Some([2, 6, 5]));
    /// assert_eq!(rows.next(), None);
    /// ```
    pub fn rows<'a>(&'a self) -> MatrixRowIterator<'a, T> {
        MatrixRowIterator {
            matrix: self,
            pos: 0,
        }
    }

    /// Get the column of the matrix. It will returns a reference to a column
    /// of the matrix.
    ///
    /// Because we store the data in row-major order, this operation is `O(m)`
    /// where `m` is a number of columns.
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::*;
    /// let W = matrix![
    ///     3.0, 1.0;
    ///     4.0, 1.0;
    ///     5.0, 9.0;
    /// ];
    /// assert_eq!(W.col(0), [3.0, 4.0, 5.0]);
    /// ```
    ///
    /// # Panics
    /// Panics if `j >= m` where `m` is number of columns.
    pub fn col<'a>(&'a self, j: usize) -> ColMatrix<'a, T> {
        self.bound_check(None, Some(j));
        // Get all element from j-th column
        let vec = (0..self.vec.len())
            .step_by(self.ncols)
            .map(|offset| self.vec[offset + j])
            .collect();
        ColMatrix { matrix: self, vec }
    }

    /// Iterates over columns of the matrix.
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::*;
    /// let w = matrix![
    ///     3, 1, 4;
    ///     1, 5, 9;
    ///     2, 6, 5
    /// ];
    /// let mut cols = w.cols();
    ///
    /// assert_eq!(cols.next(), Some([3, 1, 2]));
    /// assert_eq!(cols.next(), Some([1, 5, 6]));
    /// assert_eq!(cols.next(), Some([4, 9, 5]));
    /// assert_eq!(cols.next(), None);
    /// ```
    pub fn cols<'a>(&'a self) -> MatrixColumnIterator<'a, T> {
        MatrixColumnIterator {
            matrix: self,
            pos: 0,
        }
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
    /// # use crabsformer::*;
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
    /// # use crabsformer::*;
    /// let W: Matrix<i32> = Matrix::zeros([5, 5]);
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
    /// # use crabsformer::*;
    /// let W1 = matrix![3.0, 1.0; 4.0, 1.0];
    /// let W2 = Matrix::zeros_like(&W1);
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
    /// # use crabsformer::*;
    /// let W: Matrix<i32> = Matrix::ones([3, 5]);
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
    /// # use crabsformer::*;
    /// let W1 = matrix![3, 1; 4, 1; 5, 9];
    /// let W2 = Matrix::ones_like(&W1);
    /// ```
    pub fn ones_like(m: &Matrix<T>) -> Matrix<T>
    where
        T: FromPrimitive,
    {
        matrix![m.shape() => T::from_i32(1).unwrap()]
    }

    /// Raises each elements of matrix to the power of `exp`,
    /// using exponentiation by squaring. A new matrix is created and
    /// filled with the result. If you want to modify existing matrix
    /// use [`power_mut`].
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::*;
    /// let W1 = matrix![3, 1, 4; 1, 5, 9];
    /// let W2 = W1.power(2);
    /// assert_eq!(W2, matrix![9, 1, 16; 1, 25, 81]);
    /// ```
    ///
    /// [`power_mut`]: #power_mut
    pub fn power(&self, exp: usize) -> Matrix<T>
    where
        T: FromPrimitive,
    {
        let powered_vec = self.vec.power(exp);
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            vec: powered_vec,
        }
    }

    /// Raises each elements of matrix to the power of `exp`,
    /// using exponentiation by squaring. An existing matrix is modified and
    /// filled with the result. If you want to create new matrix
    /// use [`power`].
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::*;
    /// let mut W1 = matrix![3, 1, 4; 1, 5, 9];
    /// W1.power(2);
    /// assert_eq!(W1, matrix![9, 1, 16; 1, 25, 81]);
    /// ```
    ///
    /// [`power`]: #power
    pub fn power_mut(&mut self, exp: usize)
    where
        T: FromPrimitive,
    {
        self.vec.power_mut(exp);
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
        T: SampleUniform,
    {
        let total_elements = shape.iter().product();
        let vec = Vector::uniform(total_elements, low, high);

        Matrix {
            nrows: shape[0],
            ncols: shape[1],
            vec,
        }
    }

    /// Load Matrix from CSV file. You need to explicitly annotate the numeric type.
    ///
    /// # Examples
    ///
    /// ```
    /// use crabsformer::*;
    ///
    /// let dataset: Matrix<f32> = Matrix::from_csv("tests/weight.csv").load().unwrap();
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
        let total_elements = shape.iter().product();
        let vec = Vector::normal(total_elements, mean, std_dev);
        Matrix {
            nrows: shape[0],
            ncols: shape[1],
            vec,
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
        // Flatten the vector
        let vec = source.into_iter().flatten().collect();

        Matrix { nrows, ncols, vec }
    }
}

// Matrix comparison
impl<T> PartialEq for Matrix<T>
where
    T: Num + Copy,
{
    fn eq(&self, other: &Matrix<T>) -> bool {
        if self.vec != other.vec {
            return false;
        }
        true
    }
    fn ne(&self, other: &Matrix<T>) -> bool {
        if self.vec == other.vec {
            return false;
        }
        true
    }
}

/// Implements sub-matrix slicing with syntax
/// `w.slice(begin1 .. end1, begin2 .. end2)`.
///
/// Returns a new matrix that have elements of
/// the given matrix from the row range [`begin1`..`end1`)
/// and column range [`begin2`..`end2`).
///
/// This operation is `O(N)`, where `N` is the number of
/// sliced rows.
///
/// # Panics
/// Requires that `begin1 <= end1`, `begin2 <= end2`, `end1 <= nrows`
/// and `end2 <= ncols` where `nrows` is the number of rows and `ncols`
/// is the number of columns. Otherwise it will panic.
///
/// # Examples
/// ```
/// # use crabsformer::*;
/// let w = matrix![
///     3, 1, 4;
///     1, 5, 9;
/// ];
///
/// // (Range, Range)
/// assert_eq!(w.slice(0..1, 0..1), matrix![3]);
///
/// // (Range, RangeTo)
/// assert_eq!(w.slice(0..2, ..2), matrix![3, 1; 1, 5]);
///
/// // (Range, RangeFrom)
/// assert_eq!(w.slice(0..2, 1..), matrix![1, 4; 5, 9]);
///
/// // (Range, RangeFull)
/// assert_eq!(w.slice(0..2, ..), matrix![3, 1, 4; 1, 5, 9]);
///
/// // (Range, RangeInclusive)
/// assert_eq!(w.slice(0..2, 0..=1), matrix![3, 1; 1, 5]);
///
/// // (Range, RangeToInclusive)
/// assert_eq!(w.slice(0..2, ..=1), matrix![3, 1; 1, 5]);
///
/// // (RangeTo, Range)
/// assert_eq!(w.slice(..1, 0..1), matrix![3]);
///
/// // (RangeTo, RangeTo)
/// assert_eq!(w.slice(..2, ..2), matrix![3, 1; 1, 5]);
///
/// // (RangeTo, RangeFrom)
/// assert_eq!(w.slice(..2, 1..), matrix![1, 4; 5, 9]);
///
/// // (RangeTo, RangeFull)
/// assert_eq!(w.slice(..2, ..), matrix![3, 1, 4; 1, 5, 9]);
///
/// // (RangeTo, RangeInclusive)
/// assert_eq!(w.slice(..2, 0..=1), matrix![3, 1; 1, 5]);
///
/// // (RangeTo, RangeToInclusive)
/// assert_eq!(w.slice(..2, ..=1), matrix![3, 1; 1, 5]);
///
/// // (RangeFrom, Range)
/// assert_eq!(w.slice(0.., 0..1), matrix![3; 1]);
///
/// // (RangeFrom, RangeTo)
/// assert_eq!(w.slice(0.., ..2), matrix![3, 1; 1, 5]);
///
/// // (RangeFrom, RangeFrom)
/// assert_eq!(w.slice(0.., 1..), matrix![1, 4; 5, 9]);
///
/// // (RangeFrom, RangeFull)
/// assert_eq!(w.slice(0.., ..), matrix![3, 1, 4; 1, 5, 9]);
///
/// // (RangeFrom, RangeInclusive)
/// assert_eq!(w.slice(0.., 0..=1), matrix![3, 1; 1, 5]);
///
/// // (RangeFrom, RangeToInclusive)
/// assert_eq!(w.slice(0.., ..=1), matrix![3, 1; 1, 5]);
///
/// // (RangeFull, Range)
/// assert_eq!(w.slice(.., 0..1), matrix![3; 1]);
///
/// // (RangeFull, RangeTo)
/// assert_eq!(w.slice(.., ..2), matrix![3, 1; 1, 5]);
///
/// // (RangeFull, RangeFrom)
/// assert_eq!(w.slice(.., 1..), matrix![1, 4; 5, 9]);
///
/// // (RangeFull, RangeFull)
/// assert_eq!(w.slice(.., ..), matrix![3, 1, 4; 1, 5, 9]);
///
/// // (RangeFull, RangeInclusive)
/// assert_eq!(w.slice(.., 0..=1), matrix![3, 1; 1, 5]);
///
/// // (RangeFull, RangeToInclusive)
/// assert_eq!(w.slice(.., ..=1), matrix![3, 1; 1, 5]);
///
/// // (RangeInclusive, Range)
/// assert_eq!(w.slice(0..=1, 0..1), matrix![3; 1]);
///
/// // (RangeInclusive, RangeTo)
/// assert_eq!(w.slice(0..=1, ..2), matrix![3, 1; 1, 5]);
///
/// // (RangeInclusive, RangeFrom)
/// assert_eq!(w.slice(0..=1, 1..), matrix![1, 4; 5, 9]);
///
/// // (RangeInclusive, RangeFull)
/// assert_eq!(w.slice(0..=1, ..), matrix![3, 1, 4; 1, 5, 9]);
///
/// // (RangeInclusive, RangeInclusive)
/// assert_eq!(w.slice(0..=1, 0..=1), matrix![3, 1; 1, 5]);
///
/// // (RangeInclusive, RangeToInclusive)
/// assert_eq!(w.slice(0..=1, ..=1), matrix![3, 1; 1, 5]);
///
/// // (RangeToInclusive, Range)
/// assert_eq!(w.slice(..=1, 0..1), matrix![3; 1]);
///
/// // (RangeToInclusive, RangeTo)
/// assert_eq!(w.slice(..=1, ..2), matrix![3, 1; 1, 5]);
///
/// // (RangeToInclusive, RangeFrom)
/// assert_eq!(w.slice(..=1, 1..), matrix![1, 4; 5, 9]);
///
/// // (RangeToInclusive, RangeFull)
/// assert_eq!(w.slice(..=1, ..), matrix![3, 1, 4; 1, 5, 9]);
///
/// // (RangeToInclusive, RangeInclusive)
/// assert_eq!(w.slice(..=1, 0..=1), matrix![3, 1; 1, 5]);
///
/// // (RangeToInclusive, RangeToInclusive)
/// assert_eq!(w.slice(..=1, ..=1), matrix![3, 1; 1, 5]);
/// ```
//macro_rules! impl_slice_ops_with_range_combination {
//    ($row_i:ty, $col_i:ty) => {
//        impl<T> MatrixSlice<$row_i, $col_i> for Matrix<T>
//        where
//            T: Num + Copy,
//        {
//            type Output = Matrix<T>;
//
//            fn slice(&self, i: $row_i, j: $col_i) -> Matrix<T> {
//                let sliced_elements = self.elements[i]
//                    .iter()
//                    .map(|row| {
//                        let sliced_column = row.vec.slice(j.clone());
//                        RowMatrix::from(sliced_column)
//                    })
//                    .collect::<Vec<RowMatrix<T>>>();
//                Matrix::from(sliced_elements)
//            }
//        }
//    };
//}
//
//impl_slice_ops_with_range_combination!(ops::Range<usize>, ops::Range<usize>);
//impl_slice_ops_with_range_combination!(
//    ops::Range<usize>,
//    ops::RangeFrom<usize>
//);
//impl_slice_ops_with_range_combination!(
//    ops::Range<usize>,
//    ops::RangeTo<usize>
//);
//impl_slice_ops_with_range_combination!(ops::Range<usize>, ops::RangeFull);
//impl_slice_ops_with_range_combination!(
//    ops::Range<usize>,
//    ops::RangeInclusive<usize>
//);
//impl_slice_ops_with_range_combination!(
//    ops::Range<usize>,
//    ops::RangeToInclusive<usize>
//);
//impl_slice_ops_with_range_combination!(
//    ops::RangeFrom<usize>,
//    ops::Range<usize>
//);
//impl_slice_ops_with_range_combination!(
//    ops::RangeFrom<usize>,
//    ops::RangeFrom<usize>
//);
//impl_slice_ops_with_range_combination!(
//    ops::RangeFrom<usize>,
//    ops::RangeTo<usize>
//);
//impl_slice_ops_with_range_combination!(ops::RangeFrom<usize>, ops::RangeFull);
//impl_slice_ops_with_range_combination!(
//    ops::RangeFrom<usize>,
//    ops::RangeInclusive<usize>
//);
//impl_slice_ops_with_range_combination!(
//    ops::RangeFrom<usize>,
//    ops::RangeToInclusive<usize>
//);
//impl_slice_ops_with_range_combination!(
//    ops::RangeTo<usize>,
//    ops::Range<usize>
//);
//impl_slice_ops_with_range_combination!(
//    ops::RangeTo<usize>,
//    ops::RangeFrom<usize>
//);
//impl_slice_ops_with_range_combination!(
//    ops::RangeTo<usize>,
//    ops::RangeTo<usize>
//);
//impl_slice_ops_with_range_combination!(ops::RangeTo<usize>, ops::RangeFull);
//impl_slice_ops_with_range_combination!(
//    ops::RangeTo<usize>,
//    ops::RangeInclusive<usize>
//);
//impl_slice_ops_with_range_combination!(
//    ops::RangeTo<usize>,
//    ops::RangeToInclusive<usize>
//);
//impl_slice_ops_with_range_combination!(ops::RangeFull, ops::Range<usize>);
//impl_slice_ops_with_range_combination!(ops::RangeFull, ops::RangeFrom<usize>);
//impl_slice_ops_with_range_combination!(ops::RangeFull, ops::RangeTo<usize>);
//impl_slice_ops_with_range_combination!(ops::RangeFull, ops::RangeFull);
//impl_slice_ops_with_range_combination!(
//    ops::RangeFull,
//    ops::RangeInclusive<usize>
//);
//impl_slice_ops_with_range_combination!(
//    ops::RangeFull,
//    ops::RangeToInclusive<usize>
//);
//impl_slice_ops_with_range_combination!(
//    ops::RangeInclusive<usize>,
//    ops::Range<usize>
//);
//impl_slice_ops_with_range_combination!(
//    ops::RangeInclusive<usize>,
//    ops::RangeFrom<usize>
//);
//impl_slice_ops_with_range_combination!(
//    ops::RangeInclusive<usize>,
//    ops::RangeTo<usize>
//);
//impl_slice_ops_with_range_combination!(
//    ops::RangeInclusive<usize>,
//    ops::RangeFull
//);
//impl_slice_ops_with_range_combination!(
//    ops::RangeInclusive<usize>,
//    ops::RangeInclusive<usize>
//);
//impl_slice_ops_with_range_combination!(
//    ops::RangeInclusive<usize>,
//    ops::RangeToInclusive<usize>
//);
//impl_slice_ops_with_range_combination!(
//    ops::RangeToInclusive<usize>,
//    ops::Range<usize>
//);
//impl_slice_ops_with_range_combination!(
//    ops::RangeToInclusive<usize>,
//    ops::RangeFrom<usize>
//);
//impl_slice_ops_with_range_combination!(
//    ops::RangeToInclusive<usize>,
//    ops::RangeTo<usize>
//);
//impl_slice_ops_with_range_combination!(
//    ops::RangeToInclusive<usize>,
//    ops::RangeFull
//);
//impl_slice_ops_with_range_combination!(
//    ops::RangeToInclusive<usize>,
//    ops::RangeInclusive<usize>
//);
//impl_slice_ops_with_range_combination!(
//    ops::RangeToInclusive<usize>,
//    ops::RangeToInclusive<usize>
//);

/// Matrix slice operation
pub trait MatrixSlice<'a, RowIdx, ColIdx>
where
    RowIdx: ?Sized,
    ColIdx: ?Sized,
{
    /// The returned type after indexing.
    type Output: ?Sized;

    /// Performs the slicing (`container.slice(index1, index2)`) operation.
    /// It returns new matrix with the sliced elements.
    fn slice(&'a self, row_index: RowIdx, col_index: ColIdx) -> Self::Output;
}

/// Sub matrix is a reference to elements in the matrix.
pub struct SubMatrix<'a, T>
where
    T: Num + Copy,
{
    nrows: usize,
    ncols: usize,
    matrix: &'a Matrix<T>,
    // Copy data from matrix
    vec: Vector<T>,
}

impl<'a, T: 'a> SubMatrix<'a, T>
where
    T: Num + Copy,
{
    // Shape
    pub fn shape(&self) -> [usize; 2] {
        [self.nrows, self.ncols]
    }
}

// TODO(pyk): Pretty print on {:?}
impl<'a, T> fmt::Debug for SubMatrix<'a, T>
where
    T: Num + Copy + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SubMatrix({:?})", self.vec)
    }
}

// Sub Matrix comparison
// TODO(pyk): test this
impl<'a, T> PartialEq for SubMatrix<'a, T>
where
    T: Num + Copy,
{
    fn eq(&self, other: &SubMatrix<'a, T>) -> bool {
        if self.shape() == other.shape() && self.vec == other.vec {
            true
        } else {
            false
        }
    }
    fn ne(&self, other: &SubMatrix<'a, T>) -> bool {
        if self.shape() != other.shape() || self.vec != other.vec {
            true
        } else {
            false
        }
    }
}

// Sub Matrix vs Matrix comparison
// TODO(pyk): test this
impl<'a, T> PartialEq<Matrix<T>> for SubMatrix<'a, T>
where
    T: Num + Copy,
{
    fn eq(&self, other: &Matrix<T>) -> bool {
        if self.shape() == other.shape() && self.vec == other.vec {
            true
        } else {
            false
        }
    }
    fn ne(&self, other: &Matrix<T>) -> bool {
        if self.shape() != other.shape() || self.vec != other.vec {
            true
        } else {
            false
        }
    }
}

impl<'a, T: 'a> MatrixSlice<'a, ops::Range<usize>, ops::Range<usize>>
    for Matrix<T>
where
    T: Num + Copy,
{
    type Output = SubMatrix<'a, T>;

    fn slice(
        &'a self,
        irange: ops::Range<usize>,
        jrange: ops::Range<usize>,
    ) -> SubMatrix<'a, T> {
        // Check the slice index first, make sure the slice index `start < end`
        if irange.start >= irange.end {
            panic!(
                "Matrix slice index row starts at {} but ends at {}",
                irange.start, irange.end
            )
        }
        if jrange.start >= jrange.end {
            panic!(
                "Matrix slice index column starts at {} but ends at {}",
                jrange.start, jrange.end
            )
        }

        // Make sure irange.end-1 < self.nrows and jrange.end-1 < self.ncols
        // NOTE: range.end is excelusive, so we substract it by 1
        self.bound_check(Some(irange.end - 1), Some(jrange.end - 1));

        // Get the new nrows and new ncols
        let nrows = irange.len();
        let ncols = jrange.len();

        // Collect the matrix elements
        let vec = irange
            .map(|i| {
                jrange.clone().map(|j| *self.at(i, j)).collect::<Vec<T>>()
            })
            .flatten()
            .collect::<Vector<T>>();

        // Return a sub matrix
        SubMatrix {
            nrows,
            ncols,
            matrix: self,
            vec,
        }
    }
}

/// Matrix row iterator.
pub struct MatrixRowIterator<'a, T: 'a>
where
    T: Num + Copy,
{
    matrix: &'a Matrix<T>,
    pos: usize,
}

impl<'a, T> Iterator for MatrixRowIterator<'a, T>
where
    T: Num + Copy,
{
    type Item = RowMatrix<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.matrix.nrows {
            // Increment the position of the row iterator.
            self.pos += 1;
            // Return the row
            Some(self.matrix.row(self.pos - 1))
        } else {
            None
        }
    }
}

/// Matrix column iterator.
pub struct MatrixColumnIterator<'a, T: 'a>
where
    T: Num + Copy,
{
    matrix: &'a Matrix<T>,
    pos: usize,
}

impl<'a, T> Iterator for MatrixColumnIterator<'a, T>
where
    T: Num + Copy,
{
    type Item = ColMatrix<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.matrix.ncols {
            // Increment the position of the row iterator.
            self.pos += 1;
            // Return the row
            Some(self.matrix.col(self.pos - 1))
        } else {
            None
        }
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

        // Add the element of the matrix
        let vec = self.vec + other.vec;
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            vec,
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
        let vec = self.vec + value;
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            vec,
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
                let vec = self + m.vec;
                Matrix {
                    nrows: m.nrows,
                    ncols: m.ncols,
                    vec,
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
        self.vec += other.vec;
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
        self.vec += value;
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
        let vec = self.vec - other.vec;
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            vec,
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
        let vec = self.vec - value;
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            vec,
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
                let vec = self - m.vec;
                Matrix {
                    nrows: m.nrows,
                    ncols: m.ncols,
                    vec,
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
        self.vec -= other.vec;
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
        self.vec -= value;
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
        let vec = self.vec * other.vec;
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            vec,
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
        let vec = self.vec * value;
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            vec,
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
                let vec = self * m.vec;
                Matrix {
                    nrows: m.nrows,
                    ncols: m.ncols,
                    vec,
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

        self.vec *= other.vec;
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
        self.vec *= value;
    }
}

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
    /// use crabsformer::*;
    ///
    /// let dataset: Matrix<f32> = Matrix::from_csv("tests/dataset.csv")
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
    /// use crabsformer::*;
    ///
    /// let dataset: Matrix<f32> = Matrix::from_csv("tests/weight.csv").load().unwrap();
    /// ```
    pub fn load(self) -> Result<Matrix<T>, LoadError>
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
                        return Err(LoadError::new(
                            LoadErrorKind::InvalidElement,
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
            return Err(LoadError::new(
                LoadErrorKind::Empty,
                String::from("Cannot load empty file"),
            ));
        }
        Ok(Matrix::from(elements))
    }
}

/// Row Matrix is a reference of a row in a Matrix.
///
/// It is a `1xm` matrix where `m` is number of columns.
pub struct RowMatrix<'a, T>
where
    T: Num + Copy,
{
    matrix: &'a Matrix<T>,
    vec: Vector<T>,
}

impl<'a, T> RowMatrix<'a, T>
where
    T: Num + Copy,
{
    // Row matrix is a `1xm` matrix
    pub fn shape(&self) -> [usize; 2] {
        [1, self.matrix.ncols]
    }

    // Iterates over row elements
    pub fn elements(&'a self) -> VectorElementIterator<'a, T> {
        self.vec.elements()
    }
}

impl<'a, T> fmt::Debug for RowMatrix<'a, T>
where
    T: Num + Copy + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        return write!(f, "RowMatrix({:?})", self.vec);
    }
}

// RowMatrix vs RowMatrix comparison
impl<'a, T> PartialEq for RowMatrix<'a, T>
where
    T: Num + Copy,
{
    fn eq(&self, other: &RowMatrix<'a, T>) -> bool {
        if self.vec == other.vec {
            true
        } else {
            false
        }
    }
    fn ne(&self, other: &RowMatrix<'a, T>) -> bool {
        if self.vec != other.vec {
            true
        } else {
            false
        }
    }
}

// RowMatrix vs Matrix comparison
impl<'a, T> PartialEq<Matrix<T>> for RowMatrix<'a, T>
where
    T: Num + Copy,
{
    fn eq(&self, m: &Matrix<T>) -> bool {
        if self.shape() == m.shape() && self.vec == m.vec {
            true
        } else {
            false
        }
    }
    fn ne(&self, m: &Matrix<T>) -> bool {
        if self.shape() != m.shape() || self.vec != m.vec {
            true
        } else {
            false
        }
    }
}

/// Column matrix is a reference of a column in a Matrix.
///
/// It is a `nx1` matrix where `n` is a number of rows.
pub struct ColMatrix<'a, T>
where
    T: Num + Copy,
{
    matrix: &'a Matrix<T>,
    vec: Vector<T>,
}

impl<'a, T> ColMatrix<'a, T>
where
    T: Num + Copy,
{
    // Column matrix is a `nx1` matrix
    pub fn shape(&self) -> [usize; 2] {
        [self.matrix.nrows, 1]
    }
}

impl<'a, T> fmt::Debug for ColMatrix<'a, T>
where
    T: Num + Copy + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        return write!(f, "ColMatrix({:?})", self.vec);
    }
}

// ColMatrix vs ColMatrix comparison
impl<'a, T> PartialEq for ColMatrix<'a, T>
where
    T: Num + Copy,
{
    fn eq(&self, other: &ColMatrix<'a, T>) -> bool {
        if self.vec == other.vec {
            true
        } else {
            false
        }
    }
    fn ne(&self, other: &ColMatrix<'a, T>) -> bool {
        if self.vec != other.vec {
            true
        } else {
            false
        }
    }
}

// ColMatrix vs Matrix comparison
impl<'a, T> PartialEq<Matrix<T>> for ColMatrix<'a, T>
where
    T: Num + Copy,
{
    fn eq(&self, m: &Matrix<T>) -> bool {
        if self.shape() == m.shape() && self.vec == m.vec {
            true
        } else {
            false
        }
    }
    fn ne(&self, m: &Matrix<T>) -> bool {
        if self.shape() != m.shape() || self.vec != m.vec {
            true
        } else {
            false
        }
    }
}
