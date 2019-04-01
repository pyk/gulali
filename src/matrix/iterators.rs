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

//! A list of function to iterate over rows and cols.
//!
//! TODO(pyk): Add docs here
//!

use crate::matrix::{ColumnMatrix, Matrix, RowMatrix, Submatrix};
use num::Num;

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
    type Item = ColumnMatrix<'a, T>;

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

impl<T> Matrix<T>
where
    T: Num + Copy,
{
    /// Iterates over rows of the matrix.
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::prelude::*;
    /// let w = matrix![
    ///     3, 1, 4;
    ///     1, 5, 9;
    ///     2, 6, 5
    /// ];
    /// for row in w.rows() {
    ///     println!("row = {:?}", row);
    /// }
    /// ```
    pub fn rows<'a>(&'a self) -> MatrixRowIterator<'a, T> {
        MatrixRowIterator {
            matrix: self,
            pos: 0,
        }
    }

    /// Iterates over columns of the matrix.
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::prelude::*;
    /// let w = matrix![
    ///     3, 1, 4;
    ///     1, 5, 9;
    ///     2, 6, 5
    /// ];
    ///
    /// for col in w.cols() {
    ///     println!("col = {:?}", col);
    /// }
    /// ```
    pub fn cols<'a>(&'a self) -> MatrixColumnIterator<'a, T> {
        MatrixColumnIterator {
            matrix: self,
            pos: 0,
        }
    }
}

/// Matrix row element iterator.
pub struct MatrixRowElementIterator<'a, T: 'a>
where
    T: Num + Copy,
{
    row: &'a RowMatrix<'a, T>,
    pos: usize,
}

impl<'a, T> Iterator for MatrixRowElementIterator<'a, T>
where
    T: Num + Copy,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.row.size {
            // Increment the position of the row iterator.
            self.pos += 1;
            // Return the element
            Some(self.row[self.pos - 1])
        } else {
            None
        }
    }
}

impl<'a, T> RowMatrix<'a, T>
where
    T: Num + Copy,
{
    /// Iterates over element of the row matrix.
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::prelude::*;
    /// let w = matrix![
    ///     3, 1, 4;
    ///     1, 5, 9;
    ///     2, 6, 5;
    /// ];
    /// let row = w.row(2);
    /// let mut elements = row.elements();
    ///
    /// assert_eq!(elements.next(), Some(2));
    /// assert_eq!(elements.next(), Some(6));
    /// assert_eq!(elements.next(), Some(5));
    /// assert_eq!(elements.next(), None);
    /// ```
    pub fn elements(&'a self) -> MatrixRowElementIterator<'a, T> {
        MatrixRowElementIterator { row: self, pos: 0 }
    }
}

/// Matrix column element iterator.
pub struct MatrixColumnElementIterator<'a, T: 'a>
where
    T: Num + Copy,
{
    col: &'a ColumnMatrix<'a, T>,
    pos: usize,
}

impl<'a, T> Iterator for MatrixColumnElementIterator<'a, T>
where
    T: Num + Copy,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.col.size {
            // Increment the position of the row iterator.
            self.pos += 1;
            // Return the element
            Some(self.col[self.pos - 1])
        } else {
            None
        }
    }
}

impl<'a, T> ColumnMatrix<'a, T>
where
    T: Num + Copy,
{
    /// Iterates over element of the column matrix.
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::prelude::*;
    /// let w = matrix![
    ///     3, 1, 4;
    ///     1, 5, 9;
    ///     2, 6, 5;
    /// ];
    /// let col = w.col(2);
    /// let mut elements = col.elements();
    ///
    /// assert_eq!(elements.next(), Some(4));
    /// assert_eq!(elements.next(), Some(9));
    /// assert_eq!(elements.next(), Some(5));
    /// assert_eq!(elements.next(), None);
    /// ```
    pub fn elements(&'a self) -> MatrixColumnElementIterator<'a, T> {
        MatrixColumnElementIterator { col: self, pos: 0 }
    }
}

/// Submatrix row iterator.
pub struct SubmatrixRowIterator<'a, T: 'a>
where
    T: Num + Copy,
{
    submatrix: &'a Submatrix<'a, T>,
    pos: usize,
}

impl<'a, T> Iterator for SubmatrixRowIterator<'a, T>
where
    T: Num + Copy,
{
    type Item = RowMatrix<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.submatrix.nrows {
            // Increment the position of the row iterator.
            self.pos += 1;
            // Return the row
            Some(self.submatrix.row(self.pos - 1))
        } else {
            None
        }
    }
}

/// Submatrix column iterator.
pub struct SubmatrixColumnIterator<'a, T: 'a>
where
    T: Num + Copy,
{
    submatrix: &'a Submatrix<'a, T>,
    pos: usize,
}

impl<'a, T> Iterator for SubmatrixColumnIterator<'a, T>
where
    T: Num + Copy,
{
    type Item = ColumnMatrix<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.submatrix.ncols {
            // Increment the position of the column iterator.
            self.pos += 1;
            // Return the row
            Some(self.submatrix.col(self.pos - 1))
        } else {
            None
        }
    }
}

impl<'a, T> Submatrix<'a, T>
where
    T: Num + Copy,
{
    /// Iterates over rows of the submatrix.
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::prelude::*;
    /// let w = matrix![
    ///     3, 1, 4;
    ///     1, 5, 9;
    ///     2, 6, 5;
    /// ];
    ///
    /// let sub = w.slice(0..2, ..);
    ///
    /// for row in sub.rows() {
    ///     println!("row = {:?}", row);
    /// }
    /// ```
    pub fn rows(&'a self) -> SubmatrixRowIterator<'a, T> {
        SubmatrixRowIterator {
            submatrix: self,
            pos: 0,
        }
    }

    /// Iterates over columns of the submatrix.
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::prelude::*;
    /// let w = matrix![
    ///     3, 1, 4;
    ///     1, 5, 9;
    ///     2, 6, 5
    /// ];
    ///
    /// let sub = w.slice(0..2, ..);
    ///
    /// for col in sub.cols() {
    ///     println!("col = {:?}", col);
    /// }
    /// ```
    pub fn cols(&'a self) -> SubmatrixColumnIterator<'a, T> {
        SubmatrixColumnIterator {
            submatrix: self,
            pos: 0,
        }
    }
}
