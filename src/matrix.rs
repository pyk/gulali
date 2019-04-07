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

//! Matrix
//!
//! TODO(pyk): Add docs here
//!

use crate::vector::*;
use num::Num;
use std::fmt;

// Import all sub modules
pub mod builders;
pub mod errors;
pub mod indexing;
pub mod iterators;
pub mod loaders;
pub mod operations;
pub mod slicing;

/// Matrix.
///
/// TODO: add overview about matrix here.
/// 1. how to create a matrix
/// 2. Matrix operation
/// 3. Indexing, etc.
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
    /// # use crabsformer::prelude::*;
    /// let w = matrix![
    ///     3.0, 1.0;
    ///     4.0, 1.0;
    ///     5.0, 9.0;
    /// ];
    /// assert_eq!(w.shape(), [3, 2]);
    /// ```
    pub fn shape(&self) -> [usize; 2] {
        [self.nrows, self.ncols]
    }
}

impl<T> fmt::Debug for Matrix<T>
where
    T: Num + Copy + fmt::Debug + ToString,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut output = String::from("[");
        for (i, row) in self.rows().enumerate() {
            let mut row_output = String::from("");
            // add space to the next line
            if i > 0 {
                row_output += " ";
            }
            row_output += "[";
            for (j, value) in row.elements().enumerate() {
                row_output += &value.to_string();
                // Don't add comma to the end of line
                if j < self.ncols - 1 {
                    row_output += ", ";
                } else {
                    row_output += "]";
                }
            }
            // Don't add new line to the end of row
            if i < self.ncols - 1 {
                row_output += ",\n";
            } else {
                row_output += "]";
            }
            output += &row_output.to_string();
        }

        write!(f, "{}", output)
    }
}

// Matrix comparison
// Matrix is equal if the shape and content are the same.
impl<T> PartialEq for Matrix<T>
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
        if self.shape() != self.shape() || self.vec != other.vec {
            true
        } else {
            false
        }
    }
}

/// Row matrix is reference to a row of a matrix.
///
/// It is a `1xm` matrix where `m` is a number of columns.
pub struct RowMatrix<'a, T>
where
    T: Num + Copy,
{
    // Row position in the matrix
    pos: usize,
    // Offset from the start of the row; we use this
    // to iterate over elements in the submatrix row
    offset: usize,
    // Row size
    size: usize,
    // Original matrix; where to get the elements from
    source: &'a Matrix<T>,
}

impl<'a, T> fmt::Debug for RowMatrix<'a, T>
where
    T: Num + Copy + fmt::Debug + ToString,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut output = String::from("[");
        for (i, value) in self.elements().enumerate() {
            output += &value.to_string();
            // Don't add comma to the end of line
            if i < self.size - 1 {
                output += ", ";
            }
        }
        output += "]";
        write!(f, "{}", output)
    }
}

// Row matrix comparison
// Row matrix is equal if the position, size and content are the same.
impl<'a, T> PartialEq for RowMatrix<'a, T>
where
    T: Num + Copy,
{
    fn eq(&self, other: &RowMatrix<'a, T>) -> bool {
        if self.pos == other.pos
            && self.size == other.size
            && self.offset == other.offset
            && self.source == other.source
        {
            true
        } else {
            false
        }
    }

    fn ne(&self, other: &RowMatrix<'a, T>) -> bool {
        if self.pos != other.pos
            || self.size != other.size
            || self.offset != other.offset
            || self.source != other.source
        {
            true
        } else {
            false
        }
    }
}

/// Column matrix is reference to a column of a matrix.
///
/// It is a `nx1` matrix where `n` is a number of rows.
pub struct ColumnMatrix<'a, T>
where
    T: Num + Copy,
{
    // Column position in the matrix
    pos: usize,
    // Offset from the start of the column; we use this
    // to iterate over elements in the submatrix row
    offset: usize,
    // Column size,
    size: usize,
    // Original matrix; where to get the elements from
    source: &'a Matrix<T>,
}

impl<'a, T> fmt::Debug for ColumnMatrix<'a, T>
where
    T: Num + Copy + fmt::Debug + ToString,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut output = String::from("[");
        for (i, value) in self.elements().enumerate() {
            output += &value.to_string();
            // Don't add comma to the end of line
            if i < self.size - 1 {
                output += "; ";
            }
        }
        output += "]";
        write!(f, "{}", output)
    }
}

// Column matrix comparison
// Column matrix is equal if the position and content are the same.
impl<'a, T> PartialEq for ColumnMatrix<'a, T>
where
    T: Num + Copy,
{
    fn eq(&self, other: &ColumnMatrix<'a, T>) -> bool {
        if self.pos == other.pos
            && self.size == other.size
            && self.offset == other.offset
            && self.source == other.source
        {
            true
        } else {
            false
        }
    }

    fn ne(&self, other: &ColumnMatrix<'a, T>) -> bool {
        if self.pos != other.pos
            || self.size != other.size
            || self.offset != other.offset
            || self.source != other.source
        {
            true
        } else {
            false
        }
    }
}

/// Submatrix is a reference to a block of the elements in the matrix.
///
/// TODO(pyk): Add visualization here based on this:
/// http://mathworld.wolfram.com/Submatrix.html
pub struct Submatrix<'a, T>
where
    T: Num + Copy,
{
    // The number of rows and columns of Submatrix
    nrows: usize,
    ncols: usize,
    // Offset row & column from original matrix;
    // we use these to access the data from original matrix
    row_offset: usize,
    col_offset: usize,
    // Original matrix; where to get the elements from
    source: &'a Matrix<T>,
}

impl<'a, T: 'a> Submatrix<'a, T>
where
    T: Num + Copy,
{
    // Shape of the submatrix
    pub fn shape(&self) -> [usize; 2] {
        [self.nrows, self.ncols]
    }

    // Offsets of the submatrix `[row_offset, col_offset]`
    pub fn offsets(&self) -> [usize; 2] {
        [self.row_offset, self.col_offset]
    }
}

impl<'a, T> fmt::Debug for Submatrix<'a, T>
where
    T: Num + Copy + fmt::Debug + ToString,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut output = String::from("[");
        for (i, row) in self.rows().enumerate() {
            let mut row_output = String::from("");
            // add space to the next line
            if i > 0 {
                row_output += " ";
            }
            row_output += "[";
            for (j, value) in row.elements().enumerate() {
                row_output += &value.to_string();
                // Don't add comma to the end of line
                if j < self.ncols - 1 {
                    row_output += ", ";
                } else {
                    row_output += "]";
                }
            }
            // Don't add new line to the end of row
            if i < self.ncols - 1 {
                row_output += ",\n";
            } else {
                row_output += "]";
            }
            output += &row_output.to_string();
        }

        write!(f, "{}", output)
    }
}

// Submatrix comparison
impl<'a, T> PartialEq for Submatrix<'a, T>
where
    T: Num + Copy,
{
    fn eq(&self, other: &Submatrix<'a, T>) -> bool {
        if self.shape() == other.shape()
            && self.offsets() == other.offsets()
            && self.source == other.source
        {
            true
        } else {
            false
        }
    }

    fn ne(&self, other: &Submatrix<'a, T>) -> bool {
        if self.shape() != other.shape()
            || self.offsets() != other.offsets()
            || self.source != other.source
        {
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix;

    #[test]
    fn test_matrix_debug() {
        let w = matrix![
            1, 2, 3;
            4, 5, 6;
            7, 8, 9;
        ];
        println!("{:?}", w);
    }

    #[test]
    fn test_submatrix_debug() {
        let w = matrix![
            1, 2, 3;
            4, 5, 6;
            7, 8, 9;
        ];
        let s = Submatrix {
            nrows: 2,
            ncols: 2,
            row_offset: 1,
            col_offset: 1,
            source: &w,
        };
        println!("{:?}", s);
    }

    #[test]
    fn test_row_matrix_debug() {
        let w = matrix![
            1, 2, 3;
            4, 5, 6;
            7, 8, 9;
        ];
        println!("{:?}", w.row(0));
        println!("{:?}", w.row(1));
        println!("{:?}", w.row(2));
    }

    #[test]
    fn test_column_matrix_debug() {
        let w = matrix![
            1, 2, 3;
            4, 5, 6;
            7, 8, 9;
        ];
        println!("{:?}", w.col(0));
        println!("{:?}", w.col(1));
        println!("{:?}", w.col(2));
    }
}
