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

//! Indexing matrix element, row and column.
//!
//! TODO(pyk): Add docs here
//!
//!

use crate::matrix::{ColumnMatrix, Matrix, RowMatrix, Submatrix};
use num::Num;
use std::ops;

impl<T> Matrix<T>
where
    T: Num + Copy,
{
    // Bound check
    pub(crate) fn check_bound(&self, i: Option<usize>, j: Option<usize>) {
        if i.is_some() && i.unwrap() >= self.nrows {
            panic!(
                "Row index {} out of range for matrix with number of rows {}",
                i.unwrap(),
                self.nrows
            )
        }
        if j.is_some() && j.unwrap() >= self.ncols {
            panic!(
                "Column index {} out of range for matrix with number of columns {}",
                j.unwrap(),
                self.ncols
            )
        }
    }

    /// Get element of the matrix at row `i` and column `j`.
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::prelude::*;
    /// let w = matrix![
    ///     3, 1, 4;
    ///     1, 5, 9;
    /// ];
    ///
    /// assert_eq!(w.at(0, 0), &3);
    /// assert_eq!(w.at(0, 1), &1);
    /// assert_eq!(w.at(0, 2), &4);
    /// assert_eq!(w.at(1, 0), &1);
    /// assert_eq!(w.at(1, 1), &5);
    /// assert_eq!(w.at(1, 2), &9);
    /// ```
    ///
    /// # Panics
    /// Panics if `i >= nrows` and `j >= ncols`.
    pub fn at(&self, i: usize, j: usize) -> &T {
        self.check_bound(Some(i), Some(j));
        &self.vec[(self.ncols * i) + j]
    }

    /// Get the row of the matrix. It will returns a reference to a row
    /// of the matrix. Row matrix is `1xm` matrix, where `m` is the number
    /// of columns.
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::prelude::*;
    /// let w = matrix![
    ///     3.0, 1.0;
    ///     4.0, 1.0;
    ///     5.0, 9.0;
    /// ];
    /// w.row(0); // [3.0, 1.0]
    /// w.row(1); // [4.0, 1.0]
    /// w.row(2); // [5.0, 9.0]
    /// ```
    ///
    /// # Panics
    /// Panics if `i >= n` where `n` is number of rows.
    pub fn row<'a>(&'a self, i: usize) -> RowMatrix<'a, T> {
        self.check_bound(Some(i), None);
        RowMatrix {
            pos: i,
            offset: 0,
            size: self.ncols,
            source: self,
        }
    }

    /// Get the column of the matrix. It will returns a reference to a column
    /// of the matrix. Column matrix is `nx1` matrix, where `n` is the number
    /// of rows.
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::prelude::*;
    /// let w = matrix![
    ///     3.0, 1.0;
    ///     4.0, 1.0;
    ///     5.0, 9.0;
    /// ];
    /// w.col(0);  // [3.0, 4.0, 5.0]
    /// ```
    ///
    /// # Panics
    /// Panics if `j >= m` where `m` is number of columns.
    pub fn col<'a>(&'a self, j: usize) -> ColumnMatrix<'a, T> {
        self.check_bound(None, Some(j));
        ColumnMatrix {
            pos: j,
            offset: 0,
            size: self.nrows,
            source: self,
        }
    }
}

impl<'a, T> Submatrix<'a, T>
where
    T: Num + Copy,
{
    // Bound check
    pub(crate) fn bound_check(&self, i: Option<usize>, j: Option<usize>) {
        if i.is_some() && i.unwrap() >= self.nrows {
            panic!(
                "Row index {} out of range for matrix with number of rows {}",
                i.unwrap(),
                self.nrows
            )
        }
        if j.is_some() && j.unwrap() >= self.ncols {
            panic!(
                "Column index {} out of range for matrix with number of columns {}",
                j.unwrap(),
                self.ncols
            )
        }
    }

    /// Get element of the sub matrix at row `i` and column `j`.
    ///
    /// # Examples
    /// ```
    /// use crabsformer::prelude::*;
    ///
    /// let w = matrix![
    ///     3, 1, 4;
    ///     1, 5, 9;
    /// ];
    /// let sub = w.slice(0..1, 1..); // [[1, 4]]
    ///
    /// assert_eq!(sub.at(0, 0), &1);
    /// assert_eq!(sub.at(0, 1), &4);
    /// ```
    ///
    /// # Panics
    /// Panics if `i >= nrows` and `j >= ncols`.
    pub fn at(&self, i: usize, j: usize) -> &T {
        self.bound_check(Some(i), Some(j));
        &self.source.at(self.row_offset + i, self.col_offset + j)
    }

    /// Get the row of the sub matrix. Row matrix is `1xm` matrix, where `m`
    /// is the number of columns.
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::prelude::*;
    /// let w = matrix![
    ///     3.0, 1.0, 4.0;
    ///     4.0, 1.0, 7.0;
    ///     5.0, 9.0, 3.0;
    /// ];
    /// let sub = w.slice(1..3, ..3); // [4.0, 1.0, 7.0; 5.0, 9.0, 3.0]
    ///
    /// sub.row(0); // [4.0, 1.0, 7.0]
    /// sub.row(1); // [5.0, 9.0, 3.0]
    /// ```
    ///
    /// # Panics
    /// Panics if `i >= n` where `n` is number of submatrix rows.
    pub fn row(&'a self, i: usize) -> RowMatrix<'a, T> {
        self.bound_check(Some(i), None);
        RowMatrix {
            // Add the row offset from the submatrix
            pos: self.row_offset + i,
            // Offset where the row elements start
            offset: self.col_offset,
            // Using the number of columns of the submatrix
            // as the size of the row
            size: self.ncols,
            // Pass the matrix reference
            source: self.source,
        }
    }

    /// Get the column of the matrix. Column matrix is `nx1` matrix, where `n`
    /// is the number of rows.
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::prelude::*;
    /// let w = matrix![
    ///     3.0, 1.0, 4.0;
    ///     4.0, 1.0, 7.0;
    ///     5.0, 9.0, 3.0;
    /// ];
    /// let sub = w.slice(1..3, ..3); // [4.0, 1.0; 5.0, 9.0]
    ///
    /// sub.col(0); // [4.0; 5.0]
    /// sub.col(1); // [1.0; 9.0]
    /// ```
    ///
    /// # Panics
    /// Panics if `j >= m` where `m` is number of columns.
    pub fn col(&'a self, j: usize) -> ColumnMatrix<'a, T> {
        self.bound_check(None, Some(j));
        ColumnMatrix {
            pos: self.col_offset + j,
            offset: self.row_offset,
            size: self.ncols,
            source: self.source,
        }
    }
}

// Implement row matrix indexing
impl<'a, T> ops::Index<usize> for RowMatrix<'a, T>
where
    T: Num + Copy,
{
    type Output = T;

    fn index(&self, j: usize) -> &T {
        // Make sure the index is valid
        if j >= self.size {
            panic!(
                "index {} out of range for row matrix with number of elements {}",
                j,
                self.size
            )
        };
        self.source.at(self.pos, self.offset + j)
    }
}

// Implement column matrix indexing
impl<'a, T> ops::Index<usize> for ColumnMatrix<'a, T>
where
    T: Num + Copy,
{
    type Output = T;

    fn index(&self, i: usize) -> &T {
        // Make sure the index is valid
        if i >= self.size {
            panic!(
                "index {} out of range for column matrix with number of elements {}",
                i,
                self.size
            )
        };
        self.source.at(self.offset + i, self.pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix;

    #[test]
    fn test_matrix_index_element() {
        let w = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        assert_eq!(*w.at(0, 0), 3);
        assert_eq!(*w.at(0, 1), 1);
        assert_eq!(*w.at(0, 2), 4);
        assert_eq!(*w.at(1, 0), 1);
        assert_eq!(*w.at(1, 1), 5);
        assert_eq!(*w.at(1, 2), 9);
    }

    #[test]
    #[should_panic]
    fn test_matrix_indexing_invalid_i() {
        let w = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        w.at(10, 0);
    }

    #[test]
    #[should_panic]
    fn test_matrix_indexing_invalid_j() {
        let w = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        w.at(0, 10);
    }

    #[test]
    fn test_matrix_row() {
        let w = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        assert_eq!(
            w.row(0),
            RowMatrix {
                pos: 0,
                offset: 0,
                size: w.ncols,
                source: &w
            }
        );
    }

    #[test]
    #[should_panic]
    fn test_matrix_row_invalid() {
        let w = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        w.row(100);
    }

    #[test]
    fn test_matrix_col() {
        let w = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        assert_eq!(
            w.col(0),
            ColumnMatrix {
                pos: 0,
                offset: 0,
                size: w.nrows,
                source: &w
            }
        );
    }

    #[test]
    #[should_panic]
    fn test_matrix_col_invalid() {
        let w = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        w.col(100);
    }

    #[test]
    fn test_submatrix_index_element() {
        let w = matrix![
            3, 1, 4;
            1, 5, 9;
            2, 7, 8;
        ];
        // Sub matrix: [5, 9; 7, 8]
        let submatrix = Submatrix {
            nrows: 2,
            ncols: 2,
            row_offset: 1,
            col_offset: 1,
            source: &w,
        };
        assert_eq!(submatrix.at(0, 0), &5);
        assert_eq!(submatrix.at(0, 1), &9);
        assert_eq!(submatrix.at(1, 0), &7);
        assert_eq!(submatrix.at(1, 1), &8);
    }

    #[test]
    #[should_panic]
    fn test_submatrix_indexing_invalid_j() {
        let w = matrix![
            3, 1, 4;
            1, 5, 9;
            2, 7, 8;
        ];
        // Sub matrix: [5, 9; 7, 8]
        let submatrix = Submatrix {
            nrows: 2,
            ncols: 2,
            row_offset: 1,
            col_offset: 1,
            source: &w,
        };
        submatrix.at(0, 10);
    }

    #[test]
    fn test_submatrix_row() {
        let w = matrix![
            3, 1, 4;
            1, 5, 9;
            2, 7, 8;
        ];
        // Sub matrix: [5, 9; 7, 8]
        let submatrix = Submatrix {
            nrows: 2,
            ncols: 2,
            row_offset: 1,
            col_offset: 1,
            source: &w,
        };
        assert_eq!(
            submatrix.row(0),
            RowMatrix {
                pos: 1,
                offset: submatrix.col_offset,
                size: submatrix.ncols,
                source: &w
            }
        );
    }

    #[test]
    #[should_panic]
    fn test_submatrix_row_invalid() {
        let w = matrix![
            3, 1, 4;
            1, 5, 9;
            2, 7, 8;
        ];
        // Sub matrix: [5, 9; 7, 8]
        let submatrix = Submatrix {
            nrows: 2,
            ncols: 2,
            row_offset: 1,
            col_offset: 1,
            source: &w,
        };
        submatrix.row(100);
    }

    #[test]
    fn test_submatrix_col() {
        let w = matrix![
            3, 1, 4;
            1, 5, 9;
            2, 7, 8;
        ];
        // Sub matrix: [5, 9; 7, 8]
        let submatrix = Submatrix {
            nrows: 2,
            ncols: 2,
            row_offset: 1,
            col_offset: 1,
            source: &w,
        };
        assert_eq!(
            submatrix.col(0),
            ColumnMatrix {
                pos: 1,
                offset: submatrix.row_offset,
                size: submatrix.nrows,
                source: &w
            }
        );
    }

    #[test]
    #[should_panic]
    fn test_submatrix_col_invalid() {
        let w = matrix![
            3, 1, 4;
            1, 5, 9;
            2, 7, 8;
        ];
        // Sub matrix: [5, 9; 7, 8]
        let submatrix = Submatrix {
            nrows: 2,
            ncols: 2,
            row_offset: 1,
            col_offset: 1,
            source: &w,
        };
        submatrix.col(100);
    }

}
