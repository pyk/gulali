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

// TODO(pyk): Add docs here

use crate::matrix::{Matrix, Submatrix};
use num::Num;
use std::ops;

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

// Check the slice index first, make sure the slice index is `start < end`
// Set the panic behaviour same as Vec<T>
fn check_range(range: &ops::Range<usize>) {
    if range.start >= range.end {
        panic!(
            "Matrix slice index starts at {} but ends at {}",
            range.start, range.end
        )
    }
}

fn check_range_inclusive(range: &ops::RangeInclusive<usize>) {
    if *range.start() > *range.end() {
        panic!(
            "Matrix slice index starts at {} but ends at {}",
            range.start(),
            range.end()
        )
    }
}

impl<'a, T: 'a> MatrixSlice<'a, ops::Range<usize>, ops::Range<usize>>
    for Matrix<T>
where
    T: Num + Copy,
{
    type Output = Submatrix<'a, T>;

    fn slice(
        &'a self,
        irange: ops::Range<usize>,
        jrange: ops::Range<usize>,
    ) -> Submatrix<'a, T> {
        // Make sure the range is valid
        check_range(&irange);
        check_range(&jrange);

        // Make sure irange.end-1 < self.nrows and jrange.end-1 < self.ncols
        // NOTE: range.end is excelusive, so we substract it by 1
        self.check_bound(Some(irange.end - 1), Some(jrange.end - 1));

        // Get the new nrows and new ncols
        let nrows = irange.end - irange.start;
        let ncols = jrange.end - jrange.start;

        // Get the row & column offset
        let row_offset = irange.start;
        let col_offset = jrange.start;

        // Return a sub matrix
        Submatrix {
            nrows,
            ncols,
            row_offset,
            col_offset,
            source: self,
        }
    }
}

impl<'a, T: 'a> MatrixSlice<'a, ops::Range<usize>, ops::RangeFrom<usize>>
    for Matrix<T>
where
    T: Num + Copy,
{
    type Output = Submatrix<'a, T>;

    fn slice(
        &'a self,
        irange: ops::Range<usize>,
        jrange: ops::RangeFrom<usize>,
    ) -> Submatrix<'a, T> {
        // Make sure the range is valid
        check_range(&irange);

        // Make sure irange.end-1 < self.nrows and jrange.start < self.ncols
        // NOTE: range.end is exclusive, so we substract it by 1
        self.check_bound(Some(irange.end - 1), Some(jrange.start));

        // Get the new nrows and new ncols
        let nrows = irange.end - irange.start;
        let ncols = self.ncols - jrange.start;

        // Get the row & column offset
        let row_offset = irange.start;
        let col_offset = jrange.start;

        // Return a sub matrix
        Submatrix {
            nrows,
            ncols,
            row_offset,
            col_offset,
            source: self,
        }
    }
}

impl<'a, T: 'a> MatrixSlice<'a, ops::Range<usize>, ops::RangeTo<usize>>
    for Matrix<T>
where
    T: Num + Copy,
{
    type Output = Submatrix<'a, T>;

    fn slice(
        &'a self,
        irange: ops::Range<usize>,
        jrange: ops::RangeTo<usize>,
    ) -> Submatrix<'a, T> {
        // Make sure the range is valid
        check_range(&irange);

        // Make sure irange.end-1 < self.nrows and jrange.end-1 < self.ncols
        // NOTE: range.end is exclusive, so we substract it by 1
        self.check_bound(Some(irange.end - 1), Some(jrange.end - 1));

        // Get the new nrows and new ncols
        let nrows = irange.end - irange.start;
        let ncols = jrange.end;

        // Get the row & column offset
        let row_offset = irange.start;
        let col_offset = 0;

        // Return a sub matrix
        Submatrix {
            nrows,
            ncols,
            row_offset,
            col_offset,
            source: self,
        }
    }
}

impl<'a, T: 'a> MatrixSlice<'a, ops::Range<usize>, ops::RangeFull>
    for Matrix<T>
where
    T: Num + Copy,
{
    type Output = Submatrix<'a, T>;

    fn slice(
        &'a self,
        irange: ops::Range<usize>,
        _jrange: ops::RangeFull,
    ) -> Submatrix<'a, T> {
        // Make sure the range is valid
        check_range(&irange);

        // Make sure irange.end-1 < self.nrows
        // NOTE: range.end is exclusive, so we substract it by 1
        self.check_bound(Some(irange.end - 1), None);

        // Get the new nrows and new ncols
        let nrows = irange.end - irange.start;
        let ncols = self.ncols;

        // Get the row & column offset
        let row_offset = irange.start;
        let col_offset = 0;

        // Return a sub matrix
        Submatrix {
            nrows,
            ncols,
            row_offset,
            col_offset,
            source: self,
        }
    }
}

impl<'a, T: 'a> MatrixSlice<'a, ops::Range<usize>, ops::RangeInclusive<usize>>
    for Matrix<T>
where
    T: Num + Copy,
{
    type Output = Submatrix<'a, T>;

    fn slice(
        &'a self,
        irange: ops::Range<usize>,
        jrange: ops::RangeInclusive<usize>,
    ) -> Submatrix<'a, T> {
        // Make sure the range is valid
        check_range(&irange);
        check_range_inclusive(&jrange);

        // Make sure irange.end-1 < self.nrows and jrange.end < self.ncols
        // NOTE: range.end is exclusive, so we substract it by 1
        self.check_bound(Some(irange.end - 1), Some(*jrange.end()));

        // Get the new nrows and new ncols
        let nrows = irange.end - irange.start;
        let ncols = (*jrange.end() + 1) - *jrange.start();

        // Get the row & column offset
        let row_offset = irange.start;
        let col_offset = *jrange.start();

        // Return a sub matrix
        Submatrix {
            nrows,
            ncols,
            row_offset,
            col_offset,
            source: self,
        }
    }
}

impl<'a, T: 'a>
    MatrixSlice<'a, ops::Range<usize>, ops::RangeToInclusive<usize>>
    for Matrix<T>
where
    T: Num + Copy,
{
    type Output = Submatrix<'a, T>;

    fn slice(
        &'a self,
        irange: ops::Range<usize>,
        jrange: ops::RangeToInclusive<usize>,
    ) -> Submatrix<'a, T> {
        // Make sure the range is valid
        check_range(&irange);

        // Make sure irange.end-1 < self.nrows and jrange.end < self.ncols
        // NOTE: range.end is exclusive, so we substract it by 1
        self.check_bound(Some(irange.end - 1), Some(jrange.end));

        // Get the new nrows and new ncols
        let nrows = irange.end - irange.start;
        let ncols = jrange.end + 1;

        // Get the row & column offset
        let row_offset = irange.start;
        let col_offset = 0;

        // Return a sub matrix
        Submatrix {
            nrows,
            ncols,
            row_offset,
            col_offset,
            source: self,
        }
    }
}

impl<'a, T: 'a> MatrixSlice<'a, ops::RangeFrom<usize>, ops::Range<usize>>
    for Matrix<T>
where
    T: Num + Copy,
{
    type Output = Submatrix<'a, T>;

    fn slice(
        &'a self,
        irange: ops::RangeFrom<usize>,
        jrange: ops::Range<usize>,
    ) -> Submatrix<'a, T> {
        // Make sure the range is valid
        check_range(&jrange);

        // Make sure irange.start < self.nrows and jrange.end-1 < self.ncols
        // NOTE: range.end is excelusive, so we substract it by 1
        self.check_bound(Some(irange.start), Some(jrange.end - 1));

        // Get the new nrows and new ncols
        let nrows = self.nrows - irange.start;
        let ncols = jrange.end - jrange.start;

        // Get the row & column offset
        let row_offset = irange.start;
        let col_offset = jrange.start;

        // Return a sub matrix
        Submatrix {
            nrows,
            ncols,
            row_offset,
            col_offset,
            source: self,
        }
    }
}

impl<'a, T: 'a> MatrixSlice<'a, ops::RangeFrom<usize>, ops::RangeFrom<usize>>
    for Matrix<T>
where
    T: Num + Copy,
{
    type Output = Submatrix<'a, T>;

    fn slice(
        &'a self,
        irange: ops::RangeFrom<usize>,
        jrange: ops::RangeFrom<usize>,
    ) -> Submatrix<'a, T> {
        // Make sure irange.start < self.nrows and jrange.start < self.ncols
        self.check_bound(Some(irange.start), Some(jrange.start));

        // Get the new nrows and new ncols
        let nrows = self.nrows - irange.start;
        let ncols = self.ncols - jrange.start;

        // Get the row & column offset
        let row_offset = irange.start;
        let col_offset = jrange.start;

        // Return a sub matrix
        Submatrix {
            nrows,
            ncols,
            row_offset,
            col_offset,
            source: self,
        }
    }
}

impl<'a, T: 'a> MatrixSlice<'a, ops::RangeFrom<usize>, ops::RangeTo<usize>>
    for Matrix<T>
where
    T: Num + Copy,
{
    type Output = Submatrix<'a, T>;

    fn slice(
        &'a self,
        irange: ops::RangeFrom<usize>,
        jrange: ops::RangeTo<usize>,
    ) -> Submatrix<'a, T> {
        // Make sure irange.start < self.nrows and jrange.end-1 < self.ncols
        // NOTE: jrange.end is exlusive, so we must substract it by 1
        self.check_bound(Some(irange.start), Some(jrange.end - 1));

        // Get the new number of rows and the new number of columns
        let nrows = self.nrows - irange.start;
        let ncols = jrange.end;

        // Get the row & column offset
        let row_offset = irange.start;
        let col_offset = 0;

        // Return a sub matrix
        Submatrix {
            nrows,
            ncols,
            row_offset,
            col_offset,
            source: self,
        }
    }
}

// TODO(pyk): Implement the following slice combination
// (RangeFrom, RangeFull)
// (RangeFrom, RangeInclusive)
// (RangeFrom, RangeToInclusive)
// (RangeTo, Range)
// (RangeTo, RangeFrom)
// (RangeTo, RangeTo)
// (RangeTo, RangeFull)
// (RangeTo, RangeInclusive)
// (RangeTo, RangeToInclusive)
// (RangeFull, Range)
// (RangeFull, RangeFrom)
// (RangeFull, RangeTo)
// (RangeFull, RangeFull)
// (RangeFull, RangeInclusive)
// (RangeFull, RangeToInclusive)
// (RangeInclusive, Range)
// (RangeInclusive, RangeFrom)
// (RangeInclusive, RangeTo)
// (RangeInclusive, RangeFull)
// (RangeInclusive, RangeInclusive)
// (RangeInclusive, RangeToInclusive)
// (RangeToInclusive, Range)
// (RangeToInclusive, RangeFrom)
// (RangeToInclusive, RangeTo)
// (RangeToInclusive, RangeFull)
// (RangeToInclusive, RangeInclusive)
// (RangeToInclusive, RangeToInclusive)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix;

    // Test Slice(Range, Range)
    // matrix.slice(start..end, start..end)
    #[test]
    fn test_slice_range_range() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        let submatrix = m.slice(0..2, 1..3);
        let expected = Submatrix {
            nrows: 2,
            ncols: 2,
            row_offset: 0,
            col_offset: 1,
            source: &m,
        };
        assert_eq!(submatrix, expected);
    }

    #[test]
    #[should_panic]
    fn test_slice_range_range_invalid_range_row() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(3..2, 1..3);
    }

    #[test]
    #[should_panic]
    fn test_slice_range_range_invalid_range_col() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(0..1, 3..1);
    }

    #[test]
    #[should_panic]
    fn test_slice_range_range_invalid_row_out_of_bond() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(0..100, 1..3);
    }

    #[test]
    #[should_panic]
    fn test_slice_range_range_invalid_col_out_of_bond() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(0..2, 1..100);
    }

    // Test Slice(Range, RangeFrom)
    // matrix.slice(start..end, start..)
    #[test]
    fn test_slice_range_rangefrom() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        let submatrix = m.slice(0..2, 1..);
        let expected = Submatrix {
            nrows: 2,
            ncols: 2,
            row_offset: 0,
            col_offset: 1,
            source: &m,
        };
        assert_eq!(submatrix, expected);
    }

    #[test]
    #[should_panic]
    fn test_slice_range_rangefrom_invalid_range_row() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(3..2, 1..);
    }

    #[test]
    #[should_panic]
    fn test_slice_range_rangefrom_invalid_row_out_of_bond() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(0..100, 1..);
    }

    #[test]
    #[should_panic]
    fn test_slice_range_rangefrom_invalid_col_out_of_bond() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(0..2, 100..);
    }

    // Test Slice(Range, RangeTo)
    // matrix.slice(start..end, ..end)
    #[test]
    fn test_slice_range_rangeto() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        let submatrix = m.slice(0..2, ..1);
        let expected = Submatrix {
            nrows: 2,
            ncols: 1,
            row_offset: 0,
            col_offset: 0,
            source: &m,
        };
        assert_eq!(submatrix, expected);
    }

    #[test]
    #[should_panic]
    fn test_slice_range_rangeto_invalid_range_row() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(3..2, ..1);
    }

    #[test]
    #[should_panic]
    fn test_slice_range_rangeto_invalid_row_out_of_bond() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(0..100, ..1);
    }

    #[test]
    #[should_panic]
    fn test_slice_range_rangeto_invalid_col_out_of_bond() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(0..2, ..100);
    }

    // Test Slice(Range, RangeFull)
    // matrix.slice(start..end, ..)
    #[test]
    fn test_slice_range_rangefull() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        let submatrix = m.slice(0..2, ..);
        let expected = Submatrix {
            nrows: 2,
            ncols: 3,
            row_offset: 0,
            col_offset: 0,
            source: &m,
        };
        assert_eq!(submatrix, expected);
    }

    #[test]
    #[should_panic]
    fn test_slice_range_rangefull_invalid_range_row() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(3..2, ..);
    }

    #[test]
    #[should_panic]
    fn test_slice_range_rangefull_invalid_row_out_of_bond() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(0..100, ..);
    }

    // Test Slice(Range, RangeInclusive)
    // matrix.slice(start..end, start..=end)
    #[test]
    fn test_slice_range_rangeinclusive() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        let submatrix = m.slice(0..2, 1..=2);
        let expected = Submatrix {
            nrows: 2,
            ncols: 2,
            row_offset: 0,
            col_offset: 1,
            source: &m,
        };
        assert_eq!(submatrix, expected);
    }

    #[test]
    #[should_panic]
    fn test_slice_range_rangeinclusive_invalid_range_row() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(3..2, 1..=2);
    }

    #[test]
    #[should_panic]
    fn test_slice_range_rangeinclusive_invalid_range_col() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(0..1, 3..=1);
    }

    #[test]
    #[should_panic]
    fn test_slice_range_rangeinclusive_invalid_row_out_of_bond() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(0..100, 1..=2);
    }

    #[test]
    #[should_panic]
    fn test_slice_range_rangeinclusive_invalid_col_out_of_bond() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(0..2, 1..=100);
    }

    // Test Slice(Range, RangeToInclusive)
    // matrix.slice(start..end, ..=end)
    #[test]
    fn test_slice_range_rangetoinclusive() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        let submatrix = m.slice(0..2, ..=1);
        let expected = Submatrix {
            nrows: 2,
            ncols: 2,
            row_offset: 0,
            col_offset: 0,
            source: &m,
        };
        assert_eq!(submatrix, expected);
    }

    #[test]
    #[should_panic]
    fn test_slice_range_rangetoinclusive_invalid_range_row() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(3..2, ..=1);
    }

    #[test]
    #[should_panic]
    fn test_slice_range_rangetoinclusive_invalid_row_out_of_bond() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(0..100, ..=2);
    }

    #[test]
    #[should_panic]
    fn test_slice_range_rangetoinclusive_invalid_col_out_of_bond() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(0..2, ..=100);
    }

    // Test Slice(RangeFrom, Range)
    // matrix.slice(start.., start..end)
    #[test]
    fn test_slice_rangefrom_range() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        let submatrix = m.slice(1.., 1..3);
        let expected = Submatrix {
            nrows: 1,
            ncols: 2,
            row_offset: 1,
            col_offset: 1,
            source: &m,
        };
        assert_eq!(submatrix, expected);
    }

    #[test]
    #[should_panic]
    fn test_slice_rangefrom_range_invalid_range_col() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(0.., 3..1);
    }

    #[test]
    #[should_panic]
    fn test_slice_rangefrom_range_invalid_row_out_of_bond() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(100.., 1..3);
    }

    #[test]
    #[should_panic]
    fn test_slice_rangefrom_range_invalid_col_out_of_bond() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(0.., 1..100);
    }

    // Test Slice(RangeFrom, RangeFrom)
    // matrix.slice(start.., start..)
    #[test]
    fn test_slice_rangefrom_rangefrom() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        let submatrix = m.slice(1.., 1..);
        let expected = Submatrix {
            nrows: 1,
            ncols: 2,
            row_offset: 1,
            col_offset: 1,
            source: &m,
        };
        assert_eq!(submatrix, expected);
    }

    #[test]
    #[should_panic]
    fn test_slice_rangefrom_rangefrom_invalid_row_out_of_bond() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(100.., 1..);
    }

    #[test]
    #[should_panic]
    fn test_slice_rangefrom_rangefrom_invalid_col_out_of_bond() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(0.., 100..);
    }

    // Test Slice(RangeFrom, RangeTo)
    // matrix.slice(start.., ..end)
    #[test]
    fn test_slice_rangefrom_rangeto() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        let submatrix = m.slice(0.., ..1);
        let expected = Submatrix {
            nrows: 2,
            ncols: 1,
            row_offset: 0,
            col_offset: 0,
            source: &m,
        };
        assert_eq!(submatrix, expected);
    }

    #[test]
    #[should_panic]
    fn test_slice_rangefrom_rangeto_invalid_row_out_of_bond() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(100.., 1..);
    }

    #[test]
    #[should_panic]
    fn test_slice_rangefrom_rangeto_invalid_col_out_of_bond() {
        let m = matrix![
            3, 1, 4;
            1, 5, 9;
        ];
        m.slice(0.., ..100);
    }
}
