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

//! Slice numeric vector.
//!
//! TODO(pyk): Add docs here
//!
//!

use crate::vector::{SubVector, Vector};
use num::Num;
use std::ops;

/// Numeric vector slice operation
/// Implements sub-numeric vector slicing with syntax
/// `x.slice(begin .. end)`.
///
/// Returns a reference to elements in numeric vector
/// from the range [`begin`..`end`). This operation is `O(1)`.
///
/// # Panics
/// Requires that `begin <= end` and `end <= len` where `len` is the
/// length of the numeric vector. Otherwise it will panic.
///
/// # Examples
/// ```
/// # use crabsformer::*;
/// let x = vector![3, 1, 2, 3];
/// // Range
/// x.slice(0..1); // [3]
/// // RangeTo
/// x.slice(..2); // [3, 1])
/// // RangeFrom
/// x.slice(2..); // [2, 3])
/// // RangeFull
/// x.slice(..); // [3, 1, 2, 3]
/// // RangeInclusive
/// x.slice(0..=1); // [3, 1]
/// // RangeToInclusive
/// x.slice(..=2); // [3, 1, 2]
/// ```
pub trait VectorSlice<'a, Idx>
where
    Idx: ?Sized,
{
    /// The returned type after indexing.
    type Output: ?Sized;

    /// Performs the slicing (`container.slice(index)`) operation.
    /// It returns sub numeric vector, a reference of elements
    /// in the numeric vector.
    fn slice(&'a self, range: Idx) -> Self::Output;
}

// Check the slice index first, make sure the slice index is `start < end`
// Set the panic behaviour same as Vec<T>
fn check_range(range: &ops::Range<usize>) {
    if range.start >= range.end {
        panic!(
            "Vector slice index starts at {} but ends at {}",
            range.start, range.end
        )
    }
}

fn check_range_inclusive(range: &ops::RangeInclusive<usize>) {
    if *range.start() > *range.end() {
        panic!(
            "Vector slice index starts at {} but ends at {}",
            range.start(),
            range.end()
        )
    }
}

// vector.slice(start..end)
impl<'a, T: 'a> VectorSlice<'a, ops::Range<usize>> for Vector<T>
where
    T: Num + Copy,
{
    type Output = SubVector<'a, T>;

    fn slice(&'a self, range: ops::Range<usize>) -> SubVector<'a, T> {
        // Make sure the range is valid
        check_range(&range);

        // Performs bound checking
        // range.end is exclusive, so we need to substract it by 1.
        self.check_bound(range.end - 1);

        // Returns new sub numeric vector
        SubVector {
            offset: range.start,
            size: range.end - range.start,
            source: self,
        }
    }
}

// vector.slice(start..)
impl<'a, T: 'a> VectorSlice<'a, ops::RangeFrom<usize>> for Vector<T>
where
    T: Num + Copy,
{
    type Output = SubVector<'a, T>;

    fn slice(&'a self, range: ops::RangeFrom<usize>) -> SubVector<'a, T> {
        // Performs bound checking
        self.check_bound(range.start);

        // Returns new sub numeric vector
        SubVector {
            offset: range.start,
            size: self.len() - range.start,
            source: self,
        }
    }
}

// vector.slice(start..)
impl<'a, T: 'a> VectorSlice<'a, ops::RangeTo<usize>> for Vector<T>
where
    T: Num + Copy,
{
    type Output = SubVector<'a, T>;

    fn slice(&'a self, range: ops::RangeTo<usize>) -> SubVector<'a, T> {
        // Performs bound checking
        // range.end is exlusive, so we need to substract it by one.
        self.check_bound(range.end - 1);

        // Returns new sub numeric vector
        SubVector {
            offset: 0,
            size: range.end,
            source: self,
        }
    }
}

// vector.slice(..)
impl<'a, T: 'a> VectorSlice<'a, ops::RangeFull> for Vector<T>
where
    T: Num + Copy,
{
    type Output = SubVector<'a, T>;

    fn slice(&'a self, _range: ops::RangeFull) -> SubVector<'a, T> {
        // Returns new sub numeric vector
        SubVector {
            offset: 0,
            size: self.len(),
            source: self,
        }
    }
}

// vector.slice(start..=end)
impl<'a, T: 'a> VectorSlice<'a, ops::RangeInclusive<usize>> for Vector<T>
where
    T: Num + Copy,
{
    type Output = SubVector<'a, T>;

    fn slice(
        &'a self,
        range: ops::RangeInclusive<usize>,
    ) -> SubVector<'a, T> {
        // Make sure the input is valid
        check_range_inclusive(&range);

        // Performs bound checking
        self.check_bound(*range.end());

        // Returns new sub numeric vector
        SubVector {
            offset: *range.start(),
            size: (*range.end() + 1) - *range.start(),
            source: self,
        }
    }
}

// vector.slice(start..=end)
impl<'a, T: 'a> VectorSlice<'a, ops::RangeToInclusive<usize>> for Vector<T>
where
    T: Num + Copy,
{
    type Output = SubVector<'a, T>;

    fn slice(
        &'a self,
        range: ops::RangeToInclusive<usize>,
    ) -> SubVector<'a, T> {
        // Performs bound checking
        self.check_bound(range.end);

        // Returns new sub numeric vector
        SubVector {
            offset: 0,
            size: range.end + 1,
            source: self,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector;

    // Test Slice(Range)
    // vector.slice(start..end)
    #[test]
    fn test_slice_range() {
        let v = vector![3, 1, 4, 1, 5, 9];
        let submatrix = v.slice(2..5);
        let expected = SubVector {
            offset: 2,
            size: 3,
            source: &v,
        };
        assert_eq!(submatrix, expected);
    }

    // Test Slice(RangeFrom)
    // vector.slice(start..)
    #[test]
    fn test_slice_rangefrom() {
        let v = vector![3, 1, 4, 1, 5, 9];
        let submatrix = v.slice(2..);
        let expected = SubVector {
            offset: 2,
            size: 4,
            source: &v,
        };
        assert_eq!(submatrix, expected);
    }

    // Test Slice(RangeTo)
    // vector.slice(..end)
    #[test]
    fn test_slice_rangeto() {
        let v = vector![3, 1, 4, 1, 5, 9];
        let submatrix = v.slice(..3);
        let expected = SubVector {
            offset: 0,
            size: 3,
            source: &v,
        };
        assert_eq!(submatrix, expected);
    }

    // Test Slice(RangeFull)
    // vector.slice(..)
    #[test]
    fn test_slice_rangefull() {
        let v = vector![3, 1, 4, 1, 5, 9];
        let submatrix = v.slice(..);
        let expected = SubVector {
            offset: 0,
            size: v.len(),
            source: &v,
        };
        assert_eq!(submatrix, expected);
    }

    // Test Slice(RangeInclusive)
    // vector.slice(start..=end)
    #[test]
    fn test_slice_rangeinclusive() {
        let v = vector![3, 1, 4, 1, 5, 9];
        let submatrix = v.slice(0..=2);
        let expected = SubVector {
            offset: 0,
            size: 3,
            source: &v,
        };
        assert_eq!(submatrix, expected);
    }

    // Test Slice(RangeToInclusive)
    // vector.slice(..=end)
    #[test]
    fn test_slice_rangetoinclusive() {
        let v = vector![3, 1, 4, 1, 5, 9];
        let submatrix = v.slice(..=2);
        let expected = SubVector {
            offset: 0,
            size: 3,
            source: &v,
        };
        assert_eq!(submatrix, expected);
    }
}
