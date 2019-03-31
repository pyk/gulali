
/// Implements matrix slicing with syntax
/// `w.slice(begin1 .. end1, begin2 .. end2)`.
///
/// Returns a reference to matrix that have elements of
/// the given matrix from the row range [`begin1`..`end1`)
/// and column range [`begin2`..`end2`).
///
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
