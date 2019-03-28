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

/// Numeric vector slice operation
pub trait VectorSlice<Idx: ?Sized> {
    /// The returned type after indexing.
    type Output: ?Sized;

    /// Performs the slicing (`container.slice[index]`) operation.
    /// It returns new numeric vector with the sliced elements.
    fn slice(&self, index: Idx) -> Self::Output;
}

/// Matrix slice operation
pub trait MatrixSlice<RowIdx: ?Sized, ColIdx: ?Sized> {
    /// The returned type after indexing.
    type Output: ?Sized;

    /// Performs the slicing (`container.slice(index1, index2)`) operation.
    /// It returns new matrix with the sliced elements.
    fn slice(&self, row_index: RowIdx, col_index: ColIdx) -> Self::Output;
}
