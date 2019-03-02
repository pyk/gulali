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
//
// This file may not be copied, modified, or distributed
// except according to those terms.

//! Vector attributes
//!
//! # Overview
//! Gulali uses Rust's [vector] standard data type extensively. We don't reinvent
//! yet-another data type to keep things simple and easy to use. Gulali added
//! the following attributes to Rust's vector:
//!
//! - [dim]: the number of dimensions of the vector.
//! - [shape]: This is a list of integers indicating the
//!   size of the vector in each dimension.
//!   For a matrix with `n` rows and `m` columns, shape will be `[n,m]`.
//!   The length of the shape is therefore the number of
//!   dimensions, `dim()`.
//! - [size]: the total number of elements of the vector.
//!   This is equal to the product of the elements of shape.
//!
//! [slice]: https://doc.rust-lang.org/rust-by-example/primitives/array.html
//! [dim]: trait.Dimension.html
//! [shape]: trait.Shape.html
//! [size]: trait.Size.html
//! [vector]: https://doc.rust-lang.org/std/vec/struct.Vec.html
//!
//! # Examples
//! ```rust
//! # use gulali::prelude::*;
//! // Create two-dimensional vector with shape [3, 3]
//! // filled with zeros
//! let matrix: Vec<Vec<i32>> = Vec::two_dim(3, 3).zeros();
//!
//! assert_eq!(matrix.dim(), 2);
//! assert_eq!(matrix.shape(), [3, 3]);
//! assert_eq!(matrix.size(), 9);
//! ```

/// Dimension of the vector
pub trait Dimension<T>
where
    T: Copy,
{
    /// Returns the number of dimensions of the vector
    ///
    /// # Examples
    /// ```
    /// # use gulali::prelude::*;
    /// let arr1: Vec<i32> = Vec::one_dim(2).zeros();
    /// assert_eq!(arr1.dim(), 1);
    ///
    /// let arr2: Vec<Vec<i32>> = Vec::two_dim(2, 2).zeros();
    /// assert_eq!(arr2.dim(), 2);
    ///
    /// let arr3: Vec<Vec<Vec<i32>>> = Vec::three_dim(2, 2, 2).zeros();
    /// assert_eq!(arr3.dim(), 3);
    ///
    /// let arr4: Vec<Vec<Vec<Vec<i32>>>> = Vec::four_dim(2, 2, 2, 3).zeros();
    /// assert_eq!(arr4.dim(), 4);
    /// ```
    fn dim(&self) -> usize;
}

impl<T> Dimension<T> for Vec<T>
where
    T: Copy,
{
    fn dim(&self) -> usize {
        1
    }
}

impl<T> Dimension<T> for Vec<Vec<T>>
where
    T: Copy,
{
    fn dim(&self) -> usize {
        2
    }
}

impl<T> Dimension<T> for Vec<Vec<Vec<T>>>
where
    T: Copy,
{
    fn dim(&self) -> usize {
        3
    }
}

impl<T> Dimension<T> for Vec<Vec<Vec<Vec<T>>>>
where
    T: Copy,
{
    fn dim(&self) -> usize {
        4
    }
}

/// A list of integers indicating the size of the vector in each dimension
pub trait Shape<T>
where
    T: Copy,
{
    /// Returns a list of integers indicating the length of the
    /// vector in each dimension.
    ///
    /// # Examples
    /// ```
    /// # use gulali::prelude::*;
    /// let arr1: Vec<i32> = Vec::one_dim(2).zeros();
    /// assert_eq!(arr1.shape(), [2]);
    ///
    /// let arr2: Vec<Vec<i32>> = Vec::two_dim(2, 2).zeros();
    /// assert_eq!(arr2.shape(), [2, 2]);
    ///
    /// let arr3: Vec<Vec<Vec<i32>>> = Vec::three_dim(2, 2, 2).zeros();
    /// assert_eq!(arr3.shape(), [2, 2, 2]);
    ///
    /// let arr4: Vec<Vec<Vec<Vec<i32>>>> = Vec::four_dim(2, 2, 2, 3).zeros();
    /// assert_eq!(arr4.shape(), [2, 2, 2, 3]);
    /// ```
    fn shape(&self) -> Vec<usize>;
}

impl<T> Shape<T> for Vec<T>
where
    T: Copy,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.len()]
    }
}

impl<T> Shape<T> for Vec<Vec<T>>
where
    T: Copy,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.len(), self[0].len()]
    }
}

impl<T> Shape<T> for Vec<Vec<Vec<T>>>
where
    T: Copy,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.len(), self[0].len(), self[0][0].len()]
    }
}

impl<T> Shape<T> for Vec<Vec<Vec<Vec<T>>>>
where
    T: Copy,
{
    fn shape(&self) -> Vec<usize> {
        vec![
            self.len(),
            self[0].len(),
            self[0][0].len(),
            self[0][0][0].len(),
        ]
    }
}

/// Total number of elements of the vector
pub trait Size<T>
where
    T: Copy,
{
    /// Returns the total number of elements of the vector.
    /// This is equal to the product of the elements of shape.
    ///
    /// # Examples
    /// ```
    /// # use gulali::prelude::*;
    /// let arr1: Vec<i32> = Vec::one_dim(2).zeros();
    /// assert_eq!(arr1.size(), 2);
    ///
    /// let arr2: Vec<Vec<i32>> = Vec::two_dim(2, 2).zeros();
    /// assert_eq!(arr2.size(), 4);
    ///
    /// let arr3: Vec<Vec<Vec<i32>>> = Vec::three_dim(2, 2, 2).zeros();
    /// assert_eq!(arr3.size(), 8);
    ///
    /// let arr4: Vec<Vec<Vec<Vec<i32>>>> = Vec::four_dim(2, 2, 2, 3).zeros();
    /// assert_eq!(arr4.size(), 24);
    /// ```
    fn size(&self) -> usize;
}

impl<T> Size<T> for Vec<T>
where
    T: Copy,
{
    fn size(&self) -> usize {
        self.shape().iter().product()
    }
}

impl<T> Size<T> for Vec<Vec<T>>
where
    T: Copy,
{
    fn size(&self) -> usize {
        let shape: Vec<usize> = self.shape();
        shape.iter().product()
    }
}

impl<T> Size<T> for Vec<Vec<Vec<T>>>
where
    T: Copy,
{
    fn size(&self) -> usize {
        self.shape().iter().product()
    }
}

impl<T> Size<T> for Vec<Vec<Vec<Vec<T>>>>
where
    T: Copy,
{
    fn size(&self) -> usize {
        self.shape().iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_dim() {
        let arr1: Vec<i32> = Vec::one_dim(2).zeros();
        assert_eq!(arr1.dim(), 1);

        let arr2: Vec<Vec<i32>> = Vec::two_dim(2, 2).zeros();
        assert_eq!(arr2.dim(), 2);

        let arr3: Vec<Vec<Vec<i32>>> = Vec::three_dim(2, 2, 2).zeros();
        assert_eq!(arr3.dim(), 3);

        let arr4: Vec<Vec<Vec<Vec<i32>>>> = Vec::four_dim(2, 2, 2, 3).zeros();
        assert_eq!(arr4.dim(), 4);
    }

    #[test]
    fn test_shape() {
        let arr1: Vec<i32> = Vec::one_dim(2).zeros();
        assert_eq!(arr1.shape(), [2]);

        let arr2: Vec<Vec<i32>> = Vec::two_dim(2, 2).zeros();
        assert_eq!(arr2.shape(), [2, 2]);

        let arr3: Vec<Vec<Vec<i32>>> = Vec::three_dim(2, 2, 2).zeros();
        assert_eq!(arr3.shape(), [2, 2, 2]);

        let arr4: Vec<Vec<Vec<Vec<i32>>>> = Vec::four_dim(2, 2, 2, 3).zeros();
        assert_eq!(arr4.shape(), [2, 2, 2, 3]);
    }

    #[test]
    fn test_size() {
        let arr1: Vec<i32> = Vec::one_dim(2).zeros();
        assert_eq!(arr1.size(), 2);

        let arr2: Vec<Vec<i32>> = Vec::two_dim(2, 2).zeros();
        assert_eq!(arr2.size(), 4);

        let arr3: Vec<Vec<Vec<i32>>> = Vec::three_dim(2, 2, 2).zeros();
        assert_eq!(arr3.size(), 8);

        let arr4: Vec<Vec<Vec<Vec<i32>>>> = Vec::four_dim(2, 2, 2, 3).zeros();
        assert_eq!(arr4.size(), 24);
    }
}
