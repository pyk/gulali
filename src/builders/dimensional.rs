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

/// One-dimensional vectors
pub trait OneDimensional<T>
where
    T: Copy,
{
    /// Returns new one-dimensional vector with specified `size`.
    // TODO(pyk): Add example
    fn one_dim(size: usize) -> Self;
}

impl<T> OneDimensional<T> for Vec<T>
where
    T: Copy,
{
    fn one_dim(size: usize) -> Vec<T> {
        Vec::with_capacity(size)
    }
}

/// Two-dimensional vectors
pub trait TwoDimensional<T>
where
    T: Copy,
{
    /// Returns new two-dimensional vector with specified
    /// number of rows and number of columns.
    ///
    // TODO(pyk): Add example
    fn two_dim(nrows: usize, ncols: usize) -> Self;
}

impl<T> TwoDimensional<T> for Vec<Vec<T>>
where
    T: Copy,
{
    fn two_dim(nrows: usize, ncols: usize) -> Vec<Vec<T>> {
        let mut array2d: Vec<Vec<T>> = Vec::with_capacity(nrows);
        for _ in 0..nrows {
            array2d.push(Vec::with_capacity(ncols));
        }
        array2d
    }
}

/// Three-dimensional vectors
pub trait ThreeDimensional<T>
where
    T: Copy,
{
    /// Returns new three-dimensional vector with specified shape.
    ///
    /// # Arguments
    ///
    /// * `n1` - A number of elements in axis 1.
    /// * `n2` - A number of elements in axis 2.
    /// * `n3` - A number of elements in axis 3.
    ///
    // TODO(pyk): Add example
    fn three_dim(n1: usize, n2: usize, n3: usize) -> Self;
}

impl<T> ThreeDimensional<T> for Vec<Vec<Vec<T>>>
where
    T: Copy,
{
    fn three_dim(n1: usize, n2: usize, n3: usize) -> Vec<Vec<Vec<T>>> {
        let mut array3d: Vec<Vec<Vec<T>>> = Vec::with_capacity(n1);
        for _ in 0..n1 {
            let mut array2d: Vec<Vec<T>> = Vec::with_capacity(n2);
            for _ in 0..n2 {
                array2d.push(Vec::with_capacity(n3));
            }
            array3d.push(array2d);
        }
        array3d
    }
}

/// Four-dimensional vectors
pub trait FourDimensional<T>
where
    T: Copy,
{
    /// Returns new four-dimensional vector with specified shape.
    ///
    /// # Arguments
    ///
    /// * `n1` - A number of elements in axis 1.
    /// * `n2` - A number of elements in axis 2.
    /// * `n3` - A number of elements in axis 3.
    /// * `n4` - A number of elements in axis 4.
    ///
    // TODO(pyk): Add example
    fn four_dim(n1: usize, n2: usize, n3: usize, n4: usize) -> Self;
}

impl<T> FourDimensional<T> for Vec<Vec<Vec<Vec<T>>>>
where
    T: Copy,
{
    fn four_dim(n1: usize, n2: usize, n3: usize, n4: usize) -> Vec<Vec<Vec<Vec<T>>>> {
        let mut array4d: Vec<Vec<Vec<Vec<T>>>> = Vec::with_capacity(n1);
        for _ in 0..n1 {
            let mut array3d: Vec<Vec<Vec<T>>> = Vec::with_capacity(n2);
            for _ in 0..n2 {
                let mut array2d: Vec<Vec<T>> = Vec::with_capacity(n3);
                for _ in 0..n3 {
                    array2d.push(Vec::with_capacity(n4));
                }
                array3d.push(array2d);
            }
            array4d.push(array3d)
        }
        array4d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_dim() {
        let arr: Vec<i64> = Vec::one_dim(5);
        assert_eq!(arr.capacity(), 5);
    }

    #[test]
    fn test_two_dim() {
        let arr: Vec<Vec<f64>> = Vec::two_dim(5, 5);
        assert_eq!(arr.capacity(), 5);
        for i in 0..5 {
            assert_eq!(arr[i].capacity(), 5);
        }
    }

    #[test]
    fn test_three_dim() {
        let arr: Vec<Vec<Vec<f64>>> = Vec::three_dim(5, 5, 5);
        assert_eq!(arr.capacity(), 5);
        for i in 0..5 {
            assert_eq!(arr[i].capacity(), 5);
            for j in 0..5 {
                assert_eq!(arr[i][j].capacity(), 5);
            }
        }
    }

    #[test]
    fn test_four_dim() {
        // TODO: Look at numpy array
        let arr: Vec<Vec<Vec<Vec<f64>>>> = Vec::four_dim(5, 5, 5, 5);
        assert_eq!(arr.capacity(), 5);
        for i in 0..5 {
            assert_eq!(arr[i].capacity(), 5);
            for j in 0..5 {
                assert_eq!(arr[i][j].capacity(), 5);
                for k in 0..5 {
                    assert_eq!(arr[i][j][k].capacity(), 5);
                }
            }
        }
    }
}
