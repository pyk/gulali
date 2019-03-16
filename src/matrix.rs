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

use num::{FromPrimitive, Num};
use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Distribution, Normal, Uniform};

#[macro_export]
macro_rules! matrix {
    // NOTE: the order of the rules is very important

    // Samples: matrix![0; [3, 3]]
    ($elem:expr; $shape:expr) => {{
        let nrows = $shape[0];
        let ncols = $shape[1];
        let elements = vec![vec![$elem; ncols]; nrows];
        // TODO: use From trait
        Matrix::from_vec(elements)
    }};

    // Samples: matrix![1, 3, 4]
    ($($x:expr),*) => {{
        let elements = vec![vec![$($x),*]];
        // TODO: use From trait
        Matrix::from_vec(elements)

    }};

    // Samples: matrix![1, 2, 3, 4,]
    ($($x:expr,)*) => {{
        let elements = vec![vec![$($x),*]];
        // TODO: use From trait
        Matrix::from_vec(elements)
    }};

    // Samples: matrix![2.0, 1.0, 4.0; 2.0, 4.0, 2.0;]
    ($($($x:expr),*;)*) => {{
        let elements = vec![$(vec![$($x),*]),*];
        // TODO: use From trait
        Matrix::from_vec(elements)
    }};

    // Samples: matrix![2.0, 1.0, 4.0; 2.0, 4.0, 2.0]
    ($($($x:expr),*);*) => {{
        let elements = vec![$(vec![$($x),*]),*];
        // TODO: use From trait
        Matrix::from_vec(elements)
    }};
}

/// Matrix elements structure
///
/// TODO: add overview about matrix here.
/// 1. how to create a matrix
/// 2. Matrix operation
/// 3. Indexing, etc.
#[derive(Debug)]
pub struct Matrix<T> {
    /// Matrix size
    nrows: usize,
    ncols: usize,
    elements: Vec<Vec<T>>,
}

impl<T> Matrix<T> {
    /// The shape of the matrix `[nrows, ncols]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate crabsformer;
    /// # use crabsformer::prelude::*;
    /// # fn main() {
    /// let m = matrix![
    ///     3.0, 1.0;
    ///     4.0, 1.0;
    ///     5.0, 9.0;
    /// ];
    /// assert_eq!(m.shape(), [3, 2]);
    /// # }
    /// ```
    pub fn shape(&self) -> [usize; 2] {
        [self.nrows, self.ncols]
    }
}

// NOTE:
// - matrix is immutable elements type
impl<T> Matrix<T>
where
    T: FromPrimitive + Num + Copy,
{
    pub fn full(shape: [usize; 2], value: T) -> Matrix<T> {
        // Initialize and populate the matrix with specified value
        let nrows = shape[0];
        let ncols = shape[1];
        let elements = vec![vec![value; ncols]; nrows];
        Matrix {
            nrows,
            ncols,
            elements,
        }
    }

    pub fn zeros(shape: [usize; 2]) -> Matrix<T> {
        Self::full(shape, T::from_i32(0).unwrap())
    }

    pub fn zeros_like(m: &Matrix<T>) -> Matrix<T> {
        Self::full([m.nrows, m.ncols], T::from_i32(0).unwrap())
    }

    pub fn ones(shape: [usize; 2]) -> Matrix<T> {
        Self::full(shape, T::from_i32(1).unwrap())
    }

    pub fn ones_like(m: &Matrix<T>) -> Matrix<T> {
        Self::full([m.nrows, m.ncols], T::from_i32(1).unwrap())
    }

    // TODO: implement trait From
    pub fn from_vec(elements: Vec<Vec<T>>) -> Matrix<T> {
        let nrows = elements.len();
        let ncols = elements[0].len();
        // Raise panic if number of columns on each row is inconsistent
        let ncols_inconsistent = elements.iter().any(|v| v.len() != ncols);
        if ncols_inconsistent {
            panic!("Invalid matrix: the number of columns is inconsistent")
        }
        Matrix {
            nrows,
            ncols,
            elements,
        }
    }
}

impl<U> Matrix<U>
where
    U: SampleUniform,
{
    pub fn uniform(shape: [usize; 2], low: U, high: U) -> Matrix<U> {
        // Get the shape of the matrix
        let nrows = shape[0];
        let ncols = shape[1];

        let mut elements = Vec::with_capacity(nrows);
        let uniform_distribution = Uniform::new(low, high);
        // Populate the Matrix with the default value
        let mut rng = rand::thread_rng();
        for _ in 0..nrows {
            let mut cols = Vec::with_capacity(ncols);
            for _ in 0..ncols {
                cols.push(uniform_distribution.sample(&mut rng));
            }
            elements.push(cols);
        }

        Matrix {
            nrows,
            ncols,
            elements,
        }
    }
}

impl Matrix<f64> {
    pub fn normal(shape: [usize; 2], mean: f64, std_dev: f64) -> Matrix<f64> {
        // Get the shape of the matrix
        let nrows = shape[0];
        let ncols = shape[1];

        let mut elements = Vec::with_capacity(nrows);
        let normal_distribution = Normal::new(mean, std_dev);
        // Populate the Matrix with the default value
        let mut rng = rand::thread_rng();
        for _ in 0..nrows {
            let mut cols = Vec::with_capacity(ncols);
            for _ in 0..ncols {
                cols.push(normal_distribution.sample(&mut rng));
            }
            elements.push(cols);
        }

        Matrix {
            nrows,
            ncols,
            elements,
        }
    }
}

// Matrix comparison
impl<T> PartialEq for Matrix<T>
where
    T: Num + Copy,
{
    fn eq(&self, other: &Matrix<T>) -> bool {
        if self.elements != other.elements {
            return false;
        }
        true
    }
    fn ne(&self, other: &Matrix<T>) -> bool {
        if self.elements == other.elements {
            return false;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macro() {
        let a = matrix![0; [12, 10]];
        assert_eq!(a.shape(), [12, 10]);
        assert_eq!(a, Matrix::full([12, 10], 0));

        let b = matrix![
            2.0, 1.0, 4.0;
            2.0, 4.0, 2.0
        ];
        assert_eq!(b.shape(), [2, 3]);

        let c = matrix![
            2.0, 1.0, 4.0;
            2.0, 4.0, 2.0;
        ];
        assert_eq!(c.shape(), [2, 3]);

        let d = matrix![2.0, 1.0, 4.0];
        assert_eq!(d.shape(), [1, 3]);

        let e = matrix![2.0, 1.0, 4.0;];
        assert_eq!(e.shape(), [1, 3]);

        let f = matrix![2.0, 1.0, 4.0,];
        assert_eq!(f.shape(), [1, 3]);
    }

    #[test]
    #[should_panic]
    fn test_invalid_macro() {
        let _x = matrix![2, 2; 2, 2, 2];
    }

    #[test]
    fn test_full() {
        let a = Matrix::full([2, 2], 5.0);
        assert_eq!(a.elements, [[5.0, 5.0], [5.0, 5.0]]);

        let b = Matrix::full([2, 2], 2);
        assert_eq!(b.elements, [[2, 2], [2, 2]]);
    }

    #[test]
    fn test_zeros() {
        let vf1: Matrix<f64> = Matrix::zeros([2, 2]);
        assert_eq!(vf1.elements, [[0.0, 0.0], [0.0, 0.0]]);

        let vf2: Matrix<f32> = Matrix::zeros([2, 2]);
        assert_eq!(vf2.elements, [[0.0, 0.0], [0.0, 0.0]]);

        let ms1: Matrix<usize> = Matrix::zeros([2, 2]);
        assert_eq!(ms1.elements, [[0, 0], [0, 0]]);

        let mu1: Matrix<u8> = Matrix::zeros([2, 2]);
        assert_eq!(mu1.elements, [[0, 0], [0, 0]]);

        let mu2: Matrix<u16> = Matrix::zeros([2, 2]);
        assert_eq!(mu2.elements, [[0, 0], [0, 0]]);

        let mu3: Matrix<u32> = Matrix::zeros([2, 2]);
        assert_eq!(mu3.elements, [[0, 0], [0, 0]]);

        let mu4: Matrix<u64> = Matrix::zeros([2, 2]);
        assert_eq!(mu4.elements, [[0, 0], [0, 0]]);

        let mu5: Matrix<u128> = Matrix::zeros([2, 2]);
        assert_eq!(mu5.elements, [[0, 0], [0, 0]]);

        let vi1: Matrix<i8> = Matrix::zeros([2, 2]);
        assert_eq!(vi1.elements, [[0, 0], [0, 0]]);

        let vi2: Matrix<i16> = Matrix::zeros([2, 2]);
        assert_eq!(vi2.elements, [[0, 0], [0, 0]]);

        let vi3: Matrix<i32> = Matrix::zeros([2, 2]);
        assert_eq!(vi3.elements, [[0, 0], [0, 0]]);

        let vi4: Matrix<i64> = Matrix::zeros([2, 2]);
        assert_eq!(vi4.elements, [[0, 0], [0, 0]]);

        let vi5: Matrix<i128> = Matrix::zeros([2, 2]);
        assert_eq!(vi5.elements, [[0, 0], [0, 0]]);
    }

    #[test]
    fn test_zeros_like() {
        let mi1: Matrix<i32> = Matrix::ones([5, 5]);
        let mi2 = Matrix::zeros_like(&mi1);
        assert_eq!(mi1.nrows, mi2.nrows);
        assert_eq!(mi1.ncols, mi2.ncols);
    }

    #[test]
    fn test_ones() {
        let mf1: Matrix<f64> = Matrix::ones([2, 2]);
        assert_eq!(mf1.elements, [[1.0, 1.0], [1.0, 1.0]]);

        let mf2: Matrix<f32> = Matrix::ones([2, 2]);
        assert_eq!(mf2.elements, [[1.0, 1.0], [1.0, 1.0]]);

        let ms1: Matrix<usize> = Matrix::ones([2, 2]);
        assert_eq!(ms1.elements, [[1, 1], [1, 1]]);

        let mu1: Matrix<u8> = Matrix::ones([2, 2]);
        assert_eq!(mu1.elements, [[1, 1], [1, 1]]);

        let mu2: Matrix<u16> = Matrix::ones([2, 2]);
        assert_eq!(mu2.elements, [[1, 1], [1, 1]]);

        let mu3: Matrix<u32> = Matrix::ones([2, 2]);
        assert_eq!(mu3.elements, [[1, 1], [1, 1]]);

        let mu4: Matrix<u64> = Matrix::ones([2, 2]);
        assert_eq!(mu4.elements, [[1, 1], [1, 1]]);

        let mu5: Matrix<u128> = Matrix::ones([2, 2]);
        assert_eq!(mu5.elements, [[1, 1], [1, 1]]);

        let mi1: Matrix<i8> = Matrix::ones([2, 2]);
        assert_eq!(mi1.elements, [[1, 1], [1, 1]]);

        let mi2: Matrix<i16> = Matrix::ones([2, 2]);
        assert_eq!(mi2.elements, [[1, 1], [1, 1]]);

        let mi3: Matrix<i32> = Matrix::ones([2, 2]);
        assert_eq!(mi3.elements, [[1, 1], [1, 1]]);

        let mi4: Matrix<i64> = Matrix::ones([2, 2]);
        assert_eq!(mi4.elements, [[1, 1], [1, 1]]);

        let mi5: Matrix<i128> = Matrix::ones([2, 2]);
        assert_eq!(mi5.elements, [[1, 1], [1, 1]]);
    }

    #[test]
    fn test_ones_like() {
        let mi1: Matrix<i32> = Matrix::ones([5, 4]);
        let mi2 = Matrix::ones_like(&mi1);
        assert_eq!(mi1.nrows, mi2.nrows);
        assert_eq!(mi1.ncols, mi2.ncols);
    }

    #[test]
    fn test_uniform() {
        let mf1: Matrix<f32> = Matrix::uniform([5, 5], 0.0, 1.0);
        for cols in mf1.elements.iter() {
            for value in cols.iter() {
                assert!((0.0 <= *value) && (*value < 1.0));
            }
        }

        let mf2: Matrix<f64> = Matrix::uniform([5, 5], 0.0, 1.0);
        for cols in mf2.elements.iter() {
            for value in cols.iter() {
                assert!((0.0 <= *value) && (*value < 1.0));
            }
        }

        let ms1: Matrix<usize> = Matrix::uniform([5, 5], 1, 10);
        for cols in ms1.elements.iter() {
            for value in cols.iter() {
                assert!((1 <= *value) && (*value < 10));
            }
        }

        let mu1: Matrix<u8> = Matrix::uniform([5, 5], 1, 10);
        for cols in mu1.elements.iter() {
            for value in cols.iter() {
                assert!((1 <= *value) && (*value < 10));
            }
        }

        let mu2: Matrix<u16> = Matrix::uniform([5, 5], 1, 10);
        for cols in mu2.elements.iter() {
            for value in cols.iter() {
                assert!((1 <= *value) && (*value < 10));
            }
        }

        let mu3: Matrix<u32> = Matrix::uniform([5, 5], 1, 10);
        for cols in mu3.elements.iter() {
            for value in cols.iter() {
                assert!((1 <= *value) && (*value < 10));
            }
        }

        let mu4: Matrix<u64> = Matrix::uniform([5, 5], 1, 10);
        for cols in mu4.elements.iter() {
            for value in cols.iter() {
                assert!((1 <= *value) && (*value < 10));
            }
        }

        let mu5: Matrix<u128> = Matrix::uniform([5, 5], 1, 10);
        for cols in mu5.elements.iter() {
            for value in cols.iter() {
                assert!((1 <= *value) && (*value < 10));
            }
        }

        let mi1: Matrix<i8> = Matrix::uniform([5, 5], -10, 10);
        for cols in mi1.elements.iter() {
            for value in cols.iter() {
                assert!((-10 <= *value) && (*value < 10));
            }
        }

        let mi2: Matrix<i16> = Matrix::uniform([5, 5], -10, 10);
        for cols in mi2.elements.iter() {
            for value in cols.iter() {
                assert!((-10 <= *value) && (*value < 10));
            }
        }

        let mi3: Matrix<i32> = Matrix::uniform([5, 5], -10, 10);
        for cols in mi3.elements.iter() {
            for value in cols.iter() {
                assert!((-10 <= *value) && (*value < 10));
            }
        }

        let mi4: Matrix<i64> = Matrix::uniform([5, 5], -10, 10);
        for cols in mi4.elements.iter() {
            for value in cols.iter() {
                assert!((-10 <= *value) && (*value < 10));
            }
        }

        let mi5: Matrix<i128> = Matrix::uniform([5, 5], -10, 10);
        for cols in mi5.elements.iter() {
            for value in cols.iter() {
                assert!((-10 <= *value) && (*value < 10));
            }
        }
    }

    #[test]
    fn test_normal() {
        let a = Matrix::normal([5, 5], 2.0, 4.0);
        let b = Matrix::normal([5, 5], 2.0, 4.0);
        assert_eq!(a.nrows, b.nrows);
        assert_eq!(a.ncols, b.ncols);
        assert_ne!(a.elements, b.elements);
    }
}
