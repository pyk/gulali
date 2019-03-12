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

/// Matrix data structure
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
    data: Vec<Vec<T>>,
}

impl<T> Matrix<T> {
    pub fn shape(&self) -> [usize; 2] {
        [self.nrows, self.ncols]
    }
}

// NOTE:
// - matrix is immutable data type
impl<T> Matrix<T>
where
    T: FromPrimitive + Num + Copy,
{
    pub fn full(shape: [usize; 2], value: T) -> Matrix<T> {
        // Initialize and populate the matrix with specified value
        let nrows = shape[0];
        let ncols = shape[1];
        let data = vec![vec![value; ncols]; nrows];
        Matrix { nrows, ncols, data }
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
}

impl<U> Matrix<U>
where
    U: SampleUniform,
{
    pub fn uniform(shape: [usize; 2], low: U, high: U) -> Matrix<U> {
        // Get the shape of the matrix
        let nrows = shape[0];
        let ncols = shape[1];

        let mut data = Vec::with_capacity(nrows);
        let uniform_distribution = Uniform::new(low, high);
        // Populate the Matrix with the default value
        let mut rng = rand::thread_rng();
        for _ in 0..nrows {
            let mut cols = Vec::with_capacity(ncols);
            for _ in 0..ncols {
                cols.push(uniform_distribution.sample(&mut rng));
            }
            data.push(cols);
        }

        Matrix { nrows, ncols, data }
    }
}

impl Matrix<f64> {
    pub fn normal(shape: [usize; 2], mean: f64, std_dev: f64) -> Matrix<f64> {
        // Get the shape of the matrix
        let nrows = shape[0];
        let ncols = shape[1];

        let mut data = Vec::with_capacity(nrows);
        let normal_distribution = Normal::new(mean, std_dev);
        // Populate the Matrix with the default value
        let mut rng = rand::thread_rng();
        for _ in 0..nrows {
            let mut cols = Vec::with_capacity(ncols);
            for _ in 0..ncols {
                cols.push(normal_distribution.sample(&mut rng));
            }
            data.push(cols);
        }

        Matrix { nrows, ncols, data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full() {
        let a = Matrix::full([2, 2], 5.0);
        assert_eq!(a.data, [[5.0, 5.0], [5.0, 5.0]]);

        let b = Matrix::full([2, 2], 2);
        assert_eq!(b.data, [[2, 2], [2, 2]]);
    }

    #[test]
    fn test_zeros() {
        let vf1: Matrix<f64> = Matrix::zeros([2, 2]);
        assert_eq!(vf1.data, [[0.0, 0.0], [0.0, 0.0]]);

        let vf2: Matrix<f32> = Matrix::zeros([2, 2]);
        assert_eq!(vf2.data, [[0.0, 0.0], [0.0, 0.0]]);

        let ms1: Matrix<usize> = Matrix::zeros([2, 2]);
        assert_eq!(ms1.data, [[0, 0], [0, 0]]);

        let mu1: Matrix<u8> = Matrix::zeros([2, 2]);
        assert_eq!(mu1.data, [[0, 0], [0, 0]]);

        let mu2: Matrix<u16> = Matrix::zeros([2, 2]);
        assert_eq!(mu2.data, [[0, 0], [0, 0]]);

        let mu3: Matrix<u32> = Matrix::zeros([2, 2]);
        assert_eq!(mu3.data, [[0, 0], [0, 0]]);

        let mu4: Matrix<u64> = Matrix::zeros([2, 2]);
        assert_eq!(mu4.data, [[0, 0], [0, 0]]);

        let mu5: Matrix<u128> = Matrix::zeros([2, 2]);
        assert_eq!(mu5.data, [[0, 0], [0, 0]]);

        let vi1: Matrix<i8> = Matrix::zeros([2, 2]);
        assert_eq!(vi1.data, [[0, 0], [0, 0]]);

        let vi2: Matrix<i16> = Matrix::zeros([2, 2]);
        assert_eq!(vi2.data, [[0, 0], [0, 0]]);

        let vi3: Matrix<i32> = Matrix::zeros([2, 2]);
        assert_eq!(vi3.data, [[0, 0], [0, 0]]);

        let vi4: Matrix<i64> = Matrix::zeros([2, 2]);
        assert_eq!(vi4.data, [[0, 0], [0, 0]]);

        let vi5: Matrix<i128> = Matrix::zeros([2, 2]);
        assert_eq!(vi5.data, [[0, 0], [0, 0]]);
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
        assert_eq!(mf1.data, [[1.0, 1.0], [1.0, 1.0]]);

        let mf2: Matrix<f32> = Matrix::ones([2, 2]);
        assert_eq!(mf2.data, [[1.0, 1.0], [1.0, 1.0]]);

        let ms1: Matrix<usize> = Matrix::ones([2, 2]);
        assert_eq!(ms1.data, [[1, 1], [1, 1]]);

        let mu1: Matrix<u8> = Matrix::ones([2, 2]);
        assert_eq!(mu1.data, [[1, 1], [1, 1]]);

        let mu2: Matrix<u16> = Matrix::ones([2, 2]);
        assert_eq!(mu2.data, [[1, 1], [1, 1]]);

        let mu3: Matrix<u32> = Matrix::ones([2, 2]);
        assert_eq!(mu3.data, [[1, 1], [1, 1]]);

        let mu4: Matrix<u64> = Matrix::ones([2, 2]);
        assert_eq!(mu4.data, [[1, 1], [1, 1]]);

        let mu5: Matrix<u128> = Matrix::ones([2, 2]);
        assert_eq!(mu5.data, [[1, 1], [1, 1]]);

        let mi1: Matrix<i8> = Matrix::ones([2, 2]);
        assert_eq!(mi1.data, [[1, 1], [1, 1]]);

        let mi2: Matrix<i16> = Matrix::ones([2, 2]);
        assert_eq!(mi2.data, [[1, 1], [1, 1]]);

        let mi3: Matrix<i32> = Matrix::ones([2, 2]);
        assert_eq!(mi3.data, [[1, 1], [1, 1]]);

        let mi4: Matrix<i64> = Matrix::ones([2, 2]);
        assert_eq!(mi4.data, [[1, 1], [1, 1]]);

        let mi5: Matrix<i128> = Matrix::ones([2, 2]);
        assert_eq!(mi5.data, [[1, 1], [1, 1]]);
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
        for cols in mf1.data.iter() {
            for value in cols.iter() {
                assert!((0.0 <= *value) && (*value < 1.0));
            }
        }

        let mf2: Matrix<f64> = Matrix::uniform([5, 5], 0.0, 1.0);
        for cols in mf2.data.iter() {
            for value in cols.iter() {
                assert!((0.0 <= *value) && (*value < 1.0));
            }
        }

        let ms1: Matrix<usize> = Matrix::uniform([5, 5], 1, 10);
        for cols in ms1.data.iter() {
            for value in cols.iter() {
                assert!((1 <= *value) && (*value < 10));
            }
        }

        let mu1: Matrix<u8> = Matrix::uniform([5, 5], 1, 10);
        for cols in mu1.data.iter() {
            for value in cols.iter() {
                assert!((1 <= *value) && (*value < 10));
            }
        }

        let mu2: Matrix<u16> = Matrix::uniform([5, 5], 1, 10);
        for cols in mu2.data.iter() {
            for value in cols.iter() {
                assert!((1 <= *value) && (*value < 10));
            }
        }

        let mu3: Matrix<u32> = Matrix::uniform([5, 5], 1, 10);
        for cols in mu3.data.iter() {
            for value in cols.iter() {
                assert!((1 <= *value) && (*value < 10));
            }
        }

        let mu4: Matrix<u64> = Matrix::uniform([5, 5], 1, 10);
        for cols in mu4.data.iter() {
            for value in cols.iter() {
                assert!((1 <= *value) && (*value < 10));
            }
        }

        let mu5: Matrix<u128> = Matrix::uniform([5, 5], 1, 10);
        for cols in mu5.data.iter() {
            for value in cols.iter() {
                assert!((1 <= *value) && (*value < 10));
            }
        }

        let mi1: Matrix<i8> = Matrix::uniform([5, 5], -10, 10);
        for cols in mi1.data.iter() {
            for value in cols.iter() {
                assert!((-10 <= *value) && (*value < 10));
            }
        }

        let mi2: Matrix<i16> = Matrix::uniform([5, 5], -10, 10);
        for cols in mi2.data.iter() {
            for value in cols.iter() {
                assert!((-10 <= *value) && (*value < 10));
            }
        }

        let mi3: Matrix<i32> = Matrix::uniform([5, 5], -10, 10);
        for cols in mi3.data.iter() {
            for value in cols.iter() {
                assert!((-10 <= *value) && (*value < 10));
            }
        }

        let mi4: Matrix<i64> = Matrix::uniform([5, 5], -10, 10);
        for cols in mi4.data.iter() {
            for value in cols.iter() {
                assert!((-10 <= *value) && (*value < 10));
            }
        }

        let mi5: Matrix<i128> = Matrix::uniform([5, 5], -10, 10);
        for cols in mi5.data.iter() {
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
        assert_ne!(a.data, b.data);
    }
}
