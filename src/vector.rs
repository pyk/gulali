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

use num::{Float, FromPrimitive, Num};
use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Distribution, Normal, Uniform};
use std::fmt::Display;
use std::ops::AddAssign;

/// Vector data structure
///
/// TODO: add overview about vector here.
/// 1. how to create a vector
/// 2. Vector operation
/// 3. Indexing, etc.
#[derive(Debug)]
pub struct Vector<T> {
    /// Vector size
    size: usize,
    data: Vec<T>,
}

impl<T> Vector<T> {
    pub fn size(&self) -> usize {
        self.size
    }
}

// NOTE:
// - vector is immutable data type
impl<T> Vector<T>
where
    T: FromPrimitive + Num + Copy,
{
    pub fn full(size: usize, value: T) -> Vector<T> {
        // Initialize and populate the vector with specified value
        let data = vec![value; size];
        Vector {
            size: size,
            data: data,
        }
    }

    pub fn zeros(size: usize) -> Vector<T> {
        Self::full(size, T::from_i32(0).unwrap())
    }

    pub fn zeros_like(v: &Vector<T>) -> Vector<T> {
        Self::full(v.size, T::from_i32(0).unwrap())
    }

    pub fn ones(size: usize) -> Vector<T> {
        Self::full(size, T::from_i32(1).unwrap())
    }

    pub fn ones_like(v: &Vector<T>) -> Vector<T> {
        Self::full(v.size, T::from_i32(1).unwrap())
    }
}

impl<U> Vector<U>
where
    U: SampleUniform,
{
    pub fn uniform(size: usize, low: U, high: U) -> Vector<U> {
        let mut data = Vec::with_capacity(size);
        let uniform_distribution = Uniform::new(low, high);
        // Populate the vector with the default value
        let mut rng = rand::thread_rng();
        for _ in 0..size {
            data.push(uniform_distribution.sample(&mut rng));
        }

        Vector { size, data }
    }
}

impl Vector<f64> {
    pub fn normal(size: usize, mean: f64, std_dev: f64) -> Vector<f64> {
        let mut data = Vec::with_capacity(size);
        let normal_distribution = Normal::new(mean, std_dev);
        // Populate the vector with the default value
        let mut rng = rand::thread_rng();
        for _ in 0..size {
            data.push(normal_distribution.sample(&mut rng));
        }

        Vector { size, data }
    }
}

impl<F> Vector<F>
where
    F: Num + FromPrimitive + Copy + PartialOrd + AddAssign + Display,
{
    pub fn range(start: F, stop: F, step: F) -> Vector<F> {
        // If interval is invalid; then panic
        if start >= stop {
            panic!("Invalid range interval start={} stop={}", start, stop)
        }
        let mut data = Vec::new();
        let mut current_step = start;
        while current_step < stop {
            data.push(current_step);
            current_step += step;
        }
        Vector {
            size: data.len(),
            data,
        }
    }
}

impl<F> Vector<F>
where
    F: Float + FromPrimitive + Copy + PartialOrd + AddAssign + Display,
{
    pub fn linspace(size: usize, start: F, stop: F) -> Vector<F> {
        // Panics if start >= stop, it should be start < stop
        if start >= stop {
            panic!("Invalid linspace interval start={} stop={}", start, stop)
        }
        // Convert size to float type
        let divisor = F::from_usize(size).unwrap();
        let mut data = Vec::with_capacity(size);
        let mut current_step = start;
        let step = (stop - start) / (divisor - F::from_f32(1.0).unwrap());
        while current_step < stop {
            data.push(current_step);
            current_step += step;
        }

        // Include the `stop` value in the generated sequences
        if data.len() == size {
            data[size - 1] = stop;
        } else {
            data.push(stop);
        }

        Vector { size, data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full() {
        let a = Vector::full(5, 5.0);
        assert_eq!(a.data, [5.0, 5.0, 5.0, 5.0, 5.0]);

        let b = Vector::full(5, 2);
        assert_eq!(b.data, [2, 2, 2, 2, 2]);
    }

    #[test]
    fn test_zeros() {
        let vf1: Vector<f64> = Vector::zeros(5);
        assert_eq!(vf1.data, [0.0, 0.0, 0.0, 0.0, 0.0]);

        let vf2: Vector<f32> = Vector::zeros(5);
        assert_eq!(vf2.data, [0.0, 0.0, 0.0, 0.0, 0.0]);

        let vs1: Vector<usize> = Vector::zeros(5);
        assert_eq!(vs1.data, [0, 0, 0, 0, 0]);

        let vu1: Vector<u8> = Vector::zeros(5);
        assert_eq!(vu1.data, [0, 0, 0, 0, 0]);

        let vu2: Vector<u16> = Vector::zeros(5);
        assert_eq!(vu2.data, [0, 0, 0, 0, 0]);

        let vu3: Vector<u32> = Vector::zeros(5);
        assert_eq!(vu3.data, [0, 0, 0, 0, 0]);

        let vu4: Vector<u64> = Vector::zeros(5);
        assert_eq!(vu4.data, [0, 0, 0, 0, 0]);

        let vu5: Vector<u128> = Vector::zeros(5);
        assert_eq!(vu5.data, [0, 0, 0, 0, 0]);

        let vi1: Vector<i8> = Vector::zeros(5);
        assert_eq!(vi1.data, [0, 0, 0, 0, 0]);

        let vi2: Vector<i16> = Vector::zeros(5);
        assert_eq!(vi2.data, [0, 0, 0, 0, 0]);

        let vi3: Vector<i32> = Vector::zeros(5);
        assert_eq!(vi3.data, [0, 0, 0, 0, 0]);

        let vi4: Vector<i64> = Vector::zeros(5);
        assert_eq!(vi4.data, [0, 0, 0, 0, 0]);

        let vi5: Vector<i128> = Vector::zeros(5);
        assert_eq!(vi5.data, [0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_zeros_like() {
        let vi1: Vector<i32> = Vector::ones(5);
        let vi2 = Vector::zeros_like(&vi1);
        assert_eq!(vi1.size, vi2.size);
    }

    #[test]
    fn test_ones() {
        let vf1: Vector<f64> = Vector::ones(5);
        assert_eq!(vf1.data, [1.0, 1.0, 1.0, 1.0, 1.0]);

        let vf2: Vector<f32> = Vector::ones(5);
        assert_eq!(vf2.data, [1.0, 1.0, 1.0, 1.0, 1.0]);

        let vs1: Vector<usize> = Vector::ones(5);
        assert_eq!(vs1.data, [1, 1, 1, 1, 1]);

        let vu1: Vector<u8> = Vector::ones(5);
        assert_eq!(vu1.data, [1, 1, 1, 1, 1]);

        let vu2: Vector<u16> = Vector::ones(5);
        assert_eq!(vu2.data, [1, 1, 1, 1, 1]);

        let vu3: Vector<u32> = Vector::ones(5);
        assert_eq!(vu3.data, [1, 1, 1, 1, 1]);

        let vu4: Vector<u64> = Vector::ones(5);
        assert_eq!(vu4.data, [1, 1, 1, 1, 1]);

        let vu5: Vector<u128> = Vector::ones(5);
        assert_eq!(vu5.data, [1, 1, 1, 1, 1]);

        let vi1: Vector<i8> = Vector::ones(5);
        assert_eq!(vi1.data, [1, 1, 1, 1, 1]);

        let vi2: Vector<i16> = Vector::ones(5);
        assert_eq!(vi2.data, [1, 1, 1, 1, 1]);

        let vi3: Vector<i32> = Vector::ones(5);
        assert_eq!(vi3.data, [1, 1, 1, 1, 1]);

        let vi4: Vector<i64> = Vector::ones(5);
        assert_eq!(vi4.data, [1, 1, 1, 1, 1]);

        let vi5: Vector<i128> = Vector::ones(5);
        assert_eq!(vi5.data, [1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_ones_like() {
        let vi1: Vector<i32> = Vector::ones(10);
        let vi2 = Vector::ones_like(&vi1);
        assert_eq!(vi1.size, vi2.size);
    }

    #[test]
    fn test_uniform() {
        let vf1: Vector<f32> = Vector::uniform(5, 0.0, 1.0);
        for value in vf1.data.iter() {
            assert!((0.0 <= *value) && (*value < 1.0));
        }

        let vf2: Vector<f64> = Vector::uniform(5, 0.0, 1.0);
        for value in vf2.data.iter() {
            assert!((0.0 <= *value) && (*value < 1.0));
        }

        let vs1: Vector<usize> = Vector::uniform(5, 1, 10);
        for value in vs1.data.iter() {
            assert!((1 <= *value) && (*value < 10));
        }

        let vu1: Vector<u8> = Vector::uniform(5, 1, 10);
        for value in vu1.data.iter() {
            assert!((1 <= *value) && (*value < 10));
        }

        let vu2: Vector<u16> = Vector::uniform(5, 1, 10);
        for value in vu2.data.iter() {
            assert!((1 <= *value) && (*value < 10));
        }

        let vu3: Vector<u32> = Vector::uniform(5, 1, 10);
        for value in vu3.data.iter() {
            assert!((1 <= *value) && (*value < 10));
        }

        let vu4: Vector<u64> = Vector::uniform(5, 1, 10);
        for value in vu4.data.iter() {
            assert!((1 <= *value) && (*value < 10));
        }

        let vu5: Vector<u128> = Vector::uniform(5, 1, 10);
        for value in vu5.data.iter() {
            assert!((1 <= *value) && (*value < 10));
        }

        let vi1: Vector<i8> = Vector::uniform(5, -10, 10);
        for value in vi1.data.iter() {
            assert!((-10 <= *value) && (*value < 10));
        }

        let vi2: Vector<i16> = Vector::uniform(5, -10, 10);
        for value in vi2.data.iter() {
            assert!((-10 <= *value) && (*value < 10));
        }

        let vi3: Vector<i32> = Vector::uniform(5, -10, 10);
        for value in vi3.data.iter() {
            assert!((-10 <= *value) && (*value < 10));
        }

        let vi4: Vector<i64> = Vector::uniform(5, -10, 10);
        for value in vi4.data.iter() {
            assert!((-10 <= *value) && (*value < 10));
        }

        let vi5: Vector<i128> = Vector::uniform(5, -10, 10);
        for value in vi5.data.iter() {
            assert!((-10 <= *value) && (*value < 10));
        }
    }

    #[test]
    fn test_normal() {
        let a = Vector::normal(5, 2.0, 4.0);
        let b = Vector::normal(5, 2.0, 4.0);
        assert_eq!(a.size, b.size);
        assert_ne!(a.data, b.data);
    }

    #[test]
    fn test_range() {
        // Integer type
        let a = Vector::range(0, 10, 1);
        assert_eq!(a.data, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

        // Float type
        let a = Vector::range(0.0, 3.0, 0.5);
        assert_eq!(a.data, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
    }

    #[test]
    fn test_linspace() {
        // Integer type
        let a = Vector::linspace(5, 1.0, 10.0);
        assert_eq!(a.data, [1.0, 3.25, 5.5, 7.75, 10.0]);

        // Float type
        let a = Vector::linspace(3, 3.0, 4.0);
        assert_eq!(a.data, [3.0, 3.5, 4.0]);
    }
}
