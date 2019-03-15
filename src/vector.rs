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
use std::fmt;
use std::ops::AddAssign;

#[macro_export]
macro_rules! vector {
    ($elem:expr; $len:expr) => (Vector::full($len, $elem));
    ($($x:expr),*) => {{
        let elements = vec![$($x),*];
        Vector::from_vec(elements)
    }};
}

/// Vector elements structure
///
/// TODO: add overview about vector here.
/// 1. how to create a vector
/// 2. Vector operation
/// 3. Indexing, etc.
pub struct Vector<T> {
    elements: Vec<T>,
}

impl<T> Vector<T> {
    pub fn len(&self) -> usize {
        self.elements.len()
    }
}

// Vector comparison
impl<T> PartialEq for Vector<T>
where
    T: Num + Copy,
{
    fn eq(&self, other: &Vector<T>) -> bool {
        if self.elements != other.elements {
            return false;
        }
        true
    }
    fn ne(&self, other: &Vector<T>) -> bool {
        if self.elements == other.elements {
            return false;
        }
        true
    }
}

impl<T> Vector<T>
where
    T: FromPrimitive + Num + Copy,
{
    /// Create a new vector of given length `len` and type `T`,
    /// filled with `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::prelude::*;
    /// let v = Vector::full(5, 2.5);
    /// ```
    pub fn full(len: usize, value: T) -> Vector<T> {
        // Initialize and populate the vector with specified value
        let elements = vec![value; len];
        Vector { elements }
    }

    /// Create a new vector that have the same length and type
    /// as vector `v`, filled with `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate crabsformer;
    /// # use crabsformer::prelude::*;
    /// # fn main() {
    /// let v1 = vector![3.0, 1.0, 4.0, 1.0, 5.0];
    /// let v2 = Vector::full_like(&v1, 3.1415);
    /// # }
    /// ```
    pub fn full_like(v: &Vector<T>, value: T) -> Vector<T> {
        // Initialize and populate the vector with specified value
        let elements = vec![value; v.len()];
        Vector { elements }
    }

    /// Create a new vector of given length `len` and type `T`,
    /// filled with zeros. You need to explicitly annotate the
    /// numeric type.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::prelude::*;
    /// let v: Vector<i32> = Vector::zeros(5);
    /// ```
    pub fn zeros(len: usize) -> Vector<T> {
        Self::full(len, T::from_i32(0).unwrap())
    }

    /// Create a new vector that have the same length and type
    /// as vector `v`, filled with zeros.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate crabsformer;
    /// # use crabsformer::prelude::*;
    /// # fn main() {
    /// let v1 = vector![3, 1, 4, 1, 5];
    /// let v2 = Vector::zeros_like(&v1);
    /// # }
    /// ```
    pub fn zeros_like(v: &Vector<T>) -> Vector<T> {
        Self::full(v.elements.len(), T::from_i32(0).unwrap())
    }

    /// Create a new vector of given length `len` and type `T`,
    /// filled with ones. You need to explicitly annotate the
    /// numeric type.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::prelude::*;
    /// let v: Vector<i32> = Vector::ones(10);
    /// ```
    pub fn ones(len: usize) -> Vector<T> {
        Self::full(len, T::from_i32(1).unwrap())
    }

    /// Create a new vector that have the same length and type
    /// as vector `v`, filled with ones.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate crabsformer;
    /// # use crabsformer::prelude::*;
    /// # fn main() {
    /// let v1 = vector![3, 1, 4, 1, 5];
    /// let v2 = Vector::ones_like(&v1);
    /// # }
    /// ```
    pub fn ones_like(v: &Vector<T>) -> Vector<T> {
        Self::full(v.elements.len(), T::from_i32(1).unwrap())
    }

    // TODO: implement trait From
    pub fn from_vec(elements: Vec<T>) -> Vector<T> {
        Vector { elements }
    }
}

impl<U> Vector<U>
where
    U: SampleUniform,
{
    /// Create a new vector of the given length `len` and populate it with
    /// random samples from a uniform distribution over the half-open interval
    /// `[low, high)` (includes `low`, but excludes `high`).
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::prelude::*;
    /// let v = Vector::uniform(5, 0.0, 1.0);
    /// ```
    pub fn uniform(len: usize, low: U, high: U) -> Vector<U> {
        let mut elements = Vec::with_capacity(len);
        let uniform_distribution = Uniform::new(low, high);
        let mut rng = rand::thread_rng();
        for _ in 0..len {
            elements.push(uniform_distribution.sample(&mut rng));
        }

        Vector { elements }
    }
}

impl Vector<f64> {
    /// Create a new vector of the given length `len` and populate it with
    /// random samples from a normal distribution `N(mean, std_dev**2)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::prelude::*;
    /// let v = Vector::normal(5, 0.0, 1.0); // Gaussian mean=0.0 std_dev=1.0
    /// ```
    pub fn normal(len: usize, mean: f64, std_dev: f64) -> Vector<f64> {
        let mut elements = Vec::with_capacity(len);
        let normal_distribution = Normal::new(mean, std_dev);
        // Populate the vector with the default value
        let mut rng = rand::thread_rng();
        for _ in 0..len {
            elements.push(normal_distribution.sample(&mut rng));
        }

        Vector { elements }
    }
}

impl<T> Vector<T>
where
    T: Num + FromPrimitive + Copy + PartialOrd + AddAssign + fmt::Display,
{
    /// Create a new vector of evenly spaced values within a given half-open
    /// interval `[start, stop)` and spacing value `step`. Values are generated
    /// within the half-open interval `[start, stop)` (in other words, the
    /// interval including `start` but excluding `stop`).
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::prelude::*;
    /// let v = Vector::range(0.0, 3.0, 0.5); // vector![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    /// ```
    ///
    /// # Panics
    /// Panics if `start >= stop`.
    pub fn range(start: T, stop: T, step: T) -> Vector<T> {
        // If interval is invalid; then panic
        if start >= stop {
            panic!("Invalid range interval start={} stop={}", start, stop)
        }
        let mut elements = Vec::new();
        let mut current_step = start;
        while current_step < stop {
            elements.push(current_step);
            current_step += step;
        }
        Vector { elements }
    }
}

impl<F> Vector<F>
where
    F: Float + FromPrimitive + Copy + PartialOrd + AddAssign + fmt::Display,
{
    /// Create a new vector of the given length `len` and populate it with
    /// linearly spaced values within a given closed interval `[start, stop]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::prelude::*;
    /// let a = Vector::linspace(5, 1.0, 10.0); // vector![1.0, 3.25, 5.5, 7.75, 10.0]
    /// ```
    ///
    /// # Panics
    /// Panics if `start >= stop`.
    pub fn linspace(len: usize, start: F, stop: F) -> Vector<F> {
        // Panics if start >= stop, it should be start < stop
        if start >= stop {
            panic!("Invalid linspace interval start={} stop={}", start, stop)
        }
        // Convert len to float type
        let divisor = F::from_usize(len).unwrap();
        let mut elements = Vec::with_capacity(len);
        let mut current_step = start;
        let step = (stop - start) / (divisor - F::from_f32(1.0).unwrap());
        while current_step < stop {
            elements.push(current_step);
            current_step += step;
        }

        // Include the `stop` value in the generated sequences
        if elements.len() == len {
            elements[len - 1] = stop;
        } else {
            elements.push(stop);
        }

        Vector { elements }
    }
}

impl<T> fmt::Debug for Vector<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        return write!(f, "Vector({:?})", self.elements);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macro() {
        // Full of elements
        let a = vector![0; 5];
        let b = Vector::full(5, 0);
        assert_eq!(a, b);

        // Vector inialization
        let c = vector![1, 2, 3, 4];
        assert_eq!(c.elements, [1, 2, 3, 4]);
    }

    #[test]
    fn test_full() {
        let a = Vector::full(5, 5.0);
        assert_eq!(a.elements, [5.0, 5.0, 5.0, 5.0, 5.0]);

        let b = Vector::full(5, 2);
        assert_eq!(b.elements, [2, 2, 2, 2, 2]);
    }

    #[test]
    fn test_full_like() {
        let v1 = vector![3.0, 1.0, 4.0, 1.0, 5.0];
        let v2 = Vector::full_like(&v1, 5.0);
        assert_eq!(v2.elements, [5.0, 5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_zeros() {
        let vf1: Vector<f64> = Vector::zeros(5);
        assert_eq!(vf1.elements, [0.0, 0.0, 0.0, 0.0, 0.0]);

        let vf2: Vector<f32> = Vector::zeros(5);
        assert_eq!(vf2.elements, [0.0, 0.0, 0.0, 0.0, 0.0]);

        let vs1: Vector<usize> = Vector::zeros(5);
        assert_eq!(vs1.elements, [0, 0, 0, 0, 0]);

        let vu1: Vector<u8> = Vector::zeros(5);
        assert_eq!(vu1.elements, [0, 0, 0, 0, 0]);

        let vu2: Vector<u16> = Vector::zeros(5);
        assert_eq!(vu2.elements, [0, 0, 0, 0, 0]);

        let vu3: Vector<u32> = Vector::zeros(5);
        assert_eq!(vu3.elements, [0, 0, 0, 0, 0]);

        let vu4: Vector<u64> = Vector::zeros(5);
        assert_eq!(vu4.elements, [0, 0, 0, 0, 0]);

        let vu5: Vector<u128> = Vector::zeros(5);
        assert_eq!(vu5.elements, [0, 0, 0, 0, 0]);

        let vi1: Vector<i8> = Vector::zeros(5);
        assert_eq!(vi1.elements, [0, 0, 0, 0, 0]);

        let vi2: Vector<i16> = Vector::zeros(5);
        assert_eq!(vi2.elements, [0, 0, 0, 0, 0]);

        let vi3: Vector<i32> = Vector::zeros(5);
        assert_eq!(vi3.elements, [0, 0, 0, 0, 0]);

        let vi4: Vector<i64> = Vector::zeros(5);
        assert_eq!(vi4.elements, [0, 0, 0, 0, 0]);

        let vi5: Vector<i128> = Vector::zeros(5);
        assert_eq!(vi5.elements, [0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_zeros_like() {
        let vi1: Vector<i32> = Vector::ones(5);
        let vi2 = Vector::zeros_like(&vi1);
        assert_eq!(vi1.len(), vi2.len());
    }

    #[test]
    fn test_ones() {
        let vf1: Vector<f64> = Vector::ones(5);
        assert_eq!(vf1.elements, [1.0, 1.0, 1.0, 1.0, 1.0]);

        let vf2: Vector<f32> = Vector::ones(5);
        assert_eq!(vf2.elements, [1.0, 1.0, 1.0, 1.0, 1.0]);

        let vs1: Vector<usize> = Vector::ones(5);
        assert_eq!(vs1.elements, [1, 1, 1, 1, 1]);

        let vu1: Vector<u8> = Vector::ones(5);
        assert_eq!(vu1.elements, [1, 1, 1, 1, 1]);

        let vu2: Vector<u16> = Vector::ones(5);
        assert_eq!(vu2.elements, [1, 1, 1, 1, 1]);

        let vu3: Vector<u32> = Vector::ones(5);
        assert_eq!(vu3.elements, [1, 1, 1, 1, 1]);

        let vu4: Vector<u64> = Vector::ones(5);
        assert_eq!(vu4.elements, [1, 1, 1, 1, 1]);

        let vu5: Vector<u128> = Vector::ones(5);
        assert_eq!(vu5.elements, [1, 1, 1, 1, 1]);

        let vi1: Vector<i8> = Vector::ones(5);
        assert_eq!(vi1.elements, [1, 1, 1, 1, 1]);

        let vi2: Vector<i16> = Vector::ones(5);
        assert_eq!(vi2.elements, [1, 1, 1, 1, 1]);

        let vi3: Vector<i32> = Vector::ones(5);
        assert_eq!(vi3.elements, [1, 1, 1, 1, 1]);

        let vi4: Vector<i64> = Vector::ones(5);
        assert_eq!(vi4.elements, [1, 1, 1, 1, 1]);

        let vi5: Vector<i128> = Vector::ones(5);
        assert_eq!(vi5.elements, [1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_ones_like() {
        let vi1: Vector<i32> = Vector::ones(10);
        let vi2 = Vector::ones_like(&vi1);
        assert_eq!(vi1.len(), vi2.len());
    }

    #[test]
    fn test_uniform() {
        let vf1: Vector<f32> = Vector::uniform(5, 0.0, 1.0);
        for value in vf1.elements.iter() {
            assert!((0.0 <= *value) && (*value < 1.0));
        }

        let vf2: Vector<f64> = Vector::uniform(5, 0.0, 1.0);
        for value in vf2.elements.iter() {
            assert!((0.0 <= *value) && (*value < 1.0));
        }

        let vs1: Vector<usize> = Vector::uniform(5, 1, 10);
        for value in vs1.elements.iter() {
            assert!((1 <= *value) && (*value < 10));
        }

        let vu1: Vector<u8> = Vector::uniform(5, 1, 10);
        for value in vu1.elements.iter() {
            assert!((1 <= *value) && (*value < 10));
        }

        let vu2: Vector<u16> = Vector::uniform(5, 1, 10);
        for value in vu2.elements.iter() {
            assert!((1 <= *value) && (*value < 10));
        }

        let vu3: Vector<u32> = Vector::uniform(5, 1, 10);
        for value in vu3.elements.iter() {
            assert!((1 <= *value) && (*value < 10));
        }

        let vu4: Vector<u64> = Vector::uniform(5, 1, 10);
        for value in vu4.elements.iter() {
            assert!((1 <= *value) && (*value < 10));
        }

        let vu5: Vector<u128> = Vector::uniform(5, 1, 10);
        for value in vu5.elements.iter() {
            assert!((1 <= *value) && (*value < 10));
        }

        let vi1: Vector<i8> = Vector::uniform(5, -10, 10);
        for value in vi1.elements.iter() {
            assert!((-10 <= *value) && (*value < 10));
        }

        let vi2: Vector<i16> = Vector::uniform(5, -10, 10);
        for value in vi2.elements.iter() {
            assert!((-10 <= *value) && (*value < 10));
        }

        let vi3: Vector<i32> = Vector::uniform(5, -10, 10);
        for value in vi3.elements.iter() {
            assert!((-10 <= *value) && (*value < 10));
        }

        let vi4: Vector<i64> = Vector::uniform(5, -10, 10);
        for value in vi4.elements.iter() {
            assert!((-10 <= *value) && (*value < 10));
        }

        let vi5: Vector<i128> = Vector::uniform(5, -10, 10);
        for value in vi5.elements.iter() {
            assert!((-10 <= *value) && (*value < 10));
        }
    }

    #[test]
    fn test_normal() {
        let a = Vector::normal(5, 2.0, 4.0);
        let b = Vector::normal(5, 2.0, 4.0);
        assert_eq!(a.len(), b.len());
        assert_ne!(a.elements, b.elements);
    }

    #[test]
    fn test_range() {
        let a = Vector::range(0.0, 3.0, 0.5);
        assert_eq!(a.elements, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);

        let b = Vector::range(0, 3, 1);
        assert_eq!(b.elements, [0, 1, 2]);
    }

    #[test]
    fn test_linspace() {
        let a = Vector::linspace(5, 1.0, 10.0);
        assert_eq!(a.elements, [1.0, 3.25, 5.5, 7.75, 10.0]);
    }
}
