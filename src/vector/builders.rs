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

//! A list of function to build a new Numeric Vector.
//!
//! TODO(pyk): Add docs here
//!
//!

use crate::vector::Vector;
use num::{Float, FromPrimitive, Num};
use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Distribution, Normal, Uniform};
use std::fmt;
use std::ops;

/// Creates a [numeric vector] containing the arguments.
///
/// `vector!` allows numeric vector to be defined with
/// the same syntax as array expressions.
///
/// There are two forms of this macro:
///
/// 1. Create a numeric vector containing a given list of elements:
///
/// ```
/// # use crabsformer::*;
/// let x = vector![1, 2, 3];
/// assert_eq!(x[0], 1);
/// assert_eq!(x[1], 2);
/// assert_eq!(x[2], 3);
/// ```
///
/// 2. Create a numeric vector from a given element and length:
///
/// ```
/// # use crabsformer::*;
/// let x = vector![1; 3];
/// assert_eq!(x, vector![1, 1, 1]);
/// ```
///
/// [numeric vector]: struct.Vector.html
#[macro_export]
macro_rules! vector {
    ($value:expr; $len:expr) => {{
        // Initialize and populate the vector with specified value
        let elements = vec![$value; $len];
        $crate::Vector::from(elements)
    }};
    ($($x:expr),*) => {{
        let elements = vec![$($x),*];
        $crate::Vector::from(elements)
    }};
}

impl<T> Vector<T>
where
    T: Num + Copy,
{
    /// Create a new numeric vector of given length `len` and type `T`,
    /// filled with `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let v = Vector::full(5, 2.5);
    /// ```
    pub fn full(len: usize, value: T) -> Vector<T>
    where
        T: FromPrimitive,
    {
        vector![value; len]
    }

    /// Create a new numeric vector that have the same length and type
    /// as vector `v`, filled with `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let v1 = vector![3.0, 1.0, 4.0, 1.0, 5.0];
    /// let v2 = Vector::full_like(&v1, 3.1415);
    /// ```
    pub fn full_like(v: &Vector<T>, value: T) -> Vector<T>
    where
        T: FromPrimitive + Num + Copy,
    {
        vector![value; v.len()]
    }

    /// Create a new numeric vector of given length `len` and type `T`,
    /// filled with zeros. You need to explicitly annotate the
    /// numeric type.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let v: Vector<i32> = Vector::zeros(5);
    /// ```
    pub fn zeros(len: usize) -> Vector<T>
    where
        T: FromPrimitive + Num + Copy,
    {
        vector![T::from_i32(0).unwrap(); len]
    }

    /// Create a new numeric vector that have the same length and type
    /// as vector `v`, filled with zeros.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let v1 = vector![3, 1, 4, 1, 5];
    /// let v2 = Vector::zeros_like(&v1);
    /// ```
    pub fn zeros_like(v: &Vector<T>) -> Vector<T>
    where
        T: FromPrimitive + Num + Copy,
    {
        vector![T::from_i32(0).unwrap(); v.len()]
    }

    /// Create a new numeric vector of given length `len` and type `T`,
    /// filled with ones. You need to explicitly annotate the
    /// numeric type.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let v: Vector<i32> = Vector::ones(10);
    /// ```
    pub fn ones(len: usize) -> Vector<T>
    where
        T: FromPrimitive + Num + Copy,
    {
        vector![T::from_i32(1).unwrap(); len]
    }

    /// Create a new numeric vector that have the same length and type
    /// as vector `v`, filled with ones.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let v1 = vector![3, 1, 4, 1, 5];
    /// let v2 = Vector::ones_like(&v1);
    /// ```
    pub fn ones_like(v: &Vector<T>) -> Vector<T>
    where
        T: FromPrimitive + Num + Copy,
    {
        vector![T::from_i32(1).unwrap(); v.len()]
    }

    /// Create a new numeric vector of the given length `len` and
    /// populate it with random samples from a uniform distribution
    /// over the half-open interval `[low, high)` (includes `low`,
    /// but excludes `high`).
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let v = Vector::uniform(5, 0.0, 1.0);
    /// ```
    pub fn uniform(len: usize, low: T, high: T) -> Vector<T>
    where
        T: SampleUniform,
    {
        let mut elements = Vec::with_capacity(len);
        let uniform_distribution = Uniform::new(low, high);
        let mut rng = rand::thread_rng();
        for _ in 0..len {
            elements.push(uniform_distribution.sample(&mut rng));
        }

        Vector { data: elements }
    }

    /// Create a new numeric vector of evenly spaced values
    /// within a given half-open interval `[start, stop)` and
    /// spacing value `step`. Values are generated within the
    /// half-open interval `[start, stop)` (in other words, the
    /// interval including `start` but excluding `stop`).
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let v = Vector::range(0.0, 3.0, 0.5);
    /// // v = vector![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    /// ```
    ///
    /// # Panics
    /// Panics if `start >= stop`.
    pub fn range(start: T, stop: T, step: T) -> Vector<T>
    where
        T: Num
            + FromPrimitive
            + Copy
            + PartialOrd
            + ops::AddAssign
            + fmt::Display,
    {
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
        Vector { data: elements }
    }

    /// Create a new numeric vector of the given length `len`
    /// and populate it with linearly spaced values within a
    /// given closed interval `[start, stop]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let a = Vector::linspace(5, 1.0, 10.0); // vector![1.0, 3.25, 5.5, 7.75, 10.0]
    /// ```
    ///
    /// # Panics
    /// Panics if `start >= stop`.
    pub fn linspace(len: usize, start: T, stop: T) -> Vector<T>
    where
        T: Float
            + FromPrimitive
            + Copy
            + PartialOrd
            + ops::AddAssign
            + fmt::Display,
    {
        // Panics if start >= stop, it should be start < stop
        if start >= stop {
            panic!("Invalid linspace interval start={} stop={}", start, stop)
        }
        // Convert len to float type
        let divisor = T::from_usize(len).unwrap();
        let mut elements = Vec::with_capacity(len);
        let mut current_step = start;
        let step = (stop - start) / (divisor - T::from_f32(1.0).unwrap());
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

        Vector { data: elements }
    }
}

impl Vector<f64> {
    /// Create a new numeric vector of the given length `len` and
    /// populate it with random samples from a normal distribution
    /// `N(mean, std_dev**2)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
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

        Vector { data: elements }
    }
}
