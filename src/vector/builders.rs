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

//! A macro and functions to create new numeric vector.
//!
//! # Overview
//! There are 4 general mechanisms for creating numeric vectors:
//!
//! 1. Conversion from other Rust primitive data types [array] & [slice].
//! 2. Conversion from other Rust data structure [`Vec<T>`].
//! 3. Crabsformer numeric vector macro [`vector!`].
//! 4. Intrinsic Crabformer numeric vector creation functions (e.g., `range`,
//! `ones`, `zeros`, etc.)
//!
//! This section will not cover means of replicating, joining, or otherwise
//! expanding or mutating existing numeric vectors. Those are covered in
//! their own sections.
//!
//! [array]: https://doc.rust-lang.org/std/primitive.array.html
//! [slice]: https://doc.rust-lang.org/std/primitive.slice.html
//! [`Vec<T>`]: https://doc.rust-lang.org/std/vec/struct.Vec.html
//! [`vector!`]: ../../macro.vector.html
//!
//! # Converting Array, Slice and Vector to Numeric Vector
//! In general, numerical data arranged in an array-like structure in Rust can
//! be converted to numeric vectors through the use of the `Vector::from`
//! function. The most obvious examples are [array] and [slice].
//!
//! Examples:
//!
//! ```
//! # use crabsformer::prelude::*;
//! let x1 = Vector::from([1, 2, 3, 4]);
//! let x2 = Vector::from(&[3, 1, 4, 5]);
//! let x3 = Vector::from(vec![1, 4, 5]);
//! ```
//!
//! # Numeric Vector Macro
//! [`vector!`] allows numeric vector to be defined with the same syntax as
//! array expressions. There are two forms of this macro:
//!
//! 1. Create a numeric vector containing a given list of elements:
//!
//! ```
//! # use crabsformer::prelude::*;
//! let x = vector![1, 2, 3];
//! assert_eq!(x[0], 1);
//! assert_eq!(x[1], 2);
//! assert_eq!(x[2], 3);
//!```
//!
//! 2. Create a numeric vector from a given element and length:
//!
//! ```
//! # use crabsformer::prelude::*;
//! let x = vector![1; 3];
//! assert_eq!(x, vector![1, 1, 1]);
//! ```
//!
//! # Intrinsic Numeric Vector Creation
//! Crabsformer has built-in functions for creating numeric vectors from
//! scratch:
//!
//! [`Vector::zeros`] will create a numeric vector filled with 0 values
//! with the specified length.
//!
//! ```
//! # use crabsformer::prelude::*;
//! let v: Vector<i32> = Vector::zeros(5);
//! ```
//!
//! [`Vector::ones`] will create an array filled with 1 values. It is
//! identical to zeros in all other respects.
//!
//! [`Vector::range`] will create numeric vectors with regularly incrementing
//! values.
//!
//! ```
//! # use crabsformer::prelude::*;
//! let v = Vector::range(0.0, 3.0, 0.5);
//! // v = vector![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
//! ```
//!
//! [`Vector::linspace`] will create numeric vectors with a specified number of
//! elements, and spaced equally between the specified beginning and end
//! values. For example:
//!
//! ```
//! # use crabsformer::prelude::*;
//! let a = Vector::linspace(5, 1.0, 10.0);
//! // vector![1.0, 3.25, 5.5, 7.75, 10.0]
//! ```
//! The advantage of this creation function is that one can guarantee the
//! number of elements and the starting and end point, which [`Vector::range`]
//! generally will not do for arbitrary `start`, `stop`, and `step` values.
//!
//! See all available functions below.
//!
//! # Functions
//!
//! - Ones and zeros
//!     - [`Vector::ones`]: Create a new numeric vector of given length and
//!         type, filled with ones.
//!     - [`Vector::ones_like`]: Create a new numeric vector that have the same
//!         length and type as given numeric vector, filled with ones.
//!     - [`Vector::zeros`]: Create a new numeric vector of given length and
//!         type, filled with zeros.
//!     - [`Vector::zeros_like`]: Create a new numeric vector that have the same
//!         length and type as given numeric vector, filled with zeros.
//!     - [`Vector::full`]: Create a new numeric vector of given length and
//!         type, filled with specified value.
//!     - [`Vector::full_like`]: Create a new numeric vector that have the same
//!         length and type as given numeric vector, filled with specified
//!         value.
//! - From existing data
//!     - `Vector::from`: Convert array, slice and [`Vec<T>`] to numeric
//!         vector.
//! - Numerical ranges
//!     - [`Vector::range`]: Create a new numeric vector of evenly spaced values
//!         within a given half-open interval and spacing value.
//!     - [`Vector::linspace`]: Create a new numeric vector of the given length
//!          and populate it with linearly spaced values within a given
//!         closed interval.
//! - Simple random data
//! - Permutations
//! - Distributions
//!
//! [`Vector::zeros`]: ../struct.Vector.html#method.zeros
//! [`Vector::zeros_like`]: ../struct.Vector.html#method.zeros_like
//! [`Vector::ones`]: ../struct.Vector.html#method.ones
//! [`Vector::ones_like`]: ../struct.Vector.html#method.ones_like
//! [`Vector::full`]: ../struct.Vector.html#method.full
//! [`Vector::full_like`]: ../struct.Vector.html#method.full_like
//! [`Vector::range`]: ../struct.Vector.html#method.range
//! [`Vector::linspace`]: ../struct.Vector.html#method.linspace
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
/// # use crabsformer::prelude::*;
/// let x = vector![1, 2, 3];
/// assert_eq!(x[0], 1);
/// assert_eq!(x[1], 2);
/// assert_eq!(x[2], 3);
/// ```
///
/// 2. Create a numeric vector from a given element and length:
///
/// ```
/// # use crabsformer::prelude::*;
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
        $crate::vector::Vector::from(elements)
    }};
    ($($x:expr),*) => {{
        let elements = vec![$($x),*];
        $crate::vector::Vector::from(elements)
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
    /// # use crabsformer::prelude::*;
    /// let v = Vector::full(5, 2.5);
    /// ```
    pub fn full(len: usize, value: T) -> Vector<T>
    where
        T: FromPrimitive,
    {
        vector![value; len]
    }

    /// Create a new numeric vector that have the same length and type
    /// as numeric vector `v`, filled with `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::prelude::*;
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
    /// # use crabsformer::prelude::*;
    /// let v: Vector<i32> = Vector::zeros(5);
    /// ```
    pub fn zeros(len: usize) -> Vector<T>
    where
        T: FromPrimitive + Num + Copy,
    {
        vector![T::from_i32(0).unwrap(); len]
    }

    /// Create a new numeric vector that have the same length and type
    /// as numeric vector `v`, filled with zeros.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::prelude::*;
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
    /// # use crabsformer::prelude::*;
    /// let v: Vector<i32> = Vector::ones(10);
    /// ```
    pub fn ones(len: usize) -> Vector<T>
    where
        T: FromPrimitive + Num + Copy,
    {
        vector![T::from_i32(1).unwrap(); len]
    }

    /// Create a new numeric vector that have the same length and type
    /// as numeric vector `v`, filled with ones.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::prelude::*;
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
    /// # use crabsformer::prelude::*;
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
    /// # use crabsformer::prelude::*;
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
    /// # use crabsformer::prelude::*;
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

        Vector { data: elements }
    }
}
