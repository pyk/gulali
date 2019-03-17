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
/// # #[macro_use] extern crate crabsformer;
/// # use crabsformer::prelude::*;
/// # fn main() {
/// let x = vector![1, 2, 3];
/// assert_eq!(x[0], 1);
/// assert_eq!(x[1], 2);
/// assert_eq!(x[2], 3);
/// # }
/// ```
///
/// 2. Create a numeric vector from a given element and length:
///
/// ```
/// # #[macro_use] extern crate crabsformer;
/// # use crabsformer::prelude::*;
/// # fn main() {
/// let x = vector![1; 3];
/// assert_eq!(x, vector![1, 1, 1]);
/// # }
/// ```
///
/// [numeric vector]: ../struct.Vector.html
#[macro_export]
macro_rules! vector {
    ($elem:expr; $len:expr) => (Vector::full($len, $elem));
    ($($x:expr),*) => {{
        let elements = vec![$($x),*];
        Vector::from(elements)
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

// Conversion from other data type
impl<T> From<Vec<T>> for Vector<T>
where
    T: Num + Copy,
{
    fn from(elements: Vec<T>) -> Self {
        Vector { elements }
    }
}

impl<T> Vector<T> {
    /// The total number of elements of the numeric vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate crabsformer;
    /// # use crabsformer::prelude::*;
    /// # fn main() {
    /// let v = vector![3.0, 1.0, 4.0, 1.0, 5.0];
    /// assert_eq!(v.len(), 5);
    /// # }
    /// ```
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
    /// Create a new numeric vector of given length `len` and type `T`,
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

    /// Create a new numeric vector that have the same length and type
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
    pub fn zeros(len: usize) -> Vector<T> {
        Self::full(len, T::from_i32(0).unwrap())
    }

    /// Create a new numeric vector that have the same length and type
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
    pub fn ones(len: usize) -> Vector<T> {
        Self::full(len, T::from_i32(1).unwrap())
    }

    /// Create a new numeric vector that have the same length and type
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

    /// Raises each elements of vector to the power of `exp`,
    /// using exponentiation by squaring.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate crabsformer;
    /// # use crabsformer::prelude::*;
    /// # fn main() {
    /// let x = vector![3, 1, 4, 1];
    /// let y = x.power(2);
    /// assert_eq!(y, vector![9, 1, 16, 1]);
    /// # }
    /// ```
    pub fn power(&self, exp: usize) -> Vector<T> {
        let elements =
            self.elements.iter().map(|x| num::pow(*x, exp)).collect();
        Vector { elements }
    }

    /// Filter out the elements that doesn't match the criteria.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate crabsformer;
    /// # use crabsformer::prelude::*;
    /// # fn main() {
    /// let x = vector![3, 1, 4, 1];
    /// let y = x.filter(|x| x >= 2);
    /// assert_eq!(y, vector![3, 4]);
    /// # }
    /// ```
    pub fn filter(&self, criteria: impl Fn(T) -> bool) -> Vector<T> {
        let elements = self
            .elements
            .iter()
            .filter(|&&x| criteria(x))
            .map(|x| *x)
            .collect();
        Vector { elements }
    }
}

impl<U> Vector<U>
where
    U: SampleUniform,
{
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

        Vector { elements }
    }
}

impl<T> Vector<T>
where
    T: Num
        + FromPrimitive
        + Copy
        + PartialOrd
        + ops::AddAssign
        + fmt::Display,
{
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
    F: Float
        + FromPrimitive
        + Copy
        + PartialOrd
        + ops::AddAssign
        + fmt::Display,
{
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

impl<T> ops::Index<usize> for Vector<T> {
    type Output = T;

    fn index(&self, i: usize) -> &T {
        &self.elements[i]
    }
}

// This trait is implemented to support for numeric vector addition
// operator
impl<T> ops::Add<Vector<T>> for Vector<T>
where
    T: Num + Copy,
{
    type Output = Vector<T>;

    fn add(self, other: Vector<T>) -> Vector<T> {
        if self.len() != other.len() {
            panic!(
                "Vector addition with invalid length: {} != {}",
                self.len(),
                other.len()
            );
        }

        // Add the vectors
        let elements = self
            .elements
            .iter()
            .enumerate()
            .map(|(i, x)| *x + other[i])
            .collect();
        Vector { elements }
    }
}

// This trait is implemented to support for numeric vector addition
// operator with scalar on the right side,
// for example:
//
// let a = vector![5, 5, 5, 5] + 6;
//
impl<T> ops::Add<T> for Vector<T>
where
    T: Num + Copy,
{
    type Output = Vector<T>;

    fn add(self, value: T) -> Vector<T> {
        // Add the vectors
        let elements = self.elements.iter().map(|x| *x + value).collect();
        Vector { elements }
    }
}

// This macro is to generate support for numeric vector addition
// operator with scalar on the left side,
// for example:
//
// let a = 6 + vector![5, 5, 5, 5];
//
macro_rules! impl_add_vector_for_type {
    ($t: ty) => {
        impl ops::Add<Vector<$t>> for $t {
            type Output = Vector<$t>;

            fn add(self, v: Vector<$t>) -> Vector<$t> {
                // Add the vectors
                let elements = v.elements.iter().map(|x| *x + self).collect();
                Vector { elements }
            }
        }
    };
}

impl_add_vector_for_type!(usize);
impl_add_vector_for_type!(i8);
impl_add_vector_for_type!(i16);
impl_add_vector_for_type!(i32);
impl_add_vector_for_type!(i64);
impl_add_vector_for_type!(i128);
impl_add_vector_for_type!(u8);
impl_add_vector_for_type!(u16);
impl_add_vector_for_type!(u32);
impl_add_vector_for_type!(u64);
impl_add_vector_for_type!(u128);
impl_add_vector_for_type!(f32);
impl_add_vector_for_type!(f64);

// This trait is implemented to support for numeric vector addition
// and assignment operator (+=)
impl<T> ops::AddAssign<Vector<T>> for Vector<T>
where
    T: Num + Copy + ops::AddAssign,
{
    fn add_assign(&mut self, other: Vector<T>) {
        if self.len() != other.len() {
            panic!(
                "Vector addition with invalid length: {} != {}",
                self.len(),
                other.len()
            );
        }

        for (i, x) in self.elements.iter_mut().enumerate() {
            *x += other[i];
        }
    }
}

// This trait is implemented to support for numeric vector addition
// assignment operator (+=) with scalar on the right side,
// for example:
//
// let a = vector![5, 5, 5, 5];
// a += 6;
//
impl<T> ops::AddAssign<T> for Vector<T>
where
    T: Num + Copy + ops::AddAssign,
{
    fn add_assign(&mut self, value: T) {
        for x in self.elements.iter_mut() {
            *x += value
        }
    }
}

// This trait is implemented to support for numeric vector
// substraction operator
impl<T> ops::Sub<Vector<T>> for Vector<T>
where
    T: Num + Copy,
{
    type Output = Vector<T>;

    fn sub(self, other: Vector<T>) -> Vector<T> {
        if self.len() != other.len() {
            panic!(
                "Vector substraction with invalid length: {} != {}",
                self.len(),
                other.len()
            );
        }

        // Add the vectors
        let elements = self
            .elements
            .iter()
            .enumerate()
            .map(|(i, x)| *x - other[i])
            .collect();
        Vector { elements }
    }
}

// This trait is implemented to support for numeric vector addition
// operator with scalar on the right side,
// for example:
//
// let a = vector![5, 5, 5, 5] - 6;
impl<T> ops::Sub<T> for Vector<T>
where
    T: Num + Copy,
{
    type Output = Vector<T>;

    fn sub(self, value: T) -> Vector<T> {
        // Add the vectors
        let elements = self.elements.iter().map(|x| *x - value).collect();
        Vector { elements }
    }
}

// This macro is to generate support for numeric vector substraction
// operator with scalar on the left side,
// for example:
//
// let a = 6 - vector![5, 5, 5, 5];
//
macro_rules! impl_sub_vector_for_type {
    ($t: ty) => {
        impl ops::Sub<Vector<$t>> for $t {
            type Output = Vector<$t>;

            fn sub(self, v: Vector<$t>) -> Vector<$t> {
                // Add the vectors
                let elements = v.elements.iter().map(|x| *x - self).collect();
                Vector { elements }
            }
        }
    };
}

impl_sub_vector_for_type!(usize);
impl_sub_vector_for_type!(i8);
impl_sub_vector_for_type!(i16);
impl_sub_vector_for_type!(i32);
impl_sub_vector_for_type!(i64);
impl_sub_vector_for_type!(i128);
impl_sub_vector_for_type!(u8);
impl_sub_vector_for_type!(u16);
impl_sub_vector_for_type!(u32);
impl_sub_vector_for_type!(u64);
impl_sub_vector_for_type!(u128);
impl_sub_vector_for_type!(f32);
impl_sub_vector_for_type!(f64);

// This trait is implemented to support for numeric vector substraction
// assignment operator (-=)
impl<T> ops::SubAssign<Vector<T>> for Vector<T>
where
    T: Num + Copy + ops::SubAssign,
{
    fn sub_assign(&mut self, other: Vector<T>) {
        if self.len() != other.len() {
            panic!(
                "Vector addition with invalid length: {} != {}",
                self.len(),
                other.len()
            );
        }

        for (i, x) in self.elements.iter_mut().enumerate() {
            *x -= other[i];
        }
    }
}

// This trait is implemented to support for numeric vector substraction
// assignment operator (-=) with scalar on the right side,
// for example:
//
// let a = vector![5, 5, 5, 5];
// a -= 6;
//
impl<T> ops::SubAssign<T> for Vector<T>
where
    T: Num + Copy + ops::SubAssign,
{
    fn sub_assign(&mut self, value: T) {
        for x in self.elements.iter_mut() {
            *x -= value
        }
    }
}

impl<T> Clone for Vector<T>
where
    T: Copy,
{
    fn clone(&self) -> Vector<T> {
        Vector {
            elements: self.elements.clone(),
        }
    }
}

// This trait is implemented to support for numeric vector multiplication operator
impl<T> ops::Mul<Vector<T>> for Vector<T>
where
    T: Num + Copy,
{
    type Output = Vector<T>;

    fn mul(self, other: Vector<T>) -> Vector<T> {
        if self.len() != other.len() {
            panic!(
                "Vector multiplication with invalid length: {} != {}",
                self.len(),
                other.len()
            );
        }

        Vector {
            elements: self
                .elements
                .iter()
                .enumerate()
                .map(|(i, v)| *v * other[i])
                .collect(),
        }
    }
}

// This trait is implemented to support for numeric vector multiplication
// operator with scalar on the right side,
// for example:
//
// let a = vector![5, 5, 5, 5] * 6;
impl<T> ops::Mul<T> for Vector<T>
where
    T: Num + Copy,
{
    type Output = Vector<T>;

    fn mul(self, value: T) -> Vector<T> {
        Vector {
            elements: self.elements.iter().map(|x| *x * value).collect(),
        }
    }
}

// This macro is to generate support for numeric vector multiplication
// operator with scalar on the left side,
// for example:
//
// let a = 6 * vector![5, 5, 5, 5];
//
macro_rules! impl_mul_vector_for_type {
    ($t: ty) => {
        impl ops::Mul<Vector<$t>> for $t {
            type Output = Vector<$t>;

            fn mul(self, v: Vector<$t>) -> Vector<$t> {
                // Add the vectors
                let elements = v.elements.iter().map(|x| *x * self).collect();
                Vector { elements }
            }
        }
    };
}

impl_mul_vector_for_type!(usize);
impl_mul_vector_for_type!(i8);
impl_mul_vector_for_type!(i16);
impl_mul_vector_for_type!(i32);
impl_mul_vector_for_type!(i64);
impl_mul_vector_for_type!(i128);
impl_mul_vector_for_type!(u8);
impl_mul_vector_for_type!(u16);
impl_mul_vector_for_type!(u32);
impl_mul_vector_for_type!(u64);
impl_mul_vector_for_type!(u128);
impl_mul_vector_for_type!(f32);
impl_mul_vector_for_type!(f64);

// This trait is implemented to support for numeric vector mul
// assignment operator (*=)
impl<T> ops::MulAssign<Vector<T>> for Vector<T>
where
    T: Num + Copy + ops::MulAssign,
{
    fn mul_assign(&mut self, other: Vector<T>) {
        if self.len() != other.len() {
            panic!(
                "Vector addition with invalid length: {} != {}",
                self.len(),
                other.len()
            );
        }

        for (i, x) in self.elements.iter_mut().enumerate() {
            *x *= other[i];
        }
    }
}

// This trait is implemented to support for numeric vector mul
// assignment operator (-=) with scalar on the right side,
// for example:
//
// let a = vector![5, 5, 5, 5];
// a *= 6;
//
impl<T> ops::MulAssign<T> for Vector<T>
where
    T: Num + Copy + ops::MulAssign,
{
    fn mul_assign(&mut self, value: T) {
        for x in self.elements.iter_mut() {
            *x *= value
        }
    }
}

// TODO: implement exponent operator
// TODO: implement all operators https://www.tutorialspoint.com/numpy/numpy_arithmetic_operations.htm

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from() {
        let a = vec![1, 2, 3];
        let b = Vector::from(a);
        assert_eq!(b.elements, vec![1, 2, 3]);
    }

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

    #[test]
    fn test_indexing() {
        let a = vector![3, 1, 4, 1, 5];
        assert_eq!(a[0], 3);
        assert_eq!(a[1], 1);
        assert_eq!(a[2], 4);
        assert_eq!(a[3], 1);
        assert_eq!(a[4], 5);
    }

    #[test]
    fn test_add() {
        let a = vector![3, 1, 4, 1, 5] + vector![3, 1, 4, 1, 5];
        assert_eq!(a, vector![6, 2, 8, 2, 10]);

        let b = vector![3.0, 1.0, 4.0, 1.0, 5.5]
            + vector![3.7, 1.7, 4.4, 1.2, 5.5];
        assert_eq!(b, vector![6.7, 2.7, 8.4, 2.2, 11.0]);

        let c = vector![3, 1, 4, 1, 5] + 2;
        assert_eq!(c, vector![5, 3, 6, 3, 7]);

        let d = vector![3.7, 1.7, 4.4, 1.2, 5.5] + 2.0;
        assert_eq!(d, vector![5.7, 3.7, 6.4, 3.2, 7.5]);

        let e = 2 + vector![3, 1, 4, 1, 5];
        assert_eq!(e, vector![5, 3, 6, 3, 7]);

        let f = 2.0 + vector![3.7, 1.7, 4.4, 1.2, 5.5];
        assert_eq!(f, vector![5.7, 3.7, 6.4, 3.2, 7.5]);
    }

    #[test]
    fn test_add_assign() {
        let mut a = vector![3, 1, 4, 1, 5];
        a += vector![3, 1, 4, 1, 5];
        assert_eq!(a, vector![6, 2, 8, 2, 10]);

        let mut b = vector![3.0, 1.0, 4.0, 1.0, 5.5];
        b += vector![3.7, 1.7, 4.4, 1.2, 5.5];
        assert_eq!(b, vector![6.7, 2.7, 8.4, 2.2, 11.0]);

        let mut c = vector![3, 1, 4, 1, 5];
        c += 2;
        assert_eq!(c, vector![5, 3, 6, 3, 7]);

        let mut d = vector![3.7, 1.7, 4.4, 1.2, 5.5];
        d += 2.0;
        assert_eq!(d, vector![5.7, 3.7, 6.4, 3.2, 7.5]);
    }

    #[test]
    fn test_sub() {
        let a = vector![3, 1, 4, 1, 5] - vector![3, 1, 4, 1, 5];
        assert_eq!(a, vector![0, 0, 0, 0, 00]);

        let b = vector![3.0, 1.0, 4.0, 1.0, 5.5]
            - vector![3.7, 1.7, 4.4, 1.2, 5.5];
        assert_eq!(
            b,
            vector![
                -0.7000000000000002,
                -0.7,
                -0.40000000000000036,
                -0.19999999999999996,
                0.0
            ]
        );

        let c = vector![3, 1, 4, 1, 5] - 2;
        assert_eq!(c, vector![1, -1, 2, -1, 3]);

        let d = vector![3.7, 1.7, 4.4, 1.2, 5.5] - 2.0;
        assert_eq!(
            d,
            vector![
                1.7000000000000002,
                -0.30000000000000004,
                2.4000000000000004,
                -0.8,
                3.5
            ]
        );

        let e = 2 - vector![3, 1, 4, 1, 5];
        assert_eq!(e, vector![1, -1, 2, -1, 3]);

        let f = 2.0 - vector![3.7, 1.7, 4.4, 1.2, 5.5];
        assert_eq!(
            f,
            vector![
                1.7000000000000002,
                -0.30000000000000004,
                2.4000000000000004,
                -0.8,
                3.5
            ]
        );
    }

    #[test]
    fn test_sub_assign() {
        let mut a = vector![3, 1, 4, 1, 5];
        a -= vector![3, 1, 4, 1, 5];
        assert_eq!(a, vector![0, 0, 0, 0, 00]);

        let mut b = vector![3.0, 1.0, 4.0, 1.0, 5.5];
        b -= vector![3.7, 1.7, 4.4, 1.2, 5.5];
        assert_eq!(
            b,
            vector![
                -0.7000000000000002,
                -0.7,
                -0.40000000000000036,
                -0.19999999999999996,
                0.0
            ]
        );

        let mut c = vector![3, 1, 4, 1, 5];
        c -= 2;
        assert_eq!(c, vector![1, -1, 2, -1, 3]);

        let mut d = vector![3.7, 1.7, 4.4, 1.2, 5.5];
        d -= 2.0;
        assert_eq!(
            d,
            vector![
                1.7000000000000002,
                -0.30000000000000004,
                2.4000000000000004,
                -0.8,
                3.5
            ]
        );
    }

    #[test]
    fn test_clone() {
        // Test clone
        let a = vector![3, 1, 4];
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn test_mul() {
        let a = vector![3, 1, 4, 1, 5] * vector![3, 1, 4, 1, 5];
        assert_eq!(a, vector![9, 1, 16, 1, 25]);

        let b = vector![3.0, 1.0, 4.0, 1.0, 5.5]
            * vector![3.7, 1.7, 4.4, 1.2, 5.5];
        assert_eq!(b, vector![11.100000000000001, 1.7, 17.6, 1.2, 30.25]);

        let c = vector![3, 1, 4, 1, 5] * 2;
        assert_eq!(c, vector![6, 2, 8, 2, 10]);

        let d = vector![3.7, 1.7, 4.4, 1.2, 5.5] * 2.0;
        assert_eq!(d, vector![7.4, 3.4, 8.8, 2.4, 11.0]);

        let e = 2 * vector![3, 1, 4, 1, 5];
        assert_eq!(e, vector![6, 2, 8, 2, 10]);

        let f = 2.0 * vector![3.7, 1.7, 4.4, 1.2, 5.5];
        assert_eq!(f, vector![7.4, 3.4, 8.8, 2.4, 11.0]);
    }

    #[test]
    fn test_mul_assign() {
        let mut a = vector![3, 1, 4, 1, 5];
        a *= vector![3, 1, 4, 1, 5];
        assert_eq!(a, vector![9, 1, 16, 1, 25]);

        let mut b = vector![3.0, 1.0, 4.0, 1.0, 5.5];
        b *= vector![3.7, 1.7, 4.4, 1.2, 5.5];
        assert_eq!(b, vector![11.100000000000001, 1.7, 17.6, 1.2, 30.25]);

        let mut c = vector![3, 1, 4, 1, 5];
        c *= 2;
        assert_eq!(c, vector![6, 2, 8, 2, 10]);

        let mut d = vector![3.7, 1.7, 4.4, 1.2, 5.5];
        d *= 2.0;
        assert_eq!(d, vector![7.4, 3.4, 8.8, 2.4, 11.0]);
    }

    #[test]
    #[should_panic]
    fn test_invalid_add() {
        let _x = vector![1, 2] + vector![2];
    }

    #[test]
    #[should_panic]
    fn test_invalid_sub() {
        let _x = vector![1, 2] - vector![2];
    }

    #[test]
    #[should_panic]
    fn test_invalid_mul() {
        let _x = vector![1, 2] * vector![2];
    }

    #[test]
    fn test_power() {
        let x = vector![3, 1, 4, 1];
        let y = x.power(2);
        assert_eq!(y, vector![9, 1, 16, 1]);
    }

    #[test]
    fn test_filter() {
        let x = vector![3, 1, 4, 1];
        let y = x.filter(|x| x >= 2);
        assert_eq!(y, vector![3, 4]);
    }
}
