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
use std::iter;
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
    ($elem:expr; $len:expr) => ($crate::Vector::full($len, $elem));
    ($($x:expr),*) => {{
        let elements = vec![$($x),*];
        $crate::Vector::from(elements)
    }};
}

/// Numeric vector.
///
/// TODO: add overview about vector here.
/// 1. how to create a vector
/// 2. Vector operation
/// 3. Indexing, etc.
pub struct Vector<T>
where
    T: Num + Copy,
{
    elements: Vec<T>,
}

impl<T> Vector<T>
where
    T: Num + Copy,
{
    /// The total number of elements of the numeric vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let v = vector![3.0, 1.0, 4.0, 1.0, 5.0];
    /// assert_eq!(v.len(), 5);
    /// ```
    pub fn len(&self) -> usize {
        self.elements.len()
    }

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
        T: FromPrimitive + Num + Copy,
    {
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
    /// # use crabsformer::*;
    /// let v1 = vector![3.0, 1.0, 4.0, 1.0, 5.0];
    /// let v2 = Vector::full_like(&v1, 3.1415);
    /// ```
    pub fn full_like(v: &Vector<T>, value: T) -> Vector<T>
    where
        T: FromPrimitive + Num + Copy,
    {
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
    /// # use crabsformer::*;
    /// let v: Vector<i32> = Vector::zeros(5);
    /// ```
    pub fn zeros(len: usize) -> Vector<T>
    where
        T: FromPrimitive + Num + Copy,
    {
        Self::full(len, T::from_i32(0).unwrap())
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
        Self::full(v.elements.len(), T::from_i32(0).unwrap())
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
        Self::full(len, T::from_i32(1).unwrap())
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
        Self::full(v.elements.len(), T::from_i32(1).unwrap())
    }

    /// Raises each elements of vector to the power of `exp`,
    /// using exponentiation by squaring. A new numeric vector
    /// is created and filled with the result. If you want to
    /// modify existing numeric vector use [`power_mut`].
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::*;
    /// let x = vector![3, 1, 4, 1];
    /// let y = x.power(2);
    /// assert_eq!(y, vector![9, 1, 16, 1]);
    /// ```
    ///
    /// [`power_mut`]: #power_mut
    pub fn power(&self, exp: usize) -> Vector<T>
    where
        T: FromPrimitive + Num + Copy,
    {
        let elements =
            self.elements.iter().map(|x| num::pow(*x, exp)).collect();
        Vector { elements }
    }

    /// Raises each elements of vector to the power of `exp`,
    /// using exponentiation by squaring. An existing numeric vector
    /// is modified with the result. If you want to
    /// create a new numeric vector use [`power`].
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::*;
    /// let mut x = vector![3, 1, 4, 1];
    /// x.power_mut(2);
    /// assert_eq!(x, vector![9, 1, 16, 1]);
    /// ```
    ///
    /// [`power`]: #power
    pub fn power_mut(&mut self, exp: usize)
    where
        T: FromPrimitive + Num + Copy,
    {
        self.elements
            .iter_mut()
            .for_each(|x| *x = num::pow(*x, exp))
    }

    /// Filter out the elements that doesn't match the criteria.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let x = vector![3, 1, 4, 1];
    /// let y = x.filter(|x| x >= 2);
    /// assert_eq!(y, vector![3, 4]);
    /// ```
    pub fn filter(&self, criteria: impl Fn(T) -> bool) -> Vector<T>
    where
        T: Copy,
    {
        let elements = self
            .elements
            .iter()
            .filter(|&&x| criteria(x))
            .map(|x| *x)
            .collect();
        Vector { elements }
    }

    /// Sum of numeric vector elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let x = Vector::uniform(5, -1.0, 1.0);
    /// let sum = x.sum();
    /// println!("sum = {}", sum);
    /// ```
    pub fn sum(&self) -> T
    where
        T: FromPrimitive + Num + Copy,
    {
        self.elements
            .iter()
            .fold(T::from_f32(0.0).unwrap(), |acc, x| acc + *x)
    }

    /// Returns the maximum element of a numeric vector.
    ///
    /// Note that, it's only work for numeric vector
    /// of integer due too the trait `std::cmp::Ord` is
    /// not implemented for `f32` and `f64` in Rust
    /// standard library. This may change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let x = vector![1, 2, 3];
    /// assert_eq!(*x.max(), 3);
    /// ```
    pub fn max(&self) -> &T
    where
        T: num::Integer + Copy,
    {
        self.elements.iter().max().unwrap()
    }

    /// Returns the minimum element of a numeric vector.
    ///
    /// Note that, it's only work for numeric vector
    /// of integer due too the trait `std::cmp::Ord` is
    /// not implemented for `f32` and `f64` in Rust
    /// standard library. This may change in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::*;
    /// let x = vector![1, 2, 3];
    /// assert_eq!(*x.min(), 1);
    /// ```
    pub fn min(&self) -> &T
    where
        T: num::Integer + Copy,
    {
        self.elements.iter().min().unwrap()
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

        Vector { elements }
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
        Vector { elements }
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

        Vector { elements }
    }

    /// Iterates over elements of the numeric vector.
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::*;
    /// let x = vector![1, 2, 3];
    /// let mut elements = x.elements();
    ///
    /// assert_eq!(elements.next(), Some(&1));
    /// assert_eq!(elements.next(), Some(&2));
    /// assert_eq!(elements.next(), Some(&3));
    /// assert_eq!(elements.next(), None);
    /// ```
    pub fn elements<'a>(&'a self) -> VectorElementIterator<'a, T> {
        VectorElementIterator {
            vector: self,
            pos: 0,
        }
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

        Vector { elements }
    }
}

// Conversion from &[T] to RowMatrix<T>
impl<T> From<&[T]> for Vector<T>
where
    T: Num + Copy,
{
    fn from(elements: &[T]) -> Self {
        Vector::from(elements.to_vec())
    }
}

// Conversion from Vec<T>
impl<T> From<Vec<T>> for Vector<T>
where
    T: Num + Copy,
{
    fn from(elements: Vec<T>) -> Self {
        Vector { elements }
    }
}

// Conversion from SubVector<'a, T> to Vector<T>
impl<'a, T> From<SubVector<'a, T>> for Vector<T>
where
    T: Num + Copy,
{
    fn from(subvec: SubVector<'a, T>) -> Self {
        Vector {
            elements: subvec.inner.to_vec(),
        }
    }
}

// Numeric vector comparison
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

// This macro is used to generate support for numeric vector
// and numeric slice comparison.
//
// assert_eq!(&[1, 2, 3], vector![1, 2, 3])
//
// TODO: add test for this
macro_rules! impl_partial_eq_slice_for_type {
    ($t: ty) => {
        // Numeric vector to numeric slice comparison
        impl PartialEq<Vector<$t>> for [$t] {
            fn eq(&self, other: &Vector<$t>) -> bool {
                if other.elements != self {
                    return false;
                }
                true
            }
            fn ne(&self, other: &Vector<$t>) -> bool {
                if other.elements == self {
                    return false;
                }
                true
            }
        }
    };
}

impl_partial_eq_slice_for_type!(usize);
impl_partial_eq_slice_for_type!(i8);
impl_partial_eq_slice_for_type!(i16);
impl_partial_eq_slice_for_type!(i32);
impl_partial_eq_slice_for_type!(i64);
impl_partial_eq_slice_for_type!(i128);
impl_partial_eq_slice_for_type!(u8);
impl_partial_eq_slice_for_type!(u16);
impl_partial_eq_slice_for_type!(u32);
impl_partial_eq_slice_for_type!(u64);
impl_partial_eq_slice_for_type!(u128);
impl_partial_eq_slice_for_type!(f32);
impl_partial_eq_slice_for_type!(f64);

impl<T> fmt::Debug for Vector<T>
where
    T: Num + Copy + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        return write!(f, "Vector({:?})", self.elements);
    }
}

// Implement vector indexing
impl<T> ops::Index<usize> for Vector<T>
where
    T: Num + Copy,
{
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
                let elements = v.elements.iter().map(|x| self - *x).collect();
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
    T: Num + Copy,
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

/// Numeric vector slice operation
/// Implements sub-numeric vector slicing with syntax
/// `x.slice(begin .. end)`.
///
/// Returns a reference to elements in numeric vector
/// from the range [`begin`..`end`). This operation is `O(1)`.
///
/// # Panics
/// Requires that `begin <= end` and `end <= len` where `len` is the
/// length of the numeric vector. Otherwise it will panic.
///
/// # Examples
/// ```
/// # use crabsformer::*;
/// let x = vector![3, 1, 2, 3];
/// // Range
/// assert_eq!(x.slice(0..1), vector![3]);
/// // RangeTo
/// assert_eq!(x.slice(..2), vector![3, 1]);
/// // RangeFrom
/// assert_eq!(x.slice(2..), vector![2, 3]);
/// // RangeFull
/// assert_eq!(x.slice(..), vector![3, 1, 2, 3]);
/// // RangeInclusive
/// assert_eq!(x.slice(0..=1), vector![3, 1]);
/// // RangeToInclusive
/// assert_eq!(x.slice(..=2), vector![3, 1, 2]);
/// ```
pub trait VectorSlice<'a, Idx>
where
    Idx: ?Sized,
{
    /// The returned type after indexing.
    type Output: ?Sized;

    /// Performs the slicing (`container.slice(index)`) operation.
    /// It returns sub numeric vector, a reference of elements
    /// in the numeric vector.
    fn slice(&'a self, index: Idx) -> Self::Output;
}

/// Sub numeric vector is a reference to elements in the numeric vector.
pub struct SubVector<'a, T>
where
    T: Num + Copy,
{
    inner: &'a [T],
}

impl<'a, T> SubVector<'a, T>
where
    T: Num + Copy,
{
    /// Create new sub numeric vector
    pub fn new(inner: &'a [T]) -> SubVector<'a, T> {
        SubVector { inner }
    }

    /// Convert sub numeric vector to numeric vector
    pub fn to_vector(&self) -> Vector<T> {
        Vector {
            elements: self.inner.to_vec(),
        }
    }
}

impl<'a, T> fmt::Debug for SubVector<'a, T>
where
    T: Num + Copy + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        return write!(f, "SubVector({:?})", self.inner);
    }
}

// Sub numeric vector comparison
impl<'a, T> PartialEq for SubVector<'a, T>
where
    T: Num + Copy,
{
    fn eq(&self, other: &SubVector<'a, T>) -> bool {
        if self.inner != other.inner {
            return false;
        }
        true
    }
    fn ne(&self, other: &SubVector<'a, T>) -> bool {
        if self.inner == other.inner {
            return false;
        }
        true
    }
}

// Sub numeric vector VS numeric vector comparison
// TODO(pyk): We can't compare sub-vector to numeric vector
impl<'a, T> PartialEq<Vector<T>> for SubVector<'a, T>
where
    T: Num + Copy,
{
    fn eq(&self, other: &Vector<T>) -> bool {
        if self.inner != &other.elements[..] {
            return false;
        }
        true
    }
    fn ne(&self, other: &Vector<T>) -> bool {
        if self.inner == &other.elements[..] {
            return false;
        }
        true
    }
}

// This macro is used to generate support for sub numeric vector
// and numeric slice comparison.
macro_rules! impl_partial_eq_sub_vector_for_slice_of {
    ($t: ty) => {
        // Sub numeric vector to numeric slice comparison
        impl<'a> PartialEq<[$t]> for SubVector<'a, $t> {
            fn eq(&self, other: &[$t]) -> bool {
                if other == self.inner {
                    true
                } else {
                    false
                }
            }
            fn ne(&self, other: &[$t]) -> bool {
                if other != self.inner {
                    true
                } else {
                    false
                }
            }
        }
    };
}

impl_partial_eq_sub_vector_for_slice_of!(usize);
impl_partial_eq_sub_vector_for_slice_of!(i8);
impl_partial_eq_sub_vector_for_slice_of!(i16);
impl_partial_eq_sub_vector_for_slice_of!(i32);
impl_partial_eq_sub_vector_for_slice_of!(i64);
impl_partial_eq_sub_vector_for_slice_of!(i128);
impl_partial_eq_sub_vector_for_slice_of!(u8);
impl_partial_eq_sub_vector_for_slice_of!(u16);
impl_partial_eq_sub_vector_for_slice_of!(u32);
impl_partial_eq_sub_vector_for_slice_of!(u64);
impl_partial_eq_sub_vector_for_slice_of!(u128);
impl_partial_eq_sub_vector_for_slice_of!(f32);
impl_partial_eq_sub_vector_for_slice_of!(f64);

macro_rules! impl_slice_ops_with_range {
    ($t:ty) => {
        impl<'a, T: 'a> VectorSlice<'a, $t> for Vector<T>
        where
            T: Num + Copy,
        {
            type Output = SubVector<'a, T>;

            fn slice(&'a self, index: $t) -> SubVector<'a, T> {
                SubVector::new(&self.elements[index])
            }
        }
    };
}

impl_slice_ops_with_range!(ops::Range<usize>);
impl_slice_ops_with_range!(ops::RangeFrom<usize>);
impl_slice_ops_with_range!(ops::RangeTo<usize>);
impl_slice_ops_with_range!(ops::RangeFull);
impl_slice_ops_with_range!(ops::RangeInclusive<usize>);
impl_slice_ops_with_range!(ops::RangeToInclusive<usize>);

// Implement row iterator for matrix
pub struct VectorElementIterator<'a, T: 'a>
where
    T: Num + Copy,
{
    vector: &'a Vector<T>,
    pos: usize,
}

impl<'a, T> Iterator for VectorElementIterator<'a, T>
where
    T: Num + Copy,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.vector.len() {
            return None;
        }
        // Increment the position of the row iterator.
        self.pos += 1;
        // Return the reference to the element
        Some(&self.vector[self.pos - 1])
    }
}

// Create numerice vector from an iterator
impl<T> iter::FromIterator<T> for Vector<T>
where
    T: Num + Copy,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut v = Vec::new();

        for i in iter {
            v.push(i);
        }

        Vector::from(v)
    }
}
// TODO: implement exponent operator
// TODO: implement all operators https://www.tutorialspoint.com/numpy/numpy_arithmetic_operations.htm
