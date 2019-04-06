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

//! Numeric vector operations.
//!
//! TODO(pyk): Add docs here
//! - https://scipy-lectures.org/intro/numpy/operations.html
//! - https://www.tutorialspoint.com/numpy/numpy_arithmetic_operations.htm
//!
//!
//!

use crate::vector::Vector;
use num::{FromPrimitive, Num};
use std::ops;

impl<T> Vector<T>
where
    T: Num + Copy,
{
    /// Raises each elements of vector to the power of `exp`,
    /// using exponentiation by squaring. A new numeric vector
    /// is created and filled with the result. If you want to
    /// modify existing numeric vector use [`power_mut`].
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::prelude::*;
    /// let x = vector![3, 1, 4, 1];
    /// let y = x.power(2);
    /// assert_eq!(y, vector![9, 1, 16, 1]);
    /// ```
    ///
    /// [`power_mut`]: #power_mut
    pub fn power(&self, exp: usize) -> Vector<T> {
        self.elements().map(|x| num::pow(*x, exp)).collect()
    }

    /// Raises each elements of vector to the power of `exp`,
    /// using exponentiation by squaring. An existing numeric vector
    /// is modified with the result. If you want to
    /// create a new numeric vector use [`power`].
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::prelude::*;
    /// let mut x = vector![3, 1, 4, 1];
    /// x.power_mut(2);
    /// assert_eq!(x, vector![9, 1, 16, 1]);
    /// ```
    ///
    /// [`power`]: #power
    pub fn power_mut(&mut self, exp: usize) {
        self.data.iter_mut().for_each(|x| *x = num::pow(*x, exp))
    }

    /// Filter out the elements that doesn't match the criteria.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::prelude::*;
    /// let x = vector![3, 1, 4, 1];
    /// let y = x.filter(|x| x >= 2);
    /// assert_eq!(y, vector![3, 4]);
    /// ```
    pub fn filter(&self, criteria: impl Fn(T) -> bool) -> Vector<T> {
        let data = self
            .elements()
            .filter(|&&x| criteria(x))
            .map(|x| *x)
            .collect();
        Vector { data }
    }

    /// Sum of numeric vector elements.
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::prelude::*;
    /// let x = Vector::uniform(5, -1.0, 1.0).unwrap();
    /// let sum = x.sum();
    /// println!("sum = {}", sum);
    /// ```
    pub fn sum(&self) -> T
    where
        T: FromPrimitive,
    {
        self.elements()
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
    /// # use crabsformer::prelude::*;
    /// let x = vector![1, 2, 3];
    /// assert_eq!(*x.max(), 3);
    /// ```
    pub fn max(&self) -> &T
    where
        T: num::Integer + Copy,
    {
        self.elements().max().unwrap()
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
    /// # use crabsformer::prelude::*;
    /// let x = vector![1, 2, 3];
    /// assert_eq!(*x.min(), 1);
    /// ```
    pub fn min(&self) -> &T
    where
        T: num::Integer + Copy,
    {
        self.elements().min().unwrap()
    }
}

// Binary operations
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
        let data = self
            .data
            .iter()
            .enumerate()
            .map(|(i, x)| *x + other[i])
            .collect();
        Vector { data }
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
        let data = self.elements().map(|x| *x + value).collect();
        Vector { data }
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
                let data = v.elements().map(|x| *x + self).collect();
                Vector { data }
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

        for (i, x) in self.data.iter_mut().enumerate() {
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
        for x in self.data.iter_mut() {
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
        let data = self
            .data
            .iter()
            .enumerate()
            .map(|(i, x)| *x - other[i])
            .collect();
        Vector { data }
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
        let data = self.elements().map(|x| *x - value).collect();
        Vector { data }
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
                let data = v.elements().map(|x| self - *x).collect();
                Vector { data }
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

        for (i, x) in self.data.iter_mut().enumerate() {
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
        for x in self.data.iter_mut() {
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
            data: self.data.clone(),
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
            data: self
                .data
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
            data: self.elements().map(|x| *x * value).collect(),
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
                let data = v.elements().map(|x| *x * self).collect();
                Vector { data }
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

        for (i, x) in self.data.iter_mut().enumerate() {
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
        for x in self.data.iter_mut() {
            *x *= value
        }
    }
}
