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

/// Two-dimensional vector parameters
pub struct TwoDimensionalVectorParams<T>
where
    T: Num + FromPrimitive + Copy,
{
    /// The shape of the vector.
    ///
    /// The shape of the vector can be specified using
    /// [`with_shape()`].
    ///
    /// [`with_shape()`]: #method.with_shape
    pub shape: Option<[usize; 2]>,

    /// Default value for each element of the vector.
    ///
    /// The default value for each element of the vector
    /// can be specified using [`full_of()`], [`zeros()`]
    /// and [`ones()`].
    ///
    /// [`full_of()`]: #method.full_of
    /// [`zeros()`]: #method.zeros
    /// [`ones()`]: #method.ones
    pub default_value: Option<T>,
}

impl<T> TwoDimensionalVectorParams<T>
where
    T: Num + FromPrimitive + Copy,
{
    /// Set the shape of the vector.
    ///
    /// # Examples
    /// ```
    /// # use gulali::prelude::*;
    /// let v: Vec<Vec<i32>> = Vec::two_dim()
    ///     .with_shape([2, 2])
    ///     .zeros()
    ///     .generate();
    /// assert_eq!(v, [[0, 0], [0, 0]]);
    /// ```
    pub fn with_shape(
        &self,
        shape: [usize; 2],
    ) -> TwoDimensionalVectorParams<T> {
        TwoDimensionalVectorParams {
            shape: Some(shape),
            default_value: None,
        }
    }

    /// Set the value of all elements of the vector
    /// with `value`.
    ///
    /// # Examples
    /// ```
    /// # use gulali::prelude::*;
    /// let v: Vec<Vec<i32>> = Vec::two_dim()
    ///     .with_shape([2, 2])
    ///     .full_of(5)
    ///     .generate();
    /// assert_eq!(v, [[5, 5], [5, 5]]);
    /// ```
    pub fn full_of(&self, value: T) -> TwoDimensionalVectorParams<T> {
        TwoDimensionalVectorParams {
            shape: self.shape,
            default_value: Some(value),
        }
    }

    /// Set the value of all elements of the vector
    /// with zeros.
    ///
    /// # Examples
    /// ```
    /// # use gulali::prelude::*;
    /// let v: Vec<Vec<i32>> = Vec::two_dim()
    ///     .with_shape([2, 2])
    ///     .zeros()
    ///     .generate();
    /// assert_eq!(v, [[0, 0], [0, 0]]);
    /// ```
    pub fn zeros(&self) -> TwoDimensionalVectorParams<T> {
        self.full_of(T::from_i32(0).unwrap())
    }

    /// Set the value of all elements of the vector
    /// with ones.
    ///
    /// # Examples
    /// ```
    /// # use gulali::prelude::*;
    /// let v: Vec<Vec<i32>> = Vec::two_dim()
    ///     .with_shape([2, 2])
    ///     .ones()
    ///     .generate();
    /// assert_eq!(v, [[1, 1], [1, 1]]);
    /// ```
    pub fn ones(&self) -> TwoDimensionalVectorParams<T> {
        self.full_of(T::from_i32(1).unwrap())
    }

    /// Generate the two-dimensional vector based on the
    /// configured parameters.
    ///
    /// # Panics
    /// Panics if the `shape` is not specified.
    ///
    /// # Examples
    /// ```
    /// # use gulali::prelude::*;
    /// let v: Vec<Vec<i32>> = Vec::two_dim()
    ///     .with_shape([2, 2])
    ///     .ones()
    ///     .generate();
    /// assert_eq!(v, [[1, 1], [1, 1]]);
    /// ```
    pub fn generate(&self) -> Vec<Vec<T>> {
        // If the shape is not specified then panic
        if self.shape.is_none() {
            panic!("Vector's shape should be specified");
        }

        // Get the size of the vector
        let shape = self.shape.unwrap();
        let n1 = shape[0];
        let n2 = shape[1];
        let mut output = Vec::with_capacity(n1);
        for _ in 0..n1 {
            let mut row = Vec::with_capacity(n2);
            // If default value is provided then populate the vector
            // with the default value
            if self.default_value.is_some() {
                let default_value = self.default_value.unwrap();
                for _ in 0..n2 {
                    row.push(default_value);
                }
            }
            output.push(row);
        }
        output
    }
}

/// Two-dimensional vector builder
pub trait TwoDimensional<T>
where
    T: Num + FromPrimitive + Copy,
{
    /// Two-dimensional vector builder.
    /// It returns [`TwoDimensionalVectorParams`]
    /// with default value:
    ///
    /// ```ignore
    /// TwoDimensionalVectorParams{
    ///     shape: None,
    ///     default_value: None,
    /// }
    /// ```
    ///
    /// The parameter can be configured using
    /// the following methods: [`with_shape()`], [`full_of()`],
    /// [`ones()`] and [`zeros()`]. After the two-dimensional vector
    /// parameters are configured, use [`generate()`] to generate
    /// the two-dimensional vector.
    ///
    /// [`TwoDimensionalVectorParams`]: struct.TwoDimensionalVectorParams.html
    /// [`with_shape()`]: struct.TwoDimensionalVectorParams.html#method.with_shape
    /// [`full_of()`]: struct.TwoDimensionalVectorParams.html#method.full_of
    /// [`zeros()`]: struct.TwoDimensionalVectorParams.html#method.zeros
    /// [`ones()`]: struct.TwoDimensionalVectorParams.html#method.ones
    /// [`generate()`]: struct.TwoDimensionalVectorParams.html#method.generate
    ///
    /// # Examples
    /// ```
    /// # use gulali::prelude::*;
    /// // Generate a two-dimensional vector with shape [2, 2]
    /// // filled with zeros; f64 can be changed into any
    /// // numeric data types.
    /// let v: Vec<Vec<f64>> = Vec::two_dim()
    ///     .with_shape([2, 2])
    ///     .zeros()
    ///     .generate();
    ///
    /// assert_eq!(v, [[0.0, 0.0], [0.0, 0.0]]);
    /// ```
    fn two_dim() -> TwoDimensionalVectorParams<T>;
}

impl<T> TwoDimensional<T> for Vec<Vec<T>>
where
    T: Num + FromPrimitive + Copy,
{
    fn two_dim() -> TwoDimensionalVectorParams<T> {
        TwoDimensionalVectorParams {
            shape: None,
            default_value: None,
        }
    }
}
