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

// Doc reference:
// https://www.mathworks.com/help/matlab/ref/linspace.html
// https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html

use num::{Float, FromPrimitive};
use std::ops::AddAssign;

/// Linearly spaced vector parameters
pub struct LinspaceVectorParams<T>
where
    T: Float + FromPrimitive + AddAssign,
{
    /// The starting value of the sequence.
    /// The default `start` value is 0.
    pub start: Option<T>,
    /// The end value of the sequence.
    pub stop: Option<T>,
    /// The size of the vectors or number
    /// of points to generate.
    /// The default `size` is 10.
    pub size: usize,
}

impl<T> LinspaceVectorParams<T>
where
    T: Float + FromPrimitive + AddAssign,
{
    /// Set the starting value of the sequence.
    ///
    /// # Examples
    /// ````
    /// # use gulali::prelude::*;
    /// let lin: Vec<f64> = Vec::linspace()
    ///     .start_at(1.0)
    ///     .stop_at(3.0)
    ///     .with_size(4)
    ///     .generate();
    ///
    /// // The first value should be 1.0
    /// assert_eq!(lin[0], 1.0);
    /// ````
    pub fn start_at(self, value: T) -> LinspaceVectorParams<T> {
        LinspaceVectorParams {
            start: Some(value),
            stop: self.stop,
            size: self.size,
        }
    }

    /// Set the end value of the sequence.
    ///
    /// # Examples
    /// ````
    /// # use gulali::prelude::*;
    /// let lin: Vec<f64> = Vec::linspace()
    ///     .start_at(1.0)
    ///     .stop_at(3.0)
    ///     .with_size(4)
    ///     .generate();
    ///
    /// // The end value should be 3.0
    /// assert_eq!(*lin.last().unwrap(), 3.0);
    /// ````
    pub fn stop_at(self, value: T) -> LinspaceVectorParams<T> {
        LinspaceVectorParams {
            start: self.start,
            stop: Some(value),
            size: self.size,
        }
    }

    /// Set the size of the generated vectors
    ///
    /// # Examples
    /// ````
    /// # use gulali::prelude::*;
    /// let lin: Vec<f64> = Vec::linspace()
    ///     .start_at(1.0)
    ///     .stop_at(3.0)
    ///     .with_size(4)
    ///     .generate();
    ///
    /// // The size of generated vectors should be 4
    /// assert_eq!(lin.size(), 4);
    /// ````
    pub fn with_size(self, value: usize) -> LinspaceVectorParams<T> {
        LinspaceVectorParams {
            start: self.start,
            stop: self.stop,
            size: value,
        }
    }

    /// Generate the linearly spaced vector based on the
    /// configured parameters.
    ///
    /// It returns `None`, if `stop` value is not specified
    /// or the `start >= stop`.
    ///
    /// # Panics
    /// Panics if the `stop` value is not specified or
    /// # Examples
    /// ````
    /// # use gulali::prelude::*;
    /// let lin: Vec<f32> = Vec::linspace()
    ///     .start_at(1.0)
    ///     .stop_at(3.0)
    ///     .with_size(4)
    ///     .generate();
    ///
    /// assert_eq!(lin, [1.0, 1.6666667, 2.3333335, 3.0]);
    /// ````
    pub fn generate(self) -> Vec<T> {
        // Panics if the `stop` value is not specified
        if self.stop.is_none() {
            panic!("Linspace: stop value should be specified")
        }
        let start = self.start.unwrap_or(T::from_i32(0).unwrap());
        let stop = self.stop.unwrap();
        // Panics if start >= stop, it should be start < stop
        if start >= stop {
            panic!("Linspace: start >= stop, it should be start < stop")
        }
        // Convert size to float type
        let size = T::from_usize(self.size).unwrap();
        let mut output = Vec::with_capacity(self.size);
        let mut current_step = start;
        let step = (stop - start) / (size - T::from_f32(1.0).unwrap());
        while current_step < stop {
            output.push(current_step);
            current_step += step;
        }

        // Include the `stop` value in the sequences
        if output.len() == self.size {
            output[self.size - 1] = stop;
        } else {
            output.push(stop);
        }

        output
    }
}

/// Linearly spaced vector builder
pub trait Linspace<T>
where
    T: Float + FromPrimitive + AddAssign,
{
    /// A linearly spaced vectors builder. It returns [`LinspaceVectorParams`]
    /// with the following default value:
    ///
    /// ```ignore
    /// LinspaceVectorParams{
    ///     start: 0,
    ///     stop: None,
    ///     size: 100
    /// }
    /// ```
    ///
    /// The parameters can be configured using
    /// the following methods: [`start_at()`], [`stop_at()`]
    /// and [`with_size()`]. The only required method
    /// is [`stop_at()`].
    ///
    /// After the parameters are configured,
    /// use [`generate()`] to generate the linearly spaced vector.
    /// The values of the linearly spaced vector are generated
    /// within the interval `[start, stop]` (in other words, the interval
    /// including `start` and `stop`). The spacing between the values
    /// is `(stop-start)/(size-1)`.
    ///
    /// [`LinspaceVectorParams`]: struct.LinspaceVectorParams.html
    /// [`start_at()`]: struct.LinspaceVectorParams.html#method.start_at
    /// [`stop_at()`]: struct.LinspaceVectorParams.html#method.stop_at
    /// [`with_size()`]: struct.LinspaceVectorParams.html#method.with_size
    /// [`generate()`]: struct.LinspaceVectorParams.html#method.generate
    ///
    /// # Examples
    /// ```
    /// # use gulali::prelude::*;
    /// // Generate linearly spaced vector within interval [2.0, 5.0]
    /// let lin: Vec<f32> = Vec::linspace()
    ///     .start_at(2.0)
    ///     .stop_at(5.0)
    ///     .with_size(10)
    ///     .generate()
    ///     ;
    ///
    /// assert_eq!(
    ///     lin,
    ///     [
    ///         2.0, 2.33333333, 2.6666665, 2.9999998, 3.333333,
    ///         3.6666663, 3.9999995, 4.333333, 4.6666665, 5.0
    ///     ]
    /// );
    /// ```
    ///
    fn linspace() -> LinspaceVectorParams<T>;
}

impl<T> Linspace<T> for Vec<T>
where
    T: Float + FromPrimitive + AddAssign,
{
    fn linspace() -> LinspaceVectorParams<T> {
        LinspaceVectorParams {
            start: None,
            stop: None,
            size: 10,
        }
    }
}
