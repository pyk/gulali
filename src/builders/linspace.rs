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
pub struct LinspaceParameters<T>
where
    T: Float + FromPrimitive + AddAssign,
{
    /// The starting value of the sequence.
    /// The default `start` value is 0.
    pub start: T,
    /// The end value of the sequence.
    /// The default `stop` value is `None`.
    pub stop: Option<T>,
    /// The size of the vectors or number
    /// of points to generate.
    /// The default `size` is 10.
    pub size: usize,
}

impl<T> LinspaceParameters<T>
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
    ///     .generate()
    ///     .unwrap();
    ///
    /// // The first value should be 1.0
    /// assert_eq!(lin[0], 1.0);
    /// ````
    pub fn start_at(self, value: T) -> LinspaceParameters<T> {
        LinspaceParameters {
            start: value,
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
    ///     .generate()
    ///     .unwrap();
    ///
    /// // The end value should be 3.0
    /// assert_eq!(*lin.last().unwrap(), 3.0);
    /// ````
    pub fn stop_at(self, value: T) -> LinspaceParameters<T> {
        LinspaceParameters {
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
    ///     .generate()
    ///     .unwrap();
    ///
    /// // The size of generated vectors should be 4
    /// assert_eq!(lin.size(), 4);
    /// ````
    pub fn with_size(self, value: usize) -> LinspaceParameters<T> {
        LinspaceParameters {
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
    /// # Examples
    /// ````
    /// # use gulali::prelude::*;
    /// let lin: Vec<f32> = Vec::linspace()
    ///     .start_at(1.0)
    ///     .stop_at(3.0)
    ///     .with_size(4)
    ///     .generate()
    ///     .unwrap();
    ///
    /// assert_eq!(lin, [1.0, 1.6666667, 2.3333335, 3.0]);
    /// ````
    pub fn generate(self) -> Option<Vec<T>> {
        // Returns None if the `stop` value is not specified
        if self.stop.is_none() {
            return None;
        }
        let mut output = Vec::with_capacity(self.size);
        let mut current_step = self.start;
        // Convert size to float type
        let size = T::from_usize(self.size).unwrap();
        let stop = self.stop.unwrap();
        // Returns None if start >= stop, it should be start < stop
        if self.start >= stop {
            return None;
        }
        let step = (stop - self.start) / (size - T::from_f32(1.0).unwrap());
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

        Some(output)
    }
}

/// Linearly spaced vector builder
pub trait Linspace<T>
where
    T: Float + FromPrimitive + AddAssign,
{
    /// A linearly spaced vectors builder. It returns [`LinspaceParameters`]
    /// with the following default value:
    ///
    /// ```ignore
    /// LinspaceParameters{
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
    /// [`LinspaceParameters`]: struct.LinspaceParameters.html
    /// [`start_at()`]: struct.LinspaceParameters.html#method.start_at
    /// [`stop_at()`]: struct.LinspaceParameters.html#method.stop_at
    /// [`with_size()`]: struct.LinspaceParameters.html#method.with_size
    /// [`generate()`]: struct.LinspaceParameters.html#method.generate
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
    ///     .unwrap();
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
    fn linspace() -> LinspaceParameters<T>;
}

impl<T> Linspace<T> for Vec<T>
where
    T: Float + FromPrimitive + AddAssign,
{
    fn linspace() -> LinspaceParameters<T> {
        LinspaceParameters {
            start: T::from_f32(0.0).unwrap(),
            stop: None,
            size: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn test_linspace_f32_none() {
        // If stop_at is not used, it should returns none
        Vec::<f32>::linspace().generate().unwrap();
    }

    #[test]
    #[should_panic]
    fn test_linspace_f64_none() {
        // If stop_at is not used, it should returns none
        Vec::<f64>::linspace().generate().unwrap();
    }

    #[test]
    fn test_linspace_default() {
        // Default
        let lin: Vec<f32> = Vec::linspace().stop_at(5.0).generate().unwrap();
        assert_eq!(
            lin,
            [
                0.0, 0.55555556, 1.11111111, 1.6666667, 2.22222222, 2.777778,
                3.3333335, 3.888889, 4.44444444, 5.0
            ]
        );
    }

    #[test]
    fn test_linspace_start_at() {
        // Default
        let lin: Vec<f32> = Vec::linspace()
            .start_at(1.0)
            .stop_at(5.0)
            .generate()
            .unwrap();
        assert_eq!(
            lin,
            [
                1.0, 1.44444444, 1.88888889, 2.33333333, 2.77777778, 3.222222,
                3.6666665, 4.11111111, 4.555556, 5.0
            ]
        );
    }

    #[test]
    fn test_linspace_with_size() {
        // Default
        let lin: Vec<f32> = Vec::linspace()
            .start_at(1.0)
            .stop_at(5.0)
            .with_size(5)
            .generate()
            .unwrap();
        assert_eq!(lin, [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

}
