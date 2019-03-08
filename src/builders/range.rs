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
use std::ops::AddAssign;

/// Range vector parameters
pub struct RangeVectorParams<T>
where
    T: Num + FromPrimitive + PartialOrd + AddAssign + Copy,
{
    /// Start of interval.
    /// The interval includes this value.
    /// The default start value is 0.
    pub start: Option<T>,
    /// End of interval.
    /// The interval does not include this value.
    pub stop: Option<T>,
    /// Spacing between values.
    /// The default step size is 1.
    pub step: Option<T>,
}

impl<T> RangeVectorParams<T>
where
    T: Num + FromPrimitive + PartialOrd + AddAssign + Copy,
{
    /// Set the start of the interval with the specified `value`.
    ///
    /// # Examples
    /// ````
    /// # use crabsformer::prelude::*;
    /// let range: Vec<f64> = Vec::range()
    ///     .start_at(1.0)
    ///     .stop_at(3.0)
    ///     .generate();
    /// assert_eq!(range[0], 1.0);
    /// ````
    pub fn start_at(self, value: T) -> RangeVectorParams<T> {
        RangeVectorParams {
            start: Some(value),
            stop: self.stop,
            step: self.step,
        }
    }

    /// Set the end of the interval with the specified `value`.
    ///
    /// # Examples
    /// ````
    /// # use crabsformer::prelude::*;
    /// let range: Vec<f64> = Vec::range()
    ///     .stop_at(5.0)
    ///     .generate();
    /// assert_eq!(range, [0.0, 1.0, 2.0, 3.0, 4.0]);
    /// ````
    pub fn stop_at(self, value: T) -> RangeVectorParams<T> {
        RangeVectorParams {
            start: self.start,
            stop: Some(value),
            step: self.step,
        }
    }

    /// Set the spacing between values.
    ///
    /// # Examples
    /// ````
    /// # use crabsformer::prelude::*;
    /// let range: Vec<f64> = Vec::range()
    ///     .stop_at(3.0)
    ///     .step_by(0.5)
    ///     .generate();
    /// assert_eq!(range, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
    /// ````
    pub fn step_by(self, value: T) -> RangeVectorParams<T> {
        RangeVectorParams {
            start: self.start,
            stop: self.stop,
            step: Some(value),
        }
    }

    /// Generate the range vector based on the configured
    /// parameters.
    ///
    /// # Panics
    /// Panics if the `stop` value is not specified or
    /// the `start >= stop`.
    ///
    ///
    /// # Examples
    /// ````
    /// # use crabsformer::prelude::*;
    /// let range: Vec<i32> = Vec::range()
    ///     .stop_at(3)
    ///     .generate();
    /// assert_eq!(range, [0, 1, 2]);
    /// ````
    pub fn generate(self) -> Vec<T> {
        // If stop value is not specified then panic
        if self.stop.is_none() {
            panic!("Stop value should be specified")
        }
        let start = self.start.unwrap_or(T::from_i32(0).unwrap());
        let stop = self.stop.unwrap();
        let step = self.step.unwrap_or(T::from_i32(1).unwrap());
        // If interval is invalid; then panic
        if start >= stop {
            panic!(
                "Range vector invalid interval, it should be start < stop"
            );
        }
        let mut output = Vec::new();
        let mut current_step = start;
        while current_step < stop {
            output.push(current_step);
            current_step += step;
        }
        output
    }
}

/// Range vectors builder
pub trait Range<T>
where
    T: Num + FromPrimitive + PartialOrd + AddAssign + Copy,
{
    /// A range vectors builder. It returns [`RangeVectorParams`] with
    /// default value:
    ///
    /// ```ignore
    /// RangeVectorParams{
    ///     start: None,
    ///     stop: None,
    ///     step: None
    /// }
    /// ```
    ///
    /// The parameter can be configured using
    /// the following methods: [`start_at()`], [`stop_at()`]
    /// and [`step_by()`]. After the range parameters are configured,
    /// use [`init()`] to initialize the range vector. Values are
    /// generated within the half-open interval `[start, stop)`
    /// (in other words, the interval including start but excluding stop).
    ///
    /// [`RangeVectorParams`]: struct.RangeVectorParams.html
    /// [`start_at()`]: struct.RangeVectorParams.html#method.start_at
    /// [`stop_at()`]: struct.RangeVectorParams.html#method.stop_at
    /// [`step_by()`]: struct.RangeVectorParams.html#method.step_by
    /// [`init()`]: struct.RangeVectorParams.html#method.init
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::prelude::*;
    /// // Create range vector within interval [0, 5)
    /// let range1: Vec<i64> = Vec::range().stop_at(5).generate();
    /// assert_eq!(range1, [0, 1, 2, 3, 4]);
    ///
    /// let range2: Vec<f64> = Vec::range()
    ///     .start_at(1.0)
    ///     .stop_at(3.0)
    ///     .step_by(0.5)
    ///     .generate();
    /// assert_eq!(range2, [1.0, 1.5, 2.0, 2.5]);
    /// ```
    ///
    fn range() -> RangeVectorParams<T>;
}

impl<T> Range<T> for Vec<T>
where
    T: Num + FromPrimitive + PartialOrd + AddAssign + Copy,
{
    fn range() -> RangeVectorParams<T> {
        RangeVectorParams {
            start: None,
            stop: None,
            step: None,
        }
    }
}
