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
use std::ops::AddAssign;

/// Range vector parameters
pub struct RangeParameters<T>
where
    T: PartialOrd + AddAssign + Copy,
{
    /// Start of interval.
    /// The interval includes this value.
    /// The default start value is 0.
    pub start: T,
    /// End of interval.
    /// The interval does not include this value.
    /// The default stop value is 0.
    pub stop: T,
    /// Spacing between values.
    /// The default step size is 1.
    pub step: T,
}

impl<T> RangeParameters<T>
where
    T: PartialOrd + AddAssign + Copy,
{
    /// Set the start of the interval with the specified `value`.
    ///
    /// # Examples
    /// ````
    /// # use gulali::prelude::*;
    /// let range: Vec<f64> = Vec::range()
    ///     .start_at(1.0)
    ///     .stop_at(3.0)
    ///     .init();
    /// assert_eq!(range, [1.0, 2.0]);
    /// ````
    pub fn start_at(self, value: T) -> RangeParameters<T> {
        RangeParameters {
            start: value,
            stop: self.stop,
            step: self.step,
        }
    }

    /// Set the end of the interval with the specified `value`.
    ///
    /// # Examples
    /// ````
    /// # use gulali::prelude::*;
    /// let range: Vec<f64> = Vec::range()
    ///     .stop_at(5.0)
    ///     .init();
    /// assert_eq!(range, [0.0, 1.0, 2.0, 3.0, 4.0]);
    /// ````
    pub fn stop_at(self, value: T) -> RangeParameters<T> {
        RangeParameters {
            start: self.start,
            stop: value,
            step: self.step,
        }
    }

    /// Set the spacing between values.
    ///
    /// # Examples
    /// ````
    /// # use gulali::prelude::*;
    /// let range: Vec<f64> = Vec::range()
    ///     .stop_at(3.0)
    ///     .step_by(0.5)
    ///     .init();
    /// assert_eq!(range, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
    /// ````
    pub fn step_by(self, value: T) -> RangeParameters<T> {
        RangeParameters {
            start: self.start,
            stop: self.stop,
            step: value,
        }
    }

    /// Initialize the range vector based on the configured
    /// parameters.
    ///
    /// # Examples
    /// ````
    /// # use gulali::prelude::*;
    /// let range: Vec<i32> = Vec::range()
    ///     .stop_at(3)
    ///     .init();
    /// assert_eq!(range, [0, 1, 2]);
    /// ````
    pub fn init(self) -> Vec<T> {
        let mut output = Vec::new();
        let mut current_step = self.start;
        while current_step < self.stop {
            output.push(current_step);
            current_step += self.step;
        }
        output
    }
}

/// Range vectors builder
pub trait Range<T>
where
    T: PartialOrd + AddAssign + Copy,
{
    /// A range vectors builder. It returns [`RangeParameters`] with
    /// default value:
    ///
    /// ```ignore
    /// RangeParameters{
    ///     start: 0,
    ///     stop: 0,
    ///     step: 1
    /// }
    /// ```
    ///
    /// The parameter can be configured using
    /// the following methods: [`start_at()`], [`stop_at()`]
    /// and [`step_by()`]. After the range parameters are configured,
    /// use [`init()`] to initialie the range vector.
    ///
    /// Values are generated within the half-open interval `[start, stop)`
    /// (in other words, the interval including start but excluding stop).
    ///
    /// [`RangeParameters`]: struct.RangeParameters.html
    /// [`start_at()`]: struct.RangeParameters.html#method.start_at
    /// [`stop_at()`]: struct.RangeParameters.html#method.stop_at
    /// [`step_by()`]: struct.RangeParameters.html#method.step_by
    /// [`init()`]: struct.RangeParameters.html#method.init
    ///
    /// # Examples
    /// ```
    /// # use gulali::prelude::*;
    /// // Create range vector within interval [0, 5)
    /// let range1: Vec<i64> = Vec::range().stop_at(5).init();
    /// assert_eq!(range1, [0, 1, 2, 3, 4]);
    ///
    /// let range2: Vec<f64> = Vec::range()
    ///     .start_at(1.0)
    ///     .stop_at(3.0)
    ///     .step_by(0.5)
    ///     .init();
    /// assert_eq!(range2, [1.0, 1.5, 2.0, 2.5]);
    /// ```
    ///
    fn range() -> RangeParameters<T>;
}

impl Range<u8> for Vec<u8> {
    fn range() -> RangeParameters<u8> {
        RangeParameters {
            start: 0,
            stop: 0,
            step: 1,
        }
    }
}

impl Range<u16> for Vec<u16> {
    fn range() -> RangeParameters<u16> {
        RangeParameters {
            start: 0,
            stop: 0,
            step: 1,
        }
    }
}

impl Range<u32> for Vec<u32> {
    fn range() -> RangeParameters<u32> {
        RangeParameters {
            start: 0,
            stop: 0,
            step: 1,
        }
    }
}

impl Range<u64> for Vec<u64> {
    fn range() -> RangeParameters<u64> {
        RangeParameters {
            start: 0,
            stop: 0,
            step: 1,
        }
    }
}

impl Range<u128> for Vec<u128> {
    fn range() -> RangeParameters<u128> {
        RangeParameters {
            start: 0,
            stop: 0,
            step: 1,
        }
    }
}

impl Range<i8> for Vec<i8> {
    fn range() -> RangeParameters<i8> {
        RangeParameters {
            start: 0,
            stop: 0,
            step: 1,
        }
    }
}

impl Range<i16> for Vec<i16> {
    fn range() -> RangeParameters<i16> {
        RangeParameters {
            start: 0,
            stop: 0,
            step: 1,
        }
    }
}

impl Range<i32> for Vec<i32> {
    fn range() -> RangeParameters<i32> {
        RangeParameters {
            start: 0,
            stop: 0,
            step: 1,
        }
    }
}

impl Range<i64> for Vec<i64> {
    fn range() -> RangeParameters<i64> {
        RangeParameters {
            start: 0,
            stop: 0,
            step: 1,
        }
    }
}

impl Range<i128> for Vec<i128> {
    fn range() -> RangeParameters<i128> {
        RangeParameters {
            start: 0,
            stop: 0,
            step: 1,
        }
    }
}

impl Range<f32> for Vec<f32> {
    fn range() -> RangeParameters<f32> {
        RangeParameters {
            start: 0.0,
            stop: 0.0,
            step: 1.0,
        }
    }
}

impl Range<f64> for Vec<f64> {
    fn range() -> RangeParameters<f64> {
        RangeParameters {
            start: 0.0,
            stop: 0.0,
            step: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_u8() {
        // Default
        let range1: Vec<u8> = Vec::range().stop_at(3).init();
        assert_eq!(range1, [0, 1, 2]);

        // start at specified value
        let range2: Vec<u8> = Vec::range().start_at(1).stop_at(3).init();
        assert_eq!(range2, [1, 2]);

        // step by specified value
        let range3: Vec<u8> =
            Vec::range().start_at(1).stop_at(3).step_by(2).init();
        assert_eq!(range3, [1]);

        // No stop_at
        let range4: Vec<u8> = Vec::range().init();
        assert_eq!(range4, []);
    }

    #[test]
    fn test_range_u16() {
        // Default
        let range1: Vec<u16> = Vec::range().stop_at(3).init();
        assert_eq!(range1, [0, 1, 2]);

        // start at specified value
        let range2: Vec<u16> = Vec::range().start_at(1).stop_at(3).init();
        assert_eq!(range2, [1, 2]);

        // step by specified value
        let range3: Vec<u16> =
            Vec::range().start_at(1).stop_at(3).step_by(2).init();
        assert_eq!(range3, [1]);

        // No stop_at
        let range4: Vec<u16> = Vec::range().init();
        assert_eq!(range4, []);
    }

    #[test]
    fn test_range_u32() {
        // Default
        let range1: Vec<u32> = Vec::range().stop_at(3).init();
        assert_eq!(range1, [0, 1, 2]);

        // start at specified value
        let range2: Vec<u32> = Vec::range().start_at(1).stop_at(3).init();
        assert_eq!(range2, [1, 2]);

        // step by specified value
        let range3: Vec<u32> =
            Vec::range().start_at(1).stop_at(3).step_by(2).init();
        assert_eq!(range3, [1]);

        // No stop_at
        let range4: Vec<u32> = Vec::range().init();
        assert_eq!(range4, []);
    }

    #[test]
    fn test_range_u64() {
        // Default
        let range1: Vec<u64> = Vec::range().stop_at(3).init();
        assert_eq!(range1, [0, 1, 2]);

        // start at specified value
        let range2: Vec<u64> = Vec::range().start_at(1).stop_at(3).init();
        assert_eq!(range2, [1, 2]);

        // step by specified value
        let range3: Vec<u64> =
            Vec::range().start_at(1).stop_at(3).step_by(2).init();
        assert_eq!(range3, [1]);

        // No stop_at
        let range4: Vec<u64> = Vec::range().init();
        assert_eq!(range4, []);
    }

    #[test]
    fn test_range_u128() {
        // Default
        let range1: Vec<u128> = Vec::range().stop_at(3).init();
        assert_eq!(range1, [0, 1, 2]);

        // start at specified value
        let range2: Vec<u128> = Vec::range().start_at(1).stop_at(3).init();
        assert_eq!(range2, [1, 2]);

        // step by specified value
        let range3: Vec<u128> =
            Vec::range().start_at(1).stop_at(3).step_by(2).init();
        assert_eq!(range3, [1]);

        // No stop_at
        let range4: Vec<u128> = Vec::range().init();
        assert_eq!(range4, []);
    }

    #[test]
    fn test_range_i8() {
        // Default
        let range1: Vec<i8> = Vec::range().stop_at(3).init();
        assert_eq!(range1, [0, 1, 2]);

        // start at specified value
        let range2: Vec<i8> = Vec::range().start_at(1).stop_at(3).init();
        assert_eq!(range2, [1, 2]);

        // step by specified value
        let range3: Vec<i8> =
            Vec::range().start_at(1).stop_at(3).step_by(2).init();
        assert_eq!(range3, [1]);

        // No stop_at
        let range4: Vec<i8> = Vec::range().init();
        assert_eq!(range4, []);
    }

    #[test]
    fn test_range_i16() {
        // Default
        let range1: Vec<i16> = Vec::range().stop_at(3).init();
        assert_eq!(range1, [0, 1, 2]);

        // start at specified value
        let range2: Vec<i16> = Vec::range().start_at(1).stop_at(3).init();
        assert_eq!(range2, [1, 2]);

        // step by specified value
        let range3: Vec<i16> =
            Vec::range().start_at(1).stop_at(3).step_by(2).init();
        assert_eq!(range3, [1]);

        // No stop_at
        let range4: Vec<i16> = Vec::range().init();
        assert_eq!(range4, []);
    }

    #[test]
    fn test_range_i32() {
        // Default
        let range1: Vec<i32> = Vec::range().stop_at(3).init();
        assert_eq!(range1, [0, 1, 2]);

        // start at specified value
        let range2: Vec<i32> = Vec::range().start_at(1).stop_at(3).init();
        assert_eq!(range2, [1, 2]);

        // step by specified value
        let range3: Vec<i32> =
            Vec::range().start_at(1).stop_at(3).step_by(2).init();
        assert_eq!(range3, [1]);

        // No stop_at
        let range4: Vec<i32> = Vec::range().init();
        assert_eq!(range4, []);
    }

    #[test]
    fn test_range_i64() {
        // Default
        let range1: Vec<i64> = Vec::range().stop_at(3).init();
        assert_eq!(range1, [0, 1, 2]);

        // start at specified value
        let range2: Vec<i64> = Vec::range().start_at(1).stop_at(3).init();
        assert_eq!(range2, [1, 2]);

        // step by specified value
        let range3: Vec<i64> =
            Vec::range().start_at(1).stop_at(3).step_by(2).init();
        assert_eq!(range3, [1]);

        // No stop_at
        let range4: Vec<i64> = Vec::range().init();
        assert_eq!(range4, []);
    }

    #[test]
    fn test_range_i128() {
        // Default
        let range1: Vec<i128> = Vec::range().stop_at(3).init();
        assert_eq!(range1, [0, 1, 2]);

        // start at specified value
        let range2: Vec<i128> = Vec::range().start_at(1).stop_at(3).init();
        assert_eq!(range2, [1, 2]);

        // step by specified value
        let range3: Vec<i128> =
            Vec::range().start_at(1).stop_at(3).step_by(2).init();
        assert_eq!(range3, [1]);

        // No stop_at
        let range4: Vec<i128> = Vec::range().init();
        assert_eq!(range4, []);
    }

    #[test]
    fn test_range_f32() {
        // Default
        let range1: Vec<f32> = Vec::range().stop_at(3.0).init();
        assert_eq!(range1, [0.0, 1.0, 2.0]);

        // start at specified value
        let range2: Vec<f32> = Vec::range().start_at(1.0).stop_at(3.0).init();
        assert_eq!(range2, [1.0, 2.0]);

        // step by specified value
        let range3: Vec<f32> =
            Vec::range().start_at(1.0).stop_at(3.0).step_by(0.5).init();
        assert_eq!(range3, [1.0, 1.5, 2.0, 2.5]);

        // No stop_at
        let range4: Vec<f32> = Vec::range().init();
        assert_eq!(range4, []);
    }

    #[test]
    fn test_range_f64() {
        // Default
        let range1: Vec<f64> = Vec::range().stop_at(3.0).init();
        assert_eq!(range1, [0.0, 1.0, 2.0]);

        // start at specified value
        let range2: Vec<f64> = Vec::range().start_at(1.0).stop_at(3.0).init();
        assert_eq!(range2, [1.0, 2.0]);

        // step by specified value
        let range3: Vec<f64> =
            Vec::range().start_at(1.0).stop_at(3.0).step_by(0.5).init();
        assert_eq!(range3, [1.0, 1.5, 2.0, 2.5]);

        // No stop_at
        let range4: Vec<f64> = Vec::range().init();
        assert_eq!(range4, []);
    }
}
