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

//! Vector creation routines
//!
//! # Overview
//!
//! There are 4 general mechanisms for creating vectors:
//!
//! 1. Conversion from other Rust primitive types: [`array`] and [`slice`].
//! 2. Using Gulali's vector builder routines (e.g., [`ones()`],
//!    [`zeros()`], etc.)
//! 3. Reading vectors from disk, either from standard or
//!    custom formats *(Not available yet)*
//! 4. Creating vectors from raw bytes through the use of
//!    strings or buffers *(Not available yet)*
//!
//! This section will not cover means of replicating, joining, or otherwise
//! expanding or mutating existing vectors. Those are covered in their
//! own sections.
//!
//! [`array`]: https://doc.rust-lang.org/std/primitive.array.html
//! [`slice`]: https://doc.rust-lang.org/std/slice/index.html
//!
//! # Converting Array and Slice to Vector
//! In general, numerical data arranged in an array-like structure
//! in Rust can be converted to vector through the use of the [`to_vec()`]
//! function. The most obvious examples are [`array`] and [`slice`].
//! See the documentation for [`to_vec()`] for details for its use.
//!
//! Examples:
//!
//! ```
//! let arr = [4, 4, 1, 6];
//! assert_eq!(arr.to_vec(), vec![4, 4, 1, 6]);
//!
//! let slice = &[2, 3, 1, 0];
//! assert_eq!(slice.to_vec(), vec![2, 3, 1, 0])
//! ```
//!
//! [`to_vec()`]: https://doc.rust-lang.org/std/vec/struct.Vec.html#method.to_vec
//!
//! # Vector Builders
//! Gulali has built-in functions for creating vectors from scratch:
//!
//! [`zeros()`] will create a vector filled with 0 values with
//! the specified dimension and shape. For example:
//!
//! ```
//! # use gulali::prelude::*;
//! // Create two-dimensional vector with shape [3, 3]
//! // filled with zeros
//! let a: Vec<Vec<i32>> = Vec::two_dim(3, 3).zeros();
//! assert_eq!(a, [[0, 0, 0], [0, 0, 0], [0, 0, 0]]);
//! ```
//!
//! [`ones()`]  will create a vector filled with 1 values with
//! the specified dimension and shape. It is identical to
//! [`zeros()`] in all other respects.
//!
//! [`zeros()`]: zeros/trait.Zero.html#tymethod.zeros
//! [`ones()`]: ones/trait.One.html#tymethod.ones
//!
//! [`range()`] will create vectors with regularly incrementing values.
//! For example:
//! ```
//! # use gulali::prelude::*;
//! let range1: Vec<i32> = Vec::range().stop_at(5).init();
//! assert_eq!(range1, [0, 1, 2, 3, 4]);
//!
//! let range2: Vec<f64> = Vec::range()
//!     .start_at(1.0)
//!     .stop_at(3.0)
//!     .step_by(0.5)
//!     .init();
//! assert_eq!(range2, [1.0, 1.5, 2.0, 2.5]);
//! ```
//!
//! [`range()`]: range/trait.Range.html#tymethod.range
//!

/// Dimension and the shape of the vectors
pub mod dimensional;

/// Fill vectors with specified value
pub mod full;

/// Fill vectors with 1 value
pub mod ones;

/// Range vector builder
pub mod range;

/// Fill vectors with 0 value
pub mod zeros;
