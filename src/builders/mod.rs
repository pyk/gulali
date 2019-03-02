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
//
// This file may not be copied, modified, or distributed
// except according to those terms.

//! # Introduction
//!
//! There are 4 general mechanisms for creating vectors:
//!
//! 1. Conversion from other Rust data type (e.g., lists, tuples)
//! 2. Using Gulali vector builder routines (e.g., ones, zeros, etc.)
//! 3. Reading vectors from disk, either from standard or
//!    custom formats *(Not available yet)*
//! 4. Creating vectors from raw bytes through the use of
//!    strings or buffers *(Not available yet)*
//!
//! This section will not cover means of replicating, joining, or otherwise
//! expanding or mutating existing vectors. Those are covered in their
//! own sections.
//!
//! # Converting Array-like objects to Vector
//! In general, numerical data arranged in an array-like structure
//! in Rust can be converted to vector through the use of the [`to_vec()`]
//! function. The most obvious examples are [array] and [slices].
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
//! [array]: https://doc.rust-lang.org/rust-by-example/primitives/array.html
//! [slices]: https://doc.rust-lang.org/rust-by-example/primitives/array.html
//!
//! # Vector Builders
//! Gulali has built-in functions for creating vectors from scratch:
//!
//! [`zeros()`] will create a vector filled with 0 values with
//! the specified dimension and shape.
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
//! the specified dimension and shape.
//!
//! ```
//! # use gulali::prelude::*;
//! // Create three-dimensional vector with shape [1, 2, 1]
//! // filled with ones
//! let a: Vec<Vec<Vec<f64>>> = Vec::three_dim(1, 2, 1).ones();
//! assert_eq!(a, [[[1.0], [1.0]]]);
//! ```
//!
//! [`zeros()`]: trait.Zero.html
//! [`ones()`]: trait.One.html
//!

pub mod dimensional;
pub mod full;
pub mod ones;
pub mod zeros;
