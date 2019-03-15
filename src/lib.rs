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

//! Crabsformer is an easy-to-use fundamental library for scientific computing with
//! Rust, highly inspired by [NumPy].
//!
//! [NumPy]: http://www.numpy.org/
//!
//! # Usage
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! crabsformer = "2019.3.8"
//! ```
//!
//! and this to your crate root:
//!
//! ```rust
//! #[macro_use] extern crate crabsformer;
//! use crabsformer::prelude::*;
//! ```
//!
//! To get started using Crabsformer, read the quickstart tutorial below.
//!
//! # Quickstart Tutorial
//!
//! ## Prerequisites
//! Before reading this quick tutorial you should know a bit of Rust.
//! If you would like to refresh your memory, take a look at the
//! [Rust book].
//!
//! [Rust book]: https://doc.rust-lang.org/book/
//!
//! ## The Basics
//! Crabsformer's main data structures are [`Vector<T>`] and [`Matrix<T>`].
//!
//! `Vector<T>` is a fixed-length list of elements of the same [numeric type].
//! The [`vector!`] macro is provided to make initialization more convenient.
//!
//! ```
//! # #[macro_use] extern crate crabsformer;
//! # use crabsformer::prelude::*;
//! # fn main() {
//! let v = vector![1, 10, 11, 314];
//! # }
//! ```
//!
//! It can also initialize each element of a `Vector<T>` with a given value.
//!
//! ```
//! # #[macro_use] extern crate crabsformer;
//! # use crabsformer::prelude::*;
//! # fn main() {
//! # use crabsformer::prelude::*;
//! let v = vector![0; 5]; // vector![0, 0, 0, 0, 0]
//! # }
//! ```
//!
//! [`uniform(len, low, high)`] function can be used to create a
//! `Vector<T>` of the given length `len` and populate it with
//! random samples from a uniform distribution over the half-open
//! interval `[low, high)` (includes `low`, but excludes `high`).
//!
//! ```
//! # use crabsformer::prelude::*;
//! let v = Vector::uniform(5, 0.0, 1.0);
//! ```
//!
//! There are also other function such as [`zeros(len)`], [`ones(len)`],
//! [`full(len, value)`], [`range(start, stop, step)`],
//! [`linspace(len, start, stop)`] etc that can be used to
//! create a `Vector<T>`.
//!
//! [`uniform(len, low, high)`]: struct.Vector.html#method.uniform
//! [`zeros(len)`]: struct.Vector.html#method.zeros
//! [`ones(len)`]: struct.Vector.html#method.ones
//! [`full(len, value)`]: struct.Vector.html#method.full
//! [`range(start, stop, step)`]: struct.Vector.html#method.range
//! [`linspace(len, start, stop)`]: struct.Vector.html#method.linspace
//!
//!
//! `Vector<T>` can be added and subtracted like the following:
//!
//! ```rust
//! # #[macro_use] extern crate crabsformer;
//! # use crabsformer::prelude::*;
//! # fn main() {
//! let a = vector![0.5, 0.6, 0.9, 1.7];
//! let b = vector![1.0, 0.4, 0.2, 0.1];
//!
//! let c = a + b;
//! assert_eq!(c, vector![1.5, 1.0, 1.1, 1.8]);
//!
//! let d = a - b;
//! assert_eq!(d, vector![-0.5, 0.2, 0.7, 1.6]);
//! # }
//! ```
//!
//! It can also multiplied by scalar or other `Vector<T>` like the
//! following:
//!
//! ```rust
//! # #[macro_use] extern crate crabsformer;
//! # use crabsformer::prelude::*;
//! # fn main() {
//! # let a = vector![0.5, 0.6, 0.9, 1.7];
//! # let b = vector![1.0, 0.4, 0.2, 0.1];
//! // Multiply vector by scalar
//! let e = 2 * a;
//! assert_eq!(e, vector![1.0, 1.2, 1.8, 3.4]);
//!
//! // Dot product
//! let f = a.dot(b);
//!
//! // Cross product
//! let g = a.cross(b);
//! # }
//! ```
//!
//!
//! [numeric type]: https://doc.rust-lang.org/reference/types/numeric.html
//! [`vector!`]: macro.vector.html
//!
//! TODO: vector indexing
//! TODO: vector operations
//!
//!
//! - each element can be accessed by zero-based index
//!
//! - [`Vector<T>`] have elements/components, accessed by zero-based index
//!
//! Points:
//! - What is [`Vector<T>`] ?
//! - What is [`Matrix<T>`] ?
//! - You can build vector and matrix
//! - you can perform opertaion on them
//! - you can
//!
//! [`Vector<T>`]: struct.Vector.html
//!
//! ### An Example
//!
//! ### Vector Creation
//!
//! ## Getting help
//! Feel free to start discussion at [GitHub issues].
//!
//! [Github issues]: https://github.com/pyk/crabsformer/issues/new/choose
//!
//! ## License
//! Crabsformer is licensed under the [Apache-2.0] license.
//!
//! Unless you explicitly state otherwise, any contribution intentionally
//! submitted for inclusion in Crabsformer by you, as defined in the Apache-2.0
//! license, shall be licensed as above, without
//! any additional terms or conditions.
//!
//! [Apache-2.0]: https://github.com/pyk/crabsformer/blob/master/LICENSE
//!
extern crate num;
extern crate rand;

mod matrix;
pub mod prelude;
mod vector;

pub use matrix::*;
pub use vector::*;
