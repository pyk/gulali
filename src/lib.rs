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
//! crabsformer = "2019.3.9"
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
//! There are two main data structures in Crabsformer:
//!
//! 1. [`Vector<T>`] is a fixed-length list of elements of the same [numeric type].
//!    It has one atribute called [`len`] to represent the total number of elements.
//! 2. [`Matrix<T>`] is a table of elements of the same [numeric type]. It has one
//!    atribute called [`shape`] that represent the number of rows and the number
//!    of columns.
//!
//! `Vector<T>` is pronounced as 'numeric vector' to avoid confussion with Rust's
//! vector [`Vec<T>`] data structure.
//!
//! [`Vector<T>`]: struct.Vector.html
//! [`Matrix<T>`]: struct.Matrix.html
//! [`len`]: struct.Vector.html#method.len
//! [`shape`]: struct.Matrix.html#method.shape
//! [`Vec<T>`]: https://doc.rust-lang.org/std/vec/struct.Vec.html
//!
//! ## Numeric Vector Creation
//! There are several ways to create numeric vector.
//!
//! For example, you can create a numeric vector from a Rust vector using
//! `Vector::from` static method. The type of the resulting numeric vector is
//! deduced from the type of the elements in the sequences.
//!
//! ```
//! # use crabsformer::prelude::*;
//! let x = vec![3, 1, 4, 1, 5];
//! let y = Vector::from(x);
//! ```
//!
//! The [`vector!`] macro is provided to make initialization of the
//! numeric vector more convenient.
//!
//! ```
//! # #[macro_use] extern crate crabsformer;
//! # use crabsformer::prelude::*;
//! # fn main() {
//! let v = vector![1, 10, 11, 314];
//! # }
//! ```
//!
//! It can also initialize each element of a numeric vector with a given value.
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
//! The function [`uniform`] creates a numeric vector of the given
//! length `len` and populate it with random samples from a uniform
//! distribution over the half-open interval `[low, high)` (includes
//! `low`, but excludes `high`).
//!
//! ```
//! # use crabsformer::prelude::*;
//! let v = Vector::uniform(5, 0.0, 1.0);
//! // Vector([0.054709196, 0.86043775, 0.21187294, 0.6413728, 0.14186311]) (Random)
//! ```
//!
//! To create a numeric vector of evenly spaced values, Crabformer provide [`range`]
//! function.
//!
//! ```
//! # #[macro_use] extern crate crabsformer;
//! # use crabsformer::prelude::*;
//! # fn main() {
//! let x = Vector::range(0, 10, 1);
//! assert_eq!(x, vector![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
//! # }
//! ```
//!
//! Values are generated within the half-open interval `[start, stop)`
//! (in other words, the interval including `start` but excluding `stop`).
//!
//! See also: [`vector!`], [`zeros`], [`zeros_like`], [`ones`], [`ones_like`],
//! [`full`], [`full_like`], [`range`], [`linspace`], [`uniform`], [`normal`].
//!
//! [`zeros`]: struct.Vector.html#method.zeros
//! [`zeros_like`]: struct.Vector.html#method.zeros_like
//! [`ones`]: struct.Vector.html#method.ones
//! [`ones_like`]: struct.Vector.html#method.ones_like
//! [`full`]: struct.Vector.html#method.full
//! [`full_like`]: struct.Vector.html#method.full_like
//! [`range`]: struct.Vector.html#method.range
//! [`linspace`]: struct.Vector.html#method.linspace
//! [`uniform`]: struct.Vector.html#method.uniform
//! [`normal`]: struct.Vector.html#method.normal
//!
//!
//! ## Numeric Vector Operation
//! You can perform arithmetic operations on a `Vector<T>`.
//! For example, if you add the `Vector<T>`, the arithmetic operator
//! will work element-wise. The output will be a `Vector<T>` of the same
//! length.
//!
//! ```rust
//! # #[macro_use] extern crate crabsformer;
//! # use crabsformer::prelude::*;
//! # fn main() {
//! let x = vector![0.5, 0.6, 0.9, 1.7];
//! let y = vector![1.0, 0.4, 0.2, 0.1];
//! let z = x + y;
//! assert_eq!(z, vector![1.5, 1.0, 1.1, 1.8]);
//! # }
//! ```
//!
//! If you try to add `Vector<T>` with a different number of elements,
//! you will get an error. For example:
//!
//! ```should_panic
//! # #[macro_use] extern crate crabsformer;
//! # use crabsformer::prelude::*;
//! # fn main() {
//! let x = vector![3, 1, 4, 1, 5] + vector![2, 10, 9];
//! # }
//! ```
//! ```text
//! thread 'guide::example::main' panicked at 'Vector addition with invalid length: 5 != 3' src/main.rs:12:13
//! ```
//!
//! You can run an arithmetic operation on the `Vector<T>` with a scalar value.
//! For example, this code multiplies each element of the `Vector<T>` by 2.
//!
//! ```
//! # #[macro_use] extern crate crabsformer;
//! # use crabsformer::prelude::*;
//! # fn main() {
//! let x = vector![3, 1, 4, 1] * 2;
//! assert_eq!(x, vector![6, 2, 8, 2]);
//! # }
//! ```
//!
//! If you would like to square of the individual elements of the `Vector<T>`,
//! or even higher up, use the [`power`] method. Here, each element of the
//! `Vector<T>` is raised to the power 2.
//!
//! ```
//! # #[macro_use] extern crate crabsformer;
//! # use crabsformer::prelude::*;
//! # fn main() {
//! let x = vector![3, 1, 4, 1];
//! let y = x.power(2);
//! assert_eq!(y, vector![9, 1, 16, 1]);
//! # }
//! ```
//!
//! [`power`]: struct.Vector.html#method.power
//!
//! You can use [`filter`] to find the elements that match your criteria.
//!
//! ```
//! # #[macro_use] extern crate crabsformer;
//! # use crabsformer::prelude::*;
//! # fn main() {
//! let x = vector![3, 1, 4, 1];
//! let y = x.filter(|x| x >= 2);
//! assert_eq!(y, vector![3, 4]);
//! # }
//! ```
//!
//! [`filter`]: struct.Vector.html#method.filter
//!
//!
//! [numeric type]: https://doc.rust-lang.org/reference/types/numeric.html
//! [`vector!`]: macro.vector.html
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
