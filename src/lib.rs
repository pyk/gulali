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
//! crabsformer = "2019.3.12"
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
//! length and populate it with random samples from a uniform
//! distribution over the half-open interval.
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
//! ## Numeric Vector Basic Operations
//! You can perform arithmetic operations on a numeric vector.
//! Arithmetic operators on numeric vectors apply elementwise.
//! A new numeric vector is created and filled with the result.
//! For example, if you add the numeric vector, the arithmetic operator
//! will work element-wise. The output will be a numeric vector of the same
//! length.
//!
//!
//! ```rust
//! # #[macro_use] extern crate crabsformer;
//! # use crabsformer::prelude::*;
//! # fn main() {
//! let x = vector![2, 4, 6] + vector![1, 3, 5];
//! assert_eq!(x, vector![3, 7, 11]);
//! # }
//! ```
//!
//! Numeric vector substraction and multiplication also works the same:
//!
//! ```rust
//! # #[macro_use] extern crate crabsformer;
//! # use crabsformer::prelude::*;
//! # fn main() {
//! let x = vector![3, 1, 5] - vector![1, 3, 5];
//! assert_eq!(x, vector![2, -2, 0]);
//!
//! let y = vector![5, 4, 1] * vector![2, 1, 4];
//! assert_eq!(y, vector![10, 4, 4]);
//! # }
//! ```
//!
//! You can run an arithmetic operation on the numeric vector with
//! a scalar value too. For example, this code multiplies each element
//! of the numeric vector by 2.
//!
//! ```
//! # #[macro_use] extern crate crabsformer;
//! # use crabsformer::prelude::*;
//! # fn main() {
//! let x = vector![3, 1, 4] * 2;
//! assert_eq!(x, vector![6, 2, 8]);
//! # }
//! ```
//!
//! Some operations, such as `+=` and `*=`, act in place to modify an
//! existing numeric vector rather than create a new one.
//!
//! ```
//! # #[macro_use] extern crate crabsformer;
//! # use crabsformer::prelude::*;
//! # fn main() {
//! let mut x = vector![3, 1, 4];
//!
//! x += 3;
//! assert_eq!(x, vector![6, 4, 7]);
//!
//! x -= 1;
//! assert_eq!(x, vector![5, 3, 6]);
//!
//! x *= 2;
//! assert_eq!(x, vector![10, 6, 12]);
//! # }
//! ```
//!
//! If you try to add, substract or multiply numeric vector with a
//! different number of elements, you will get an error. For example:
//!
//! ```should_panic
//! # #[macro_use] extern crate crabsformer;
//! # use crabsformer::prelude::*;
//! # fn main() {
//! let x = vector![3, 1, 4, 1, 5] + vector![2, 10, 9];
//! # }
//! // thread 'main' panicked at 'Vector addition with invalid length: 5 != 3' src/main.rs:12:13
//! ```
//!
//! If you would like to square of the individual elements of the numeric
//! vector, or even higher up, use the [`power`] method. Here, each element of the
//! numeric vector is raised to the power 2.
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
//! When operating with numeric vectors of different types,
//! the Rust compiler will raise error like the following:
//!
//! ```text
//! cannot add `vector::Vector<{integer}>` to `vector::Vector<{float}>`
//! ```
//!
//! Many unary operations, such as computing the sum of all the elements in the
//! numeric vector, are implemented as methods.
//!
//! ```
//! # #[macro_use] extern crate crabsformer;
//! # use crabsformer::prelude::*;
//! # fn main() {
//! let x = vector![3, 1, 4];
//! let sum = x.sum();
//! assert_eq!(sum, 8);
//!
//! let max = x.max();
//! assert_eq!(max, 4);
//!
//! let min = x.min();
//! assert_eq!(min, 1);
//! # }
//! ```
//!
//! See also: [`power`], [`filter`], [`sum`], [`max`], [`min`].
//!
//! [`power`]: struct.Vector.html#method.power
//! [`filter`]: struct.Vector.html#method.filter
//! [`sum`]: struct.Vector.html#method.sum
//! [`max`]: struct.Vector.html#method.max
//! [`min`]: struct.Vector.html#method.min
//!
//! ---
//!
//! **TODO([pyk])**: Continue here, numeric vector indexing, slicing and iterating
//!
//! ---
//!
//! [numeric type]: https://doc.rust-lang.org/reference/types/numeric.html
//! [`vector!`]: macro.vector.html
//! [pyk]: https://github.com/pyk
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
