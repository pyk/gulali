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
//! **Notice!** This project is in early phase. Expect bugs and missing features.
//!
//! [NumPy]: http://www.numpy.org/
//!
//! # Usage
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! crabsformer = "2019.3.17"
//! ```
//!
//! and this to your crate root:
//!
//! ```rust
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
//! [`Vector<T>`]: vector/struct.Vector.html
//! [`Matrix<T>`]: matrix/struct.Matrix.html
//! [`len`]: vector/struct.Vector.html#method.len
//! [`shape`]: matrix/struct.Matrix.html#method.shape
//! [`Vec<T>`]: https://doc.rust-lang.org/std/vec/struct.Vec.html
//!
//! ### Numeric Vector Creation
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
//! # use crabsformer::prelude::*;
//! let v = vector![1, 10, 11, 314];
//! ```
//!
//! It can also initialize each element of a numeric vector with a given value.
//!
//! ```
//! # use crabsformer::prelude::*;
//! let v = vector![0; 5]; // vector![0, 0, 0, 0, 0]
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
//! # use crabsformer::prelude::*;
//! let x = Vector::range(0, 10, 1);
//! assert_eq!(x, vector![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
//! ```
//!
//! See also: [`vector!`], [`zeros`], [`zeros_like`], [`ones`], [`ones_like`],
//! [`full`], [`full_like`], [`range`], [`linspace`], [`uniform`], [`normal`].
//!
//! [`vector!`]: macro.vector.html
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
//! ### Numeric Vector Basic Operations
//! You can perform arithmetic operations on a numeric vector.
//! Arithmetic operators on numeric vectors apply elementwise.
//! A new numeric vector is created and filled with the result.
//! For example, if you add the numeric vector, the arithmetic operator
//! will work element-wise. The output will be a numeric vector of the same
//! length.
//!
//!
//! ```rust
//! # use crabsformer::prelude::*;
//! let x = vector![2, 4, 6] + vector![1, 3, 5];
//! assert_eq!(x, vector![3, 7, 11]);
//! ```
//!
//! Numeric vector substraction and multiplication also works the same:
//!
//! ```rust
//! # use crabsformer::prelude::*;
//! let x = vector![3, 1, 5] - vector![1, 3, 5];
//! assert_eq!(x, vector![2, -2, 0]);
//!
//! let y = vector![5, 4, 1] * vector![2, 1, 4];
//! assert_eq!(y, vector![10, 4, 4]);
//! ```
//!
//! You can run an arithmetic operation on the numeric vector with
//! a scalar value too. For example, this code multiplies each element
//! of the numeric vector by 2.
//!
//! ```
//! # use crabsformer::prelude::*;
//! let x = vector![3, 1, 4] * 2;
//! assert_eq!(x, vector![6, 2, 8]);
//! ```
//!
//! Some operations, such as `+=` and `*=`, act in place to modify an
//! existing numeric vector rather than create a new one.
//!
//! ```
//! # use crabsformer::prelude::*;
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
//! ```
//!
//! If you try to add, substract or multiply numeric vector with a
//! different number of elements, you will get an error. For example:
//!
//! ```should_panic
//! # use crabsformer::prelude::*;
//! let x = vector![3, 1, 4, 1, 5] + vector![2, 10, 9];
//! // thread 'main' panicked at 'Vector addition with invalid length: 5 != 3' src/main.rs:12:13
//! ```
//!
//! If you would like to square of the individual elements of the numeric
//! vector, or even higher up, use the [`power`] method. Here, each element of the
//! numeric vector is raised to the power 2.
//!
//! ```
//! # use crabsformer::prelude::*;
//! let x = vector![3, 1, 4, 1];
//! let y = x.power(2);
//! assert_eq!(y, vector![9, 1, 16, 1]);
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
//! # use crabsformer::prelude::*;
//! let x = vector![3, 1, 4];
//! let sum = x.sum();
//! assert_eq!(sum, 8);
//! assert_eq!(*x.max(), 4);
//! assert_eq!(*x.min(), 1);
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
//! ### Indexing, Slicing and Iterating Numeric Vector
//! Numeric vectors can be indexed, sliced and iterated over, much like
//! Rust's vector.
//!
//! ```
//! # use crabsformer::prelude::*;
//! let x = vector![3, 1, 4, 1];
//!
//! // Indexing numeric vector
//! assert_eq!(x[0], 3);
//! assert_eq!(x[2], 4);
//!
//! // Slicing numeric vector
//! x.slice(0..2); // [3, 1]
//! x.slice(2..); // [4, 1]
//! x.slice(..2); // [3, 1]
//!
//! // Iterating over element of numeric vector
//! for element in x.elements() {
//!     println!("element = {:?}", element);
//! }
//! ```
//!
//! ### Matrix Creation
//! There are several ways to create matrix too.
//!
//! For example, you can create a matrix from a Rust's vector using
//! `Matrix::from` static method. The type of the resulting matrix is
//! deduced from the type of the elements in the sequences.
//!
//! ```
//! # use crabsformer::prelude::*;
//! let x = vec![
//!     vec![3, 1, 4],
//!     vec![1, 5, 9],
//!     vec![0, 1, 2],
//! ];
//! let w = Matrix::from(x);
//! ```
//!
//! The number of the columns should be consistent
//! otherwise it will panic. For example:
//!
//! ```should_panic
//! # use crabsformer::prelude::*;
//! let x = vec![
//!     vec![3, 1, 4],
//!     vec![1, 5],
//! ];
//! let w = Matrix::from(x);
//! // thread 'main' panicked at 'Invalid matrix: the number of columns is inconsistent',
//! ```
//!
//!
//! The [`matrix!`] macro is provided to make initialization of the
//! matrix more convenient.
//!
//! ```
//! # use crabsformer::prelude::*;
//! let w = matrix![
//!     3.0, 1.0, 4.0;
//!     1.0, 5.0, 9.0;
//! ];
//! ```
//!
//! It can also initialize each element of a matrix with a given shape
//! and value.
//!
//! ```
//! # use crabsformer::prelude::*;
//! let w = matrix![[3, 3] => 0]; // matrix![0, 0, 0; 0, 0, 0; 0, 0, 0]
//! ```
//!
//! The function [`uniform`][m.uniform] creates a matrix of the given
//! shape and populate it with random samples from a uniform
//! distribution over the half-open interval.
//!
//! ```
//! # use crabsformer::prelude::*;
//! let w = Matrix::uniform([2, 2], 0.0, 1.0);
//! ```
//!
//! See also: [`matrix!`], [`zeros`][m.zeros], [`zeros_like`][m.zeros_like],
//! [`ones`][m.ones], [`ones_like`][m.ones_like], [`full`][m.full],
//! [`full_like`][m.full_like], [`uniform`][m.uniform], [`normal`][m.normal].
//!
//! [`matrix!`]: macro.matrix.html
//! [m.zeros]: struct.Matrix.html#method.zeros
//! [m.zeros_like]: struct.Matrix.html#method.zeros_like
//! [m.ones]: struct.Matrix.html#method.ones
//! [m.ones_like]: struct.Matrix.html#method.ones_like
//! [m.full]: struct.Matrix.html#method.full
//! [m.full_like]: struct.Matrix.html#method.full_like
//! [m.uniform]: struct.Matrix.html#method.uniform
//! [m.normal]: struct.Matrix.html#method.normal
//!
//! ### Matrix Basic Operations
//! You can perform arithmetic operations on a matrix.
//! Arithmetic operators on matrices apply elementwise.
//! A new matrix is created and filled with the result.
//! For example, if you add the matrix, the arithmetic operator
//! will work element-wise. The output will be a matrix of the same
//! shape.
//!
//!
//! ```rust
//! # use crabsformer::prelude::*;
//! let w1 = matrix![
//!     2, 4, 6;
//!     3, 1, 1;
//!     4, 5, 6;
//! ];
//!
//! let w2 = matrix![
//!     1, 3, 5;
//!     3, 1, 3;
//!     1, 1, 1;
//! ];
//!
//! let w3 = w1 + w2;
//!
//! assert_eq!(w3, matrix![
//!     3, 7, 11;
//!     6, 2, 4;
//!     5, 6, 7;
//! ]);
//! ```
//!
//! Matrix substraction and multiplication also works the same:
//!
//! ```rust
//! # use crabsformer::prelude::*;
//! let w1 = matrix![2, 4; 3, 1] - matrix![1, 3; 3, 1];
//! assert_eq!(w1, matrix![
//!     1, 1;
//!     0, 0;
//! ]);
//!
//! let w2 = matrix![0, 1; 2, 0] - matrix![1, 1; 0, 1];
//! assert_eq!(w2, matrix![
//!     -1, 0;
//!     2, -1;
//! ]);
//!
//! let w3 = matrix![0, 1; 1, 0] * matrix![1, 1; 1, 1];
//! assert_eq!(w3, matrix![
//!     0, 1;
//!     1, 0;
//! ]);
//! ```
//!
//! You can run an arithmetic operation on the matrix with
//! a scalar value too. For example, this code multiplies each element
//! of the matrix by 2.
//!
//! ```
//! # use crabsformer::prelude::*;
//! let w = matrix![3, 1; 4, 1] * 2;
//! assert_eq!(w, matrix![6, 2; 8, 2]);
//! ```
//!
//! Some operations, such as `+=` and `*=`, act in place to modify an
//! existing matrix rather than create a new one.
//!
//! ```
//! # use crabsformer::prelude::*;
//! let mut w = matrix![3, 1; 4, 1];
//!
//! w += 3;
//! assert_eq!(w, matrix![6, 4; 7, 4]);
//!
//! w -= 1;
//! assert_eq!(w, matrix![5, 3; 6, 3]);
//!
//! w *= 2;
//! assert_eq!(w, matrix![10, 6; 12, 6]);
//! ```
//!
//! If you try to add, substract or multiply matrix with a
//! different shape, you will get an error. For example:
//!
//! ```should_panic
//! # use crabsformer::prelude::*;
//! let x = matrix![3, 1; 4, 1] + matrix![2, 10, 9; 1, 4, 7];
//! // thread 'main' panicked at 'Matrix addition with invalid shape: [2, 2] != [3, 3]' src/main.rs:12:13
//! ```
//!
//! If you would like to square of the individual elements of the matrix
//! , or even higher up, use the [`power`][m.power] method. Here, each element
//! of the matrix is raised to the power 2.
//!
//! ```
//! # use crabsformer::prelude::*;
//! let w1 = matrix![3, 1; 4, 1];
//! let w2 = w1.power(2);
//! assert_eq!(w2, matrix![9, 1; 16, 1]);
//! ```
//!
//! [m.power]: struct.Matrix.html#method.power
//!
//! When operating with matrices of different types,
//! the Rust compiler will raise error like the following:
//!
//! ```text
//! cannot add `matrix::Matrix<{integer}>` to `matrix::Matrix<{float}>`
//! ```
//!
//! ---
//! TODO(pyk): Continue quick tutorial here
//!
//! ---
//!
//! [numeric type]: https://doc.rust-lang.org/reference/types/numeric.html
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

pub mod error;
pub mod matrix;
pub mod prelude;
pub mod utils;
pub mod vector;
