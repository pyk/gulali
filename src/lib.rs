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

//! Gulali is an easy-to-use fundamental library for scientific computing with
//! Rust, highly inspired by [NumPy].
//!
//! [NumPy]: http://www.numpy.org/
//!
//!
//! ## Usage
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! gulali = "2019.3.8"
//! ```
//!
//! and this to your crate root:
//!
//! ```rust
//! extern crate gulali;
//!
//! // Import all required traits
//! use gulali::prelude::*;
//! ```
//!
//! To get started using Gulali, read the quickstart tutorial below.
//!
//! ## Quickstart Tutorial
//!
//! ### Prerequisites
//! Before reading this quick tutorial you should know a bit of Rust.
//! If you would like to refresh your memory, take a look at the
//! [Rust book].
//!
//! [Rust book]: https://doc.rust-lang.org/book/
//!
//! ### The Basics
//! Gulali's main data structure is the homogeneous multidimensional [vector].
//! It is a table of elements (usually numbers), all of the same type,
//! indexed by a tuple of positive integers.
//!
//! [vector]: https://doc.rust-lang.org/std/vec/struct.Vec.html
//!
//! For example, the coordinates of a point in 3D space `[1, 2, 1]` has
//! one dimension. That dimension has 3 elements in it, so we say
//! it has a length of 3. In the example pictured below,
//! the vector has 2 dimensions. The first dimension has a length of 2,
//! the second dimension has a length of 3.
//!
//! ```ignore
//! [
//!     [ 1., 0., 0.],
//!     [ 0., 1., 2.]
//! ]
//! ```
//!
//! Gulali uses Rust's [vector] standard data structure extensively.
//! Gulali don't reinvent yet-another data strucutre to keep things
//! simple and easy to use. Gulali add attributes to Rust's [vector]
//! like the following:
//!
//! - [`dim()`]: the number of dimensions of the vector.
//! - [`shape()`]: This is a list of integers indicating the
//!   size of the vector in each dimension.
//!   For a matrix with `n` rows and `m` columns, shape will be `[n,m]`.
//!   The length of the shape is therefore the number of
//!   dimensions, [`dim()`].
//! - [`size()`]: the total number of elements of the vector.
//!   This is equal to the product of the elements of shape.
//!
//! [slice]: https://doc.rust-lang.org/rust-by-example/primitives/array.html
//! [`dim()`]: attributes/trait.Dimension.html#tymethod.dim
//! [`shape()`]: attributes/trait.Shape.html#tymethod.shape
//! [`size()`]: attributes/trait.Size.html#tymethod.size
//!
//! ### An Example
//! ```rust
//! # use gulali::prelude::*;
//! // Generate a new two-dimensional vector with shape [3, 3]
//! // filled with zeros; i32 can be changed into any
//! // numeric data types.
//! let matrix: Vec<Vec<i32>> = Vec::two_dim()
//!     .with_shape([3, 3])
//!     .zeros()
//!     .generate();
//!
//! assert_eq!(matrix.dim(), 2);
//! assert_eq!(matrix.shape(), [3, 3]);
//! assert_eq!(matrix.size(), 9);
//! ```
//!
//! ### Vector Creation
//! There are several ways to create vectors.
//!
//! For example, you can create a vector using [`vec!`][vec!] macro
//! like the following:
//!
//! ```rust
//! # use gulali::prelude::*;
//! let a = vec![1, 2, 3];
//! assert_eq!(a.dim(), 1);
//! assert_eq!(a.shape(), [3]);
//! assert_eq!(a.size(), 3);
//!
//! let b = vec![
//!     vec![1, 2, 3],
//!     vec![4, 5, 6],
//!     vec![8, 9, 10],
//! ];
//! assert_eq!(b.dim(), 2);
//! assert_eq!(b.shape(), [3, 3]);
//! assert_eq!(b.size(), 9);
//! ```
//!
//! [vec!]: https://doc.rust-lang.org/std/macro.vec.html
//!
//! The type of the resulting vector is deduced from the type
//! of the elements in the macro.
//!
//! Often, the elements of a vector are originally unknown, but
//! its shape is known. Hence, Gulali offers several functions to
//! create vectors with initial placeholder content. These minimize
//! the necessity of growing vectors, an expensive operation.
//!
//! [`one_dim()`] will create a one-dimensional vector with specified
//! shape and values. For example:
//!
//! ```
//! # use gulali::prelude::*;
//! // Generate a one-dimensional vector with shape [5]
//! // filled with zeros; f64 can be changed into any
//! // numeric data types.
//! let bias: Vec<f64> = Vec::one_dim()
//!     .with_shape([5])
//!     .zeros()
//!     .generate();
//!
//! assert_eq!(bias, [0.0, 0.0, 0.0, 0.0, 0.0]);
//! ```
//!
//! [`two_dim()`] will create a two-dimensional vector with specified
//! shape and values. For example:
//!
//! ```
//! # use gulali::prelude::*;
//! // Generate a two-dimensional vector with shape [2, 2]
//! // filled with ones; f64 can be changed into any
//! // numeric data types.
//! let matrix: Vec<Vec<f64>> = Vec::two_dim()
//!     .with_shape([2, 2])
//!     .ones()
//!     .generate();
//!
//! assert_eq!(matrix, [[1.0, 1.0], [1.0, 1.0]]);
//! ```
//!
//! [`three_dim()`] will create a three-dimensional vector with specified
//! shape and values. For example:
//!
//! ```
//! # use gulali::prelude::*;
//! // Generate a three-dimensional vector with shape [1, 1, 2]
//! // filled with 5.0; f64 can be changed into any
//! // numeric data types.
//! let test: Vec<Vec<Vec<f64>>> = Vec::three_dim()
//!     .with_shape([1, 1, 2])
//!     .full_of(5.0)
//!     .generate();
//!
//! assert_eq!(test, [[[5.0, 5.0]]]);
//! ```
//!
//! [`four_dim()`] will create a four-dimensional vector with specified
//! shape and values. For example:
//!
//! ```
//! # use gulali::prelude::*;
//! // Generate a four-dimensional vector with shape [1, 1, 1, 2]
//! // filled with ones; f64 can be changed into any
//! // numeric data types.
//! let test: Vec<Vec<Vec<Vec<f64>>>> = Vec::four_dim()
//!     .with_shape([1, 1, 1, 2])
//!     .ones()
//!     .generate();
//!
//! assert_eq!(test, [[[[1.0, 1.0]]]]);
//! ```
//!
//! [`one_dim()`]: builders/trait.OneDimensional.html#tymethod.one_dim
//! [`two_dim()`]: builders/trait.TwoDimensional.html#tymethod.two_dim
//! [`three_dim()`]: builders/trait.ThreeDimensional.html#tymethod.three_dim
//! [`four_dim()`]: builders/trait.FourDimensional.html#tymethod.four_dim
//!
//!
//! ## Getting help
//! Feel free to start discussion at [GitHub issues].
//!
//! [Github issues]: https://github.com/pyk/gulali/issues/new/choose
//!
//! ## License
//! Gulali is licensed under the [Apache-2.0] license.
//!
//! Unless you explicitly state otherwise, any contribution intentionally
//! submitted for inclusion in Gulali by you, as defined in the Apache-2.0
//! license, shall be licensed as above, without
//! any additional terms or conditions.
//!
//! [Apache-2.0]: https://github.com/pyk/gulali/blob/master/LICENSE
//!
extern crate num;

pub mod builders;

pub mod attributes;

pub mod prelude;
