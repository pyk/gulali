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
//!
//! ## Usage
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
//! extern crate crabsformer;
//!
//! // Import all required traits
//! use crabsformer::prelude::*;
//! ```
//!
//! To get started using Crabsformer, read the quickstart tutorial below.
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
//! Crabsformer's main data structures are [`Vector<T>`] and [`Matrix<T>`].
//! You can build vector and matrix
//! you can perform opertaion on them
//! you can
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
