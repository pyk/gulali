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

//! Gulali's prelude
//!
//! Like the standard library's prelude, this module simplifies importing of
//! common items. Unlike the standard prelude, the contents of this module must
//! be imported manually:
//!
//! ```
//! use gulali::prelude::*;
//!
//! # let matrix: Vec<Vec<i32>> = Vec::two_dim(3, 3).zeros();
//! # assert_eq!(matrix.dim(), 2);
//! ```

pub use crate::attributes::*;
pub use crate::builders::dimensional::*;
pub use crate::builders::full::*;
pub use crate::builders::ones::*;
pub use crate::builders::range::*;
pub use crate::builders::zeros::*;
