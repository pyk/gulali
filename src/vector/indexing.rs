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

//! Numeric vector indexing.
//!
//! # Overview
//! TODO(pyk): https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
//!
//!

use crate::vector::Vector;
use num::Num;
use std::ops;

impl<T> Vector<T>
where
    T: Num + Copy,
{
    // Bound checking
    pub(crate) fn check_bound(&self, i: usize) {
        if i >= self.len() {
            panic!(
                "Vector index {} out of range for vector with length {}",
                i,
                self.len()
            )
        }
    }
}

// Implement vector indexing
// vector[index]
impl<T> ops::Index<usize> for Vector<T>
where
    T: Num + Copy,
{
    type Output = T;

    fn index(&self, i: usize) -> &T {
        self.check_bound(i);
        &self.data[i]
    }
}
// Implement vector indexing in mutable context
// vector[index] = lut
impl<T> ops::IndexMut<usize> for Vector<T>
where
    T: Num + Copy,
{
    fn index_mut(&mut self, i: usize) -> &mut T {
        self.check_bound(i);
        &mut self.data[i]
    }
}
