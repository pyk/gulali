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

//! Iterates over elements of the numeric vector.
//!
//! TODO(pyk): Add docs here
//!
//!

use crate::vector::Vector;
use num::Num;
use std::iter;

// Implement row iterator for matrix
pub struct VectorElementIterator<'a, T: 'a>
where
    T: Num + Copy,
{
    vector: &'a Vector<T>,
    pos: usize,
}

impl<'a, T> Iterator for VectorElementIterator<'a, T>
where
    T: Num + Copy,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.vector.len() {
            return None;
        }
        // Increment the position of the row iterator.
        self.pos += 1;
        // Return the reference to the element
        Some(&self.vector[self.pos - 1])
    }
}

// Implement mutable row iterator for matrix
// Currently we can't implement a safe mutable Iterator
// https://www.reddit.com/r/rust/comments/6ffrbs/implementing_a_safe_mutable_iterator/
// https://stackoverflow.com/a/30422716
//pub struct VectorElementIMutableterator<'a, T: 'a>
//where
//    T: Num + Copy,
//{
//    vector: &'a mut Vector<T>,
//    pos: usize,
//}
//
//impl<'a, T> Iterator for VectorElementIMutableterator<'a, T>
//where
//    T: Num + Copy,
//{
//    type Item = &'a mut T;
//
//    fn next(&mut self) -> Option<Self::Item> {
//        if self.pos >= self.vector.len() {
//            return None;
//        }
//        // Increment the position of the row iterator.
//        self.pos += 1;
//        // Return the reference to the element
//        Some(&mut self.vector.data[self.pos])
//    }
//}

// Create numeric vector from an iterator
impl<T> iter::FromIterator<T> for Vector<T>
where
    T: Num + Copy,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut v = Vec::new();

        for i in iter {
            v.push(i);
        }

        Vector::from(v)
    }
}

impl<T> Vector<T>
where
    T: Num + Copy,
{
    /// Iterates over elements of the numeric vector.
    ///
    /// # Examples
    /// ```
    /// # use crabsformer::*;
    /// let x = vector![1, 2, 3];
    /// let mut elements = x.elements();
    ///
    /// assert_eq!(elements.next(), Some(&1));
    /// assert_eq!(elements.next(), Some(&2));
    /// assert_eq!(elements.next(), Some(&3));
    /// assert_eq!(elements.next(), None);
    /// ```
    pub fn elements<'a>(&'a self) -> VectorElementIterator<'a, T> {
        VectorElementIterator {
            vector: self,
            pos: 0,
        }
    }

    // NOTE: Currently we can't implement mutable iterator in safe
    // manner. https://stackoverflow.com/a/30422716
    // Iterates over elements of the numeric vector with mutable
    // references.
    //
    // # Examples
    // ```
    // # use crabsformer::*;
    // let mut x = vector![1, 2, 3];
    // for value in x.elements_mut() {
    //     *value = 314;
    // }
    // ```
    //    pub fn elements_mut<'a>(&'a mut self) -> VectorElementIterator<'a, T> {
    //        VectorElementIterator {
    //            vector: self,
    //            pos: 0,
    //        }
    //    }
}
