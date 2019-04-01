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
use num::Num;
use std::fmt;

// Import all sub modules
pub mod builders;
pub mod indexing;
pub mod iterators;
pub mod loaders;
pub mod operations;
pub mod slicing;

/// Numeric vector.
///
/// TODO: add overview about vector here.
/// 1. how to create a vector
/// 2. Vector operation
/// 3. Indexing, etc.
pub struct Vector<T>
where
    T: Num + Copy,
{
    data: Vec<T>,
}

impl<T> Vector<T>
where
    T: Num + Copy,
{
    /// The total number of elements of the numeric vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crabsformer::prelude::*;
    /// let v = vector![3.0, 1.0, 4.0, 1.0, 5.0];
    /// assert_eq!(v.len(), 5);
    /// ```
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

// Conversion from &[T] to RowMatrix<T>
impl<T> From<&[T]> for Vector<T>
where
    T: Num + Copy,
{
    fn from(elements: &[T]) -> Self {
        Vector::from(elements.to_vec())
    }
}

// Conversion from Vec<T>
impl<T> From<Vec<T>> for Vector<T>
where
    T: Num + Copy,
{
    fn from(elements: Vec<T>) -> Self {
        Vector { data: elements }
    }
}

// Numeric vector comparison
impl<T> PartialEq for Vector<T>
where
    T: Num + Copy,
{
    fn eq(&self, other: &Vector<T>) -> bool {
        if self.data != other.data {
            return false;
        }
        true
    }
    fn ne(&self, other: &Vector<T>) -> bool {
        if self.data == other.data {
            return false;
        }
        true
    }
}

impl<T> fmt::Debug for Vector<T>
where
    T: Num + Copy + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        return write!(f, "{:?}", self.data);
    }
}

/// Sub numeric vector is a reference to contiguous elements
/// in the numeric vector.
#[derive(Debug)]
pub struct SubVector<'a, T>
where
    T: Num + Copy,
{
    // Offset sub numeric vector from the start of the vector
    offset: usize,
    // The size of the sub numeric vector
    size: usize,
    // The original numeric vector; where to get the elements from
    source: &'a Vector<T>,
}

//impl<'a, T> SubVector<'a, T>
//where
//    T: Num + Copy,
//{
//    /// Create new sub numeric vector
//    pub fn new(inner: &'a [T]) -> SubVector<'a, T> {
//        SubVector { inner }
//    }
//
//    /// Convert sub numeric vector to numeric vector
//    pub fn to_vector(&self) -> Vector<T> {
//        Vector {
//            elements: self.inner.to_vec(),
//        }
//    }
//}

// TODO: implement debug
//impl<'a, T> fmt::Debug for SubVector<'a, T>
//where
//    T: Num + Copy + fmt::Debug,
//{
//    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//        return write!(f, "SubVector({:?})", self.inner);
//    }
//}

// Sub numeric vector comparison
impl<'a, T> PartialEq for SubVector<'a, T>
where
    T: Num + Copy,
{
    fn eq(&self, other: &SubVector<'a, T>) -> bool {
        if self.offset == other.offset
            && self.size == other.size
            && self.source == other.source
        {
            true
        } else {
            false
        }
    }

    fn ne(&self, other: &SubVector<'a, T>) -> bool {
        if self.offset != other.offset
            || self.size != other.size
            || self.source != other.source
        {
            true
        } else {
            false
        }
    }
}

// TODO: implement exponent operator
// TODO: implement all operators https://www.tutorialspoint.com/numpy/numpy_arithmetic_operations.htm
