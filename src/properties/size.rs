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
//
// This file may not be copied, modified, or distributed
// except according to those terms.

use crate::properties::shape::*;

/// Total number of elements of the vector
pub trait Size<T>
where
    T: Copy,
{
    /// Returns the total number of elements of the vector.
    /// This is equal to the product of the elements of shape.
    fn size(&self) -> usize;
}

impl<T> Size<T> for Vec<T>
where
    T: Copy,
{
    fn size(&self) -> usize {
        self.shape().iter().product()
    }
}

impl<T> Size<T> for Vec<Vec<T>>
where
    T: Copy,
{
    fn size(&self) -> usize {
        let shape: Vec<usize> = self.shape();
        shape.iter().product()
    }
}

impl<T> Size<T> for Vec<Vec<Vec<T>>>
where
    T: Copy,
{
    fn size(&self) -> usize {
        self.shape().iter().product()
    }
}

impl<T> Size<T> for Vec<Vec<Vec<Vec<T>>>>
where
    T: Copy,
{
    fn size(&self) -> usize {
        self.shape().iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_size() {
        let arr1: Vec<i32> = Vec::one_dim(2).zeros();
        assert_eq!(arr1.size(), 2);

        let arr2: Vec<Vec<i32>> = Vec::two_dim(2, 2).zeros();
        assert_eq!(arr2.size(), 4);

        let arr3: Vec<Vec<Vec<i32>>> = Vec::three_dim(2, 2, 2).zeros();
        assert_eq!(arr3.size(), 8);

        let arr4: Vec<Vec<Vec<Vec<i32>>>> = Vec::four_dim(2, 2, 2, 3).zeros();
        assert_eq!(arr4.size(), 24);
    }
}
