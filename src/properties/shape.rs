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

/// A list of integers indicating the size of the vector in each dimension
pub trait Shape<T>
where
    T: Copy,
{
    /// Returns a list of integers indicating the length of the
    /// vector in each dimension
    fn shape(&self) -> Vec<usize>;
}

impl<T> Shape<T> for Vec<T>
where
    T: Copy,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.len()]
    }
}

impl<T> Shape<T> for Vec<Vec<T>>
where
    T: Copy,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.len(), self[0].len()]
    }
}

impl<T> Shape<T> for Vec<Vec<Vec<T>>>
where
    T: Copy,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.len(), self[0].len(), self[0][0].len()]
    }
}

impl<T> Shape<T> for Vec<Vec<Vec<Vec<T>>>>
where
    T: Copy,
{
    fn shape(&self) -> Vec<usize> {
        vec![
            self.len(),
            self[0].len(),
            self[0][0].len(),
            self[0][0][0].len(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_shape() {
        let arr1: Vec<i32> = Vec::one_dim(2).zeros();
        assert_eq!(arr1.shape(), [2]);

        let arr2: Vec<Vec<i32>> = Vec::two_dim(2, 2).zeros();
        assert_eq!(arr2.shape(), [2, 2]);

        let arr3: Vec<Vec<Vec<i32>>> = Vec::three_dim(2, 2, 2).zeros();
        assert_eq!(arr3.shape(), [2, 2, 2]);

        let arr4: Vec<Vec<Vec<Vec<i32>>>> = Vec::four_dim(2, 2, 2, 3).zeros();
        assert_eq!(arr4.shape(), [2, 2, 2, 3]);
    }
}
