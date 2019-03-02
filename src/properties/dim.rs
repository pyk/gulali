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

/// Dimension of the vector
pub trait Dimension<T>
where
    T: Copy,
{
    /// Returns the number of dimensions of the vector
    fn dim(&self) -> usize;
}

impl<T> Dimension<T> for Vec<T>
where
    T: Copy,
{
    fn dim(&self) -> usize {
        1
    }
}

impl<T> Dimension<T> for Vec<Vec<T>>
where
    T: Copy,
{
    fn dim(&self) -> usize {
        2
    }
}

impl<T> Dimension<T> for Vec<Vec<Vec<T>>>
where
    T: Copy,
{
    fn dim(&self) -> usize {
        3
    }
}

impl<T> Dimension<T> for Vec<Vec<Vec<Vec<T>>>>
where
    T: Copy,
{
    fn dim(&self) -> usize {
        4
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_dim() {
        let arr1: Vec<i32> = Vec::one_dim(2).zeros();
        assert_eq!(arr1.dim(), 1);

        let arr2: Vec<Vec<i32>> = Vec::two_dim(2, 2).zeros();
        assert_eq!(arr2.dim(), 2);

        let arr3: Vec<Vec<Vec<i32>>> = Vec::three_dim(2, 2, 2).zeros();
        assert_eq!(arr3.dim(), 3);

        let arr4: Vec<Vec<Vec<Vec<i32>>>> = Vec::four_dim(2, 2, 2, 3).zeros();
        assert_eq!(arr4.dim(), 4);
    }

}
