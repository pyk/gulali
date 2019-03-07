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

extern crate gulali;

use gulali::prelude::*;

#[test]
fn test_full_one_dim() {
    // 1D array
    let arr1: Vec<i32> = Vec::one_dim().with_shape([5]).full_of(0).generate();
    assert_eq!(arr1, [0, 0, 0, 0, 0]);
}

#[test]
fn test_full_two_dim() {
    let arr2: Vec<Vec<f64>> =
        Vec::two_dim().with_shape([1, 2]).full_of(5.0).generate();
    assert_eq!(arr2, [[5.0, 5.0]]);
}

#[test]
fn test_full_three_dim() {
    let arr3: Vec<Vec<Vec<f64>>> = Vec::three_dim()
        .with_shape([1, 1, 2])
        .full_of(5.0)
        .generate();
    assert_eq!(arr3, [[[5.0, 5.0]]]);
}

#[test]
fn test_full_four_dim() {
    let arr4: Vec<Vec<Vec<Vec<f64>>>> = Vec::four_dim()
        .with_shape([1, 1, 1, 2])
        .full_of(5.0)
        .generate();
    assert_eq!(arr4, [[[[5.0, 5.0]]]]);
}
