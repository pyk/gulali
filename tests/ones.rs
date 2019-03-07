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
fn test_ones_u8_one_dim() {
    let arr: Vec<u8> = Vec::one_dim().with_shape([2]).ones().generate();
    assert_eq!(arr, [1, 1]);
}

#[test]
fn test_ones_u8_two_dim() {
    let arr: Vec<Vec<u8>> =
        Vec::two_dim().with_shape([1, 2]).ones().generate();
    assert_eq!(arr, [[1, 1]]);
}

#[test]
fn test_ones_u8_three_dim() {
    let arr: Vec<Vec<Vec<u8>>> =
        Vec::three_dim().with_shape([1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[1, 1]]]);
}

#[test]
fn test_ones_u8_four_dim() {
    let arr: Vec<Vec<Vec<Vec<u8>>>> =
        Vec::four_dim().with_shape([1, 1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[[1, 1]]]]);
}

#[test]
fn test_ones_u16_one_dim() {
    let arr: Vec<u16> = Vec::one_dim().with_shape([2]).ones().generate();
    assert_eq!(arr, [1, 1]);
}

#[test]
fn test_ones_u16_two_dim() {
    let arr: Vec<Vec<u16>> =
        Vec::two_dim().with_shape([1, 2]).ones().generate();
    assert_eq!(arr, [[1, 1]]);
}

#[test]
fn test_ones_u16_three_dim() {
    let arr: Vec<Vec<Vec<u16>>> =
        Vec::three_dim().with_shape([1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[1, 1]]]);
}

#[test]
fn test_ones_u16_four_dim() {
    let arr: Vec<Vec<Vec<Vec<u16>>>> =
        Vec::four_dim().with_shape([1, 1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[[1, 1]]]]);
}

#[test]
fn test_ones_u32_one_dim() {
    let arr: Vec<u32> = Vec::one_dim().with_shape([2]).ones().generate();
    assert_eq!(arr, [1, 1]);
}

#[test]
fn test_ones_u32_two_dim() {
    let arr: Vec<Vec<u32>> =
        Vec::two_dim().with_shape([1, 2]).ones().generate();
    assert_eq!(arr, [[1, 1]]);
}

#[test]
fn test_ones_u32_three_dim() {
    let arr: Vec<Vec<Vec<u32>>> =
        Vec::three_dim().with_shape([1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[1, 1]]]);
}

#[test]
fn test_ones_u32_four_dim() {
    let arr: Vec<Vec<Vec<Vec<u32>>>> =
        Vec::four_dim().with_shape([1, 1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[[1, 1]]]]);
}

#[test]
fn test_ones_u64_one_dim() {
    let arr: Vec<u64> = Vec::one_dim().with_shape([2]).ones().generate();
    assert_eq!(arr, [1, 1]);
}

#[test]
fn test_ones_u64_two_dim() {
    let arr: Vec<Vec<u64>> =
        Vec::two_dim().with_shape([1, 2]).ones().generate();
    assert_eq!(arr, [[1, 1]]);
}

#[test]
fn test_ones_u64_three_dim() {
    let arr: Vec<Vec<Vec<u64>>> =
        Vec::three_dim().with_shape([1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[1, 1]]]);
}

#[test]
fn test_ones_u64_four_dim() {
    let arr: Vec<Vec<Vec<Vec<u64>>>> =
        Vec::four_dim().with_shape([1, 1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[[1, 1]]]]);
}

#[test]
fn test_ones_u128_one_dim() {
    let arr: Vec<u128> = Vec::one_dim().with_shape([2]).ones().generate();
    assert_eq!(arr, [1, 1]);
}

#[test]
fn test_ones_u128_two_dim() {
    let arr: Vec<Vec<u128>> =
        Vec::two_dim().with_shape([1, 2]).ones().generate();
    assert_eq!(arr, [[1, 1]]);
}

#[test]
fn test_ones_u128_three_dim() {
    let arr: Vec<Vec<Vec<u128>>> =
        Vec::three_dim().with_shape([1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[1, 1]]]);
}

#[test]
fn test_ones_u128_four_dim() {
    let arr: Vec<Vec<Vec<Vec<u128>>>> =
        Vec::four_dim().with_shape([1, 1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[[1, 1]]]]);
}

#[test]
fn test_ones_i8_one_dim() {
    let arr: Vec<i8> = Vec::one_dim().with_shape([2]).ones().generate();
    assert_eq!(arr, [1, 1]);
}

#[test]
fn test_ones_i8_two_dim() {
    let arr: Vec<Vec<i8>> =
        Vec::two_dim().with_shape([1, 2]).ones().generate();
    assert_eq!(arr, [[1, 1]]);
}

#[test]
fn test_ones_i8_three_dim() {
    let arr: Vec<Vec<Vec<i8>>> =
        Vec::three_dim().with_shape([1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[1, 1]]]);
}

#[test]
fn test_ones_i8_four_dim() {
    let arr: Vec<Vec<Vec<Vec<i8>>>> =
        Vec::four_dim().with_shape([1, 1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[[1, 1]]]]);
}

#[test]
fn test_ones_i16_one_dim() {
    let arr: Vec<i16> = Vec::one_dim().with_shape([2]).ones().generate();
    assert_eq!(arr, [1, 1]);
}

#[test]
fn test_ones_i16_two_dim() {
    let arr: Vec<Vec<i16>> =
        Vec::two_dim().with_shape([1, 2]).ones().generate();
    assert_eq!(arr, [[1, 1]]);
}

#[test]
fn test_ones_i16_three_dim() {
    let arr: Vec<Vec<Vec<i16>>> =
        Vec::three_dim().with_shape([1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[1, 1]]]);
}

#[test]
fn test_ones_i16_four_dim() {
    let arr: Vec<Vec<Vec<Vec<i16>>>> =
        Vec::four_dim().with_shape([1, 1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[[1, 1]]]]);
}

#[test]
fn test_ones_i32_one_dim() {
    let arr: Vec<i32> = Vec::one_dim().with_shape([2]).ones().generate();
    assert_eq!(arr, [1, 1]);
}

#[test]
fn test_ones_i32_two_dim() {
    let arr: Vec<Vec<i32>> =
        Vec::two_dim().with_shape([1, 2]).ones().generate();
    assert_eq!(arr, [[1, 1]]);
}

#[test]
fn test_ones_i32_three_dim() {
    let arr: Vec<Vec<Vec<i32>>> =
        Vec::three_dim().with_shape([1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[1, 1]]]);
}

#[test]
fn test_ones_i32_four_dim() {
    let arr: Vec<Vec<Vec<Vec<i32>>>> =
        Vec::four_dim().with_shape([1, 1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[[1, 1]]]]);
}

#[test]
fn test_ones_i64_one_dim() {
    let arr: Vec<i64> = Vec::one_dim().with_shape([2]).ones().generate();
    assert_eq!(arr, [1, 1]);
}

#[test]
fn test_ones_i64_two_dim() {
    let arr: Vec<Vec<i64>> =
        Vec::two_dim().with_shape([1, 2]).ones().generate();
    assert_eq!(arr, [[1, 1]]);
}

#[test]
fn test_ones_i64_three_dim() {
    let arr: Vec<Vec<Vec<i64>>> =
        Vec::three_dim().with_shape([1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[1, 1]]]);
}

#[test]
fn test_ones_i64_four_dim() {
    let arr: Vec<Vec<Vec<Vec<i64>>>> =
        Vec::four_dim().with_shape([1, 1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[[1, 1]]]]);
}

#[test]
fn test_ones_i128_one_dim() {
    let arr: Vec<i128> = Vec::one_dim().with_shape([2]).ones().generate();
    assert_eq!(arr, [1, 1]);
}

#[test]
fn test_ones_i128_two_dim() {
    let arr: Vec<Vec<i128>> =
        Vec::two_dim().with_shape([1, 2]).ones().generate();
    assert_eq!(arr, [[1, 1]]);
}

#[test]
fn test_ones_i128_three_dim() {
    let arr: Vec<Vec<Vec<i128>>> =
        Vec::three_dim().with_shape([1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[1, 1]]]);
}

#[test]
fn test_ones_i128_four_dim() {
    let arr: Vec<Vec<Vec<Vec<i128>>>> =
        Vec::four_dim().with_shape([1, 1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[[1, 1]]]]);
}

#[test]
fn test_ones_f32_one_dim() {
    let arr: Vec<f32> = Vec::one_dim().with_shape([2]).ones().generate();
    assert_eq!(arr, [1.0, 1.0]);
}

#[test]
fn test_ones_f32_two_dim() {
    let arr: Vec<Vec<f32>> =
        Vec::two_dim().with_shape([1, 2]).ones().generate();
    assert_eq!(arr, [[1.0, 1.0]]);
}

#[test]
fn test_ones_f32_three_dim() {
    let arr: Vec<Vec<Vec<f32>>> =
        Vec::three_dim().with_shape([1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[1.0, 1.0]]]);
}

#[test]
fn test_ones_f32_four_dim() {
    let arr: Vec<Vec<Vec<Vec<f32>>>> =
        Vec::four_dim().with_shape([1, 1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[[1.0, 1.0]]]]);
}

#[test]
fn test_ones_f64_one_dim() {
    let arr: Vec<f64> = Vec::one_dim().with_shape([2]).ones().generate();
    assert_eq!(arr, [1.0, 1.0]);
}

#[test]
fn test_ones_f64_two_dim() {
    let arr: Vec<Vec<f64>> =
        Vec::two_dim().with_shape([1, 2]).ones().generate();
    assert_eq!(arr, [[1.0, 1.0]]);
}

#[test]
fn test_ones_f64_three_dim() {
    let arr: Vec<Vec<Vec<f64>>> =
        Vec::three_dim().with_shape([1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[1.0, 1.0]]]);
}

#[test]
fn test_ones_f64_four_dim() {
    let arr: Vec<Vec<Vec<Vec<f64>>>> =
        Vec::four_dim().with_shape([1, 1, 1, 2]).ones().generate();
    assert_eq!(arr, [[[[1.0, 1.0]]]]);
}
