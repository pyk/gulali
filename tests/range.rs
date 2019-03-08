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

extern crate crabsformer;

use crabsformer::prelude::*;

#[test]
#[should_panic]
fn test_range_no_stop() {
    let _a: Vec<i32> = Vec::range().generate();
}

#[test]
#[should_panic]
fn test_range_invalid_interval() {
    let _a: Vec<i32> = Vec::range().start_at(5).stop_at(3).generate();
}

#[test]
fn test_range_u8() {
    // Default
    let range1: Vec<u8> = Vec::range().stop_at(3).generate();
    assert_eq!(range1, [0, 1, 2]);

    // start at specified value
    let range2: Vec<u8> = Vec::range().start_at(1).stop_at(3).generate();
    assert_eq!(range2, [1, 2]);

    // step by specified value
    let range3: Vec<u8> =
        Vec::range().start_at(1).stop_at(3).step_by(2).generate();
    assert_eq!(range3, [1]);
}

#[test]
fn test_range_u16() {
    // Default
    let range1: Vec<u16> = Vec::range().stop_at(3).generate();
    assert_eq!(range1, [0, 1, 2]);

    // start at specified value
    let range2: Vec<u16> = Vec::range().start_at(1).stop_at(3).generate();
    assert_eq!(range2, [1, 2]);

    // step by specified value
    let range3: Vec<u16> =
        Vec::range().start_at(1).stop_at(3).step_by(2).generate();
    assert_eq!(range3, [1]);
}

#[test]
fn test_range_u32() {
    // Default
    let range1: Vec<u32> = Vec::range().stop_at(3).generate();
    assert_eq!(range1, [0, 1, 2]);

    // start at specified value
    let range2: Vec<u32> = Vec::range().start_at(1).stop_at(3).generate();
    assert_eq!(range2, [1, 2]);

    // step by specified value
    let range3: Vec<u32> =
        Vec::range().start_at(1).stop_at(3).step_by(2).generate();
    assert_eq!(range3, [1]);
}

#[test]
fn test_range_u64() {
    // Default
    let range1: Vec<u64> = Vec::range().stop_at(3).generate();
    assert_eq!(range1, [0, 1, 2]);

    // start at specified value
    let range2: Vec<u64> = Vec::range().start_at(1).stop_at(3).generate();
    assert_eq!(range2, [1, 2]);

    // step by specified value
    let range3: Vec<u64> =
        Vec::range().start_at(1).stop_at(3).step_by(2).generate();
    assert_eq!(range3, [1]);
}

#[test]
fn test_range_u128() {
    // Default
    let range1: Vec<u128> = Vec::range().stop_at(3).generate();
    assert_eq!(range1, [0, 1, 2]);

    // start at specified value
    let range2: Vec<u128> = Vec::range().start_at(1).stop_at(3).generate();
    assert_eq!(range2, [1, 2]);

    // step by specified value
    let range3: Vec<u128> =
        Vec::range().start_at(1).stop_at(3).step_by(2).generate();
    assert_eq!(range3, [1]);
}

#[test]
fn test_range_i8() {
    // Default
    let range1: Vec<i8> = Vec::range().stop_at(3).generate();
    assert_eq!(range1, [0, 1, 2]);

    // start at specified value
    let range2: Vec<i8> = Vec::range().start_at(1).stop_at(3).generate();
    assert_eq!(range2, [1, 2]);

    // step by specified value
    let range3: Vec<i8> =
        Vec::range().start_at(1).stop_at(3).step_by(2).generate();
    assert_eq!(range3, [1]);
}

#[test]
fn test_range_i16() {
    // Default
    let range1: Vec<i16> = Vec::range().stop_at(3).generate();
    assert_eq!(range1, [0, 1, 2]);

    // start at specified value
    let range2: Vec<i16> = Vec::range().start_at(1).stop_at(3).generate();
    assert_eq!(range2, [1, 2]);

    // step by specified value
    let range3: Vec<i16> =
        Vec::range().start_at(1).stop_at(3).step_by(2).generate();
    assert_eq!(range3, [1]);
}

#[test]
fn test_range_i32() {
    // Default
    let range1: Vec<i32> = Vec::range().stop_at(3).generate();
    assert_eq!(range1, [0, 1, 2]);

    // start at specified value
    let range2: Vec<i32> = Vec::range().start_at(1).stop_at(3).generate();
    assert_eq!(range2, [1, 2]);

    // step by specified value
    let range3: Vec<i32> =
        Vec::range().start_at(1).stop_at(3).step_by(2).generate();
    assert_eq!(range3, [1]);
}

#[test]
fn test_range_i64() {
    // Default
    let range1: Vec<i64> = Vec::range().stop_at(3).generate();
    assert_eq!(range1, [0, 1, 2]);

    // start at specified value
    let range2: Vec<i64> = Vec::range().start_at(1).stop_at(3).generate();
    assert_eq!(range2, [1, 2]);

    // step by specified value
    let range3: Vec<i64> =
        Vec::range().start_at(1).stop_at(3).step_by(2).generate();
    assert_eq!(range3, [1]);
}

#[test]
fn test_range_i128() {
    // Default
    let range1: Vec<i128> = Vec::range().stop_at(3).generate();
    assert_eq!(range1, [0, 1, 2]);

    // start at specified value
    let range2: Vec<i128> = Vec::range().start_at(1).stop_at(3).generate();
    assert_eq!(range2, [1, 2]);

    // step by specified value
    let range3: Vec<i128> =
        Vec::range().start_at(1).stop_at(3).step_by(2).generate();
    assert_eq!(range3, [1]);
}

#[test]
fn test_range_f32() {
    // Default
    let range1: Vec<f32> = Vec::range().stop_at(3.0).generate();
    assert_eq!(range1, [0.0, 1.0, 2.0]);

    // start at specified value
    let range2: Vec<f32> = Vec::range().start_at(1.0).stop_at(3.0).generate();
    assert_eq!(range2, [1.0, 2.0]);

    // step by specified value
    let range3: Vec<f32> = Vec::range()
        .start_at(1.0)
        .stop_at(3.0)
        .step_by(0.5)
        .generate();
    assert_eq!(range3, [1.0, 1.5, 2.0, 2.5]);
}

#[test]
fn test_range_f64() {
    // Default
    let range1: Vec<f64> = Vec::range().stop_at(3.0).generate();
    assert_eq!(range1, [0.0, 1.0, 2.0]);

    // start at specified value
    let range2: Vec<f64> = Vec::range().start_at(1.0).stop_at(3.0).generate();
    assert_eq!(range2, [1.0, 2.0]);

    // step by specified value
    let range3: Vec<f64> = Vec::range()
        .start_at(1.0)
        .stop_at(3.0)
        .step_by(0.5)
        .generate();
    assert_eq!(range3, [1.0, 1.5, 2.0, 2.5]);
}
