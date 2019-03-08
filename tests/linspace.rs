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
fn test_linspace_f32_none() {
    // If stop_at is not used, it should returns none
    Vec::<f32>::linspace().generate();
}

#[test]
#[should_panic]
fn test_linspace_f64_none() {
    // If stop_at is not used, it should returns none
    Vec::<f64>::linspace().generate();
}

#[test]
fn test_linspace_default() {
    // Default
    let lin: Vec<f32> = Vec::linspace().stop_at(5.0).generate();
    assert_eq!(
        lin,
        [
            0.0, 0.55555556, 1.11111111, 1.6666667, 2.22222222, 2.777778,
            3.3333335, 3.888889, 4.44444444, 5.0
        ]
    );
}

#[test]
fn test_linspace_start_at() {
    // Default
    let lin: Vec<f32> = Vec::linspace().start_at(1.0).stop_at(5.0).generate();
    assert_eq!(
        lin,
        [
            1.0, 1.44444444, 1.88888889, 2.33333333, 2.77777778, 3.222222,
            3.6666665, 4.11111111, 4.555556, 5.0
        ]
    );
}

#[test]
fn test_linspace_with_size() {
    // Default
    let lin: Vec<f32> = Vec::linspace()
        .start_at(1.0)
        .stop_at(5.0)
        .with_size(5)
        .generate();
    assert_eq!(lin, [1.0, 2.0, 3.0, 4.0, 5.0]);
}
