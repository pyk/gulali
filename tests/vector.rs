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

use crabsformer::prelude::*;

#[test]
fn test_from() {
    let a = vec![1, 2, 3];
    let b = Vector::from(a);
    assert_eq!(b, vector![1, 2, 3]);
}

#[test]
fn test_clone() {
    // Test clone
    let a = vector![3, 1, 4];
    let b = a.clone();
    assert_eq!(a, b);
}
