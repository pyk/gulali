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

use crabsformer::*;

#[test]
fn test_macro() {
    // Full of elements
    let a = vector![0; 5];
    let b = Vector::full(5, 0);
    assert_eq!(a, b);

    // Vector inialization
    let c = vector![1, 2, 3, 4];
    assert_eq!(c, vector![1, 2, 3, 4]);
}

#[test]
fn test_full() {
    let a = Vector::full(5, 5.0);
    assert_eq!(a, vector![5.0, 5.0, 5.0, 5.0, 5.0]);

    let b = Vector::full(5, 2);
    assert_eq!(b, vector![2, 2, 2, 2, 2]);
}

#[test]
fn test_full_like() {
    let v1 = vector![3.0, 1.0, 4.0, 1.0, 5.0];
    let v2 = Vector::full_like(&v1, 5.0);
    assert_eq!(v2, vector![5.0, 5.0, 5.0, 5.0, 5.0]);
}

#[test]
fn test_zeros() {
    let vf1: Vector<f64> = Vector::zeros(5);
    assert_eq!(vf1, vector![0.0, 0.0, 0.0, 0.0, 0.0]);

    let vf2: Vector<f32> = Vector::zeros(5);
    assert_eq!(vf2, vector![0.0, 0.0, 0.0, 0.0, 0.0]);

    let vs1: Vector<usize> = Vector::zeros(5);
    assert_eq!(vs1, vector![0, 0, 0, 0, 0]);

    let vu1: Vector<u8> = Vector::zeros(5);
    assert_eq!(vu1, vector![0, 0, 0, 0, 0]);

    let vu2: Vector<u16> = Vector::zeros(5);
    assert_eq!(vu2, vector![0, 0, 0, 0, 0]);

    let vu3: Vector<u32> = Vector::zeros(5);
    assert_eq!(vu3, vector![0, 0, 0, 0, 0]);

    let vu4: Vector<u64> = Vector::zeros(5);
    assert_eq!(vu4, vector![0, 0, 0, 0, 0]);

    let vu5: Vector<u128> = Vector::zeros(5);
    assert_eq!(vu5, vector![0, 0, 0, 0, 0]);

    let vi1: Vector<i8> = Vector::zeros(5);
    assert_eq!(vi1, vector![0, 0, 0, 0, 0]);

    let vi2: Vector<i16> = Vector::zeros(5);
    assert_eq!(vi2, vector![0, 0, 0, 0, 0]);

    let vi3: Vector<i32> = Vector::zeros(5);
    assert_eq!(vi3, vector![0, 0, 0, 0, 0]);

    let vi4: Vector<i64> = Vector::zeros(5);
    assert_eq!(vi4, vector![0, 0, 0, 0, 0]);

    let vi5: Vector<i128> = Vector::zeros(5);
    assert_eq!(vi5, vector![0, 0, 0, 0, 0]);
}

#[test]
fn test_zeros_like() {
    let vi1: Vector<i32> = Vector::ones(5);
    let vi2 = Vector::zeros_like(&vi1);
    assert_eq!(vi1.len(), vi2.len());
}

#[test]
fn test_ones() {
    let vf1: Vector<f64> = Vector::ones(5);
    assert_eq!(vf1, vector![1.0, 1.0, 1.0, 1.0, 1.0]);

    let vf2: Vector<f32> = Vector::ones(5);
    assert_eq!(vf2, vector![1.0, 1.0, 1.0, 1.0, 1.0]);

    let vs1: Vector<usize> = Vector::ones(5);
    assert_eq!(vs1, vector![1, 1, 1, 1, 1]);

    let vu1: Vector<u8> = Vector::ones(5);
    assert_eq!(vu1, vector![1, 1, 1, 1, 1]);

    let vu2: Vector<u16> = Vector::ones(5);
    assert_eq!(vu2, vector![1, 1, 1, 1, 1]);

    let vu3: Vector<u32> = Vector::ones(5);
    assert_eq!(vu3, vector![1, 1, 1, 1, 1]);

    let vu4: Vector<u64> = Vector::ones(5);
    assert_eq!(vu4, vector![1, 1, 1, 1, 1]);

    let vu5: Vector<u128> = Vector::ones(5);
    assert_eq!(vu5, vector![1, 1, 1, 1, 1]);

    let vi1: Vector<i8> = Vector::ones(5);
    assert_eq!(vi1, vector![1, 1, 1, 1, 1]);

    let vi2: Vector<i16> = Vector::ones(5);
    assert_eq!(vi2, vector![1, 1, 1, 1, 1]);

    let vi3: Vector<i32> = Vector::ones(5);
    assert_eq!(vi3, vector![1, 1, 1, 1, 1]);

    let vi4: Vector<i64> = Vector::ones(5);
    assert_eq!(vi4, vector![1, 1, 1, 1, 1]);

    let vi5: Vector<i128> = Vector::ones(5);
    assert_eq!(vi5, vector![1, 1, 1, 1, 1]);
}

#[test]
fn test_ones_like() {
    let vi1: Vector<i32> = Vector::ones(10);
    let vi2 = Vector::ones_like(&vi1);
    assert_eq!(vi1.len(), vi2.len());
}

#[test]
fn test_uniform() {
    let vf1: Vector<f32> = Vector::uniform(5, 0.0, 1.0);
    for value in vf1.elements() {
        assert!((0.0 <= *value) && (*value < 1.0));
    }

    let vf2: Vector<f64> = Vector::uniform(5, 0.0, 1.0);
    for value in vf2.elements() {
        assert!((0.0 <= *value) && (*value < 1.0));
    }

    let vs1: Vector<usize> = Vector::uniform(5, 1, 10);
    for value in vs1.elements() {
        assert!((1 <= *value) && (*value < 10));
    }

    let vu1: Vector<u8> = Vector::uniform(5, 1, 10);
    for value in vu1.elements() {
        assert!((1 <= *value) && (*value < 10));
    }

    let vu2: Vector<u16> = Vector::uniform(5, 1, 10);
    for value in vu2.elements() {
        assert!((1 <= *value) && (*value < 10));
    }

    let vu3: Vector<u32> = Vector::uniform(5, 1, 10);
    for value in vu3.elements() {
        assert!((1 <= *value) && (*value < 10));
    }

    let vu4: Vector<u64> = Vector::uniform(5, 1, 10);
    for value in vu4.elements() {
        assert!((1 <= *value) && (*value < 10));
    }

    let vu5: Vector<u128> = Vector::uniform(5, 1, 10);
    for value in vu5.elements() {
        assert!((1 <= *value) && (*value < 10));
    }

    let vi1: Vector<i8> = Vector::uniform(5, -10, 10);
    for value in vi1.elements() {
        assert!((-10 <= *value) && (*value < 10));
    }

    let vi2: Vector<i16> = Vector::uniform(5, -10, 10);
    for value in vi2.elements() {
        assert!((-10 <= *value) && (*value < 10));
    }

    let vi3: Vector<i32> = Vector::uniform(5, -10, 10);
    for value in vi3.elements() {
        assert!((-10 <= *value) && (*value < 10));
    }

    let vi4: Vector<i64> = Vector::uniform(5, -10, 10);
    for value in vi4.elements() {
        assert!((-10 <= *value) && (*value < 10));
    }

    let vi5: Vector<i128> = Vector::uniform(5, -10, 10);
    for value in vi5.elements() {
        assert!((-10 <= *value) && (*value < 10));
    }
}

#[test]
fn test_normal() {
    let a = Vector::normal(5, 2.0, 4.0);
    let b = Vector::normal(5, 2.0, 4.0);
    assert_eq!(a.len(), b.len());
    assert_ne!(a, b);
}

#[test]
fn test_range() {
    let a = Vector::range(0.0, 3.0, 0.5);
    assert_eq!(a, vector![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);

    let b = Vector::range(0, 3, 1);
    assert_eq!(b, vector![0, 1, 2]);
}

#[test]
fn test_linspace() {
    let a = Vector::linspace(5, 1.0, 10.0);
    assert_eq!(a, vector![1.0, 3.25, 5.5, 7.75, 10.0]);
}
