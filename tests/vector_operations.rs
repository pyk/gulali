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

// Unary operations
#[test]
fn test_power() {
    let x = vector![3, 1, 4, 1];
    let y = x.power(2);
    assert_eq!(y, vector![9, 1, 16, 1]);
}

#[test]
fn test_power_mut() {
    let mut x = vector![3, 1, 4, 1];
    x.power_mut(2);
    assert_eq!(x, vector![9, 1, 16, 1]);
}

#[test]
fn test_filter() {
    let x = vector![3, 1, 4, 1];
    let y = x.filter(|x| x >= 2);
    assert_eq!(y, vector![3, 4]);
}

#[test]
fn test_sum() {
    let x = vector![3, 1, 4, 1];
    assert_eq!(x.sum(), 9);

    let y = vector![3.0, 1.0, 4.0, 1.0];
    assert_eq!(y.sum(), 9.0);
}

#[test]
fn test_max() {
    let x = vector![3, 1, 4, 1];
    assert_eq!(x.max(), &4);

    // let y = vector![3.0, 1.0, 4.0, 1.0];
    // assert_eq!(y.max(), 4.0);
}

#[test]
fn test_min() {
    let x = vector![3, 1, 4, 1];
    assert_eq!(x.min(), &1);

    // let y = vector![3.0, 1.0, 4.0, 1.0];
    // assert_eq!(y.min(), 1.0);
}

// Binary operations

#[test]
fn test_add() {
    let a = vector![3, 1, 4, 1, 5] + vector![3, 1, 4, 1, 5];
    assert_eq!(a, vector![6, 2, 8, 2, 10]);

    let b =
        vector![3.0, 1.0, 4.0, 1.0, 5.5] + vector![3.7, 1.7, 4.4, 1.2, 5.5];
    assert_eq!(b, vector![6.7, 2.7, 8.4, 2.2, 11.0]);

    let c = vector![3, 1, 4, 1, 5] + 2;
    assert_eq!(c, vector![5, 3, 6, 3, 7]);

    let d = vector![3.7, 1.7, 4.4, 1.2, 5.5] + 2.0;
    assert_eq!(d, vector![5.7, 3.7, 6.4, 3.2, 7.5]);

    let e = 2 + vector![3, 1, 4, 1, 5];
    assert_eq!(e, vector![5, 3, 6, 3, 7]);

    let f = 2.0 + vector![3.7, 1.7, 4.4, 1.2, 5.5];
    assert_eq!(f, vector![5.7, 3.7, 6.4, 3.2, 7.5]);
}

#[test]
fn test_add_assign() {
    let mut a = vector![3, 1, 4, 1, 5];
    a += vector![3, 1, 4, 1, 5];
    assert_eq!(a, vector![6, 2, 8, 2, 10]);

    let mut b = vector![3.0, 1.0, 4.0, 1.0, 5.5];
    b += vector![3.7, 1.7, 4.4, 1.2, 5.5];
    assert_eq!(b, vector![6.7, 2.7, 8.4, 2.2, 11.0]);

    let mut c = vector![3, 1, 4, 1, 5];
    c += 2;
    assert_eq!(c, vector![5, 3, 6, 3, 7]);

    let mut d = vector![3.7, 1.7, 4.4, 1.2, 5.5];
    d += 2.0;
    assert_eq!(d, vector![5.7, 3.7, 6.4, 3.2, 7.5]);
}

#[test]
#[should_panic]
fn test_add_invalid() {
    let _a = vector![3, 1, 4, 1, 5] + vector![3, 1, 4, 1];
}

#[test]
fn test_sub() {
    let a = vector![3, 1, 4, 1, 5] - vector![3, 1, 4, 1, 5];
    assert_eq!(a, vector![0, 0, 0, 0, 00]);

    let b =
        vector![3.0, 1.0, 4.0, 1.0, 5.5] - vector![3.7, 1.7, 4.4, 1.2, 5.5];
    assert_eq!(
        b,
        vector![
            -0.7000000000000002,
            -0.7,
            -0.40000000000000036,
            -0.19999999999999996,
            0.0
        ]
    );

    let c = vector![3, 1, 4, 1, 5] - 2;
    assert_eq!(c, vector![1, -1, 2, -1, 3]);

    let d = vector![3.7, 1.7, 4.4, 1.2, 5.5] - 2.0;
    assert_eq!(
        d,
        vector![
            1.7000000000000002,
            -0.30000000000000004,
            2.4000000000000004,
            -0.8,
            3.5
        ]
    );

    let e = 2 - vector![3, 1, 4, 1, 5];
    assert_eq!(e, vector![-1, 1, -2, 1, -3]);

    let f = 2.0 - vector![3.7, 1.7, 4.4, 1.2, 5.5];
    assert_eq!(
        f,
        vector![
            -1.7000000000000002,
            0.30000000000000004,
            -2.4000000000000004,
            0.8,
            -3.5
        ]
    );
}

#[test]
fn test_sub_assign() {
    let mut a = vector![3, 1, 4, 1, 5];
    a -= vector![3, 1, 4, 1, 5];
    assert_eq!(a, vector![0, 0, 0, 0, 00]);

    let mut b = vector![3.0, 1.0, 4.0, 1.0, 5.5];
    b -= vector![3.7, 1.7, 4.4, 1.2, 5.5];
    assert_eq!(
        b,
        vector![
            -0.7000000000000002,
            -0.7,
            -0.40000000000000036,
            -0.19999999999999996,
            0.0
        ]
    );

    let mut c = vector![3, 1, 4, 1, 5];
    c -= 2;
    assert_eq!(c, vector![1, -1, 2, -1, 3]);

    let mut d = vector![3.7, 1.7, 4.4, 1.2, 5.5];
    d -= 2.0;
    assert_eq!(
        d,
        vector![
            1.7000000000000002,
            -0.30000000000000004,
            2.4000000000000004,
            -0.8,
            3.5
        ]
    );
}

#[test]
#[should_panic]
fn test_sub_invalid() {
    let _a = vector![3, 1, 4, 1, 5] - vector![3, 1, 4, 1];
}

#[test]
fn test_mul() {
    let a = vector![3, 1, 4, 1, 5] * vector![3, 1, 4, 1, 5];
    assert_eq!(a, vector![9, 1, 16, 1, 25]);

    let b =
        vector![3.0, 1.0, 4.0, 1.0, 5.5] * vector![3.7, 1.7, 4.4, 1.2, 5.5];
    assert_eq!(b, vector![11.100000000000001, 1.7, 17.6, 1.2, 30.25]);

    let c = vector![3, 1, 4, 1, 5] * 2;
    assert_eq!(c, vector![6, 2, 8, 2, 10]);

    let d = vector![3.7, 1.7, 4.4, 1.2, 5.5] * 2.0;
    assert_eq!(d, vector![7.4, 3.4, 8.8, 2.4, 11.0]);

    let e = 2 * vector![3, 1, 4, 1, 5];
    assert_eq!(e, vector![6, 2, 8, 2, 10]);

    let f = 2.0 * vector![3.7, 1.7, 4.4, 1.2, 5.5];
    assert_eq!(f, vector![7.4, 3.4, 8.8, 2.4, 11.0]);
}

#[test]
fn test_mul_assign() {
    let mut a = vector![3, 1, 4, 1, 5];
    a *= vector![3, 1, 4, 1, 5];
    assert_eq!(a, vector![9, 1, 16, 1, 25]);

    let mut b = vector![3.0, 1.0, 4.0, 1.0, 5.5];
    b *= vector![3.7, 1.7, 4.4, 1.2, 5.5];
    assert_eq!(b, vector![11.100000000000001, 1.7, 17.6, 1.2, 30.25]);

    let mut c = vector![3, 1, 4, 1, 5];
    c *= 2;
    assert_eq!(c, vector![6, 2, 8, 2, 10]);

    let mut d = vector![3.7, 1.7, 4.4, 1.2, 5.5];
    d *= 2.0;
    assert_eq!(d, vector![7.4, 3.4, 8.8, 2.4, 11.0]);
}

#[test]
#[should_panic]
fn test_mul_invalid() {
    let _x = vector![1, 2] * vector![2];
}
