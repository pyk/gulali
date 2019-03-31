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
    let w1 = matrix![3, 1; 4, 1];
    let w2 = w1.power(2);
    assert_eq!(w2, matrix![9, 1; 16, 1]);
}

#[test]
fn test_power_mut() {
    let mut w1 = matrix![3, 1; 4, 1];
    w1.power_mut(2);
    assert_eq!(w1, matrix![9, 1; 16, 1]);
}

// Binary operations
#[test]
fn test_add() {
    let a = matrix![3, 1; 4, 1] + matrix![3, 1; 4, 1];
    assert_eq!(a, matrix![6, 2; 8, 2]);

    let b = matrix![3.0, 1.0; 4.0, 1.0] + matrix![3.7, 1.7; 4.4, 1.2];
    assert_eq!(b, matrix![6.7, 2.7; 8.4, 2.2]);

    let c = matrix![3, 1; 4, 1] + 2;
    assert_eq!(c, matrix![5, 3; 6, 3]);

    let d = matrix![3.7, 1.7; 4.4, 1.2] + 2.0;
    assert_eq!(d, matrix![5.7, 3.7; 6.4, 3.2]);

    let e = 2 + matrix![3, 1; 4, 1];
    assert_eq!(e, matrix![5, 3; 6, 3]);

    let f = 2.0 + matrix![3.7, 1.7; 4.4, 1.2];
    assert_eq!(f, matrix![5.7, 3.7; 6.4, 3.2]);
}

#[test]
fn test_add_assign() {
    let mut a = matrix![3, 1; 4, 1];
    a += matrix![3, 1; 4, 1];
    assert_eq!(a, matrix![6, 2; 8, 2]);

    let mut b = matrix![3.0, 1.0; 4.0, 1.0];
    b += matrix![3.7, 1.7; 4.4, 1.2];
    assert_eq!(b, matrix![6.7, 2.7; 8.4, 2.2]);

    let mut c = matrix![3, 1; 4, 1];
    c += 2;
    assert_eq!(c, matrix![5, 3; 6, 3]);

    let mut d = matrix![3.7, 1.7; 4.4, 1.2];
    d += 2.0;
    assert_eq!(d, matrix![5.7, 3.7; 6.4, 3.2]);
}

#[test]
#[should_panic]
fn test_add_invalid() {
    let _a = matrix![3, 1, 4; 1, 5, 5] + matrix![3, 1; 4, 1];
}

#[test]
fn test_sub() {
    let a = matrix![3, 1; 4, 1] - matrix![3, 1; 4, 1];
    assert_eq!(a, matrix![0, 0; 0, 0]);

    let b = matrix![3.0, 1.0; 4.0, 1.0] - matrix![3.7, 1.7; 4.4, 1.2];
    assert_eq!(
        b,
        matrix![
            -0.7000000000000002, -0.7;
            -0.40000000000000036, -0.19999999999999996;
        ]
    );

    let c = matrix![3, 1; 4, 1] - 2;
    assert_eq!(c, matrix![1, -1; 2, -1]);

    let d = matrix![3.7, 1.7; 4.4, 1.2] - 2.0;
    assert_eq!(
        d,
        matrix![
            1.7000000000000002, -0.30000000000000004;
            2.4000000000000004, -0.8;
        ]
    );

    let e = 2 - matrix![3, 1; 4, 1];
    assert_eq!(e, matrix![-1, 1; -2, 1]);

    let f = 2.0 - matrix![3.7, 1.7; 4.4, 1.2];
    assert_eq!(
        f,
        matrix![
            -1.7000000000000002, 0.30000000000000004;
            -2.4000000000000004, 0.8;
        ]
    );
}

#[test]
fn test_sub_assign() {
    let mut a = matrix![3, 1; 4, 1];
    a -= matrix![3, 1; 4, 1];
    assert_eq!(a, matrix![0, 0; 0, 0]);

    let mut b = matrix![3.0, 1.0; 4.0, 1.0];
    b -= matrix![3.7, 1.7; 4.4, 1.2];
    assert_eq!(
        b,
        matrix![
            -0.7000000000000002, -0.7;
            -0.40000000000000036, -0.19999999999999996;
        ]
    );

    let mut c = matrix![3, 1; 4, 1];
    c -= 2;
    assert_eq!(c, matrix![1, -1; 2, -1]);

    let mut d = matrix![3.7, 1.7; 4.4, 1.2];
    d -= 2.0;
    assert_eq!(
        d,
        matrix![
            1.7000000000000002, -0.30000000000000004;
            2.4000000000000004, -0.8;
        ]
    );
}

#[test]
#[should_panic]
fn test_sub_invalid() {
    let _a = matrix![3, 1, 4; 1, 5, 5] - matrix![3, 1; 4, 1];
}

#[test]
fn test_mul() {
    let a = matrix![3, 1; 4, 1] * matrix![3, 1; 4, 1];
    assert_eq!(a, matrix![9, 1; 16, 1]);

    let b = matrix![3.0, 1.0; 4.0, 1.0] * matrix![3.7, 1.7; 4.4, 1.2];
    assert_eq!(b, matrix![11.100000000000001, 1.7; 17.6, 1.2]);

    let c = matrix![3, 1; 4, 1] * 2;
    assert_eq!(c, matrix![6, 2; 8, 2]);

    let d = matrix![3.7, 1.7; 4.4, 1.2] * 2.0;
    assert_eq!(d, matrix![7.4, 3.4; 8.8, 2.4]);

    let e = 2 * matrix![3, 1; 4, 1];
    assert_eq!(e, matrix![6, 2; 8, 2]);

    let f = 2.0 * matrix![3.7, 1.7; 4.4, 1.2];
    assert_eq!(f, matrix![7.4, 3.4; 8.8, 2.4]);
}

#[test]
fn test_mul_assign() {
    let mut a = matrix![3, 1; 4, 1];
    a *= matrix![3, 1; 4, 1];
    assert_eq!(a, matrix![9, 1; 16, 1]);

    let mut b = matrix![3.0, 1.0; 4.0, 1.0];
    b *= matrix![3.7, 1.7; 4.4, 1.2];
    assert_eq!(b, matrix![11.100000000000001, 1.7; 17.6, 1.2]);

    let mut c = matrix![3, 1; 4, 1];
    c *= 2;
    assert_eq!(c, matrix![6, 2; 8, 2]);

    let mut d = matrix![3.7, 1.7; 4.4, 1.2];
    d *= 2.0;
    assert_eq!(d, matrix![7.4, 3.4; 8.8, 2.4]);
}

#[test]
#[should_panic]
fn test_mul_invalid() {
    let _x = matrix![[4, 4] => 1] * matrix![[3, 3] => 2];
}
