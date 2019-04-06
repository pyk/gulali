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
fn test_macro() {
    let a = matrix![[12, 10] => 0];
    assert_eq!(a.shape(), [12, 10]);
    assert_eq!(a, Matrix::full([12, 10], 0));

    let b = matrix![
        2.0, 1.0, 4.0;
        2.0, 4.0, 2.0
    ];
    assert_eq!(b.shape(), [2, 3]);

    let c = matrix![
        2.0, 1.0, 4.0;
        2.0, 4.0, 2.0;
    ];
    assert_eq!(c.shape(), [2, 3]);

    let d = matrix![2.0, 1.0, 4.0];
    assert_eq!(d.shape(), [1, 3]);

    let e = matrix![2.0, 1.0, 4.0;];
    assert_eq!(e.shape(), [1, 3]);

    let f = matrix![2.0, 1.0, 4.0];
    assert_eq!(f.shape(), [1, 3]);
}

#[test]
#[should_panic]
fn test_invalid_macro() {
    let _x = matrix![2, 2; 2, 2, 2];
}

#[test]
fn test_from() {
    let x = vec![vec![1, 2, 3], vec![1, 2, 3]];
    let m = Matrix::from(x);
    assert_eq!(m.shape(), [2, 3]);
}

#[test]
fn test_copy() {
    let x = vector![1, 2, 3, 4];
    let y = Vector::copy(&x);
    assert_eq!(x, y);
}

#[test]
fn test_full() {
    let a = Matrix::full([2, 2], 5.0);
    assert_eq!(a, matrix![5.0, 5.0; 5.0, 5.0]);

    let b = Matrix::full([2, 2], 2);
    assert_eq!(b, matrix![2, 2; 2, 2]);
}

#[test]
fn test_full_like() {
    let a = matrix![3.0, 4.0; 4.0, 5.0];
    let b = Matrix::full_like(&a, 5.0);
    assert_eq!(b, matrix![5.0, 5.0; 5.0, 5.0]);
}

#[test]
fn test_zeros() {
    let vf1: Matrix<f64> = Matrix::zeros([2, 2]);
    assert_eq!(vf1, matrix![0.0, 0.0; 0.0, 0.0]);

    let vf2: Matrix<f32> = Matrix::zeros([2, 2]);
    assert_eq!(vf2, matrix![0.0, 0.0; 0.0, 0.0]);

    let ms1: Matrix<usize> = Matrix::zeros([2, 2]);
    assert_eq!(ms1, matrix![0, 0; 0, 0]);

    let mu1: Matrix<u8> = Matrix::zeros([2, 2]);
    assert_eq!(mu1, matrix![0, 0; 0, 0]);

    let mu2: Matrix<u16> = Matrix::zeros([2, 2]);
    assert_eq!(mu2, matrix![0, 0; 0, 0]);

    let mu3: Matrix<u32> = Matrix::zeros([2, 2]);
    assert_eq!(mu3, matrix![0, 0; 0, 0]);

    let mu4: Matrix<u64> = Matrix::zeros([2, 2]);
    assert_eq!(mu4, matrix![0, 0; 0, 0]);

    let mu5: Matrix<u128> = Matrix::zeros([2, 2]);
    assert_eq!(mu5, matrix![0, 0; 0, 0]);

    let vi1: Matrix<i8> = Matrix::zeros([2, 2]);
    assert_eq!(vi1, matrix![0, 0; 0, 0]);

    let vi2: Matrix<i16> = Matrix::zeros([2, 2]);
    assert_eq!(vi2, matrix![0, 0; 0, 0]);

    let vi3: Matrix<i32> = Matrix::zeros([2, 2]);
    assert_eq!(vi3, matrix![0, 0; 0, 0]);

    let vi4: Matrix<i64> = Matrix::zeros([2, 2]);
    assert_eq!(vi4, matrix![0, 0; 0, 0]);

    let vi5: Matrix<i128> = Matrix::zeros([2, 2]);
    assert_eq!(vi5, matrix![0, 0; 0, 0]);
}

#[test]
fn test_zeros_like() {
    let mi1: Matrix<i32> = Matrix::ones([2, 2]);
    let mi2 = Matrix::zeros_like(&mi1);
    assert_eq!(mi2, matrix![0, 0; 0, 0]);
}

#[test]
fn test_ones() {
    let mf1: Matrix<f64> = Matrix::ones([2, 2]);
    assert_eq!(mf1, matrix![1.0, 1.0; 1.0, 1.0]);

    let mf2: Matrix<f32> = Matrix::ones([2, 2]);
    assert_eq!(mf2, matrix![1.0, 1.0; 1.0, 1.0]);

    let ms1: Matrix<usize> = Matrix::ones([2, 2]);
    assert_eq!(ms1, matrix![1, 1; 1, 1]);

    let mu1: Matrix<u8> = Matrix::ones([2, 2]);
    assert_eq!(mu1, matrix![1, 1; 1, 1]);

    let mu2: Matrix<u16> = Matrix::ones([2, 2]);
    assert_eq!(mu2, matrix![1, 1; 1, 1]);

    let mu3: Matrix<u32> = Matrix::ones([2, 2]);
    assert_eq!(mu3, matrix![1, 1; 1, 1]);

    let mu4: Matrix<u64> = Matrix::ones([2, 2]);
    assert_eq!(mu4, matrix![1, 1; 1, 1]);

    let mu5: Matrix<u128> = Matrix::ones([2, 2]);
    assert_eq!(mu5, matrix![1, 1; 1, 1]);

    let mi1: Matrix<i8> = Matrix::ones([2, 2]);
    assert_eq!(mi1, matrix![1, 1; 1, 1]);

    let mi2: Matrix<i16> = Matrix::ones([2, 2]);
    assert_eq!(mi2, matrix![1, 1; 1, 1]);

    let mi3: Matrix<i32> = Matrix::ones([2, 2]);
    assert_eq!(mi3, matrix![1, 1; 1, 1]);

    let mi4: Matrix<i64> = Matrix::ones([2, 2]);
    assert_eq!(mi4, matrix![1, 1; 1, 1]);

    let mi5: Matrix<i128> = Matrix::ones([2, 2]);
    assert_eq!(mi5, matrix![1, 1; 1, 1]);
}

#[test]
fn test_ones_like() {
    let mi1: Matrix<i32> = Matrix::ones([2, 2]);
    let mi2 = Matrix::ones_like(&mi1);
    assert_eq!(mi2, matrix![1, 1; 1, 1]);
}

#[test]
fn test_uniform() {
    let mf1: Matrix<f32> = Matrix::uniform([5, 5], 0.0, 1.0).unwrap();
    for rows in mf1.rows() {
        for value in rows.elements() {
            assert!((0.0 <= value) && (value < 1.0));
        }
    }

    let mf2: Matrix<f64> = Matrix::uniform([5, 5], 0.0, 1.0).unwrap();
    for rows in mf2.rows() {
        for value in rows.elements() {
            assert!((0.0 <= value) && (value < 1.0));
        }
    }

    let ms1: Matrix<usize> = Matrix::uniform([5, 5], 1, 10).unwrap();
    for rows in ms1.rows() {
        for value in rows.elements() {
            assert!((1 <= value) && (value < 10));
        }
    }

    let mu1: Matrix<u8> = Matrix::uniform([5, 5], 1, 10).unwrap();
    for rows in mu1.rows() {
        for value in rows.elements() {
            assert!((1 <= value) && (value < 10));
        }
    }

    let mu2: Matrix<u16> = Matrix::uniform([5, 5], 1, 10).unwrap();
    for rows in mu2.rows() {
        for value in rows.elements() {
            assert!((1 <= value) && (value < 10));
        }
    }

    let mu3: Matrix<u32> = Matrix::uniform([5, 5], 1, 10).unwrap();
    for rows in mu3.rows() {
        for value in rows.elements() {
            assert!((1 <= value) && (value < 10));
        }
    }

    let mu4: Matrix<u64> = Matrix::uniform([5, 5], 1, 10).unwrap();
    for rows in mu4.rows() {
        for value in rows.elements() {
            assert!((1 <= value) && (value < 10));
        }
    }

    let mu5: Matrix<u128> = Matrix::uniform([5, 5], 1, 10).unwrap();
    for rows in mu5.rows() {
        for value in rows.elements() {
            assert!((1 <= value) && (value < 10));
        }
    }

    let mi1: Matrix<i8> = Matrix::uniform([5, 5], -10, 10).unwrap();
    for rows in mi1.rows() {
        for value in rows.elements() {
            assert!((-10 <= value) && (value < 10));
        }
    }

    let mi2: Matrix<i16> = Matrix::uniform([5, 5], -10, 10).unwrap();
    for rows in mi2.rows() {
        for value in rows.elements() {
            assert!((-10 <= value) && (value < 10));
        }
    }

    let mi3: Matrix<i32> = Matrix::uniform([5, 5], -10, 10).unwrap();
    for rows in mi3.rows() {
        for value in rows.elements() {
            assert!((-10 <= value) && (value < 10));
        }
    }

    let mi4: Matrix<i64> = Matrix::uniform([5, 5], -10, 10).unwrap();
    for rows in mi4.rows() {
        for value in rows.elements() {
            assert!((-10 <= value) && (value < 10));
        }
    }

    let mi5: Matrix<i128> = Matrix::uniform([5, 5], -10, 10).unwrap();
    for rows in mi5.rows() {
        for value in rows.elements() {
            assert!((-10 <= value) && (value < 10));
        }
    }
}

#[test]
fn test_uniform_interval() {
    // low < high
    let x1 = Matrix::uniform([5, 5], -10, 10);
    assert_eq!(x1.is_ok(), true);

    // low = high
    let x2 = Matrix::uniform([5, 5], 10, 10);
    assert_eq!(x2.is_err(), true);

    // low > high
    let x3 = Matrix::uniform([5, 5], 10, -10);
    assert_eq!(x3.is_err(), true);
}

#[test]
fn test_normal() {
    let a = Matrix::normal([5, 5], 2.0, 4.0);
    let b = Matrix::normal([5, 5], 2.0, 4.0);
    assert_eq!(a.shape(), b.shape());
    assert_ne!(a, b);
}
