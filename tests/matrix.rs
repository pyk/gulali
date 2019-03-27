use crabsformer::*;

#[test]
fn test_macro() {
    let a = matrix![0; [12, 10]];
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

    let f = matrix![2.0, 1.0, 4.0,];
    assert_eq!(f.shape(), [1, 3]);
}

#[test]
#[should_panic]
fn test_invalid_macro() {
    let _x = matrix![2, 2; 2, 2, 2];
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
fn test_power() {
    let w1 = matrix![3, 1; 4, 1];
    let w2 = w1.power(2);
    assert_eq!(w2, matrix![9, 1; 16, 1]);
}

#[test]
fn test_uniform() {
    let mf1: Matrix<f32> = Matrix::uniform([5, 5], 0.0, 1.0);
    for cols in mf1.into_iter() {
        for value in cols.into_iter() {
            assert!((0.0 <= value) && (value < 1.0));
        }
    }

    let mf2: Matrix<f64> = Matrix::uniform([5, 5], 0.0, 1.0);
    for cols in mf2.into_iter() {
        for value in cols.into_iter() {
            assert!((0.0 <= value) && (value < 1.0));
        }
    }

    let ms1: Matrix<usize> = Matrix::uniform([5, 5], 1, 10);
    for cols in ms1.into_iter() {
        for value in cols.into_iter() {
            assert!((1 <= value) && (value < 10));
        }
    }

    let mu1: Matrix<u8> = Matrix::uniform([5, 5], 1, 10);
    for cols in mu1.into_iter() {
        for value in cols.into_iter() {
            assert!((1 <= value) && (value < 10));
        }
    }

    let mu2: Matrix<u16> = Matrix::uniform([5, 5], 1, 10);
    for cols in mu2.into_iter() {
        for value in cols.into_iter() {
            assert!((1 <= value) && (value < 10));
        }
    }

    let mu3: Matrix<u32> = Matrix::uniform([5, 5], 1, 10);
    for cols in mu3.into_iter() {
        for value in cols.into_iter() {
            assert!((1 <= value) && (value < 10));
        }
    }

    let mu4: Matrix<u64> = Matrix::uniform([5, 5], 1, 10);
    for cols in mu4.into_iter() {
        for value in cols.into_iter() {
            assert!((1 <= value) && (value < 10));
        }
    }

    let mu5: Matrix<u128> = Matrix::uniform([5, 5], 1, 10);
    for cols in mu5.into_iter() {
        for value in cols.into_iter() {
            assert!((1 <= value) && (value < 10));
        }
    }

    let mi1: Matrix<i8> = Matrix::uniform([5, 5], -10, 10);
    for cols in mi1.into_iter() {
        for value in cols.into_iter() {
            assert!((-10 <= value) && (value < 10));
        }
    }

    let mi2: Matrix<i16> = Matrix::uniform([5, 5], -10, 10);
    for cols in mi2.into_iter() {
        for value in cols.into_iter() {
            assert!((-10 <= value) && (value < 10));
        }
    }

    let mi3: Matrix<i32> = Matrix::uniform([5, 5], -10, 10);
    for cols in mi3.into_iter() {
        for value in cols.into_iter() {
            assert!((-10 <= value) && (value < 10));
        }
    }

    let mi4: Matrix<i64> = Matrix::uniform([5, 5], -10, 10);
    for cols in mi4.into_iter() {
        for value in cols.into_iter() {
            assert!((-10 <= value) && (value < 10));
        }
    }

    let mi5: Matrix<i128> = Matrix::uniform([5, 5], -10, 10);
    for cols in mi5.into_iter() {
        for value in cols.into_iter() {
            assert!((-10 <= value) && (value < 10));
        }
    }
}

#[test]
fn test_normal() {
    let a = Matrix::normal([5, 5], 2.0, 4.0);
    let b = Matrix::normal([5, 5], 2.0, 4.0);
    assert_eq!(a.shape(), b.shape());
    assert_ne!(a, b);
}

#[test]
fn test_indexing() {
    let w = matrix![
        3, 1, 4;
        1, 5, 9;
    ];
    assert_eq!(w[0][0], 3);
    assert_eq!(w[0][1], 1);
    assert_eq!(w[0][2], 4);
    assert_eq!(w[1][0], 1);
    assert_eq!(w[1][1], 5);
    assert_eq!(w[1][2], 9);

    assert_eq!(w[0], vector![3, 1, 4]);
    assert_eq!(w[1], vector![1, 5, 9]);
}

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
    let _x = matrix![1; [4, 4]] * matrix![2; [3, 3]];
}
