use crabsformer::*;

#[test]
fn test_from() {
    let a = vec![1, 2, 3];
    let b = Vector::from(a);
    assert_eq!(b, vector![1, 2, 3]);
}

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
fn test_power() {
    let x = vector![3, 1, 4, 1];
    let y = x.power(2);
    assert_eq!(y, vector![9, 1, 16, 1]);
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
    assert_eq!(x.max(), 4);

    // let y = vector![3.0, 1.0, 4.0, 1.0];
    // assert_eq!(y.max(), 4.0);
}

#[test]
fn test_min() {
    let x = vector![3, 1, 4, 1];
    assert_eq!(x.min(), 1);

    // let y = vector![3.0, 1.0, 4.0, 1.0];
    // assert_eq!(y.min(), 1.0);
}

#[test]
fn test_uniform() {
    let vf1: Vector<f32> = Vector::uniform(5, 0.0, 1.0);
    for value in vf1.into_iter() {
        assert!((0.0 <= value) && (value < 1.0));
    }

    let vf2: Vector<f64> = Vector::uniform(5, 0.0, 1.0);
    for value in vf2.into_iter() {
        assert!((0.0 <= value) && (value < 1.0));
    }

    let vs1: Vector<usize> = Vector::uniform(5, 1, 10);
    for value in vs1.into_iter() {
        assert!((1 <= value) && (value < 10));
    }

    let vu1: Vector<u8> = Vector::uniform(5, 1, 10);
    for value in vu1.into_iter() {
        assert!((1 <= value) && (value < 10));
    }

    let vu2: Vector<u16> = Vector::uniform(5, 1, 10);
    for value in vu2.into_iter() {
        assert!((1 <= value) && (value < 10));
    }

    let vu3: Vector<u32> = Vector::uniform(5, 1, 10);
    for value in vu3.into_iter() {
        assert!((1 <= value) && (value < 10));
    }

    let vu4: Vector<u64> = Vector::uniform(5, 1, 10);
    for value in vu4.into_iter() {
        assert!((1 <= value) && (value < 10));
    }

    let vu5: Vector<u128> = Vector::uniform(5, 1, 10);
    for value in vu5.into_iter() {
        assert!((1 <= value) && (value < 10));
    }

    let vi1: Vector<i8> = Vector::uniform(5, -10, 10);
    for value in vi1.into_iter() {
        assert!((-10 <= value) && (value < 10));
    }

    let vi2: Vector<i16> = Vector::uniform(5, -10, 10);
    for value in vi2.into_iter() {
        assert!((-10 <= value) && (value < 10));
    }

    let vi3: Vector<i32> = Vector::uniform(5, -10, 10);
    for value in vi3.into_iter() {
        assert!((-10 <= value) && (value < 10));
    }

    let vi4: Vector<i64> = Vector::uniform(5, -10, 10);
    for value in vi4.into_iter() {
        assert!((-10 <= value) && (value < 10));
    }

    let vi5: Vector<i128> = Vector::uniform(5, -10, 10);
    for value in vi5.into_iter() {
        assert!((-10 <= value) && (value < 10));
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

#[test]
fn test_indexing() {
    let a = vector![3, 1, 4, 1, 5];
    assert_eq!(a[0], 3);
    assert_eq!(a[1], 1);
    assert_eq!(a[2], 4);
    assert_eq!(a[3], 1);
    assert_eq!(a[4], 5);
}

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
fn test_clone() {
    // Test clone
    let a = vector![3, 1, 4];
    let b = a.clone();
    assert_eq!(a, b);
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

#[test]
fn test_index() {
    let x = vector![3, 1, 2, 3];
    assert_eq!(x[0], 3);
    assert_eq!(x[1], 1);
    assert_eq!(x[2], 2);
    assert_eq!(x[3], 3);
}

#[test]
#[should_panic]
fn test_invalid_index() {
    let x = vector![3, 1, 2, 3];
    x[12];
}

#[test]
fn test_slice() {
    let x = vector![3, 1, 2, 3];

    // Range
    assert_eq!(x.slice(0..1), vector![3]);

    // RangeTo
    assert_eq!(x.slice(..2), vector![3, 1]);

    // RangeFrom
    assert_eq!(x.slice(2..), vector![2, 3]);

    // RangeFull
    assert_eq!(x.slice(..), vector![3, 1, 2, 3]);

    // RangeInclusive
    assert_eq!(x.slice(0..=1), vector![3, 1]);

    // RangeToInclusive
    assert_eq!(x.slice(..=2), vector![3, 1, 2]);
}

#[test]
#[should_panic]
fn test_invalid_slice() {
    let x = vector![3, 1, 2, 3];
    x.slice(1..100);
}

#[test]
fn test_iteration() {
    let x = vector![1, 2, 3, 5];
    for value in x.into_iter() {
        let _a = value;
    }
}
