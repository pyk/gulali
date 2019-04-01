use crabsformer::*;

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
