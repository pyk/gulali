//! `np` is an easy-to-use fundamental library for scientific computing with
//! Rust, highly inspired by [NumPy].
//!
//! [NumPy]: http://www.numpy.org/
//!
//!
//! ## Usage
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! np = "2019.3.1"
//! ```
//!
//! To get started using `np`, read the quickstart tutorial below.
//!
//! ## Quickstart Tutorial
//! ### Prerequisites
//! Before reading this quick tutorial you should know a bit of Rust.
//! If you would like to refresh your memory, take a look at the
//! [Rust book].
//!
//! [Rust book]: https://doc.rust-lang.org/book/
//!
//! ### The Basics
//! np's main data type is the homogeneous multidimensional [vector].
//! It is a table of elements (usually numbers), all of the same type,
//! indexed by a tuple of positive integers.
//!
//! [vector]: https://doc.rust-lang.org/std/vec/struct.Vec.html
//!
//! For example, the coordinates of a point in 3D space `[1, 2, 1]` has
//! one dimension. That dimension has 3 elements in it, so we say
//! it has a length of 3. In the example pictured below,
//! the vector has 2 dimensions. The first dimension has a length of 2,
//! the second dimension has a length of 3.
//!
//! ```ignore
//! [
//!     [ 1., 0., 0.],
//!     [ 0., 1., 2.]
//! ]
//! ```
//!
//! np uses Rust's [vector] standard data type extensively. We don't reinvent
//! yet-another data type to keep things simple and easy to use.
//! np added useful attributes for Rust's vector like
//! the following:
//!
//! - [dim]: the number of dimensions of the vector.
//! - [shape]: This is a list of integers indicating the
//!   size of the vector in each dimension.
//!   For a matrix with `n` rows and `m` columns, shape will be `[n,m]`.
//!   The length of the shape is therefore the number of
//!   dimensions, `dim()`.
//! - [size]: the total number of elements of the vector.
//!   This is equal to the product of the elements of shape.
//!
//! [slice]: https://doc.rust-lang.org/rust-by-example/primitives/array.html
//! [dim]: trait.Dimension.html
//! [shape]: trait.Shape.html
//! [size]: trait.Size.html
//!
//! ### An Example
//! ```rust
//! # use np::*;
//! // Create two-dimensional vector with shape [3, 3]
//! // filled with zeros
//! let matrix: Vec<Vec<i32>> = Vec::two_dim(3, 3).zeros();
//!
//! assert_eq!(matrix.dim(), 2);
//! assert_eq!(matrix.shape(), [3, 3]);
//! assert_eq!(matrix.size(), 9);
//! ```
//!
//! ## Getting help
//! Feel free to start discussion at [GitHub issues].
//!
//! [Github issues]: https://github.com/pyk/np/issues/new/choose
//!
//! ## License
//! `np` is licensed under the BSD 3-Clause license.
//!

pub fn zeros(shape: [usize; 2]) -> Vec<Vec<i64>> {
    let mut rows = Vec::with_capacity(shape[0]);
    for _ in 0..shape[0] {
        let mut cols = Vec::with_capacity(shape[1]);
        for _ in 0..shape[1] {
            cols.push(0);
        }
        rows.push(cols);
    }
    rows
}

/// One-dimensional vectors
pub trait OneDimensional<T>
where
    T: Copy,
{
    /// Returns new one-dimensional vector with specified `size`.
    // TODO(pyk): Add example
    fn one_dim(size: usize) -> Self;
}

impl<T> OneDimensional<T> for Vec<T>
where
    T: Copy,
{
    fn one_dim(size: usize) -> Vec<T> {
        Vec::with_capacity(size)
    }
}

/// Two-dimensional vectors
pub trait TwoDimensional<T>
where
    T: Copy,
{
    /// Returns new two-dimensional vector with specified
    /// number of rows and number of columns.
    ///
    // TODO(pyk): Add example
    fn two_dim(nrows: usize, ncols: usize) -> Self;
}

impl<T> TwoDimensional<T> for Vec<Vec<T>>
where
    T: Copy,
{
    fn two_dim(nrows: usize, ncols: usize) -> Vec<Vec<T>> {
        let mut array2d: Vec<Vec<T>> = Vec::with_capacity(nrows);
        for _ in 0..nrows {
            array2d.push(Vec::with_capacity(ncols));
        }
        array2d
    }
}

/// Three-dimensional vectors
pub trait ThreeDimensional<T>
where
    T: Copy,
{
    /// Returns new three-dimensional vector with specified shape.
    ///
    /// # Arguments
    ///
    /// * `n1` - A number of elements in axis 1.
    /// * `n2` - A number of elements in axis 2.
    /// * `n3` - A number of elements in axis 3.
    ///
    // TODO(pyk): Add example
    fn three_dim(n1: usize, n2: usize, n3: usize) -> Self;
}

impl<T> ThreeDimensional<T> for Vec<Vec<Vec<T>>>
where
    T: Copy,
{
    fn three_dim(n1: usize, n2: usize, n3: usize) -> Vec<Vec<Vec<T>>> {
        let mut array3d: Vec<Vec<Vec<T>>> = Vec::with_capacity(n1);
        for _ in 0..n1 {
            let mut array2d: Vec<Vec<T>> = Vec::with_capacity(n2);
            for _ in 0..n2 {
                array2d.push(Vec::with_capacity(n3));
            }
            array3d.push(array2d);
        }
        array3d
    }
}

/// Four-dimensional vectors
pub trait FourDimensional<T>
where
    T: Copy,
{
    /// Returns new four-dimensional vector with specified shape.
    ///
    /// # Arguments
    ///
    /// * `n1` - A number of elements in axis 1.
    /// * `n2` - A number of elements in axis 2.
    /// * `n3` - A number of elements in axis 3.
    /// * `n4` - A number of elements in axis 4.
    ///
    // TODO(pyk): Add example
    fn four_dim(n1: usize, n2: usize, n3: usize, n4: usize) -> Self;
}

impl<T> FourDimensional<T> for Vec<Vec<Vec<Vec<T>>>>
where
    T: Copy,
{
    fn four_dim(n1: usize, n2: usize, n3: usize, n4: usize) -> Vec<Vec<Vec<Vec<T>>>> {
        let mut array4d: Vec<Vec<Vec<Vec<T>>>> = Vec::with_capacity(n1);
        for _ in 0..n1 {
            let mut array3d: Vec<Vec<Vec<T>>> = Vec::with_capacity(n2);
            for _ in 0..n2 {
                let mut array2d: Vec<Vec<T>> = Vec::with_capacity(n3);
                for _ in 0..n3 {
                    array2d.push(Vec::with_capacity(n4));
                }
                array3d.push(array2d);
            }
            array4d.push(array3d)
        }
        array4d
    }
}

/// Dimension of the vector
pub trait Dimension<T>
where
    T: Copy,
{
    /// Returns the number of dimensions of the vector
    fn dim(&self) -> usize;
}

impl<T> Dimension<T> for Vec<T>
where
    T: Copy,
{
    fn dim(&self) -> usize {
        1
    }
}

impl<T> Dimension<T> for Vec<Vec<T>>
where
    T: Copy,
{
    fn dim(&self) -> usize {
        2
    }
}

impl<T> Dimension<T> for Vec<Vec<Vec<T>>>
where
    T: Copy,
{
    fn dim(&self) -> usize {
        3
    }
}

impl<T> Dimension<T> for Vec<Vec<Vec<Vec<T>>>>
where
    T: Copy,
{
    fn dim(&self) -> usize {
        4
    }
}

/// A list of integers indicating the size of the vector in each dimension
pub trait Shape<T>
where
    T: Copy,
{
    /// Returns a list of integers indicating the length of the
    /// vector in each dimension
    fn shape(&self) -> Vec<usize>;
}

impl<T> Shape<T> for Vec<T>
where
    T: Copy,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.len()]
    }
}

impl<T> Shape<T> for Vec<Vec<T>>
where
    T: Copy,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.len(), self[0].len()]
    }
}

impl<T> Shape<T> for Vec<Vec<Vec<T>>>
where
    T: Copy,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.len(), self[0].len(), self[0][0].len()]
    }
}

impl<T> Shape<T> for Vec<Vec<Vec<Vec<T>>>>
where
    T: Copy,
{
    fn shape(&self) -> Vec<usize> {
        vec![
            self.len(),
            self[0].len(),
            self[0][0].len(),
            self[0][0][0].len(),
        ]
    }
}

/// Total number of elements of the vector
pub trait Size<T>
where
    T: Copy,
{
    /// Returns the total number of elements of the vector.
    /// This is equal to the product of the elements of shape.
    fn size(&self) -> usize;
}

impl<T> Size<T> for Vec<T>
where
    T: Copy,
{
    fn size(&self) -> usize {
        self.shape().iter().product()
    }
}

impl<T> Size<T> for Vec<Vec<T>>
where
    T: Copy,
{
    fn size(&self) -> usize {
        let shape: Vec<usize> = self.shape();
        shape.iter().product()
    }
}

impl<T> Size<T> for Vec<Vec<Vec<T>>>
where
    T: Copy,
{
    fn size(&self) -> usize {
        self.shape().iter().product()
    }
}

impl<T> Size<T> for Vec<Vec<Vec<Vec<T>>>>
where
    T: Copy,
{
    fn size(&self) -> usize {
        self.shape().iter().product()
    }
}

/// Fillable vectors
pub trait Full<T>
where
    T: Copy,
{
    /// Returns a new array of given shape and type, filled with `value`.
    fn full(&mut self, value: T) -> Self;
}

impl<T> Full<T> for Vec<T>
where
    T: Copy,
{
    // Return a new 1D array of given shape and type, filled with `value`.
    fn full(&mut self, value: T) -> Vec<T> {
        for _ in 0..self.capacity() {
            self.push(value);
        }
        self.to_vec()
    }
}

impl<T> Full<T> for Vec<Vec<T>>
where
    T: Copy,
{
    // Return a new 2D array of given shape and type, filled with `value`.
    fn full(&mut self, value: T) -> Vec<Vec<T>> {
        for i in 0..self.capacity() {
            for _ in 0..self[i].capacity() {
                self[i].push(value);
            }
        }
        self.to_vec()
    }
}

impl<T> Full<T> for Vec<Vec<Vec<T>>>
where
    T: Copy,
{
    // Return a new 3D array of given shape and type, filled with `value`.
    fn full(&mut self, value: T) -> Vec<Vec<Vec<T>>> {
        for i in 0..self.capacity() {
            for j in 0..self[i].capacity() {
                for _ in 0..self[i][j].capacity() {
                    self[i][j].push(value);
                }
            }
        }
        self.to_vec()
    }
}

impl<T> Full<T> for Vec<Vec<Vec<Vec<T>>>>
where
    T: Copy,
{
    // Return a new 4D array of given shape and type, filled with `value`.
    fn full(&mut self, value: T) -> Vec<Vec<Vec<Vec<T>>>> {
        for i in 0..self.capacity() {
            for j in 0..self[i].capacity() {
                for k in 0..self[i][j].capacity() {
                    for _ in 0..self[i][j][k].capacity() {
                        self[i][j][k].push(value);
                    }
                }
            }
        }
        self.to_vec()
    }
}

/// A zero-able vectors
pub trait Zero {
    /// Return a new vector of given data type and shape,
    /// filled with zeros.
    fn zeros(&mut self) -> Self;
}

impl Zero for Vec<u8> {
    fn zeros(&mut self) -> Vec<u8> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<u8>> {
    fn zeros(&mut self) -> Vec<Vec<u8>> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<Vec<u8>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<u8>>> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<Vec<Vec<u8>>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<Vec<u8>>>> {
        self.full(0)
    }
}

impl Zero for Vec<u16> {
    fn zeros(&mut self) -> Vec<u16> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<u16>> {
    fn zeros(&mut self) -> Vec<Vec<u16>> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<Vec<u16>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<u16>>> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<Vec<Vec<u16>>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<Vec<u16>>>> {
        self.full(0)
    }
}

impl Zero for Vec<u32> {
    fn zeros(&mut self) -> Vec<u32> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<u32>> {
    fn zeros(&mut self) -> Vec<Vec<u32>> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<Vec<u32>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<u32>>> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<Vec<Vec<u32>>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<Vec<u32>>>> {
        self.full(0)
    }
}

impl Zero for Vec<u64> {
    fn zeros(&mut self) -> Vec<u64> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<u64>> {
    fn zeros(&mut self) -> Vec<Vec<u64>> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<Vec<u64>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<u64>>> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<Vec<Vec<u64>>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<Vec<u64>>>> {
        self.full(0)
    }
}

impl Zero for Vec<u128> {
    fn zeros(&mut self) -> Vec<u128> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<u128>> {
    fn zeros(&mut self) -> Vec<Vec<u128>> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<Vec<u128>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<u128>>> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<Vec<Vec<u128>>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<Vec<u128>>>> {
        self.full(0)
    }
}

impl Zero for Vec<i8> {
    fn zeros(&mut self) -> Vec<i8> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<i8>> {
    fn zeros(&mut self) -> Vec<Vec<i8>> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<Vec<i8>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<i8>>> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<Vec<Vec<i8>>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<Vec<i8>>>> {
        self.full(0)
    }
}

impl Zero for Vec<i16> {
    fn zeros(&mut self) -> Vec<i16> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<i16>> {
    fn zeros(&mut self) -> Vec<Vec<i16>> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<Vec<i16>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<i16>>> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<Vec<Vec<i16>>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<Vec<i16>>>> {
        self.full(0)
    }
}

impl Zero for Vec<i32> {
    fn zeros(&mut self) -> Vec<i32> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<i32>> {
    fn zeros(&mut self) -> Vec<Vec<i32>> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<Vec<i32>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<i32>>> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<Vec<Vec<i32>>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<Vec<i32>>>> {
        self.full(0)
    }
}

impl Zero for Vec<i64> {
    fn zeros(&mut self) -> Vec<i64> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<i64>> {
    fn zeros(&mut self) -> Vec<Vec<i64>> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<Vec<i64>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<i64>>> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<Vec<Vec<i64>>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<Vec<i64>>>> {
        self.full(0)
    }
}

impl Zero for Vec<i128> {
    fn zeros(&mut self) -> Vec<i128> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<i128>> {
    fn zeros(&mut self) -> Vec<Vec<i128>> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<Vec<i128>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<i128>>> {
        self.full(0)
    }
}

impl Zero for Vec<Vec<Vec<Vec<i128>>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<Vec<i128>>>> {
        self.full(0)
    }
}

impl Zero for Vec<f32> {
    fn zeros(&mut self) -> Vec<f32> {
        self.full(0.0)
    }
}

impl Zero for Vec<Vec<f32>> {
    fn zeros(&mut self) -> Vec<Vec<f32>> {
        self.full(0.0)
    }
}

impl Zero for Vec<Vec<Vec<f32>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<f32>>> {
        self.full(0.0)
    }
}

impl Zero for Vec<Vec<Vec<Vec<f32>>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<Vec<f32>>>> {
        self.full(0.0)
    }
}

impl Zero for Vec<f64> {
    fn zeros(&mut self) -> Vec<f64> {
        self.full(0.0)
    }
}

impl Zero for Vec<Vec<f64>> {
    fn zeros(&mut self) -> Vec<Vec<f64>> {
        self.full(0.0)
    }
}

impl Zero for Vec<Vec<Vec<f64>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<f64>>> {
        self.full(0.0)
    }
}

impl Zero for Vec<Vec<Vec<Vec<f64>>>> {
    fn zeros(&mut self) -> Vec<Vec<Vec<Vec<f64>>>> {
        self.full(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fn_zeros() {
        let arr2d: Vec<Vec<i64>> = zeros([2, 2]);
        assert_eq!(arr2d, [[0, 0], [0, 0],])
    }

    #[test]
    fn test_one_dim() {
        let arr: Vec<i64> = Vec::one_dim(5);
        assert_eq!(arr.capacity(), 5);
    }

    #[test]
    fn test_two_dim() {
        let arr: Vec<Vec<f64>> = Vec::two_dim(5, 5);
        assert_eq!(arr.capacity(), 5);
        for i in 0..5 {
            assert_eq!(arr[i].capacity(), 5);
        }
    }

    #[test]
    fn test_three_dim() {
        let arr: Vec<Vec<Vec<f64>>> = Vec::three_dim(5, 5, 5);
        assert_eq!(arr.capacity(), 5);
        for i in 0..5 {
            assert_eq!(arr[i].capacity(), 5);
            for j in 0..5 {
                assert_eq!(arr[i][j].capacity(), 5);
            }
        }
    }

    #[test]
    fn test_four_dim() {
        // TODO: Look at numpy array
        let arr: Vec<Vec<Vec<Vec<f64>>>> = Vec::four_dim(5, 5, 5, 5);
        assert_eq!(arr.capacity(), 5);
        for i in 0..5 {
            assert_eq!(arr[i].capacity(), 5);
            for j in 0..5 {
                assert_eq!(arr[i][j].capacity(), 5);
                for k in 0..5 {
                    assert_eq!(arr[i][j][k].capacity(), 5);
                }
            }
        }
    }

    #[test]
    fn test_full_one_dim() {
        // 1D array
        let arr1: Vec<i32> = Vec::one_dim(5).full(0);
        assert_eq!(arr1, [0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_full_two_dim() {
        let arr2: Vec<Vec<f64>> = Vec::two_dim(1, 2).full(5.0);
        assert_eq!(arr2, [[5.0, 5.0]]);
    }

    #[test]
    fn test_full_three_dim() {
        let arr3: Vec<Vec<Vec<f64>>> = Vec::three_dim(1, 1, 2).full(5.0);
        assert_eq!(arr3, [[[5.0, 5.0]]]);
    }

    #[test]
    fn test_full_four_dim() {
        let arr4: Vec<Vec<Vec<Vec<f64>>>> = Vec::four_dim(1, 1, 1, 2).full(5.0);
        assert_eq!(arr4, [[[[5.0, 5.0]]]]);
    }

    #[test]
    fn test_zeros_u8_one_dim() {
        let arr: Vec<u8> = Vec::one_dim(2).zeros();
        assert_eq!(arr, [0, 0]);
    }

    #[test]
    fn test_zeros_u8_two_dim() {
        let arr: Vec<Vec<u8>> = Vec::two_dim(1, 2).zeros();
        assert_eq!(arr, [[0, 0]]);
    }

    #[test]
    fn test_zeros_u8_three_dim() {
        let arr: Vec<Vec<Vec<u8>>> = Vec::three_dim(1, 1, 2).zeros();
        assert_eq!(arr, [[[0, 0]]]);
    }

    #[test]
    fn test_zeros_u8_four_dim() {
        let arr: Vec<Vec<Vec<Vec<u8>>>> = Vec::four_dim(1, 1, 1, 2).zeros();
        assert_eq!(arr, [[[[0, 0]]]]);
    }

    #[test]
    fn test_zeros_u16_one_dim() {
        let arr: Vec<u16> = Vec::one_dim(2).zeros();
        assert_eq!(arr, [0, 0]);
    }

    #[test]
    fn test_zeros_u16_two_dim() {
        let arr: Vec<Vec<u16>> = Vec::two_dim(1, 2).zeros();
        assert_eq!(arr, [[0, 0]]);
    }

    #[test]
    fn test_zeros_u16_three_dim() {
        let arr: Vec<Vec<Vec<u16>>> = Vec::three_dim(1, 1, 2).zeros();
        assert_eq!(arr, [[[0, 0]]]);
    }

    #[test]
    fn test_zeros_u16_four_dim() {
        let arr: Vec<Vec<Vec<Vec<u16>>>> = Vec::four_dim(1, 1, 1, 2).zeros();
        assert_eq!(arr, [[[[0, 0]]]]);
    }

    #[test]
    fn test_zeros_u32_one_dim() {
        let arr: Vec<u32> = Vec::one_dim(2).zeros();
        assert_eq!(arr, [0, 0]);
    }

    #[test]
    fn test_zeros_u32_two_dim() {
        let arr: Vec<Vec<u32>> = Vec::two_dim(1, 2).zeros();
        assert_eq!(arr, [[0, 0]]);
    }

    #[test]
    fn test_zeros_u32_three_dim() {
        let arr: Vec<Vec<Vec<u32>>> = Vec::three_dim(1, 1, 2).zeros();
        assert_eq!(arr, [[[0, 0]]]);
    }

    #[test]
    fn test_zeros_u32_four_dim() {
        let arr: Vec<Vec<Vec<Vec<u32>>>> = Vec::four_dim(1, 1, 1, 2).zeros();
        assert_eq!(arr, [[[[0, 0]]]]);
    }

    #[test]
    fn test_zeros_u64_one_dim() {
        let arr: Vec<u64> = Vec::one_dim(2).zeros();
        assert_eq!(arr, [0, 0]);
    }

    #[test]
    fn test_zeros_u64_two_dim() {
        let arr: Vec<Vec<u64>> = Vec::two_dim(1, 2).zeros();
        assert_eq!(arr, [[0, 0]]);
    }

    #[test]
    fn test_zeros_u64_three_dim() {
        let arr: Vec<Vec<Vec<u64>>> = Vec::three_dim(1, 1, 2).zeros();
        assert_eq!(arr, [[[0, 0]]]);
    }

    #[test]
    fn test_zeros_u64_four_dim() {
        let arr: Vec<Vec<Vec<Vec<u64>>>> = Vec::four_dim(1, 1, 1, 2).zeros();
        assert_eq!(arr, [[[[0, 0]]]]);
    }

    #[test]
    fn test_zeros_u128_one_dim() {
        let arr: Vec<u128> = Vec::one_dim(2).zeros();
        assert_eq!(arr, [0, 0]);
    }

    #[test]
    fn test_zeros_u128_two_dim() {
        let arr: Vec<Vec<u128>> = Vec::two_dim(1, 2).zeros();
        assert_eq!(arr, [[0, 0]]);
    }

    #[test]
    fn test_zeros_u128_three_dim() {
        let arr: Vec<Vec<Vec<u128>>> = Vec::three_dim(1, 1, 2).zeros();
        assert_eq!(arr, [[[0, 0]]]);
    }

    #[test]
    fn test_zeros_u128_four_dim() {
        let arr: Vec<Vec<Vec<Vec<u128>>>> = Vec::four_dim(1, 1, 1, 2).zeros();
        assert_eq!(arr, [[[[0, 0]]]]);
    }

    #[test]
    fn test_zeros_i8_one_dim() {
        let arr: Vec<i8> = Vec::one_dim(2).zeros();
        assert_eq!(arr, [0, 0]);
    }

    #[test]
    fn test_zeros_i8_two_dim() {
        let arr: Vec<Vec<i8>> = Vec::two_dim(1, 2).zeros();
        assert_eq!(arr, [[0, 0]]);
    }

    #[test]
    fn test_zeros_i8_three_dim() {
        let arr: Vec<Vec<Vec<i8>>> = Vec::three_dim(1, 1, 2).zeros();
        assert_eq!(arr, [[[0, 0]]]);
    }

    #[test]
    fn test_zeros_i8_four_dim() {
        let arr: Vec<Vec<Vec<Vec<i8>>>> = Vec::four_dim(1, 1, 1, 2).zeros();
        assert_eq!(arr, [[[[0, 0]]]]);
    }

    #[test]
    fn test_zeros_i16_one_dim() {
        let arr: Vec<i16> = Vec::one_dim(2).zeros();
        assert_eq!(arr, [0, 0]);
    }

    #[test]
    fn test_zeros_i16_two_dim() {
        let arr: Vec<Vec<i16>> = Vec::two_dim(1, 2).zeros();
        assert_eq!(arr, [[0, 0]]);
    }

    #[test]
    fn test_zeros_i16_three_dim() {
        let arr: Vec<Vec<Vec<i16>>> = Vec::three_dim(1, 1, 2).zeros();
        assert_eq!(arr, [[[0, 0]]]);
    }

    #[test]
    fn test_zeros_i16_four_dim() {
        let arr: Vec<Vec<Vec<Vec<i16>>>> = Vec::four_dim(1, 1, 1, 2).zeros();
        assert_eq!(arr, [[[[0, 0]]]]);
    }

    #[test]
    fn test_zeros_i32_one_dim() {
        let arr: Vec<i32> = Vec::one_dim(2).zeros();
        assert_eq!(arr, [0, 0]);
    }

    #[test]
    fn test_zeros_i32_two_dim() {
        let arr: Vec<Vec<i32>> = Vec::two_dim(1, 2).zeros();
        assert_eq!(arr, [[0, 0]]);
    }

    #[test]
    fn test_zeros_i32_three_dim() {
        let arr: Vec<Vec<Vec<i32>>> = Vec::three_dim(1, 1, 2).zeros();
        assert_eq!(arr, [[[0, 0]]]);
    }

    #[test]
    fn test_zeros_i32_four_dim() {
        let arr: Vec<Vec<Vec<Vec<i32>>>> = Vec::four_dim(1, 1, 1, 2).zeros();
        assert_eq!(arr, [[[[0, 0]]]]);
    }

    #[test]
    fn test_zeros_i64_one_dim() {
        let arr: Vec<i64> = Vec::one_dim(2).zeros();
        assert_eq!(arr, [0, 0]);
    }

    #[test]
    fn test_zeros_i64_two_dim() {
        let arr: Vec<Vec<i64>> = Vec::two_dim(1, 2).zeros();
        assert_eq!(arr, [[0, 0]]);
    }

    #[test]
    fn test_zeros_i64_three_dim() {
        let arr: Vec<Vec<Vec<i64>>> = Vec::three_dim(1, 1, 2).zeros();
        assert_eq!(arr, [[[0, 0]]]);
    }

    #[test]
    fn test_zeros_i64_four_dim() {
        let arr: Vec<Vec<Vec<Vec<i64>>>> = Vec::four_dim(1, 1, 1, 2).zeros();
        assert_eq!(arr, [[[[0, 0]]]]);
    }

    #[test]
    fn test_zeros_i128_one_dim() {
        let arr: Vec<i128> = Vec::one_dim(2).zeros();
        assert_eq!(arr, [0, 0]);
    }

    #[test]
    fn test_zeros_i128_two_dim() {
        let arr: Vec<Vec<i128>> = Vec::two_dim(1, 2).zeros();
        assert_eq!(arr, [[0, 0]]);
    }

    #[test]
    fn test_zeros_i128_three_dim() {
        let arr: Vec<Vec<Vec<i128>>> = Vec::three_dim(1, 1, 2).zeros();
        assert_eq!(arr, [[[0, 0]]]);
    }

    #[test]
    fn test_zeros_i128_four_dim() {
        let arr: Vec<Vec<Vec<Vec<i128>>>> = Vec::four_dim(1, 1, 1, 2).zeros();
        assert_eq!(arr, [[[[0, 0]]]]);
    }

    #[test]
    fn test_zeros_f32_one_dim() {
        let arr: Vec<f32> = Vec::one_dim(2).zeros();
        assert_eq!(arr, [0.0, 0.0]);
    }

    #[test]
    fn test_zeros_f32_two_dim() {
        let arr: Vec<Vec<f32>> = Vec::two_dim(1, 2).zeros();
        assert_eq!(arr, [[0.0, 0.0]]);
    }

    #[test]
    fn test_zeros_f32_three_dim() {
        let arr: Vec<Vec<Vec<f32>>> = Vec::three_dim(1, 1, 2).zeros();
        assert_eq!(arr, [[[0.0, 0.0]]]);
    }

    #[test]
    fn test_zeros_f32_four_dim() {
        let arr: Vec<Vec<Vec<Vec<f32>>>> = Vec::four_dim(1, 1, 1, 2).zeros();
        assert_eq!(arr, [[[[0.0, 0.0]]]]);
    }

    #[test]
    fn test_zeros_f64_one_dim() {
        let arr: Vec<f64> = Vec::one_dim(2).zeros();
        assert_eq!(arr, [0.0, 0.0]);
    }

    #[test]
    fn test_zeros_f64_two_dim() {
        let arr: Vec<Vec<f64>> = Vec::two_dim(1, 2).zeros();
        assert_eq!(arr, [[0.0, 0.0]]);
    }

    #[test]
    fn test_zeros_f64_three_dim() {
        let arr: Vec<Vec<Vec<f64>>> = Vec::three_dim(1, 1, 2).zeros();
        assert_eq!(arr, [[[0.0, 0.0]]]);
    }

    #[test]
    fn test_zeros_f64_four_dim() {
        let arr: Vec<Vec<Vec<Vec<f64>>>> = Vec::four_dim(1, 1, 1, 2).zeros();
        assert_eq!(arr, [[[[0.0, 0.0]]]]);
    }

    #[test]
    fn test_dim() {
        let arr1: Vec<i32> = Vec::one_dim(2).zeros();
        assert_eq!(arr1.dim(), 1);

        let arr2: Vec<Vec<i32>> = Vec::two_dim(2, 2).zeros();
        assert_eq!(arr2.dim(), 2);

        let arr3: Vec<Vec<Vec<i32>>> = Vec::three_dim(2, 2, 2).zeros();
        assert_eq!(arr3.dim(), 3);

        let arr4: Vec<Vec<Vec<Vec<i32>>>> = Vec::four_dim(2, 2, 2, 3).zeros();
        assert_eq!(arr4.dim(), 4);
    }

    #[test]
    fn test_shape() {
        let arr1: Vec<i32> = Vec::one_dim(2).zeros();
        assert_eq!(arr1.shape(), [2]);

        let arr2: Vec<Vec<i32>> = Vec::two_dim(2, 2).zeros();
        assert_eq!(arr2.shape(), [2, 2]);

        let arr3: Vec<Vec<Vec<i32>>> = Vec::three_dim(2, 2, 2).zeros();
        assert_eq!(arr3.shape(), [2, 2, 2]);

        let arr4: Vec<Vec<Vec<Vec<i32>>>> = Vec::four_dim(2, 2, 2, 3).zeros();
        assert_eq!(arr4.shape(), [2, 2, 2, 3]);
    }

    #[test]
    fn test_size() {
        let arr1: Vec<i32> = Vec::one_dim(2).zeros();
        assert_eq!(arr1.size(), 2);

        let arr2: Vec<Vec<i32>> = Vec::two_dim(2, 2).zeros();
        assert_eq!(arr2.size(), 4);

        let arr3: Vec<Vec<Vec<i32>>> = Vec::three_dim(2, 2, 2).zeros();
        assert_eq!(arr3.size(), 8);

        let arr4: Vec<Vec<Vec<Vec<i32>>>> = Vec::four_dim(2, 2, 2, 3).zeros();
        assert_eq!(arr4.size(), 24);
    }
}
