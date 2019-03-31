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

// TODO(pyk): Add docs here

use crate::matrix::Matrix;
use num::Num;
use std::ops;

// This trait is implemented to support for matrix addition operator
impl<T> ops::Add<Matrix<T>> for Matrix<T>
where
    T: Num + Copy,
{
    type Output = Matrix<T>;

    fn add(self, other: Matrix<T>) -> Matrix<T> {
        if self.shape() != other.shape() {
            panic!(
                "Matrix addition with invalid shape: {:?} != {:?}",
                self.shape(),
                other.shape()
            );
        }

        // Add the element of the matrix
        let vec = self.vec + other.vec;
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            vec,
        }
    }
}

// This trait is implemented to support for matrix addition
// operator with scalar on the right side,
// for example:
//
// let a = matrix![5, 5; 5, 5] + 6;
//
impl<T> ops::Add<T> for Matrix<T>
where
    T: Num + Copy,
{
    type Output = Matrix<T>;

    fn add(self, value: T) -> Matrix<T> {
        let vec = self.vec + value;
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            vec,
        }
    }
}

// This macro is to generate support for matrix addition
// operator with scalar on the left side,
// for example:
//
// let a = 6 + matrix![5, 5; 5, 5];
//
macro_rules! impl_add_matrix_for_type {
    ($t: ty) => {
        impl ops::Add<Matrix<$t>> for $t {
            type Output = Matrix<$t>;

            fn add(self, m: Matrix<$t>) -> Matrix<$t> {
                let vec = self + m.vec;
                Matrix {
                    nrows: m.nrows,
                    ncols: m.ncols,
                    vec,
                }
            }
        }
    };
}

impl_add_matrix_for_type!(usize);
impl_add_matrix_for_type!(i8);
impl_add_matrix_for_type!(i16);
impl_add_matrix_for_type!(i32);
impl_add_matrix_for_type!(i64);
impl_add_matrix_for_type!(i128);
impl_add_matrix_for_type!(u8);
impl_add_matrix_for_type!(u16);
impl_add_matrix_for_type!(u32);
impl_add_matrix_for_type!(u64);
impl_add_matrix_for_type!(u128);
impl_add_matrix_for_type!(f32);
impl_add_matrix_for_type!(f64);

// This trait is implemented to support for matrix addition
// and assignment operator (+=)
impl<T> ops::AddAssign<Matrix<T>> for Matrix<T>
where
    T: Num + Copy + ops::AddAssign,
{
    fn add_assign(&mut self, other: Matrix<T>) {
        if self.shape() != other.shape() {
            panic!(
                "Matrix addition with invalid length: {:?} != {:?}",
                self.shape(),
                other.shape()
            );
        }
        self.vec += other.vec;
    }
}

// This trait is implemented to support for matrix addition
// assignment operator (+=) with scalar on the right side,
// for example:
//
// let a = matrix![5, 5; 5, 5];
// a += 6;
//
impl<T> ops::AddAssign<T> for Matrix<T>
where
    T: Num + Copy + ops::AddAssign,
{
    fn add_assign(&mut self, value: T) {
        self.vec += value;
    }
}

// This trait is implemented to support for matrix
// substraction operator
impl<T> ops::Sub<Matrix<T>> for Matrix<T>
where
    T: Num + Copy,
{
    type Output = Matrix<T>;

    fn sub(self, other: Matrix<T>) -> Matrix<T> {
        if self.shape() != other.shape() {
            panic!(
                "Matrix substraction with invalid shape: {:?} != {:?}",
                self.shape(),
                other.shape()
            );
        }

        // Substract the matrix
        let vec = self.vec - other.vec;
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            vec,
        }
    }
}

// This trait is implemented to support for matrix substraction
// operator with scalar on the right side,
// for example:
//
// let a = matrix![5, 5; 5, 5] - 6;
//
impl<T> ops::Sub<T> for Matrix<T>
where
    T: Num + Copy,
{
    type Output = Matrix<T>;

    fn sub(self, value: T) -> Matrix<T> {
        // Substract the matrix
        let vec = self.vec - value;
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            vec,
        }
    }
}

// This macro is to generate support for matrix substraction
// operator with scalar on the left side,
// for example:
//
// let a = 6 - matrix![5, 5; 5, 5];
//
macro_rules! impl_sub_matrix_for_type {
    ($t: ty) => {
        impl ops::Sub<Matrix<$t>> for $t {
            type Output = Matrix<$t>;

            fn sub(self, m: Matrix<$t>) -> Matrix<$t> {
                // Substract the matrix
                let vec = self - m.vec;
                Matrix {
                    nrows: m.nrows,
                    ncols: m.ncols,
                    vec,
                }
            }
        }
    };
}

impl_sub_matrix_for_type!(usize);
impl_sub_matrix_for_type!(i8);
impl_sub_matrix_for_type!(i16);
impl_sub_matrix_for_type!(i32);
impl_sub_matrix_for_type!(i64);
impl_sub_matrix_for_type!(i128);
impl_sub_matrix_for_type!(u8);
impl_sub_matrix_for_type!(u16);
impl_sub_matrix_for_type!(u32);
impl_sub_matrix_for_type!(u64);
impl_sub_matrix_for_type!(u128);
impl_sub_matrix_for_type!(f32);
impl_sub_matrix_for_type!(f64);

// This trait is implemented to support for matrix substraction
// and assignment operator (-=)
impl<T> ops::SubAssign<Matrix<T>> for Matrix<T>
where
    T: Num + Copy + ops::SubAssign,
{
    fn sub_assign(&mut self, other: Matrix<T>) {
        if self.shape() != other.shape() {
            panic!(
                "Matrix substraction with invalid length: {:?} != {:?}",
                self.shape(),
                other.shape()
            );
        }
        self.vec -= other.vec;
    }
}

// This trait is implemented to support for matrix substraction
// assignment operator (-=) with scalar on the right side,
// for example:
//
// let a = matrix![5, 5; 5, 5];
// a -= 6;
//
impl<T> ops::SubAssign<T> for Matrix<T>
where
    T: Num + Copy + ops::SubAssign,
{
    fn sub_assign(&mut self, value: T) {
        self.vec -= value;
    }
}

// This trait is implemented to support for matrix
// multiplication operator
impl<T> ops::Mul<Matrix<T>> for Matrix<T>
where
    T: Num + Copy,
{
    type Output = Matrix<T>;

    fn mul(self, other: Matrix<T>) -> Matrix<T> {
        if self.shape() != other.shape() {
            panic!(
                "Matrix multiplication with invalid shape: {:?} != {:?}",
                self.shape(),
                other.shape()
            );
        }
        let vec = self.vec * other.vec;
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            vec,
        }
    }
}

// This trait is implemented to support for matrix multiplication
// operator with scalar on the right side,
// for example:
//
// let a = matrix![5, 5; 5, 5] * 6;
//
impl<T> ops::Mul<T> for Matrix<T>
where
    T: Num + Copy,
{
    type Output = Matrix<T>;

    fn mul(self, value: T) -> Matrix<T> {
        let vec = self.vec * value;
        Matrix {
            nrows: self.nrows,
            ncols: self.ncols,
            vec,
        }
    }
}

// This macro is to generate support for matrix multiplication
// operator with scalar on the left side,
// for example:
//
// let a = 6 * matrix![5, 5; 5, 5];
//
macro_rules! impl_sub_matrix_for_type {
    ($t: ty) => {
        impl ops::Mul<Matrix<$t>> for $t {
            type Output = Matrix<$t>;

            fn mul(self, m: Matrix<$t>) -> Matrix<$t> {
                let vec = self * m.vec;
                Matrix {
                    nrows: m.nrows,
                    ncols: m.ncols,
                    vec,
                }
            }
        }
    };
}

impl_sub_matrix_for_type!(usize);
impl_sub_matrix_for_type!(i8);
impl_sub_matrix_for_type!(i16);
impl_sub_matrix_for_type!(i32);
impl_sub_matrix_for_type!(i64);
impl_sub_matrix_for_type!(i128);
impl_sub_matrix_for_type!(u8);
impl_sub_matrix_for_type!(u16);
impl_sub_matrix_for_type!(u32);
impl_sub_matrix_for_type!(u64);
impl_sub_matrix_for_type!(u128);
impl_sub_matrix_for_type!(f32);
impl_sub_matrix_for_type!(f64);

// This trait is implemented to support for matrix substraction
// and assignment operator (-=)
impl<T> ops::MulAssign<Matrix<T>> for Matrix<T>
where
    T: Num + Copy + ops::MulAssign,
{
    fn mul_assign(&mut self, other: Matrix<T>) {
        if self.shape() != other.shape() {
            panic!(
                "Matrix multiplication with invalid length: {:?} != {:?}",
                self.shape(),
                other.shape()
            );
        }

        self.vec *= other.vec;
    }
}

// This trait is implemented to support for matrix multiplication
// assignment operator (*=) with scalar on the right side,
// for example:
//
// let a = matrix![5, 5; 5, 5];
// a *= 6;
//
impl<T> ops::MulAssign<T> for Matrix<T>
where
    T: Num + Copy + ops::MulAssign,
{
    fn mul_assign(&mut self, value: T) {
        self.vec *= value;
    }
}
