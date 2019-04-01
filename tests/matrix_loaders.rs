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
#[should_panic]
fn test_load_invalid_not_found_csv() {
    let _w: Matrix<i32> = Matrix::from_csv("tests/data/random_file.csv")
        .load()
        .unwrap();
}

#[test]
#[should_panic]
fn test_load_invalid_format_csv() {
    let _w: Matrix<i32> =
        Matrix::from_csv("tests/data/matrix_invalid_empty.csv")
            .load()
            .unwrap();
}

#[test]
#[should_panic]
fn test_load_invalid_element() {
    let _w: Matrix<i32> =
        Matrix::from_csv("tests/data/matrix_invalid_element.csv")
            .load()
            .unwrap();
}

#[test]
#[should_panic]
fn test_load_invalid_element_different_data_type_expected_u32() {
    let _w: Matrix<u32> =
        Matrix::from_csv("tests/data/matrix_valid_without_header.csv")
            .load()
            .unwrap();
}

#[test]
#[should_panic]
fn test_load_invalid_inconsistent_column() {
    let _w: Matrix<u32> =
        Matrix::from_csv("tests/data/matrix_invalid_inconsistent_column.csv")
            .load()
            .unwrap();
}

#[test]
fn test_load_valid_csv_without_header() {
    let w: Matrix<i32> =
        Matrix::from_csv("tests/data/matrix_valid_without_header.csv")
            .load()
            .unwrap();
    assert_eq!(
        w,
        matrix![
            -1, 2, 3;
            4, -5, 6;
            7, 8, -9;
        ]
    );
}

#[test]
fn test_load_valid_csv_with_header() {
    let w: Matrix<f32> = Matrix::from_csv("tests/data/dataset.csv")
        .has_headers(true)
        .load()
        .unwrap();
    assert_eq!(
        w,
        matrix![
            1.0,2.0;
            3.0,4.0;
            5.0,6.0;
        ]
    );
}
