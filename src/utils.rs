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

pub trait TypeName {
    fn type_name() -> &'static str;
}

macro_rules! implement_type_name_for_type {
    ($t: ty, $name: expr) => {
        impl TypeName for $t {
            fn type_name() -> &'static str {
                $name
            }
        }
    };
}

implement_type_name_for_type!(usize, "usize");
implement_type_name_for_type!(i8, "i8");
implement_type_name_for_type!(i16, "i16");
implement_type_name_for_type!(i32, "i32");
implement_type_name_for_type!(i64, "i64");
implement_type_name_for_type!(i128, "i128");
implement_type_name_for_type!(u8, "u8");
implement_type_name_for_type!(u16, "u16");
implement_type_name_for_type!(u32, "u32");
implement_type_name_for_type!(u64, "u64");
implement_type_name_for_type!(u128, "u128");
implement_type_name_for_type!(f32, "f32");
implement_type_name_for_type!(f64, "f64");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name() {
        assert_eq!(<usize>::type_name(), "usize");
        assert_eq!(<i8>::type_name(), "i8");
        assert_eq!(<i16>::type_name(), "i16");
        assert_eq!(<i32>::type_name(), "i32");
        assert_eq!(<i64>::type_name(), "i64");
        assert_eq!(<i128>::type_name(), "i128");
        assert_eq!(<u8>::type_name(), "u8");
        assert_eq!(<u16>::type_name(), "u16");
        assert_eq!(<u32>::type_name(), "u32");
        assert_eq!(<u64>::type_name(), "u64");
        assert_eq!(<u128>::type_name(), "u128");
        assert_eq!(<f32>::type_name(), "f32");
        assert_eq!(<f64>::type_name(), "f64");
    }
}