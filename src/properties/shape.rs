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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

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
}
