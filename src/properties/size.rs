use crate::properties::shape::*;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

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
