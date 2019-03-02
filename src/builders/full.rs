
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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

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
}