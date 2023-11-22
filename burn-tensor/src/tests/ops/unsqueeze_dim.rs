#[burn_tensor_testgen::testgen(unsqueeze_dim)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Shape, Tensor};

    /// Test if the function can successfully unsqueeze a 4D tensor to a 5D tensor in the first
    /// dimension.
    #[test]
    fn should_unsqueeze_dim_4d_to_5d_first_dimension() {
        let tensor = Tensor::<TestBackend, 4>::ones(Shape::new([2, 3, 4, 5]));
        let unsqueezed_tensor: Tensor<TestBackend, 5> = tensor.unsqueeze_dim(0);
        let expected_shape = Shape::new([1, 2, 3, 4, 5]);
        assert_eq!(unsqueezed_tensor.shape(), expected_shape);
    }

    /// Test if the function can successfully unsqueeze a 4D tensor to a 5D tensor in some midddle
    /// dimension.
    #[test]
    fn should_unsqueeze_dim_4d_to_5d_middle_dimension() {
        let tensor = Tensor::<TestBackend, 4>::ones(Shape::new([2, 3, 4, 5]));
        let unsqueezed_tensor: Tensor<TestBackend, 5> = tensor.unsqueeze_dim(2);
        let expected_shape = Shape::new([2, 3, 1, 4, 5]);
        assert_eq!(unsqueezed_tensor.shape(), expected_shape);
    }

    /// Test if the function can successfully unsqueeze a 4D tensor to a 5D tensor in the last
    /// dimension.
    #[test]
    fn should_unsqueeze_dim_4d_to_5d_last_dimension() {
        let tensor = Tensor::<TestBackend, 4>::ones(Shape::new([2, 3, 4, 5]));
        let unsqueezed_tensor: Tensor<TestBackend, 5> = tensor.unsqueeze_dim(4);
        let expected_shape = Shape::new([2, 3, 4, 5, 1]);
        assert_eq!(unsqueezed_tensor.shape(), expected_shape);
    }

    /// Test if the function panics unsqueezing a 1D tensor to a 5D tensor i.e more than one
    /// dimension.
    #[test]
    #[should_panic]
    fn should_panic_unsqueeze_dim_1d_to_5d() {
        let tensor = Tensor::<TestBackend, 1>::ones(Shape::new([5]));
        let unsqueezed_tensor: Tensor<TestBackend, 5> = tensor.unsqueeze_dim(0);
    }

    /// Test that the function panics when unsqueezing to the same dimension.
    #[test]
    #[should_panic]
    fn should_panic_unsqueeze_dim_5d_to_5d() {
        let tensor = Tensor::<TestBackend, 5>::ones(Shape::new([1, 2, 3, 4, 5]));
        let unsqueezed_tensor: Tensor<TestBackend, 5> = tensor.unsqueeze_dim(2);
    }

    /// Test that the function can't unsqueeze into a lower dimension
    #[test]
    #[should_panic]
    fn should_panic_unsqueeze_dim_5d_to_3d() {
        let tensor = Tensor::<TestBackend, 5>::ones(Shape::new([1, 2, 3, 4, 5]));
        let unsqueezed_tensor: Tensor<TestBackend, 3> = tensor.unsqueeze_dim(4);
    }
}
