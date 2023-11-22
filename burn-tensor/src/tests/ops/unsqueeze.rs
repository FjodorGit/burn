#[burn_tensor_testgen::testgen(unsqueeze)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Shape, Tensor};

    /// Test if the function can successfully unsqueeze a 4D tensor to a 5D tensor.
    #[test]
    fn should_unsqueeze_4d_to_5d() {
        let tensor = Tensor::<TestBackend, 4>::ones(Shape::new([2, 3, 4, 5]));
        let unsqueezed_tensor: Tensor<TestBackend, 5> = tensor.unsqueeze();
        let expected_shape = Shape::new([1, 2, 3, 4, 5]);
        assert_eq!(unsqueezed_tensor.shape(), expected_shape);
    }

    /// Test if the function can successfully unsqueeze a 1D tensor to a 5D tensor.
    #[test]
    fn should_unsqueeze_1d_to_5d() {
        let tensor = Tensor::<TestBackend, 1>::ones(Shape::new([5]));
        let unsqueezed_tensor: Tensor<TestBackend, 5> = tensor.unsqueeze();
        let expected_shape = Shape::new([1, 1, 1, 1, 5]);
        assert_eq!(unsqueezed_tensor.shape(), expected_shape);
    }

    /// Test that the function does not change the tensor when unsqueezing to the same dimension.
    #[test]
    fn should_not_alter_5d_to_5d() {
        let tensor = Tensor::<TestBackend, 5>::ones(Shape::new([1, 2, 3, 4, 5]));
        let unsqueezed_tensor: Tensor<TestBackend, 5> = tensor.unsqueeze();
        let expected_shape = Shape::new([1, 2, 3, 4, 5]);
        assert_eq!(unsqueezed_tensor.shape(), expected_shape);
    }

    /// Test that the function can't unsqueeze into a lower dimension
    #[test]
    #[should_panic]
    fn should_not_alter_5d_to_3d() {
        let tensor = Tensor::<TestBackend, 5>::ones(Shape::new([1, 2, 3, 4, 5]));
        let unsqueezed_tensor: Tensor<TestBackend, 3> = tensor.unsqueeze();
    }
}
