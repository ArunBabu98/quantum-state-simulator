#[cfg(test)]
mod StateVectorTest {
    use crate::{complex_math::Complex, gates::Gate, state_vector::StateVec};

    /*---------tests------------*/
    #[test]
    fn single_qubit_norm_is_one() {
        let psi = StateVec::from(vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]);
        assert!((psi.norm2() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn normalization_works() {
        let mut psi = StateVec::from(vec![Complex::new(3.0, 0.0), Complex::new(4.0, 0.0)]);
        psi.normalize();
        assert!((psi.norm2() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn inner_product_self_is_one() {
        let psi = StateVec::basis_zero();
        let ip = psi.inner(&psi);
        assert!((ip.re - 1.0).abs() < 1e-10);
        assert!(ip.im.abs() < 1e-10);
    }

    #[test]
    fn basis_states_are_orthogonal() {
        let zero = StateVec::basis_zero();
        let one = StateVec::basis_one();
        let ip = zero.inner(&one);
        assert!(ip.abs2() < 1e-10);
    }

    #[test]
    fn tensor_product_dimension_doubles() {
        let a = StateVec::basis_zero();
        let b = StateVec::basis_one();
        let ab = a.tensor(&b);
        assert_eq!(ab.data.len(), 4);
    }

    #[test]
    fn tensor_product_basis_states() {
        let zero = StateVec::basis_zero();
        let one = StateVec::basis_one();

        let zero_one = zero.tensor(&one);

        let expected = StateVec::from(vec![
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ]);

        assert!(zero_one.approx_eq(&expected));
    }
    #[test]
    fn tensor_product_one_zero() {
        let one = StateVec::basis_one();
        let zero = StateVec::basis_zero();
        let one_zero = one.tensor(&zero);
        // |10⟩ = index 2
        let expected = StateVec::from(vec![
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
        ]);
        assert!(one_zero.approx_eq(&expected));
    }

    #[test]
    fn tensor_product_preserves_norm() {
        let a = StateVec::basis_zero();
        let b = StateVec::basis_one();
        let ab = a.tensor(&b);
        assert!((ab.norm2() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn measurement_of_basis_zero_is_deterministic() {
        let mut psi = StateVec::basis_zero();
        let result = psi.measure();
        assert_eq!(result, 0);
        assert!(psi.approx_eq(&StateVec::basis_zero()));
    }

    #[test]
    fn measurement_collapses_superposition() {
        let h = Gate::hadamard();
        let mut psi = h.apply(&StateVec::basis_zero());

        let _ = psi.measure();

        // After measurement, must be basis state
        let is_zero = psi.approx_eq(&StateVec::basis_zero());
        let is_one = psi.approx_eq(&StateVec::basis_one());

        assert!(is_zero || is_one);
    }

    #[test]
    fn repeated_measurement_is_stable() {
        let mut psi = StateVec::basis_one();
        let first = psi.measure();
        let second = psi.measure();
        assert_eq!(first, second);
    }
}
