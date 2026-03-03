#[cfg(test)]
mod GatesTest {
    use crate::{complex_math::Complex, gates::Gate, state_vector::StateVec};

    #[test]
    fn x_gate_flips_basis_states() {
        let x = Gate::pauli_x();
        let zero = StateVec::basis_zero();
        let one = x.apply(&zero);
        assert!(one.approx_eq(&StateVec::basis_one()));
    }

    #[test]
    fn x_gate_is_its_own_inverse() {
        let x = Gate::pauli_x();
        let zero = StateVec::basis_zero();
        let out = x.apply(&x.apply(&zero));
        assert!(out.approx_eq(&StateVec::basis_zero()));
    }

    #[test]
    fn x_gate_preserves_norm() {
        let x = Gate::pauli_x();
        let mut psi = StateVec::from(vec![Complex::new(0.3, 0.4), Complex::new(0.5, 0.7)]);
        psi.normalize();
        let out = x.apply(&psi);
        assert!((out.norm2() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn identity_leaves_zero_unchanged() {
        let i = Gate::identity();
        let zero = StateVec::basis_zero();
        let out = i.apply(&zero);
        assert!(out.approx_eq(&zero));
    }

    #[test]
    fn identity_leaves_one_unchanged() {
        let i = Gate::identity();
        let one = StateVec::basis_one();
        let out = i.apply(&one);
        assert!(out.approx_eq(&one));
    }

    #[test]
    fn identity_preserves_norm() {
        let i = Gate::identity();
        let psi = StateVec::from(vec![Complex::new(0.6, 0.0), Complex::new(0.8, 0.0)]);
        let out = i.apply(&psi);
        assert!((out.norm2() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn z_gate_leaves_zero_unchanged() {
        let z = Gate::pauli_z();
        let zero = StateVec::basis_zero();
        let out = z.apply(&zero);
        assert!(out.approx_eq(&zero));
    }

    #[test]
    fn z_gate_flips_phase_of_one() {
        let z = Gate::pauli_z();
        let one = StateVec::basis_one();
        let out = z.apply(&one);

        let expected = StateVec::from(vec![Complex::new(0.0, 0.0), Complex::new(-1.0, 0.0)]);

        assert!(out.approx_eq(&expected));
    }

    #[test]
    fn hadamard_on_zero_creates_equal_superposition() {
        let h = Gate::hadamard();
        let zero = StateVec::basis_zero();
        let out = h.apply(&zero);

        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let expected = StateVec::from(vec![
            Complex::new(inv_sqrt2, 0.0),
            Complex::new(inv_sqrt2, 0.0),
        ]);

        assert!(out.approx_eq(&expected));
    }

    #[test]
    fn hadamard_on_one_creates_signed_superposition() {
        let h = Gate::hadamard();
        let one = StateVec::basis_one();
        let out = h.apply(&one);

        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let expected = StateVec::from(vec![
            Complex::new(inv_sqrt2, 0.0),
            Complex::new(-inv_sqrt2, 0.0),
        ]);

        assert!(out.approx_eq(&expected));
    }

    #[test]
    fn z_gate_preserves_norm() {
        let z = Gate::pauli_z();
        let mut psi = StateVec::from(vec![Complex::new(0.5, 0.0), Complex::new(0.5, 0.5)]);
        psi.normalize();
        let out = z.apply(&psi);
        assert!((out.norm2() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn hadamard_is_its_own_inverse() {
        let h = Gate::hadamard();
        let psi = StateVec::basis_zero();
        let out = h.apply(&h.apply(&psi));
        assert!(out.approx_eq(&psi));
    }

    #[test]
    fn hadamard_preserves_norm() {
        let h = Gate::hadamard();
        let psi = StateVec::basis_zero();
        let out = h.apply(&psi);
        assert!((out.norm2() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn gate_does_not_resize_state() {
        let h = Gate::hadamard();
        let psi = StateVec::basis_zero();
        let out = h.apply(&psi);
        assert_eq!(out.data.len(), psi.data.len());
    }

    #[test]
    fn y_gate_maps_zero_to_i_one() {
        let y = Gate::pauli_y();
        let zero = StateVec::basis_zero();
        let out = y.apply(&zero);

        let expected = StateVec::from(vec![Complex::new(0.0, 0.0), Complex::new(0.0, 1.0)]);

        assert!(out.approx_eq(&expected));
    }

    #[test]
    fn y_gate_maps_one_to_minus_i_zero() {
        let y = Gate::pauli_y();
        let one = StateVec::basis_one();
        let out = y.apply(&one);

        let expected = StateVec::from(vec![Complex::new(0.0, -1.0), Complex::new(0.0, 0.0)]);

        assert!(out.approx_eq(&expected));
    }

    #[test]
    fn y_gate_is_its_own_inverse_up_to_phase() {
        let y = Gate::pauli_y();
        let zero = StateVec::basis_zero();
        let out = y.apply(&y.apply(&zero));

        // Y² = I (global phase irrelevant)
        assert!(out.approx_eq(&StateVec::basis_zero()));
    }

    #[test]
    fn y_gate_preserves_norm() {
        let y = Gate::pauli_y();
        let mut psi = StateVec::from(vec![Complex::new(0.4, 0.3), Complex::new(0.1, 0.8)]);
        psi.normalize();

        let out = y.apply(&psi);
        assert!((out.norm2() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn s_gate_leaves_zero_unchanged() {
        let s = Gate::S();
        let zero = StateVec::basis_zero();
        let out = s.apply(&zero);
        assert!(out.approx_eq(&zero));
    }

    #[test]
    fn s_gate_multiplies_one_by_i() {
        let s = Gate::S();
        let one = StateVec::basis_one();
        let out = s.apply(&one);

        let expected = StateVec::from(vec![Complex::new(0.0, 0.0), Complex::new(0.0, 1.0)]);

        assert!(out.approx_eq(&expected));
    }

    #[test]
    fn s_gate_preserves_norm() {
        let s = Gate::S();
        let mut psi = StateVec::from(vec![Complex::new(0.6, 0.0), Complex::new(0.0, 0.8)]);
        psi.normalize();

        let out = s.apply(&psi);
        assert!((out.norm2() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn t_gate_leaves_zero_unchanged() {
        let t = Gate::T();
        let zero = StateVec::basis_zero();
        let out = t.apply(&zero);
        assert!(out.approx_eq(&zero));
    }

    #[test]
    fn t_gate_applies_pi_over_4_phase_to_one() {
        let t = Gate::T();
        let one = StateVec::basis_one();
        let out = t.apply(&one);

        let factor = 1.0 / 2.0_f64.sqrt();
        let expected = StateVec::from(vec![Complex::new(0.0, 0.0), Complex::new(factor, factor)]);

        assert!(out.approx_eq(&expected));
    }

    #[test]
    fn t_gate_preserves_norm() {
        let t = Gate::T();
        let mut psi = StateVec::from(vec![Complex::new(0.7, 0.0), Complex::new(0.2, 0.6)]);
        psi.normalize();

        let out = t.apply(&psi);
        assert!((out.norm2() - 1.0).abs() < 1e-10);
    }
}
