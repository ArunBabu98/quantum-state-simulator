#[cfg(test)]
mod gate_test {
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

#[cfg(test)]
mod gate_tensor_tests {
    use crate::{complex_math::Complex, gates::Gate, state_vector::StateVec};

    // H⊗I applied to |00⟩ should give (1/√2)(|00⟩ + |10⟩)
    #[test]
    fn tensor_h_i_on_00() {
        let hi = Gate::tensor_product(&Gate::hadamard(), &Gate::identity());
        let psi = StateVec::zero_state(2);
        let result = hi.apply(&psi);
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let expected = StateVec::from(vec![
            Complex::new(inv_sqrt2, 0.0), // |00⟩
            Complex::new(0.0, 0.0),
            Complex::new(inv_sqrt2, 0.0), // |10⟩
            Complex::new(0.0, 0.0),
        ]);
        assert!(result.approx_eq(&expected));
    }

    // I⊗H applied to |00⟩ should give (1/√2)(|00⟩ + |01⟩)
    #[test]
    fn tensor_i_h_on_00() {
        let ih = Gate::tensor_product(&Gate::identity(), &Gate::hadamard());
        let psi = StateVec::zero_state(2);
        let result = ih.apply(&psi);
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let expected = StateVec::from(vec![
            Complex::new(inv_sqrt2, 0.0), // |00⟩
            Complex::new(inv_sqrt2, 0.0), // |01⟩
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ]);
        assert!(result.approx_eq(&expected));
    }

    // Tensor product of two 2x2 gates produces a 4x4 gate
    #[test]
    fn tensor_product_dimension() {
        let g = Gate::tensor_product(&Gate::hadamard(), &Gate::pauli_x());
        assert_eq!(g.data.len(), 4);
        assert!(g.data.iter().all(|row| row.len() == 4));
    }

    // (A⊗B)(C⊗D) = (AC)⊗(BD) — mixed product property
    #[test]
    fn tensor_product_mixed_product_property() {
        let psi = StateVec::zero_state(2);

        // Apply H⊗X as one tensor gate
        let hx = Gate::tensor_product(&Gate::hadamard(), &Gate::pauli_x());
        let result_a = hx.apply(&psi);

        // Apply H to qubit 0, X to qubit 1 via lift — must match
        let lifted_h = Gate::lift(&Gate::hadamard(), 0, 2);
        let lifted_x = Gate::lift(&Gate::pauli_x(), 1, 2);
        let result_b = lifted_x.apply(&lifted_h.apply(&psi));

        assert!(result_a.approx_eq(&result_b));
    }

    // I⊗I = identity on 2-qubit space
    #[test]
    fn tensor_identity_is_identity() {
        let ii = Gate::tensor_product(&Gate::identity(), &Gate::identity());
        let psi = StateVec::zero_state(2);
        let result = ii.apply(&psi);
        assert!(result.approx_eq(&psi));
    }
}

#[cfg(test)]
mod gate_lift_tests {
    use crate::{complex_math::Complex, gates::Gate, state_vector::StateVec};

    // Lifting identity to any qubit does nothing
    #[test]
    fn lift_identity_is_noop_qubit0() {
        let psi = StateVec::zero_state(3);
        let lifted = Gate::lift(&Gate::identity(), 0, 3);
        assert!(lifted.apply(&psi).approx_eq(&psi));
    }

    #[test]
    fn lift_identity_is_noop_qubit1() {
        let psi = StateVec::zero_state(3);
        let lifted = Gate::lift(&Gate::identity(), 1, 3);
        assert!(lifted.apply(&psi).approx_eq(&psi));
    }

    #[test]
    fn lift_identity_is_noop_qubit2() {
        let psi = StateVec::zero_state(3);
        let lifted = Gate::lift(&Gate::identity(), 2, 3);
        assert!(lifted.apply(&psi).approx_eq(&psi));
    }

    // Lifting X to qubit 0 of |000⟩ gives |100⟩
    #[test]
    fn lift_x_to_qubit0_flips_first() {
        let psi = StateVec::zero_state(3); // |000⟩ = index 0
        let lifted = Gate::lift(&Gate::pauli_x(), 0, 3);
        let result = lifted.apply(&psi);
        // |100⟩ = index 4 in 3-qubit system
        assert!((result.data[4].re - 1.0).abs() < 1e-10);
        assert!((result.norm2() - 1.0).abs() < 1e-10);
    }

    // Lifting X to qubit 2 of |000⟩ gives |001⟩
    #[test]
    fn lift_x_to_qubit2_flips_last() {
        let psi = StateVec::zero_state(3); // |000⟩ = index 0
        let lifted = Gate::lift(&Gate::pauli_x(), 2, 3);
        let result = lifted.apply(&psi);
        // |001⟩ = index 1
        assert!((result.data[1].re - 1.0).abs() < 1e-10);
    }

    // Lifting H to qubit 1 of 3-qubit state produces correct superposition
    #[test]
    fn lift_h_to_middle_qubit() {
        let psi = StateVec::zero_state(3); // |000⟩
        let lifted = Gate::lift(&Gate::hadamard(), 1, 3);
        let result = lifted.apply(&psi);
        // Should give (1/√2)(|000⟩ + |010⟩) = amplitudes at index 0 and 2
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        assert!((result.data[0].re - inv_sqrt2).abs() < 1e-10);
        assert!((result.data[2].re - inv_sqrt2).abs() < 1e-10);
        assert!((result.norm2() - 1.0).abs() < 1e-10);
    }

    // Lifted gate produces correct output dimension
    #[test]
    fn lift_produces_correct_dimension() {
        let g = Gate::lift(&Gate::hadamard(), 1, 3);
        assert_eq!(g.data.len(), 8);
        assert!(g.data.iter().all(|row| row.len() == 8));
    }

    // H lifted to qubit 0 then again = identity (H is self-inverse)
    #[test]
    fn lift_h_twice_is_identity() {
        let psi = StateVec::zero_state(2);
        let lh = Gate::lift(&Gate::hadamard(), 0, 2);
        let result = lh.apply(&lh.apply(&psi));
        assert!(result.approx_eq(&psi));
    }
}

#[cfg(test)]
mod controlled_gate_tests {
    use crate::{complex_math::Complex, gates::Gate, state_vector::StateVec};

    // Controlled-X (built generically) must equal hardcoded CNOT
    #[test]
    fn controlled_x_equals_cnot() {
        let cx = Gate::controlled(&Gate::pauli_x());
        let cnot = Gate::cnot();
        let psi = StateVec::zero_state(2);

        // Test on all 4 basis states
        for idx in 0..4 {
            let mut state_data = vec![Complex::new(0.0, 0.0); 4];
            state_data[idx] = Complex::new(1.0, 0.0);
            let state = StateVec::from(state_data);

            assert!(
                cx.apply(&state).approx_eq(&cnot.apply(&state)),
                "Mismatch on basis state |{:02b}⟩",
                idx
            );
        }
    }

    // Controlled-Z (built generically) must equal hardcoded CZ
    #[test]
    fn controlled_z_equals_cz() {
        let cz_generic = Gate::controlled(&Gate::pauli_z());
        let cz = Gate::cz();
        for idx in 0..4 {
            let mut state_data = vec![Complex::new(0.0, 0.0); 4];
            state_data[idx] = Complex::new(1.0, 0.0);
            let state = StateVec::from(state_data);
            assert!(cz_generic.apply(&state).approx_eq(&cz.apply(&state)));
        }
    }

    // Controlled-U with control=|0⟩ never applies U
    #[test]
    fn controlled_gate_noop_when_control_zero() {
        let cu = Gate::controlled(&Gate::pauli_x());
        // |00⟩ and |01⟩ both have control=0, should be unchanged
        for idx in [0usize, 1usize] {
            let mut state_data = vec![Complex::new(0.0, 0.0); 4];
            state_data[idx] = Complex::new(1.0, 0.0);
            let state = StateVec::from(state_data.clone());
            let result = cu.apply(&state);
            assert!((result.data[idx].re - 1.0).abs() < 1e-10);
        }
    }

    // Controlled-U with control=|1⟩ always applies U
    #[test]
    fn controlled_gate_applies_when_control_one() {
        let cx = Gate::controlled(&Gate::pauli_x());
        // |10⟩ (index 2): control=1, target=0 → should flip to |11⟩ (index 3)
        let mut state_data = vec![Complex::new(0.0, 0.0); 4];
        state_data[2] = Complex::new(1.0, 0.0);
        let state = StateVec::from(state_data);
        let result = cx.apply(&state);
        assert!((result.data[3].re - 1.0).abs() < 1e-10);
    }

    // Controlled-I = identity on full space
    #[test]
    fn controlled_identity_is_noop() {
        let ci = Gate::controlled(&Gate::identity());
        let psi = StateVec::zero_state(2);
        assert!(ci.apply(&psi).approx_eq(&psi));
    }

    // Controlled-H is unitary: norm must be preserved
    #[test]
    fn controlled_h_preserves_norm() {
        let ch = Gate::controlled(&Gate::hadamard());
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let psi = StateVec::from(vec![
            Complex::new(inv_sqrt2, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(inv_sqrt2, 0.0),
            Complex::new(0.0, 0.0),
        ]);
        let result = ch.apply(&psi);
        assert!((result.norm2() - 1.0).abs() < 1e-10);
    }
}
