#[cfg(test)]
mod circuit_tests {
    use crate::{circuit::Circuit, complex_math::Complex, state_vector::StateVec};

    // Empty circuit returns |00...0⟩ unchanged
    #[test]
    fn empty_circuit_is_identity() {
        let c = Circuit::new(2);
        let result = c.run();
        assert!(result.approx_eq(&StateVec::zero_state(2)));
    }

    // Single H on qubit 0 of 1-qubit circuit
    #[test]
    fn single_h_gate_circuit() {
        let mut c = Circuit::new(1);
        c.h(0);
        let result = c.run();
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let expected = StateVec::from(vec![
            Complex::new(inv_sqrt2, 0.0),
            Complex::new(inv_sqrt2, 0.0),
        ]);
        assert!(result.approx_eq(&expected));
    }

    // H then H = identity (self-inverse check through circuit)
    #[test]
    fn h_h_is_identity_circuit() {
        let mut c = Circuit::new(1);
        c.h(0).h(0);
        let result = c.run();
        assert!(result.approx_eq(&StateVec::zero_state(1)));
    }

    // X on qubit 0 flips |0⟩ to |1⟩
    #[test]
    fn x_gate_flips_qubit() {
        let mut c = Circuit::new(1);
        c.x(0);
        let result = c.run();
        assert!(result.approx_eq(&StateVec::basis_one()));
    }

    // Bell state circuit: H(0) → CNOT(0,1)
    #[test]
    fn bell_state_circuit() {
        let mut c = Circuit::new(2);
        c.h(0).cnot(0, 1);
        let result = c.run();
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let expected = StateVec::from(vec![
            Complex::new(inv_sqrt2, 0.0), // |00⟩
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(inv_sqrt2, 0.0), // |11⟩
        ]);
        assert!(result.approx_eq(&expected));
    }

    // GHZ state: H(0) → CNOT(0,1) → CNOT(0,2) on 3 qubits
    // (1/√2)(|000⟩ + |111⟩)
    #[test]
    fn ghz_state_circuit() {
        let mut c = Circuit::new(3);
        c.h(0).cnot(0, 1).cnot(0, 2);
        let result = c.run();
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        // |000⟩ = index 0, |111⟩ = index 7
        assert!((result.data[0].re - inv_sqrt2).abs() < 1e-10);
        assert!((result.data[7].re - inv_sqrt2).abs() < 1e-10);
        assert!((result.norm2() - 1.0).abs() < 1e-10);
    }

    // Circuit norm is always 1 after run
    #[test]
    fn circuit_always_preserves_norm() {
        let mut c = Circuit::new(3);
        c.h(0).x(1).h(2).cnot(0, 1);
        let result = c.run();
        assert!((result.norm2() - 1.0).abs() < 1e-10);
    }

    // Gate order matters — H then X ≠ X then H
    #[test]
    fn gate_order_is_not_commutative() {
        let mut c1 = Circuit::new(1);
        c1.h(0).x(0);

        let mut c2 = Circuit::new(1);
        c2.x(0).h(0);

        let r1 = c1.run();
        let r2 = c2.run();
        assert!(!r1.approx_eq(&r2));
    }

    // Circuit with only identity gates = |00...0⟩
    #[test]
    fn identity_gates_noop() {
        let mut c = Circuit::new(2);
        c.i(0).i(1);
        let result = c.run();
        assert!(result.approx_eq(&StateVec::zero_state(2)));
    }
}
