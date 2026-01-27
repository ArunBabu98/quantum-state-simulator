/* ----------Gates---------

Predefined gates: I, X, Z, H

Invariants
----------
1. Norm before =  norm after
2. Dimensions must match
3. No silent resizing

*/

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

use crate::{complex_math::Complex, state_vector::StateVec};

pub struct Gate {
    pub data: Vec<Vec<Complex>>,
}

impl Gate {
    pub fn apply(&self, state: &StateVec) -> StateVec {
        let a = self.data[0][0] * state.data[0] + self.data[0][1] * state.data[1];
        let b = self.data[1][0] * state.data[0] + self.data[1][1] * state.data[1];
        StateVec::from(vec![a, b])
    }
    pub fn identity() -> Self {
        Self {
            data: vec![
                vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
                vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
            ],
        }
    }
    // Pauli-X (NOT) Gate
    // Swaps the amplites of |0> and |1>
    // [0  1]
    // [1  0]
    pub fn pauli_x() -> Self {
        Self {
            data: vec![
                vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
                vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
            ],
        }
    }

    // Pauli-Z (Phase flip) Gate
    // [1  0]
    // [0  -1]
    pub fn pauli_z() -> Self {
        Self {
            data: vec![
                vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
                vec![Complex::new(0.0, 0.0), Complex::new(-1.0, 0.0)],
            ],
        }
    }

    pub fn hadamard() -> Self {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let val = Complex::new(inv_sqrt2, 0.0);
        let neg_val = Complex::new(-inv_sqrt2, 0.0);
        Self {
            data: vec![vec![val, val], vec![val, neg_val]],
        }
    }
}
