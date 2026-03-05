/* ----------Gates---------

Predefined gates: I, X, Z, H

Invariants
----------
1. Norm before =  norm after
2. Dimensions must match
3. No silent resizing

*/

use crate::{complex_math::Complex, state_vector::StateVec};

#[derive(Clone)]
pub struct Gate {
    pub data: Vec<Vec<Complex>>,
}

impl Gate {
    pub fn apply(&self, state: &StateVec) -> StateVec {
        let dim = self.data.len();
        assert_eq!(
            state.data.len(),
            dim,
            "Gate dimension {} does not match state dimension {}",
            dim,
            state.data.len()
        );
        // Also assert gate is square
        assert!(
            self.data.iter().all(|row| row.len() == dim),
            "Gate matrix is not square"
        );

        let mut new_data = vec![Complex::new(0.0, 0.0); dim];
        for i in 0..dim {
            for j in 0..dim {
                let product = self.data[i][j] * state.data[j];
                new_data[i] = new_data[i] + product;
            }
        }
        StateVec::from(new_data)
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

    // Pauli-y (Phase flip) Gate
    // [0  -i]
    // [i  0]
    pub fn pauli_y() -> Self {
        Self {
            data: vec![
                vec![Complex::new(0.0, 0.0), Complex::new(0.0, -1.0)],
                vec![Complex::new(0.0, 1.0), Complex::new(0.0, 0.0)],
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

    // phase gate
    pub fn S() -> Self {
        Self {
            data: vec![
                vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
                vec![Complex::new(0.0, 0.0), Complex::new(0.0, 1.0)],
            ],
        }
    }

    // pi/8 gate
    pub fn T() -> Self {
        let (re, im) = (
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        );
        Self {
            data: vec![
                vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
                vec![Complex::new(0.0, 0.0), Complex::new(re, im)],
            ],
        }
    }
    pub fn cnot() -> Self {
        let zero = Complex::new(0.0, 0.0);
        let one = Complex::new(1.0, 0.0);
        Self {
            data: vec![
                vec![one, zero, zero, zero],
                vec![zero, one, zero, zero],
                vec![zero, zero, zero, one], // Row 2: Target flips if control is 1
                vec![zero, zero, one, zero], // Row 3: Target flips if control is 1
            ],
        }
    }

    pub fn cz() -> Self {
        let zero = Complex::new(0.0, 0.0);
        let one = Complex::new(1.0, 0.0);
        Self {
            data: vec![
                vec![one, zero, zero, zero],
                vec![zero, one, zero, zero],
                vec![zero, zero, one, zero],
                vec![zero, zero, zero, Complex::new(-1.0, 0.0)],
            ],
        }
    }

    pub fn tensor_product(gate1: &Gate, gate2: &Gate) -> Gate {
        let n = gate1.data.len();
        let m = gate2.data.len();

        let mut data = vec![vec![Complex::new(0.0, 0.0); n * m]; n * m];

        for i in 0..n {
            for j in 0..n {
                for k in 0..m {
                    for l in 0..m {
                        data[i * m + k][j * m + l] = gate1.data[i][j] * gate2.data[k][l]
                    }
                }
            }
        }
        Gate { data }
    }

    pub fn lift(gate: &Gate, target: usize, n: usize) -> Gate {
        let identity = Self::identity();

        let mut result = if target == 0 {
            gate.clone()
        } else {
            identity.clone()
        };

        for i in 1..n {
            let next_gate = if i == target { gate } else { &identity };
            result = Self::tensor_product(&result, next_gate);
        }

        result
    }
}
