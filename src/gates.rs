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

    pub fn lift_multi(controlled_gate: &Gate, control: usize, target: usize, n: usize) -> Gate {
        // For a 2-qubit gate, we assume it's acting on 'control' and 'target'.
        // We decompose it into: Projector on control=0 + Projector on control=1 * U on target.

        let identity_single = Self::identity();

        // 1. Component for Control = |0> (Identity applied to target)
        // Projector |0><0| = [[1, 0], [0, 0]]
        let p0 = Gate {
            data: vec![
                vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
                vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
            ],
        };

        // 2. Component for Control = |1> (U applied to target)
        // Projector |1><1| = [[0, 0], [0, 1]]
        let p1 = Gate {
            data: vec![
                vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
                vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
            ],
        };

        // Extract the U gate from the bottom-right of the controlled gate
        let u_data = vec![
            vec![controlled_gate.data[2][2], controlled_gate.data[2][3]],
            vec![controlled_gate.data[3][2], controlled_gate.data[3][3]],
        ];
        let u_gate = Gate { data: u_data };

        // Helper to build the full-space tensor chain
        let build_chain = |control_gate: &Gate, target_gate: &Gate| {
            let mut res = if 0 == control {
                control_gate.clone()
            } else if 0 == target {
                target_gate.clone()
            } else {
                identity_single.clone()
            };

            for i in 1..n {
                let next = if i == control {
                    control_gate
                } else if i == target {
                    target_gate
                } else {
                    &identity_single
                };
                res = Self::tensor_product(&res, next);
            }
            res
        };

        let part1 = build_chain(&p0, &identity_single);
        let part2 = build_chain(&p1, &u_gate);

        // Matrix Addition: part1 + part2
        let mut final_data = part1.data;
        for i in 0..final_data.len() {
            for j in 0..final_data.len() {
                final_data[i][j] = final_data[i][j] + part2.data[i][j];
            }
        }

        Gate { data: final_data }
    }

    pub fn controlled(gate: &Gate) -> Self {
        let size = gate.data.len();
        let new_dim = size * 2;
        let mut data = vec![vec![Complex::new(0.0, 0.0); new_dim]; new_dim];
        let zero = Complex::new(0.0, 0.0);
        let one = Complex::new(1.0, 0.0);
        for i in 0..new_dim {
            for j in 0..new_dim {
                if i < size && j < size {
                    data[i][j] = if i == j { one } else { zero };
                } else if i >= size && j >= size {
                    data[i][j] = gate.data[i - size][j - size];
                } else {
                    data[i][j] = zero;
                }
            }
        }

        Gate { data }
    }
}
