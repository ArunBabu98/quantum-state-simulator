use rand::Rng;

use crate::{complex_math::Complex, gates::Gate};

pub struct StateVec {
    pub data: Vec<Complex>,
}

impl StateVec {
    pub fn from(data: Vec<Complex>) -> Self {
        Self { data }
    }

    pub fn norm2(&self) -> f64 {
        self.data.iter().map(|c| c.abs2()).sum()
    }

    pub fn normalize(&mut self) {
        let norm = self.norm2().sqrt();
        if norm > 1e-15 {
            let inv = 1.0 / norm;
            for amplitude in self.data.iter_mut() {
                *amplitude = amplitude.scale(inv);
            }
        }
    }

    pub fn basis_zero() -> Self {
        Self {
            data: vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
        }
    }

    pub fn basis_one() -> Self {
        Self {
            data: vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
        }
    }

    pub fn inner(&self, rhs: &StateVec) -> Complex {
        assert_eq!(self.data.len(), rhs.data.len(), "Dimension mismatch");
        self.data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a.conj() * *b)
            .fold(Complex::new(0.0, 0.0), |acc, x| acc + x)
    }

    pub fn approx_eq(&self, other: &StateVec) -> bool {
        let epsilon = 1e-10; // Standard tolerance for quantum math

        if self.data.len() != other.data.len() {
            return false;
        }

        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(a, b)| a.approx_eq(b, epsilon))
    }

    pub fn is_physically_equivalent(&self, other: &StateVec) -> bool {
        let overlap = self.inner(other);
        // If the absolute overlap squared is 1, they are the same state
        (overlap.abs2() - 1.0).abs() < 1e-10
    }

    pub fn tensor(&self, other: &StateVec) -> StateVec {
        let mut new_data = Vec::with_capacity(self.data.len() * other.data.len());
        for a in &self.data {
            for b in &other.data {
                new_data.push(*a * *b);
            }
        }
        StateVec::from(new_data)
    }

    pub fn measure(&mut self) -> usize {
        // Invariant: state must be normalized
        debug_assert!(
            (self.norm2() - 1.0).abs() < 1e-9,
            "State is not normalized before measurement"
        );

        let mut rng = rand::rng();
        let r: f64 = rng.random_range(0.0..1.0);

        let mut cumulative = 0.0;
        // Default to last index to handle floating point rounding at the tail
        let mut measured_index = self.data.len() - 1;

        for (i, amplitude) in self.data.iter().enumerate() {
            cumulative += amplitude.abs2();
            if r < cumulative {
                measured_index = i;
                break;
            }
        }

        // Collapse: zero out everything, set measured basis state to 1
        for (i, amplitude) in self.data.iter_mut().enumerate() {
            *amplitude = if i == measured_index {
                Complex::new(1.0, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
        }

        measured_index
    }

    pub fn measure_qubit(&mut self, qubit: usize) -> usize {
        let num_amplitudes = self.data.len();
        let num_qubits = num_amplitudes.trailing_zeros() as usize; // log2(len)

        debug_assert!(qubit < num_qubits, "Qubit index out of range");
        debug_assert!(
            (self.norm2() - 1.0).abs() < 1e-9,
            "State is not normalized before measurement"
        );

        // Step 1: Compute probability that qubit = 1
        // Bit position in index: qubit 0 = MSB = highest bit
        let bit_pos = num_qubits - 1 - qubit;

        let prob_one: f64 = self
            .data
            .iter()
            .enumerate()
            .filter(|(i, _)| (i >> bit_pos) & 1 == 1)
            .map(|(_, amp)| amp.abs2())
            .sum();

        // Step 2: Throw the dart
        let mut rng = rand::rng();
        let r: f64 = rng.random_range(0.0..1.0);
        let outcome = if r < prob_one { 1 } else { 0 };

        // Step 3: Zero out inconsistent amplitudes
        for (i, amp) in self.data.iter_mut().enumerate() {
            let bit = (i >> bit_pos) & 1;
            if bit != outcome {
                *amp = Complex::new(0.0, 0.0);
            }
        }

        // Step 4: Re-normalize surviving amplitudes
        self.normalize();

        outcome
    }
}
