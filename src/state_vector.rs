use rand::Rng;

use crate::{complex_math::Complex, gates::Gate};

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
        let norm_facor = self.norm2().sqrt();
        if norm_facor > 0.0 {
            let inv = 1.0 / norm_facor;
            self.data[0] = self.data[0].scale(inv);
            self.data[1] = self.data[1].scale(inv);
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
        let bra = self.data[0].conj() * rhs.data[0];
        let ket = self.data[1].conj() * rhs.data[1];
        bra + ket
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
        let a = self.data[0];
        let b = self.data[1];
        let c = other.data[0];
        let d = other.data[1];

        Self {
            data: vec![a * c, a * d, b * c, b * d],
        }
    }
    pub fn measure(&mut self) -> usize {
        let mut rng = rand::rng();
        let r = rng.random_range(0.0..1.0);

        let mut cumulative_probability = 0.0;
        let mut measured_index = 0;
        for (i, amplitude) in self.data.iter().enumerate() {
            cumulative_probability += amplitude.abs2();
            if r < cumulative_probability {
                measured_index = i;
                break;
            }
        }
        for i in 0..self.data.len() {
            if i == measured_index {
                self.data[i] = Complex::new(1.0, 0.0);
            } else {
                self.data[i] = Complex::new(0.0, 0.0);
            }
        }

        measured_index
    }

    // pub fn measure_qubit(&mut self, qubit: usize) -> usize{};
}
