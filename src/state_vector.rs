use crate::complex_math::Complex;

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

pub struct StateVec {
    pub data: Vec<Complex>,
}

impl StateVec {
    pub fn from(data: Vec<Complex>) -> Self {
        Self { data }
    }

    pub fn norm2(&self) -> f64 {
        self.data[0].abs2() + self.data[1].abs2()
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
}
