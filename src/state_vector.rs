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

pub struct StateVec {
    data: Vec<Complex>,
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
}
