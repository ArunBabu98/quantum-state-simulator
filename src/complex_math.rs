/*
----------------------Complex Number Math---------------------------
Implements:
1. Complex struct
2. add, sub. mul
3. conj()
4. abs2 (absolute square)

Invariant:

1. (z * z.conj()).im == 0
2. abs2(z) >= 0
3. conj(conj(z)) == z

*/

/* ---------------------------tests------------------------ */

use std::ops::{Add, Mul, Sub};

#[test]
fn complex_conjugate() {
    let z = Complex::new(3.0, 4.0);
    let c = z.conj();
    assert_eq!(c.re, 3.0);
    assert_eq!(c.im, -4.0);
}

#[test]
fn complex_abs2() {
    let z = Complex::new(3.0, 4.0);
    assert_eq!(z.abs2(), 25.0);
}

#[test]
fn complex_mul_conj_is_real() {
    let z = Complex::new(1.2, -0.7);
    let conj = z.conj();
    let p = z * conj;
    assert!(p.im.abs() < 1e-10);
}

/* -------------------------End tests------------------------ */

#[derive(Clone, Copy)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn conj(&self) -> Complex {
        Complex {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn abs2(&self) -> f64 {
        (self.re.powi(2) + self.im.powi(2)).abs()
    }

    pub fn scale(&self, factor: f64) -> Complex {
        Complex {
            re: self.re * factor,
            im: self.im * factor,
        }
    }
}

impl Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Complex {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}
