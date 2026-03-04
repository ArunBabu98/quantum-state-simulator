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

use std::ops::{Add, Mul, Sub};

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
        self.re * self.re + self.im * self.im
    }

    pub fn scale(&self, factor: f64) -> Complex {
        Complex {
            re: self.re * factor,
            im: self.im * factor,
        }
    }
    
    pub fn approx_eq(&self, other: &Complex, epsilon: f64) -> bool {
        (self.re - other.re).abs() < epsilon && (self.im - other.im).abs() < epsilon
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
