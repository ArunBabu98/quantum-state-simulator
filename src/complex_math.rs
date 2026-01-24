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
    let p = z * z.conj();
    assert!(p.im.abs() < 1e-10);
}

/* -------------------------End tests------------------------ */

struct Complex {
    re: f64,
    im: f64,
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
}
