#[cfg(test)]
mod complextest {
    use crate::complex_math::Complex;

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
}
