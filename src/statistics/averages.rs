// gwaihir
// Essential Statistical Functions for Rust
// Copyright: 2018, Maani Beigy <manibeygi@gmail.com>, 
// License: MIT/Apache-2.0
//! # gwaihir
//!
//! `gwaihir` is a library for Essential Statistical Functions for Rust.
//! 
// ------------------ bring external functions/traits ------------------------
use std::iter::Sum;

pub fn mean<'a, T, I>(numerics: I) -> f64
where
    T: Into<f64> + Sum<&'a T> + 'a,
    I: Iterator<Item = &'a T>,
{
    let mut len = 0;
    let sum = numerics
        .map(|t| {
            len += 1;
            t
        })
        .sum::<T>();
    use std::f64::NAN;
    match len {
        0 => NAN.abs(),
        _ => (sum.into() / len as f64),
    }
}

