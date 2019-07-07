// gwaihir
// Essential Statistical Functions for Rust
// Copyright: 2018, Maani Beigy <manibeygi@gmail.com>, 
// License: MIT/Apache-2.0
//! # gwaihir
//!
//! `gwaihir` is a library for Essential Statistical Functions for Rust.
//! 
// ------------------ bring external libraries/crates -------------------------
extern crate rayon;
extern crate rgsl;
// ------------------ bring external functions/traits -------------------------
use rayon::prelude::*;
use std::iter::Sum;

pub fn sum<'a, T, I>(numerics: I) -> f64
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
        _ => sum.into(),
    }
}



pub fn minimum<T>(numerics: &[T]) -> f64 where
    T: Into<f64>,
    T: PartialOrd + Copy,
{
    
    let len = numerics.len();
    use std::f64::NAN;
    if len == 0 { 
        NAN.abs()
    } else {
        let mut smallest = numerics[0];
        for &item in numerics.iter() {
            if item < smallest {
                smallest = item;
            }
        }
        smallest.into()
    }
    
}

pub fn maximum<T>(numerics: &[T]) -> f64 where
    T: Into<f64>,
    T: PartialOrd + Copy,
{
    
    let len = numerics.len();
    use std::f64::NAN;
    if len == 0 { 
        NAN.abs()
    } else {
        let mut largest = numerics[0];
        for &item in numerics.iter() {
            if item > largest {
                largest = item;
            }
        }
        largest.into()
    }
    
}



pub fn length_i32(numbers: &[i32]) -> usize {
    numbers.par_iter()
        .count()
}

pub fn sum_i32(numbers: &[i32]) -> i32 {
    numbers.par_iter()
        .map(|&x| x)
        .reduce(|| 0, |x, y| x + y)
}