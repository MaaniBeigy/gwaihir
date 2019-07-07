// gwaihir
// Essential Statistical Functions for Rust
// Copyright: 2018, Maani Beigy <manibeygi@gmail.com>, 
// License: MIT/Apache-2.0
//! # gwaihir
//!
//! `gwaihir` is a library for Essential Statistical Functions for Rust.
//! 
// ----------------------------- bring Modules --------------------------------
mod statistics;
// ------------------ bring external libraries/crates -------------------------
extern crate rayon;
extern crate rgsl;
// ------------------ bring external functions/traits -------------------------

// ------------------ bring internal functions/traits -------------------------

fn main() {
    let numbers = [42, 1, 36, 34, 76, 378, 43, 1, 43, 54, 2, 3, 43];
    let numbers_two = [
        42.0, 1.0, 36.0, 34.0, 76.0, 378.0, 43.0, 1.0, 43.0, 54.0, 2.0, 3.0, 
        43.0
        ];
    let empty: Vec<i32> = Vec::new();

    println!("generic mean: {}", statistics::averages::mean(
        numbers.iter())
        );
    println!("empty vector mean: {}", statistics::averages::mean(
        empty.iter())
        );
    println!("generic sum: {}", statistics::base::sum(
        numbers.iter())
        );
    println!("empty vector sum: {}", statistics::base::sum(
        empty.iter())
        );
    println!("generic min: {:?}", statistics::base::minimum(&numbers_two));
    println!("min empty: {:?}", statistics::base::minimum(&empty));
    println!("generic max: {:?}", statistics::base::maximum(&numbers_two));
    println!("max empty: {:?}", statistics::base::maximum(&empty));
}