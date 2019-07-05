// gwaihir
// Essential Statistical Functions for Rust
// Copyright: 2018, Maani Beigy <manibeygi@gmail.com>, 
// License: MIT/Apache-2.0
//! # gwaihir
//!
//! `gwaihir` is a library for Essential Statistical Functions for Rust.
//! 
extern crate rayon;
extern crate rgsl;
use std::collections::HashMap;
use rayon::prelude::*;

fn length(numbers: &[i32]) -> usize {
    numbers.par_iter()
        .count()
}

fn sum(numbers: &[i32]) -> i32 {
    numbers.par_iter()
        .map(|&x| x)
        .reduce(|| 0, |x, y| x + y)
}

fn average(numbers: &[i32]) -> f64 {
    numbers.iter().sum::<i32>() as f64 / numbers.len() as f64
}

fn arithmetic_mean(numbers: &[i32]) -> f64 {
    (sum(&numbers) as f64)/(length(&numbers) as f64)
}


fn median(numbers: &mut [i32]) -> i32 {
    numbers.sort();
    let mid = numbers.len() / 2;
    numbers[mid]
}

fn mode(numbers: &[i32]) -> i32 {
    let mut occurrences = HashMap::new();

    for &value in numbers {
        *occurrences.entry(value).or_insert(0) += 1;
    }

    occurrences
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(val, _)| val)
        .expect("Cannot compute the mode of zero numbers")
}

fn main() {
    let mut numbers = [42, 1, 36, 34, 76, 378, 43, 1, 43, 54, 2, 3, 43];
    let mut numbers_two = [42.0, 1.0, 36.0, 34.0, 76.0, 378.0, 43.0, 1.0, 43.0, 54.0, 2.0, 3.0, 43.0];
    
    println!("COUNT: {}", length(&numbers));
    println!("SUM: {}", sum(&numbers));
    println!("ARITHMETIC MEAN: {}", arithmetic_mean(&numbers));
    println!("AVERAGE: {}", average(&numbers));
    println!("GSL MEAN: {}", rgsl::statistics::mean(&numbers_two, 1, length(&numbers)));
    println!("MEDIAN: {}", median(&mut numbers));
    println!("MODE: {}", mode(&numbers));
}