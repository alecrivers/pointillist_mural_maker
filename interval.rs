use std::ops::{Add, Sub};

// An interval from 0 to 1
pub trait Interval: Add + Sub + Sized {
    fn from_f64(f: f64) -> Self;
    fn as_f64(&self) -> f64;
    fn increment(&mut self) -> bool;
    fn decrement(&mut self) -> bool;
    fn zero() -> Self {
        Self::from_f64(0.0)
    }
    fn one() -> Self {
        Self::from_f64(1.0)
    }
}

#[derive(Clone, PartialEq, Debug, Copy)]
pub struct Interval100 {
    value: u8,
}

impl Add for Interval100 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Interval100 {
            value: self.value + other.value,
        }
    }
}

impl Sub for Interval100 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Interval100 {
            value: self.value - other.value,
        }
    }
}

impl Interval for Interval100 {
    fn from_f64(f: f64) -> Self {
        Interval100 {
            value: (f * 100.0).round() as u8,
        }
    }

    fn as_f64(&self) -> f64 {
        self.value as f64 / 100.
    }

    fn increment(&mut self) -> bool {
        if self.value < 100 {
            self.value += 1;
            true
        } else {
            false
        }
    }

    fn decrement(&mut self) -> bool {
        if self.value > 0 {
            self.value -= 1;
            true
        } else {
            false
        }
    }
}
