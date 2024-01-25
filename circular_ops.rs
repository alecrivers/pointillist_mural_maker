use std::ops::{Add, Rem, Sub, Neg, Div, Mul};

pub trait CircularOps: Sized + Copy + PartialOrd + From<i32> + Add<Output = Self> + Sub<Output = Self> + Rem<Output = Self> + Neg<Output = Self> + Div<Output = Self> + Mul<Output = Self> {}

impl<T> CircularOps for T where T: Sized + Copy + PartialOrd + From<i32> + Add<Output = Self> + Sub<Output = Self> + Rem<Output = Self> + Neg<Output = Self> + Div<Output = Self> + Mul<Output = Self> {}

pub fn circular_wrap<T: CircularOps>(value: T, wrap_val: T) -> T {
    if value == T::from(0) && wrap_val == T::from(0) {
        return T::from(0); // Special case
    }

    let mut kx = value;
    if value < T::from(0) {
        kx = kx + wrap_val * (-kx / wrap_val + T::from(1));
    }

    kx % wrap_val
}

pub fn circular_offset<T: CircularOps>(v0: T, v1: T, wrap_val: T) -> T {
    assert!(T::from(-1) < T::from(0)); // Ensure T is a signed type

    let v0_wrap = circular_wrap(v0, wrap_val);
    let v1_wrap = circular_wrap(v1, wrap_val);

    let mut o = v1_wrap - v0_wrap;
    if o < -wrap_val / T::from(2) {
        o = o + wrap_val;
    } else if o > wrap_val / T::from(2) {
        o = o - wrap_val;
    }
    o
}
