use palette::{Hsl, IntoColor, Srgb};
use image::Rgb;
use anyhow::{ensure, Result};
use crate::circular_ops::*;

pub fn color_to_f64(color: &Rgb<u8>) -> [f64; 3] {
    [
        color[0] as f64 / 255.0,
        color[1] as f64 / 255.0,
        color[2] as f64 / 255.0,
    ]
}

pub fn f64_to_color(color: &[f64; 3]) -> Rgb<u8> {
    Rgb([
        (color[0] * 255.0) as u8,
        (color[1] * 255.0) as u8,
        (color[2] * 255.0) as u8,
    ])
}

pub fn hex_to_rgb(hex: &str) -> Result<Rgb<u8>> {
    ensure!(hex.len() == 7, "Hex color must be 7 characters long, including the leading #");
    let r = u8::from_str_radix(&hex[1..3], 16)?;
    let g = u8::from_str_radix(&hex[3..5], 16)?;
    let b = u8::from_str_radix(&hex[5..7], 16)?;

    Ok(Rgb([r, g, b]))
}

pub fn rgb_to_hex(color: &Rgb<u8>) -> String {
    format!("#{:02X}{:02X}{:02X}", color[0], color[1], color[2])
}

fn rgb_to_hsl(color: &[f64; 3]) -> (f64, f64, f64) {
    // Create an Srgb color from the f64 values
    let rgb_color = Srgb::from_components((color[0] as f32, color[1] as f32, color[2] as f32));

    // Convert to HSL
    let hsl_color: Hsl = rgb_color.into_color();
    
    // Extract HSL values
    (
        (hsl_color.hue.into_positive_degrees() / 360.0) as f64,
        hsl_color.saturation as f64,
        hsl_color.lightness as f64
    )
}

pub fn score(color: &[f64; 3], target: &[f64; 3]) -> f64 {
    //(computed_color[0] - color[0]).powi(2) + (computed_color[1] - color[1]).powi(2) + (computed_color[2] - color[2]).powi(2)
    // Rather than just comparing RGB raw values, the score should depend primarily on the HUE
    // To do this, convert to HSL
    let color_hsl = rgb_to_hsl(color);
    let target_hsl = rgb_to_hsl(target);
    // Then, compare the hue, saturation, and lightness
    // The hue should be the most important, then saturation, then lightness
    // The hue should be a circular distance, so that 0.0 and 1.0 are close together
    let hue_diff = circular_offset(color_hsl.0, target_hsl.0, 1.0).abs();
    let saturation_diff = (color_hsl.1 - target_hsl.1).abs();
    let lightness_diff = (color_hsl.2 - target_hsl.2).abs();
    hue_diff.powi(2) + saturation_diff.powi(2) + lightness_diff.powi(2)
}
