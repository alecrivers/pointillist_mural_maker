use clap::Parser;
use image::Rgb;
use anyhow::{ensure, Result};
use palette::{Srgb, Hsl};
use palette::convert::*;

type Palette = Vec<Rgb<u8>>;

fn hex_to_rgb(hex: &str) -> Result<Rgb<u8>> {
    ensure!(hex.len() == 7, "Hex color must be 7 characters long, including the leading #");
    let r = u8::from_str_radix(&hex[1..3], 16)?;
    let g = u8::from_str_radix(&hex[3..5], 16)?;
    let b = u8::from_str_radix(&hex[5..7], 16)?;

    Ok(Rgb([r, g, b]))
}

fn rgb_to_hex(color: &Rgb<u8>) -> String {
    format!("#{:02X}{:02X}{:02X}", color[0], color[1], color[2])
}

// fn rgb_to_hsl(color: [f64; 3]) -> (f64, f64, f64) {
//     let r = color[0] / 255.0;
//     let g = color[1] / 255.0;
//     let b = color[2] / 255.0;

//     let max = r.max(g).max(b);
//     let min = r.min(g).min(b);
//     let delta = max - min;

//     // Hue calculation
//     let hue = if delta == 0.0 {
//         0.0
//     } else if max == r {
//         60.0 * (((g - b) / delta) % 6.0)
//     } else if max == g {
//         60.0 * (((b - r) / delta) + 2.0)
//     } else { // max == b
//         60.0 * (((r - g) / delta) + 4.0)
//     };

//     // Lightness calculation
//     let lightness = (max + min) / 2.0;

//     // Saturation calculation
//     let saturation = if delta == 0.0 {
//         0.0
//     } else {
//         delta / (1.0 - (2.0 * lightness - 1.0).abs())
//     };

//     (hue, saturation, lightness)
// }

// ...
fn rgb_to_hsl(color: [f64; 3]) -> (f64, f64, f64) {
    // Create an Srgb color from the f64 values
    let rgb_color = Srgb::from_components((color[0] as f32, color[1] as f32, color[2] as f32));

    // Convert to HSL
    let hsl_color: Hsl = rgb_color.into_color();
    
    // Extract HSL values
    (
        hsl_color.hue.into_positive_degrees() as f64,
        hsl_color.saturation as f64,
        hsl_color.lightness as f64
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MuralPixel {
    // The color of the pixel
    color: image::Rgb<u8>,
    // The current mixture of palette colors, from 0 to 255 where 0 is 0% and 255 is 100%
    mixture: Vec<u8>,
}

impl MuralPixel {
    fn compute_color(&self, palette: &Palette) -> Rgb<u8> {
        // Compute the color of this pixel, given a palette of colors
        // The color is the weighted average of the palette colors, weighted by the mixture
        Rgb([
            self.mixture.iter().zip(palette.iter()).map(|(&mix, color)| (mix as f64 / 255.0) * color[0] as f64).sum::<f64>() as u8,
            self.mixture.iter().zip(palette.iter()).map(|(&mix, color)| (mix as f64 / 255.0) * color[1] as f64).sum::<f64>() as u8,
            self.mixture.iter().zip(palette.iter()).map(|(&mix, color)| (mix as f64 / 255.0) * color[2] as f64).sum::<f64>() as u8,
        ])
    }

    fn compute_color_f64(&self, palette: &Palette) -> [f64; 3] {
        // Compute the color of this pixel, given a palette of colors
        // The color is the weighted average of the palette colors, weighted by the mixture
        [
            self.mixture.iter().zip(palette.iter()).map(|(&mix, color)| (mix as f64 / 255.0) * color[0] as f64).sum::<f64>(),
            self.mixture.iter().zip(palette.iter()).map(|(&mix, color)| (mix as f64 / 255.0) * color[1] as f64).sum::<f64>(),
            self.mixture.iter().zip(palette.iter()).map(|(&mix, color)| (mix as f64 / 255.0) * color[2] as f64).sum::<f64>(),
        ]
    }

    fn score(&self, palette: &Palette) -> f64 {
        // Compute the score of this pixel, given a palette of colors
        // The score is the distance between the current color and the computed color
        let computed_color = self.compute_color_f64(palette);
        let color = [
            self.color[0] as f64,
            self.color[1] as f64,
            self.color[2] as f64,
        ];
        //(computed_color[0] - color[0]).powi(2) + (computed_color[1] - color[1]).powi(2) + (computed_color[2] - color[2]).powi(2)
        // Rather than just comparing RGB raw values, the score should depend primarily on the HUE
        // To do this, convert to HSL
        let computed_color = rgb_to_hsl(computed_color);
        let color = rgb_to_hsl(color);
        // Then, compare the hue, saturation, and lightness
        // The hue should be the most important, then saturation, then lightness
        // The hue should be a circular distance, so that 0.0 and 1.0 are close together
        let hue_diff = (computed_color.0 - color.0).abs().min((computed_color.0 - color.0).abs() - 1.0).abs();
        let saturation_diff = (computed_color.1 - color.1).abs();
        let lightness_diff = (computed_color.2 - color.2).abs();
        hue_diff.powi(2) + saturation_diff.powi(2) + lightness_diff.powi(2)
    }

    fn greedily_compute_mixture(&mut self, palette: &Palette) {
        // Greedily compute the mixture of palette colors for this pixel
        // Given a palette of colors, c_0…c_i.
        //   Initialize this pixel to be 100% c_0.
        //   Greedily, for this pixel, repeat this process:
        //     For all colors represented in this pixel:
        //       Contemplate switching 5% to another color.
        //       Take the resulting mixture, and compute a score, where the score is highest if it is closest to the original color.
        //     Make the move that has the highest resulting score.
        //   If no pixels changed, stop.

        loop {
            let mut best_score = self.score(palette);
            let mut best_mixture = self.mixture.clone();

            for (i, &mix) in self.mixture.iter().enumerate() {
                if mix > 0 {
                    // Try moving 1 step of this color to another color
                    for j in 0..palette.len() {
                        if i != j {
                            let mut new_pixel = self.clone();
                            new_pixel.mixture[i] -= 1;
                            new_pixel.mixture[j] += 1;
                            let score = new_pixel.score(palette);
                            if score < best_score {
                                best_score = score;
                                best_mixture = new_pixel.mixture;
                            }
                        }
                    }
                }
            }

            if best_mixture != self.mixture {
                self.mixture = best_mixture;
            } else {
                break;
            }
        }
    }
}

struct Mural {
    palette: Vec<image::Rgb<u8>>,
    pixels: ndarray::Array2<MuralPixel>,
}

impl Mural {
    fn new(palette: Vec<image::Rgb<u8>>, image: image::RgbImage) -> Self {
        let pixels = ndarray::Array2::from_shape_fn((image.width() as usize, image.height() as usize), |(x, y)| {
            MuralPixel {
                color: image.get_pixel(x as u32, y as u32).clone(),
                // Initialize each pixel's mixture to be 1.0 of the first palette color, and 0.0 of the rest
                mixture: palette.iter().enumerate().map(|(i, _)| if i == 0 { 255 } else { 0 }).collect(),
            }
        });

        Mural {
            palette,
            pixels
        }
    }

    fn get_image(&self) -> image::RgbImage {
        let mut img = image::RgbImage::new(self.pixels.shape()[0] as u32, self.pixels.shape()[1] as u32);
        for ((x, y), pixel) in self.pixels.indexed_iter() {
            // For each pixel, convert the mixture of palette colors to a single color
            img.put_pixel(x as u32, y as u32, pixel.compute_color(&self.palette));
        }
        img
    }

    fn greedily_compute_mixture(&mut self) {
        // Greedily compute the mixture of palette colors for each pixel
        // For each pixel, greedily compute the mixture of palette colors
        for pixel in self.pixels.iter_mut() {
            pixel.greedily_compute_mixture(&self.palette);
        }
    }
}

// Use clap to take command line arguments: a single image file, plus a list of four palette colors as hex strings
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short,long)]
    image: String,

    #[arg(short,long)]
    palette: String,

    #[arg(short, long)]
    output: String,
}

// Add a main function
fn main() {
    // Parse the command line arguments
    let args = Cli::parse();

    // Load an image from a file
    let img = image::open(&args.image).unwrap().to_rgb8();

    // Initialize mural
    let mut mural = Mural::new(
        // Parse the palette colors from the command line arguments
        args.palette.split(',').map(|s| hex_to_rgb(s).unwrap()).collect(),
        img,
    );

    // Run greedy algorithm
    mural.greedily_compute_mixture();

    // Save out mural
    mural.get_image().save(args.output.clone() + ".png").unwrap();

    // Save also an SVG with a circle at each pixel location the size of which corresponds to the mixture
    let mut svg = svg::Document::new()
        .set("viewBox", (0, 0, mural.pixels.shape()[0], mural.pixels.shape()[1]))
        .add(svg::node::element::Rectangle::new().set("width", mural.pixels.shape()[0]).set("height", mural.pixels.shape()[1]).set("fill", "white"));
    for ((x, y), pixel) in mural.pixels.indexed_iter() {
        // For each palette color, add a circle, where the size of the circle corresponds to the mixture
        // For now, assume we have four palette colors, and put the circles in a 2x2 grid
        for (i, &mix) in pixel.mixture.iter().enumerate() {
            let color = rgb_to_hex(&mural.palette[i]);
            let (x, y) = (x as f64, y as f64);
            let (x, y) = match i {
                0 => (x + 0.25, y + 0.25),
                1 => (x + 0.75, y + 0.25),
                2 => (x + 0.25, y + 0.75),
                3 => (x + 0.75, y + 0.75),
                _ => unreachable!(),
            };
            let radius = (mix as f64 / 255.0) * 0.5;
            svg = svg.add(svg::node::element::Circle::new().set("cx", x).set("cy", y).set("r", radius).set("fill", color));
        }
    }
    svg::save(args.output + ".svg", &svg).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_to_rgb() {
        assert_eq!(hex_to_rgb("#000000").unwrap(), Rgb([0, 0, 0]));
        assert_eq!(hex_to_rgb("#ffffff").unwrap(), Rgb([255, 255, 255]));
        assert_eq!(hex_to_rgb("#ff0000").unwrap(), Rgb([255, 0, 0]));
        assert_eq!(hex_to_rgb("#00ff00").unwrap(), Rgb([0, 255, 0]));
        assert_eq!(hex_to_rgb("#0000ff").unwrap(), Rgb([0, 0, 255]));
    }

    #[test]
    fn test_mural() {
        let palette = vec![
            Rgb([0, 0, 0]),
            Rgb([255, 255, 255]),
        ];
        let image = image::RgbImage::from_fn(2, 2, |x, y| {
            if x == 0 && y == 0 {
                Rgb([0, 0, 0])
            } else if x == 1 && y == 0 {
                Rgb([255, 255, 255])
            } else if x == 0 && y == 1 {
                Rgb([255, 255, 255])
            } else {
                Rgb([0, 0, 0])
            }
        });
        let mut mural = Mural::new(palette, image);
        mural.greedily_compute_mixture();
        assert_eq!(mural.pixels[[0, 0]].mixture, vec![255, 0]);
        assert_eq!(mural.pixels[[1, 0]].mixture, vec![0, 255]);
        assert_eq!(mural.pixels[[0, 1]].mixture, vec![0, 255]);
        assert_eq!(mural.pixels[[1, 1]].mixture, vec![255, 0]);
    }

    #[test]
    fn test_greedily_compute_mixture() {
        let palette = vec![
            Rgb([0, 0, 0]),
            Rgb([255, 0, 0,]),
            Rgb([0, 255, 0,]),
            Rgb([122, 122, 122,]),
        ];
        // Test a few possible pixel colors, individually
        let mut pixel = MuralPixel {
            color: Rgb([0, 0, 0]),
            mixture: vec![255, 0, 0, 0],
        };
        pixel.greedily_compute_mixture(&palette);
        assert_eq!(pixel.mixture, vec![255, 0, 0, 0]);
        let mut pixel = MuralPixel {
            color: Rgb([255, 0, 0]),
            mixture: vec![255, 0, 0, 0],
        };
        pixel.greedily_compute_mixture(&palette);
        assert_eq!(pixel.mixture, vec![0, 255, 0, 0]);
        let mut pixel = MuralPixel {
            color: Rgb([0, 255, 0]),
            mixture: vec![255, 0, 0, 0],
        };
        pixel.greedily_compute_mixture(&palette);
        assert_eq!(pixel.mixture, vec![0, 0, 255, 0]);
        let mut pixel = MuralPixel {
            color: Rgb([122, 122, 122]),
            mixture: vec![255, 0, 0, 0],
        };
        pixel.greedily_compute_mixture(&palette);
        assert_eq!(pixel.mixture, vec![0, 0, 0, 255]);
    }
}
