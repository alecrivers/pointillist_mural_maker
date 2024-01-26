use clap::Parser;
use image::Rgb;
use anyhow::{ensure, Result};
use palette::convert::*;
mod colors;
mod circular_ops;
use colors::*;

// type Palette = Vec<Rgb<u8>>;

// An interval from 0 to 1
trait Interval {
    fn from_f64(f: f64) -> Self;
    fn as_f64(&self) -> f64;
    fn increment(&mut self) -> bool;
    fn decrement(&mut self) -> bool;
}

#[derive(Clone, PartialEq, Debug, Copy)]
struct Interval100 {
    value: u8,
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

pub trait ColorMixture<const PaletteCount: usize>
    where Self: Sized
{
    fn compute_color(&self) -> [f64; 3];
    fn possible_moves(&self) -> Vec<Self>;
}

#[derive(Clone, PartialEq, Debug)]
struct MuralColorMixture<const PaletteCount: usize> {
    background_color: Rgb<u8>,
    palette: [Rgb<u8>; PaletteCount],
    mixture: [Interval100; PaletteCount],
}

impl<const PaletteCount: usize> MuralColorMixture<PaletteCount> {
    fn new(background_color: Rgb<u8>, palette: [Rgb<u8>; PaletteCount]) -> Self {
        MuralColorMixture {
            background_color,
            palette,
            // The mixture should be an array of Interval100 of length PaletteCount, each initialized to zero
            mixture: [Interval100::from_f64(0.); PaletteCount],
        }
    }
}

impl<const PaletteCount: usize> ColorMixture<PaletteCount> for MuralColorMixture<PaletteCount> {
    fn compute_color(&self) -> [f64; 3] {
        let mut color = self.background_color.clone();
        panic!("TODO");
    }

    fn possible_moves(&self) -> Vec<Self> {
        let mut moves = Vec::new();
        for (i, mix) in self.mixture.iter().enumerate() {
            {
                let mut c = self.mixture.clone();
                if c[i].decrement() {
                    moves.push(MuralColorMixture {
                        background_color: self.background_color,
                        palette: self.palette,
                        mixture: c,
                    });
                }
            }
            {
                let mut c = self.mixture.clone();
                if c[i].increment() {
                    moves.push(MuralColorMixture {
                        background_color: self.background_color,
                        palette: self.palette,
                        mixture: c,
                    });
                }
            }
        }
        moves
    }
}

struct MuralPixel<const PaletteCount: usize> {
    target_color: Rgb<u8>,
    mixture: MuralColorMixture<PaletteCount>
}

impl<const PaletteCount: usize> MuralPixel<PaletteCount> {
    fn new(color: Rgb<u8>, palette: [Rgb<u8>; PaletteCount]) -> Self {
        MuralPixel {
            target_color: color,
            mixture: MuralColorMixture::new(color, palette),
        }
    }

    fn greedily_compute_mixture(&mut self) {
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
            let target_color_f64 = [
                self.target_color[0] as f64 / 255.,
                self.target_color[1] as f64 / 255.,
                self.target_color[2] as f64 / 255.,
            ];
            let mut best_score = score(&target_color_f64, &self.mixture.compute_color());
            let mut best_mixture = self.mixture.clone();

            for mix in self.mixture.possible_moves() {
                let score = score(&target_color_f64, &mix.compute_color());
                if score < best_score {
                    best_score = score;
                    best_mixture = mix;
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

struct Mural<const PaletteCount: usize> {
    pixels: ndarray::Array2<MuralPixel<PaletteCount>>,
}

impl<const PaletteCount: usize> Mural<PaletteCount> {
    fn new(palette: Vec<image::Rgb<u8>>, image: image::RgbImage) -> Self {
        let pixels = ndarray::Array2::from_shape_fn((image.width() as usize, image.height() as usize), |(x, y)| {
            MuralPixel {
                target_color: image.get_pixel(x as u32, y as u32).clone(),
                // Initialize each pixel's mixture to be 1.0 of the first palette color, and 0.0 of the rest
                mixture: MuralColorMixture::new(image.get_pixel(x as u32, y as u32).clone(), palette.clone())
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
            img.put_pixel(x as u32, y as u32, compute_color(&pixel.mixture, &self.palette));
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
    let mut mural = Mural::<MixtureChannel100>::new(
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
        for (i, &mix) in pixel.mixture.0.iter().enumerate() {
            let color = rgb_to_hex(&mural.palette[i]);
            let (x, y) = (x as f64, y as f64);
            let (x, y) = match i {
                0 => (x + 0.25, y + 0.25),
                1 => (x + 0.75, y + 0.25),
                2 => (x + 0.25, y + 0.75),
                3 => (x + 0.75, y + 0.75),
                _ => unreachable!(),
            };
            let radius = mix.as_f64() * 0.5;
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
        let mut mural = Mural::<MixtureChannel100>::new(palette, image);
        mural.greedily_compute_mixture();
        assert_eq!(mural.pixels[[0, 0]].mixture.as_f64(), vec![1.0, 0.]);
        assert_eq!(mural.pixels[[1, 0]].mixture.as_f64(), vec![0., 1.0]);
        assert_eq!(mural.pixels[[0, 1]].mixture.as_f64(), vec![0., 1.0]);
        assert_eq!(mural.pixels[[1, 1]].mixture.as_f64(), vec![1.0, 0.]);
    }

    fn run_pixel_test<T: MixtureChannel>(color: Rgb<u8>, palette: Vec<Rgb<u8>>, expected_mixture: ColorMixture<T>) {
        let mut pixel = MuralPixel {
            target_color: color,
            mixture: ColorMixture::<T>::from_f64(vec![1., 0., 0., 0.]),
        };
        println!("===========================");
        println!("Target color: {:?}", pixel.target_color);
        println!("Initial mixture: {:?}", pixel.mixture);
        println!("Color of initial mixture: {:?}", compute_color_f64(&pixel.mixture, &palette));
        println!("Score of initial mixture: {}", score(&pixel.target_color, &pixel.mixture, &palette));
        pixel.greedily_compute_mixture(&palette);
        println!("Final mixture: {:?}", pixel.mixture);
        println!("Color of final mixture: {:?}", compute_color(&pixel.mixture, &palette));
        println!("Score of final mixture: {}", score(&pixel.target_color, &pixel.mixture, &palette));
        println!("Color of expected mixture: {:?}", compute_color(&expected_mixture, &palette));
        println!("Score of expected mixture: {}", score(&pixel.target_color, &expected_mixture, &palette));
        assert!(pixel.mixture == expected_mixture);
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
        run_pixel_test(Rgb([0,0,0]), palette.clone(), ColorMixture::<MixtureChannel100>::from_f64(vec![1., 0., 0., 0.]));
        run_pixel_test(Rgb([255,0,0]), palette.clone(), ColorMixture::<MixtureChannel100>::from_f64(vec![0., 1., 0., 0.]));
        run_pixel_test(Rgb([0,255,0]), palette.clone(), ColorMixture::<MixtureChannel100>::from_f64(vec![0., 0., 1., 0.]));
        run_pixel_test(Rgb([122,122,122]), palette.clone(), ColorMixture::<MixtureChannel100>::from_f64(vec![0., 0., 0., 1.]));
    }
}
