use std::f64::consts::PI;

use clap::Parser;
use image::Rgb;
mod colors;
mod circular_ops;
mod interval;
use interval::*;
use colors::*;

type Palette = Vec<Rgb<u8>>;

pub trait ColorMixture
    where Self: Sized
{
    fn compute_color(&self) -> [f64; 3];
    fn possible_moves(&self) -> Vec<Self>;
    fn add_to_svg(&self, svg: svg::Document, x: f64, y: f64) -> svg::Document;
}

#[derive(Clone, PartialEq, Debug)]
struct MuralColorMixture {
    background_color: Rgb<u8>,
    channels: Vec<(Rgb<u8>, Interval100)>
}

impl MuralColorMixture {
    fn new(background_color: Rgb<u8>, palette: &Palette) -> Self {
        MuralColorMixture {
            background_color,
            channels: palette.iter().map(|color| (color.clone(), Interval100::from_f64(0.))).collect()
        }
    }

    fn as_f64(&self) -> Vec<f64> {
        self.channels.iter().map(|(_, interval)| interval.as_f64()).collect()
    }

    fn from_f64(background_color: Rgb<u8>, palette: &Palette, f: Vec<f64>) -> Self {
        MuralColorMixture {
            background_color,
            channels: palette.iter().zip(f).map(|(color, f)| (color.clone(), Interval100::from_f64(f))).collect()
        }
    }
}

fn radius_from_area(area: f64) -> f64 {
    (area / PI).sqrt()
}

impl ColorMixture for MuralColorMixture {
    fn compute_color(&self) -> [f64; 3] {
        // I'm going to ignore reality and just assume that the colors are additive
        let mut background_intensity = Interval100::one();
        for (_, intensity) in self.channels.iter() {
            background_intensity = background_intensity - intensity.clone();
        }
        // Add in the bg color to make a full list of colors
        let mut colors_plus_background = self.channels.clone();
        colors_plus_background.push((self.background_color, background_intensity));
        // OK, now let's blend.
        let mut color = [0., 0., 0.];
        for (c, intensity) in colors_plus_background {
            let c = color_to_f64(&c);
            color[0] += c[0] * intensity.as_f64();
            color[1] += c[1] * intensity.as_f64();
            color[2] += c[2] * intensity.as_f64();
        }
        color
    }

    fn possible_moves(&self) -> Vec<Self> {
        let mut moves = Vec::new();
        // For each channel, I can decrement it if it's nonzero.
        for (i, mix) in self.channels.iter().enumerate() {
            let mut c = self.channels.clone();
            if c[i].1.decrement() {
                moves.push(MuralColorMixture {
                    background_color: self.background_color,
                    channels: c
                });
            }
        }
        // If we're not maxed out, for each channel I can increment it if it's not maxed out.
        // First add all channels together and see if that's incrementable (i.e. not maxxed out):
        let mut sum = Interval100::from_f64(0.);
        for mix in self.channels.iter() {
            sum = sum + mix.1;
        }
        if sum.increment() {
            // OK, we're not maxxed out.
            for (i, mix) in self.channels.iter().enumerate() {
                let mut c = self.channels.clone();
                if c[i].1.increment() {
                    moves.push(MuralColorMixture {
                        background_color: self.background_color,
                        channels: c
                    });
                }
            }
        }

        moves
    }

    fn add_to_svg(&self, svg: svg::Document, x: f64, y: f64) -> svg::Document {
        // For each palette color, add a circle, where the size of the circle corresponds to the mixture
        // For now, assume we have no more than four palette colors, and put the circles in a 2x2 grid
        let mut svg = svg;
        for (i, &channel) in self.channels.iter().enumerate() {
            let color = rgb_to_hex(&channel.0);
            let (x, y) = (x as f64, y as f64);
            let (x, y) = match i {
                0 => (x + 0.25, y + 0.25),
                1 => (x + 0.75, y + 0.25),
                2 => (x + 0.25, y + 0.75),
                3 => (x + 0.75, y + 0.75),
                _ => unreachable!(),
            };
            // We want the AREA of the circle to be the intensity
            let area = channel.1.as_f64();
            let radius = radius_from_area(area);
            svg = svg.add(svg::node::element::Circle::new().set("cx", x).set("cy", y).set("r", radius).set("fill", color));
        }
        svg
    }
}

struct MuralPixel {
    target_color: Rgb<u8>,
    mixture: MuralColorMixture
}

impl MuralPixel {
    fn new(color: Rgb<u8>, background_color: Rgb<u8>, palette: &Palette) -> Self {
        MuralPixel {
            target_color: color,
            mixture: MuralColorMixture::new(background_color, palette),
        }
    }

    fn compute_color(&self) -> [f64; 3] {
        self.mixture.compute_color()
    }

    fn score(&self) -> f64 {
        let target_color_f64 = [
            self.target_color[0] as f64 / 255.,
            self.target_color[1] as f64 / 255.,
            self.target_color[2] as f64 / 255.,
        ];
        score(&target_color_f64, &self.mixture.compute_color())
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

struct Mural {
    pixels: ndarray::Array2<MuralPixel>,
}

impl Mural {
    fn new(background_color: Rgb<u8>, palette: Vec<image::Rgb<u8>>, image: image::RgbImage) -> Self {
        let pixels = ndarray::Array2::from_shape_fn((image.width() as usize, image.height() as usize), |(x, y)| {
            MuralPixel::new(image.get_pixel(x as u32, y as u32).clone(), background_color, &palette)
        });

        Mural {
            pixels
        }
    }

    fn get_image(&self) -> image::RgbImage {
        let mut img = image::RgbImage::new(self.pixels.shape()[0] as u32, self.pixels.shape()[1] as u32);
        for ((x, y), pixel) in self.pixels.indexed_iter() {
            // For each pixel, convert the mixture of palette colors to a single color
            img.put_pixel(x as u32, y as u32, f64_to_color(&pixel.mixture.compute_color()));
        }
        img
    }

    fn greedily_compute_mixture(&mut self) {
        // Greedily compute the mixture of palette colors for each pixel
        // For each pixel, greedily compute the mixture of palette colors
        for pixel in self.pixels.iter_mut() {
            pixel.greedily_compute_mixture();
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
    let mut all_colors = args.palette.split(',').map(|s| hex_to_rgb(s).unwrap());
    // The first color is the background color.
    let background_color = all_colors.next().unwrap();
    let palette_colors: Palette = all_colors.collect();
    assert!(palette_colors.len() <= 4, "Must specify no more than four palette colors");
    let mut mural = Mural::new(
        // Parse the palette colors from the command line arguments
        background_color,
        palette_colors,
        img,
    );

    // Run greedy algorithm
    mural.greedily_compute_mixture();

    // Save out mural
    mural.get_image().save(args.output.clone() + ".png").unwrap();

    // Save also an SVG with a circle at each pixel location the size of which corresponds to the mixture
    let mut svg = svg::Document::new()
        .set("viewBox", (0, 0, mural.pixels.shape()[0], mural.pixels.shape()[1]))
        .add(svg::node::element::Rectangle::new().set("width", mural.pixels.shape()[0]).set("height", mural.pixels.shape()[1]).set("fill", rgb_to_hex(&background_color)));
    for ((x, y), pixel) in mural.pixels.indexed_iter() {
        svg = pixel.mixture.add_to_svg(svg, x as f64, y as f64);
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
        let background_color = Rgb([0, 0, 0]);
        let palette = vec![
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
        let mut mural = Mural::new(background_color, palette, image);
        mural.greedily_compute_mixture();
        assert_eq!(mural.pixels[[0, 0]].mixture.as_f64(), vec![0.]);
        assert_eq!(mural.pixels[[1, 0]].mixture.as_f64(), vec![1.0]);
        assert_eq!(mural.pixels[[0, 1]].mixture.as_f64(), vec![1.0]);
        assert_eq!(mural.pixels[[1, 1]].mixture.as_f64(), vec![0.]);
    }

    fn run_pixel_test(color: Rgb<u8>, background_color: Rgb<u8>, palette: Palette, expected_mixture_intensities: Vec<f64>) {
        let mut pixel = MuralPixel::new(color, background_color, &palette);
        println!("===========================");
        println!("Target color: {:?}", pixel.target_color);
        println!("Initial mixture: {:?}", pixel.mixture);
        println!("Color of initial mixture: {:?}", pixel.compute_color());
        println!("Score of initial mixture: {}", pixel.score());
        pixel.greedily_compute_mixture();
        println!("Final mixture: {:?}", pixel.mixture);
        println!("Color of final mixture: {:?}", pixel.compute_color());
        println!("Score of final mixture: {}", pixel.score());
        let expected_mixture = MuralColorMixture::from_f64(background_color, &palette, expected_mixture_intensities);
        println!("Color of expected mixture: {:?}", expected_mixture.compute_color());
        println!("Score of expected mixture: {}", MuralPixel { target_color: color, mixture: expected_mixture.clone() }.score());
        assert!(pixel.mixture == expected_mixture);
    }

    #[test]
    fn test_greedily_compute_mixture() {
        let background_color = Rgb([0, 0, 0]);
        let palette = vec![
            Rgb([255, 0, 0,]),
            Rgb([0, 255, 0,]),
            Rgb([122, 122, 122,]),
        ];
        // Test a few possible pixel colors, individually
        run_pixel_test(Rgb([0,0,0]), background_color, palette.clone(), vec![0., 0., 0.]);
        run_pixel_test(Rgb([255,0,0]), background_color, palette.clone(), vec![1., 0., 0.]);
        run_pixel_test(Rgb([0,255,0]), background_color, palette.clone(), vec![0., 1., 0.]);
        run_pixel_test(Rgb([122,122,122]), background_color, palette.clone(), vec![0., 0., 1.]);
    }
}
