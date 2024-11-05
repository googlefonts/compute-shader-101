//! Visualizations for sparse strip rendering.

use crate::Strip;

const SCALE: f32 = 10.0;
const INSET: f32 = 1.5;
const CELL: f32 = SCALE - 1.0 * INSET;

#[allow(unused)]
pub fn visualize_strips(strips: &[Strip], alphas: &[u32]) {
    println!("<svg width='1000' height='1000' viewBox='0 0 1000 1000' xmlns='http://www.w3.org/2000/svg'>");
    for i in 0..strips.len() - 1 {
        let strip = &strips[i];
        let next_strip = &strips[i + 1];
        let strip_x = strip.xy & 0xffff;
        let strip_y = strip.xy >> 16;
        let width = next_strip.col - strip.col;
        for x in 0..width {
            let a = alphas[(strip.col + x) as usize];
            for y in 0..4 {
                let g = (a >> (y * 8)) & 0xff;
                let rgb = (255 - g) * 0x10101;
                println!("  <rect x='{}' y='{}' width='{CELL}' height='{CELL}' fill='#{rgb:06x}' />",
                    (strip_x + x) as f32 * SCALE + INSET,
                    strip_y as f32 * SCALE + y as f32 * (CELL + INSET) + INSET,
                );
            }
        }
        println!("  <rect x='{}' y='{}' width='{}' height='{}' fill='none' stroke='#000' />",
            strip_x as f32 * SCALE + INSET,
            strip_y as f32 * SCALE + INSET,
            width as f32 * SCALE - 2.0 * INSET,
            4.0 * SCALE - 2.0 * INSET,
        );
        if next_strip.winding != 0 {
            let next_x = next_strip.xy & 0xffff;
            let next_y = next_strip.xy >> 16;
            if strip_y == next_y {
                println!("  <rect x='{}' y='{}' width='{}' height='{}' fill='#888' />",
                    (strip_x + width) as f32 * SCALE + INSET,
                    strip_y as f32 * SCALE + INSET,
                    (next_x - strip_x - width) as f32 * SCALE - 2.0 * INSET,
                    4.0 * SCALE - 2.0 * INSET,
                );

            }
        }
    }
    println!("</svg>");
}
