//! Utilities for flattening

use flatten::stroke::LoweredPath;
use kurbo::{BezPath, Line, Stroke};

use crate::tiling::LineSoup;

pub fn flatten(path: &BezPath, path_ix: u32, soup: &mut Vec<LineSoup>) {
    let style = Stroke::new(1.0);
    let lines: LoweredPath<Line> = flatten::stroke::stroke_undashed(path, &style, 0.25);
    for line in &lines.path {
        let p0 = [line.p0.x as f32, line.p0.y as f32];
        let p1 = [line.p1.x as f32, line.p1.y as f32];
        soup.push(LineSoup::new(path_ix, p0, p1));
    }
}
