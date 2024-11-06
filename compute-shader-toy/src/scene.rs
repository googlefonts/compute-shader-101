use kurbo::{Affine, BezPath, Stroke};
use peniko::Color;

use crate::{flatten::SoupBowl, pico_svg::Item};

pub fn create_basic_scene(lines: &mut SoupBowl) {
    let mut path = BezPath::new();
    path.move_to((2., 2.));
    path.line_to((100., 30.));
    path.line_to((20., 100.));
    path.close_path();
    let style = Stroke::new(1.0);
    lines.stroke(&path, &style, 1.0, Color::YELLOW);
    let path2 = Affine::translate((0.0, 200.0)) * path;
    lines.fill(&path2, 1.0, Color::MAGENTA);
}

pub fn flatten_svg(lines: &mut SoupBowl, items: &[Item], scale: f64) {
    for item in items {
        match item {
            crate::pico_svg::Item::Fill(fill_item) => {
                lines.fill(&fill_item.path, scale, fill_item.color)
            }
            crate::pico_svg::Item::Stroke(stroke_item) => {
                let style = Stroke::new(stroke_item.width);
                lines.stroke(&stroke_item.path, &style, scale, stroke_item.color);
            }
            crate::pico_svg::Item::Group(group_item) => {
                flatten_svg(lines, &group_item.children, scale)
            }
        }
    }
}
