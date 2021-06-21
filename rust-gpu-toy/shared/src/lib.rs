#![no_std]

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Config {
    pub width: u32,
    pub height: u32,
    pub time: f32,
}
