#![no_std]

#[cfg(feature = "bytemuck")]
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(feature = "bytemuck", derive(Pod, Zeroable))]
pub struct Config {
    pub width: u32,
    pub height: u32,
    pub time: f32,
}
