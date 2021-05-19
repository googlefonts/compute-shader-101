// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

// Note: doing this by hand is silly and it should probably be replaced by
// the bytemuck crate.

pub trait Codable: Sized {
    const SIZE: usize;

    fn decode(buf: &[u8]) -> Self;

    fn encode(&self, buf: &mut [u8]);

    // It might be just as useful to provide the iterator.
    fn decode_vec(buf: &[u8]) -> Vec<Self> {
        buf.chunks_exact(Self::SIZE).map(Codable::decode).collect()
    }

    fn encode_vec(src: &[Self]) -> Vec<u8> {
        let mut result = vec![0; src.len() * Self::SIZE];
        for (chunk, val) in result.chunks_exact_mut(Self::SIZE).zip(src) {
            val.encode(chunk);
        }
        result
    }
}

impl Codable for u64 {
    const SIZE: usize = 8;

    fn decode(buf: &[u8]) -> Self {
        let mut mybuf = [0; 8];
        mybuf[..].copy_from_slice(buf);
        u64::from_le_bytes(mybuf)
    }

    fn encode(&self, buf: &mut [u8]) {
        buf.copy_from_slice(&u64::to_le_bytes(*self))
    }
}

impl Codable for f32 {
    const SIZE: usize = 4;

    fn decode(buf: &[u8]) -> Self {
        let mut mybuf = [0; 4];
        mybuf[..].copy_from_slice(buf);
        f32::from_le_bytes(mybuf)
    }

    fn encode(&self, buf: &mut [u8]) {
        buf.copy_from_slice(&f32::to_le_bytes(*self))
    }
}
