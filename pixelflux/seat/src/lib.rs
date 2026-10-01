/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! # pixelflux-seat
//!
//! The part of pixelflux a remote-desktop seat needs, as a Rust library: capture one rectangle of
//! an X screen through MIT-SHM (only when the screen or the cursor changed), draw the cursor into
//! the picture, and encode it with the NVIDIA encoder at pixelflux's low-latency settings
//! (ultra-low-latency tuning, infinite GOP with key frames on request, CBR with a VBV of 1.5
//! frames, no reordering, parameter sets repeated on every key frame).
//!
//! The full pixelflux crate is a Python extension that also carries a Wayland compositor, a
//! virtual camera and software encoders (GPL x264/x265 in its default build). This crate leaves
//! all of that out and links only the NVENC/CUDA bindings (`nvcodec-sys`) and libxcb-free X11
//! (`x11rb`), so it builds on any Linux with a C-free toolchain and loads the NVIDIA driver at
//! run time.

pub mod cuda;
pub mod nvenc;
pub mod x11;

pub use nvenc::{Codec, Encoder, EncoderConfig, EncodedFrame, H264Profile};
pub use x11::{Capture, Monitor, XDisplay};
