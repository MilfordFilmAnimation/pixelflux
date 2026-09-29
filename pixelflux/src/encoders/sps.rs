//! The color a stream declares, read from an H.264 sequence parameter set and written into one.
//!
//! An encoder that converts RGB itself decides the matrix and the range, and the stream is the
//! only place a decoder learns which it was: with no `video_signal_type` in the VUI a browser
//! guesses from the frame size, so one session is colored one way at 1080p and another at
//! 800x600. Devices differ in whether they say: a Raspberry Pi with firmware from August 2024
//! or later converts and declares it, and the same board on an older firmware converts to full
//! range BT.601 and declares nothing at all.
//!
//! So this reads first. A stream that declares its color is left alone, whatever it declares,
//! because the device knows what it did and we do not. Only a stream that declares nothing is
//! written into, and then only with a value the caller can justify.
//!
//! The VUI sits at the end of the SPS, ahead of the trailing bits, and the fields after the
//! insertion point are copied as opaque bits rather than re-encoded: a writer that re-emits what
//! it does not understand is a writer that corrupts streams on devices it was never run against.
//!
//! The same VUI bounds reordering. Without a `bitstream_restriction` a decoder has to assume the
//! stream may reorder as deep as its level's decoded picture buffer, and Chromium's hardware H.264
//! decoder (VA-API, D3D11, VideoToolbox) holds that many pictures back before it outputs one, up to
//! sixteen at a small size. A stream whose pictures leave in the order they are shown is written a
//! bound of zero. That write reads through the timing and HRD parameters to reach the restriction,
//! still copying them as they came, and the read has to land on the stop bit, so a field it
//! stepped over wrongly refuses the write instead of corrupting the set.

/// What a stream says about the color it carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorSignal {
    pub full_range: bool,
    pub primaries: u8,
    pub transfer: u8,
    pub matrix: u8,
}

impl ColorSignal {
    /// BT.601 at full range, which is what a VideoCore firmware older than August 2024 converts
    /// RGB with, both by measurement against a decoded chart and by the driver author's account.
    pub const BT601_FULL: Self = Self {
        full_range: true,
        primaries: 6,
        transfer: 6,
        matrix: 6,
    };

    /// BT.709 at limited range, the crate's own convention for 4:2:0.
    pub const BT709_LIMITED: Self = Self {
        full_range: false,
        primaries: 1,
        transfer: 1,
        matrix: 1,
    };
}

struct Reader<'a> {
    bits: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bits: &'a [u8]) -> Self {
        Self { bits, pos: 0 }
    }

    fn bit(&mut self) -> Result<u32, String> {
        let byte = self
            .bits
            .get(self.pos >> 3)
            .ok_or("the SPS ends mid-field")?;
        let value = (byte >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Ok(value as u32)
    }

    fn bits(&mut self, count: usize) -> Result<u32, String> {
        let mut value = 0;
        for _ in 0..count {
            value = (value << 1) | self.bit()?;
        }
        Ok(value)
    }

    /// Exp-Golomb, the coding every length in an SPS is written with.
    fn ue(&mut self) -> Result<u32, String> {
        let mut zeros = 0;
        while self.bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return Err("an exp-Golomb code longer than the field it names".into());
            }
        }
        if zeros == 0 {
            return Ok(0);
        }
        Ok((1 << zeros) - 1 + self.bits(zeros)?)
    }

    fn se(&mut self) -> Result<i32, String> {
        let value = self.ue()?;
        Ok(if value.is_multiple_of(2) {
            -((value / 2) as i32)
        } else {
            value.div_ceil(2) as i32
        })
    }
}

struct Writer {
    bytes: Vec<u8>,
    pos: usize,
}

impl Writer {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            pos: 0,
        }
    }

    fn bit(&mut self, value: u32) {
        if self.pos.is_multiple_of(8) {
            self.bytes.push(0);
        }
        if value & 1 != 0 {
            let index = self.pos >> 3;
            self.bytes[index] |= 1 << (7 - (self.pos & 7));
        }
        self.pos += 1;
    }

    fn bits(&mut self, value: u32, count: usize) {
        for shift in (0..count).rev() {
            self.bit((value >> shift) & 1);
        }
    }

    /// Copy bits `[from, to)` of `source` verbatim, which is how everything this code does not
    /// interpret survives an insertion that shifts it.
    fn copy_from(&mut self, source: &[u8], from: usize, to: usize) {
        for index in from..to {
            let byte = source[index >> 3];
            self.bit(((byte >> (7 - (index & 7))) & 1) as u32);
        }
    }

    /// Exp-Golomb, as `Reader::ue` reads it.
    fn ue(&mut self, value: u32) {
        let coded = value as u64 + 1;
        let length = 64 - coded.leading_zeros() as usize;
        for _ in 1..length {
            self.bit(0);
        }
        for shift in (0..length).rev() {
            self.bit(((coded >> shift) & 1) as u32);
        }
    }

    fn trailing_bits(&mut self) {
        self.bit(1);
        while !self.pos.is_multiple_of(8) {
            self.bit(0);
        }
    }
}

/// Where the video signal type sits in a VUI, and what it says today.
struct Located {
    /// The bit the block starts at: the `video_signal_type_present_flag` itself.
    start: usize,
    /// The bit after the block, where the rest of the VUI resumes.
    end: usize,
    /// Present when the VUI exists at all; a VUI has to be created otherwise.
    vui_present: bool,
    /// The bit the `vui_parameters_present_flag` sits at, for a stream that carries no VUI.
    vui_flag: usize,
    declared: Option<ColorSignal>,
    pic_order_cnt_type: u32,
    max_num_ref_frames: u32,
}

fn unescape(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal.len());
    let mut index = 0;
    while index < nal.len() {
        if index + 2 < nal.len() && nal[index] == 0 && nal[index + 1] == 0 && nal[index + 2] == 3 {
            out.push(0);
            out.push(0);
            index += 3;
        } else {
            out.push(nal[index]);
            index += 1;
        }
    }
    out
}

fn escape(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + 4);
    let mut zeros = 0;
    for &byte in rbsp {
        if zeros >= 2 && byte <= 3 {
            out.push(3);
            zeros = 0;
        }
        out.push(byte);
        zeros = if byte == 0 { zeros + 1 } else { 0 };
    }
    out
}

/// The last bit that carries meaning: the RBSP stop bit, with alignment zeros after it.
fn stop_bit(rbsp: &[u8]) -> Result<usize, String> {
    for index in (0..rbsp.len() * 8).rev() {
        let byte = rbsp[index >> 3];
        if (byte >> (7 - (index & 7))) & 1 == 1 {
            return Ok(index);
        }
    }
    Err("the SPS carries no stop bit".into())
}

fn locate(rbsp: &[u8]) -> Result<Located, String> {
    let mut r = Reader::new(rbsp);
    let profile = r.bits(8)?;
    r.bits(8)?;
    r.bits(8)?;
    r.ue()?;
    if matches!(
        profile,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    ) {
        let chroma = r.ue()?;
        if chroma == 3 {
            r.bit()?;
        }
        r.ue()?;
        r.ue()?;
        r.bit()?;
        if r.bit()? == 1 {
            // Scaling lists are the one part an insertion cannot step over blindly, so a stream
            // carrying them is refused rather than guessed at.
            return Err("the SPS carries scaling lists".into());
        }
    }
    r.ue()?;
    let poc_type = r.ue()?;
    if poc_type == 0 {
        r.ue()?;
    } else if poc_type == 1 {
        r.bit()?;
        r.se()?;
        r.se()?;
        let cycle = r.ue()?;
        for _ in 0..cycle {
            r.se()?;
        }
    }
    let max_num_ref_frames = r.ue()?;
    r.bit()?;
    r.ue()?;
    r.ue()?;
    if r.bit()? == 0 {
        r.bit()?;
    }
    r.bit()?;
    if r.bit()? == 1 {
        r.ue()?;
        r.ue()?;
        r.ue()?;
        r.ue()?;
    }
    let vui_flag = r.pos;
    if r.bit()? == 0 {
        return Ok(Located {
            start: vui_flag,
            end: vui_flag,
            vui_present: false,
            vui_flag,
            declared: None,
            pic_order_cnt_type: poc_type,
            max_num_ref_frames,
        });
    }
    if r.bit()? == 1 && r.bits(8)? == 255 {
        r.bits(16)?;
        r.bits(16)?;
    }
    if r.bit()? == 1 {
        r.bit()?;
    }
    let start = r.pos;
    let declared = if r.bit()? == 1 {
        r.bits(3)?;
        let full_range = r.bit()? == 1;
        if r.bit()? == 1 {
            Some(ColorSignal {
                full_range,
                primaries: r.bits(8)? as u8,
                transfer: r.bits(8)? as u8,
                matrix: r.bits(8)? as u8,
            })
        } else {
            // A range without a matrix says half of what a decoder needs, and the half it leaves
            // out is the one that colors the picture, so this counts as undeclared.
            None
        }
    } else {
        None
    };
    Ok(Located {
        start,
        end: r.pos,
        vui_present: true,
        vui_flag,
        declared,
        pic_order_cnt_type: poc_type,
        max_num_ref_frames,
    })
}

/// Where a VUI's bitstream restriction sits, and the reordering it declares.
struct Restriction {
    /// The bit the `bitstream_restriction_flag` sits at, or the VUI's own flag where the SPS
    /// carries no VUI.
    flag: usize,
    /// `max_num_reorder_frames` and `max_dec_frame_buffering` with the bits each code spans, or
    /// None where the stream declares no restriction.
    declared: Option<((u32, usize, usize), (u32, usize, usize))>,
}

/// The restriction of the VUI `at` located, read through what follows its video signal type.
/// The read has to end on the stop bit, the check that no field was stepped over wrongly.
fn restriction(rbsp: &[u8], at: &Located) -> Result<Restriction, String> {
    let last = stop_bit(rbsp)?;
    if !at.vui_present {
        return if at.vui_flag + 1 == last {
            Ok(Restriction {
                flag: at.vui_flag,
                declared: None,
            })
        } else {
            Err("the SPS does not end after its VUI flag".into())
        };
    }
    let mut r = Reader {
        bits: rbsp,
        pos: at.end,
    };
    if r.bit()? == 1 {
        r.ue()?;
        r.ue()?;
    }
    if r.bit()? == 1 {
        r.bits(32)?;
        r.bits(32)?;
        r.bit()?;
    }
    let nal_hrd = r.bit()? == 1;
    if nal_hrd {
        hrd(&mut r)?;
    }
    let vcl_hrd = r.bit()? == 1;
    if vcl_hrd {
        hrd(&mut r)?;
    }
    if nal_hrd || vcl_hrd {
        r.bit()?;
    }
    r.bit()?;
    let flag = r.pos;
    let declared = if r.bit()? == 1 {
        r.bit()?;
        for _ in 0..4 {
            r.ue()?;
        }
        let at = r.pos;
        let reorder = (r.ue()?, at, r.pos);
        let at = r.pos;
        Some((reorder, (r.ue()?, at, r.pos)))
    } else {
        None
    };
    if r.pos != last {
        return Err("the VUI does not end where the SPS does".into());
    }
    Ok(Restriction { flag, declared })
}

/// Step over one `hrd_parameters` structure.
fn hrd(r: &mut Reader) -> Result<(), String> {
    let schedules = r.ue()? + 1;
    if schedules > 32 {
        return Err("an HRD with more schedules than the syntax allows".into());
    }
    r.bits(8)?;
    for _ in 0..schedules {
        r.ue()?;
        r.ue()?;
        r.bit()?;
    }
    r.bits(20)?;
    Ok(())
}

/// What the stream says about its color, or `None` when it says nothing a decoder can use.
pub fn read_color(nal: &[u8]) -> Option<ColorSignal> {
    let rbsp = unescape(nal.get(1..)?);
    locate(&rbsp).ok()?.declared
}

/// The same SPS with `signal` declared in its VUI, replacing whatever it declared before.
pub fn write_color(nal: &[u8], signal: ColorSignal) -> Result<Vec<u8>, String> {
    let header = *nal.first().ok_or("an empty NAL unit")?;
    let rbsp = unescape(&nal[1..]);
    let at = locate(&rbsp)?;
    let last = stop_bit(&rbsp)?;

    let mut w = Writer::new();
    if at.vui_present {
        w.copy_from(&rbsp, 0, at.start);
        write_signal(&mut w, signal);
        w.copy_from(&rbsp, at.end, last);
    } else {
        w.copy_from(&rbsp, 0, at.vui_flag);
        w.bit(1);
        w.bit(0);
        w.bit(0);
        write_signal(&mut w, signal);
        // The rest of a VUI this stream never had: chroma location, timing, both HRDs, picture
        // structure, and bitstream restriction, each absent.
        for _ in 0..6 {
            w.bit(0);
        }
    }
    w.trailing_bits();

    let mut out = Vec::with_capacity(w.bytes.len() + 8);
    out.push(header);
    out.extend_from_slice(&escape(&w.bytes));
    Ok(out)
}

fn write_signal(w: &mut Writer, signal: ColorSignal) {
    w.bit(1);
    w.bits(5, 3);
    w.bit(u32::from(signal.full_range));
    w.bit(1);
    w.bits(signal.primaries as u32, 8);
    w.bits(signal.transfer as u32, 8);
    w.bits(signal.matrix as u32, 8);
}

/// The same SPS bounding reordering at zero, or None where it already does: a restriction
/// written where the VUI carries none (a VUI created where there is none), with every other
/// restriction field at the value a decoder infers in its absence and `max_dec_frame_buffering`
/// at `max_num_ref_frames`; a declared `max_num_reorder_frames` above zero rewritten to zero.
/// Only a stream that cannot reorder takes it: `in_order` where the session proves its pictures
/// leave in the order they are shown, else a stream at picture order count type 2, whose
/// output order is its decoding order.
pub fn bound_reorder(nal: &[u8], in_order: bool) -> Result<Option<Vec<u8>>, String> {
    let header = *nal.first().ok_or("an empty NAL unit")?;
    let rbsp = unescape(&nal[1..]);
    let at = locate(&rbsp)?;
    let bound = restriction(&rbsp, &at)?;
    if let Some(((0, _, _), (buffering, ..))) = bound.declared
        && buffering >= at.max_num_ref_frames
    {
        return Ok(None);
    }
    if !in_order && at.pic_order_cnt_type != 2 {
        return Err(format!(
            "picture order count type {} may reorder",
            at.pic_order_cnt_type
        ));
    }
    let last = stop_bit(&rbsp)?;
    let mut w = Writer::new();
    match bound.declared {
        Some(((_, reorder_at, _), (buffering, _, buffering_end))) => {
            w.copy_from(&rbsp, 0, reorder_at);
            w.ue(0);
            w.ue(buffering.max(at.max_num_ref_frames));
            w.copy_from(&rbsp, buffering_end, last);
        }
        None => {
            w.copy_from(&rbsp, 0, bound.flag);
            if !at.vui_present {
                w.bit(1);
                for _ in 0..8 {
                    w.bit(0);
                }
            }
            w.bit(1);
            w.bit(1);
            w.ue(2);
            w.ue(1);
            w.ue(15);
            w.ue(15);
            w.ue(0);
            w.ue(at.max_num_ref_frames);
        }
    }
    w.trailing_bits();
    let mut out = Vec::with_capacity(w.bytes.len() + 8);
    out.push(header);
    out.extend_from_slice(&escape(&w.bytes));
    Ok(Some(out))
}

/// Where each H.264 sequence parameter set sits in an Annex B access unit, as `(start, end)` of
/// the NAL unit itself, the start code excluded. A set repeats with every key frame when the
/// encoder is asked for headers on each one, and a stream whose first set alone is written says
/// one thing to the client that connected first and another to the one that joined later.
pub fn sequence_parameter_sets(unit: &[u8]) -> Vec<(usize, usize)> {
    let mut starts = Vec::new();
    let mut index = 0;
    while index + 3 < unit.len() {
        if unit[index] == 0 && unit[index + 1] == 0 && unit[index + 2] == 1 {
            starts.push(index + 3);
            index += 3;
        } else {
            index += 1;
        }
    }
    let mut sets = Vec::new();
    for (position, &start) in starts.iter().enumerate() {
        if unit[start] & 0x1f != 7 {
            continue;
        }
        let mut end = starts.get(position + 1).map_or(unit.len(), |next| next - 3);
        if end > start && unit[end - 1] == 0 {
            end -= 1;
        }
        sets.push((start, end));
    }
    sets
}

/// A device's H.264 stream held to a reorder bound of zero: each sequence parameter set it
/// sends is rewritten once and matched by its bytes after that, since a device repeats the
/// same set with every key frame.
pub struct NoReorder {
    /// What the stream is named as in the log.
    source: &'static str,
    in_order: bool,
    /// The last set seen and what replaces it, None where it is sent as it came.
    last: Option<(Vec<u8>, Option<Vec<u8>>)>,
    /// Whether the last line logged said the sets are rewritten, so each outcome is said once.
    logged: Option<bool>,
}

impl NoReorder {
    pub fn new(source: &'static str, in_order: bool) -> Self {
        Self {
            source,
            in_order,
            last: None,
            logged: None,
        }
    }

    /// The set that replaces `sps`, or None where it goes out as it came.
    pub fn set(&mut self, sps: &[u8]) -> Option<Vec<u8>> {
        if let Some((was, now)) = &self.last
            && was == sps
        {
            return now.clone();
        }
        let now = match bound_reorder(sps, self.in_order) {
            Ok(bounded) => {
                if bounded.is_some() && self.logged != Some(true) {
                    self.logged = Some(true);
                    println!(
                        "[pixelflux] {} declares no bound on reordering; its sequence parameter sets are written a bound of zero.",
                        self.source
                    );
                }
                bounded
            }
            Err(e) => {
                if self.logged != Some(false) {
                    self.logged = Some(false);
                    eprintln!(
                        "[pixelflux] {}'s sequence parameter set was left as it came, without a reorder bound of zero: {e}",
                        self.source
                    );
                }
                None
            }
        };
        self.last = Some((sps.to_vec(), now.clone()));
        now
    }

    /// `unit` with every sequence parameter set in it replaced, or None where none is.
    pub fn apply(&mut self, unit: &[u8]) -> Option<Vec<u8>> {
        let mut out: Option<Vec<u8>> = None;
        let mut copied = 0;
        for (start, end) in sequence_parameter_sets(unit) {
            if let Some(set) = self.set(&unit[start..end]) {
                let out = out.get_or_insert_with(|| Vec::with_capacity(unit.len() + 16));
                out.extend_from_slice(&unit[copied..start]);
                out.extend_from_slice(&set);
                copied = end;
            }
        }
        let mut out = out?;
        out.extend_from_slice(&unit[copied..]);
        Some(out)
    }
}

/// The sequence parameter set read an H.264 session makes of its own key frames, and the
/// reference checks of every backend share.
mod dpb {
    struct Bits<'a> {
        rbsp: &'a [u8],
        pos: usize,
    }

    impl Bits<'_> {
        fn bits(&mut self, n: u32) -> u32 {
            (0..n).fold(0, |acc, _| {
                let bit = self
                    .rbsp
                    .get(self.pos / 8)
                    .map_or(0, |b| (b >> (7 - self.pos % 8)) & 1);
                self.pos += 1;
                (acc << 1) | bit as u32
            })
        }

        fn ue(&mut self) -> u32 {
            let mut zeros = 0;
            while self.bits(1) == 0 && zeros < 32 {
                zeros += 1;
            }
            (1 << zeros) - 1 + self.bits(zeros)
        }

        fn se(&mut self) -> i32 {
            let k = self.ue() as i32;
            if k % 2 == 1 { (k + 1) / 2 } else { -(k / 2) }
        }
    }

    /// `(log2_max_frame_num, max_num_ref_frames, chroma_format_idc)` of the first SPS in an
    /// Annex-B H.264 stream.
    fn h264_sps(stream: &[u8]) -> Option<(u32, u32, u32)> {
        let nal = crate::encoders::codec::annexb_nals(stream).find(|n| n[0] & 0x1f == 7)?;
        let mut rbsp = Vec::with_capacity(nal.len());
        let mut zeros = 0;
        for &b in &nal[1..] {
            if zeros >= 2 && b == 3 {
                zeros = 0;
                continue;
            }
            zeros = if b == 0 { zeros + 1 } else { 0 };
            rbsp.push(b);
        }
        let mut r = Bits {
            rbsp: &rbsp,
            pos: 0,
        };
        let profile = r.bits(8);
        r.bits(16);
        r.ue();
        let mut chroma = 1;
        if matches!(
            profile,
            100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
        ) {
            chroma = r.ue();
            if chroma == 3 {
                r.bits(1);
            }
            r.ue();
            r.ue();
            r.bits(1);
            if r.bits(1) == 1 {
                for list in 0..(if chroma == 3 { 12 } else { 8 }) {
                    if r.bits(1) == 1 {
                        let (mut last, mut next) = (8i32, 8i32);
                        for _ in 0..(if list < 6 { 16 } else { 64 }) {
                            if next != 0 {
                                next = (last + r.se()).rem_euclid(256);
                            }
                            if next != 0 {
                                last = next;
                            }
                        }
                    }
                }
            }
        }
        let log2_max_frame_num = r.ue() + 4;
        match r.ue() {
            0 => {
                r.ue();
            }
            1 => {
                r.bits(1);
                r.se();
                r.se();
                for _ in 0..r.ue() {
                    r.se();
                }
            }
            _ => {}
        }
        Some((log2_max_frame_num, r.ue(), chroma))
    }

    /// How many values `frame_num` takes before it wraps, from the first SPS of an Annex-B
    /// H.264 stream.
    pub fn h264_frame_num_range(stream: &[u8]) -> Option<u32> {
        h264_sps(stream).map(|(log2, _, _)| 1 << log2)
    }

    /// `max_num_ref_frames` of the first SPS in an Annex-B H.264 stream.
    #[cfg(test)]
    pub fn h264_max_num_ref_frames(stream: &[u8]) -> Option<u32> {
        h264_sps(stream).map(|(_, refs, _)| refs)
    }

    /// `chroma_format_idc` of the first SPS in an Annex-B H.264 stream: 3 for 4:4:4.
    #[cfg(test)]
    pub fn h264_chroma_format_idc(stream: &[u8]) -> Option<u32> {
        h264_sps(stream).map(|(_, _, chroma)| chroma)
    }
}

pub use dpb::h264_frame_num_range;
#[cfg(test)]
pub use dpb::{h264_chroma_format_idc, h264_max_num_ref_frames};

/// The reordering the first SPS of an Annex B H.264 stream bounds, as `(max_num_reorder_frames,
/// max_dec_frame_buffering)`, or None where it declares no bound or cannot be read.
#[cfg(test)]
pub fn h264_reorder(stream: &[u8]) -> Option<(u32, u32)> {
    let nal = crate::encoders::codec::annexb_nals(stream).find(|n| n[0] & 0x1f == 7)?;
    let rbsp = unescape(&nal[1..]);
    let bound = restriction(&rbsp, &locate(&rbsp).ok()?).ok()?;
    bound
        .declared
        .map(|((reorder, ..), (buffering, ..))| (reorder, buffering))
}

/// The timing the first SPS of an Annex B H.264 stream declares, as `(num_units_in_tick,
/// time_scale)`, or None where it declares none or cannot be read.
#[cfg(test)]
pub fn h264_timing(stream: &[u8]) -> Option<(u32, u32)> {
    let nal = crate::encoders::codec::annexb_nals(stream).find(|n| n[0] & 0x1f == 7)?;
    let rbsp = unescape(&nal[1..]);
    let at = locate(&rbsp).ok()?;
    if !at.vui_present {
        return None;
    }
    let mut r = Reader {
        bits: &rbsp,
        pos: at.end,
    };
    if r.bit().ok()? == 1 {
        r.ue().ok()?;
        r.ue().ok()?;
    }
    if r.bit().ok()? == 0 {
        return None;
    }
    Some((r.bits(32).ok()?, r.bits(32).ok()?))
}

/// Sequence parameter sets devices wrote, and the edits the tests make of them.
#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;

    /// An SPS a Raspberry Pi 4 produced at 1280x720: a VUI with timing and no color at all.
    pub const PI4_SPS: &[u8] = &[
        0x27, 0x64, 0x00, 0x28, 0xac, 0x2b, 0x40, 0x28, 0x02, 0xdd, 0x08, 0x00, 0x00, 0x03, 0x00,
        0x08, 0x00, 0x00, 0x03, 0x01, 0xe7, 0x15, 0x00, 0x01, 0xe8, 0x48, 0x00, 0x02, 0xfa, 0xf3,
        0x7b, 0xdc, 0x03, 0xc4, 0x89, 0xa8,
    ];

    /// The SPS a Radeon Pro VII's VCE firmware writes for 1920x1080 under Mesa 24.0.5, whatever
    /// the session packed: color and timing in the VUI, no bitstream restriction, picture order
    /// count type 0, two reference frames.
    pub const VCE_SPS: &[u8] = &[
        0x67, 0x64, 0x40, 0x2a, 0xac, 0x26, 0xc0, 0x78, 0x02, 0x27, 0xe5, 0xc0, 0x5a, 0x20, 0x00,
        0x00, 0x03, 0x00, 0x20, 0x00, 0x00, 0x0f, 0x10, 0x80,
    ];

    /// The restriction an SPS declares, or None where it declares none.
    pub fn reorder_of(nal: &[u8]) -> Option<(u32, u32)> {
        let rbsp = unescape(&nal[1..]);
        let bound =
            restriction(&rbsp, &locate(&rbsp).expect("located")).expect("read to the stop bit");
        bound
            .declared
            .map(|((reorder, ..), (buffering, ..))| (reorder, buffering))
    }

    /// `nal` rebuilt with its bitstream restriction taken out, or its whole VUI.
    pub fn without(nal: &[u8], whole_vui: bool) -> Vec<u8> {
        let rbsp = unescape(&nal[1..]);
        let at = locate(&rbsp).expect("located");
        let bound = restriction(&rbsp, &at).expect("read");
        let mut w = Writer::new();
        w.copy_from(&rbsp, 0, if whole_vui { at.vui_flag } else { bound.flag });
        w.bit(0);
        w.trailing_bits();
        let mut out = vec![nal[0]];
        out.extend_from_slice(&escape(&w.bytes));
        out
    }

    /// Assert the first SPS of the Annex B `stream` bounds reordering at zero in a buffer no
    /// smaller than its references: the set a hardware decoder outputs each picture at once
    /// for, and one Chromium's parser accepts.
    pub fn assert_no_reorder(stream: &[u8], what: &str) {
        let refs = h264_max_num_ref_frames(stream).unwrap_or_else(|| panic!("{what}: no SPS"));
        match h264_reorder(stream) {
            Some((0, buffering)) => {
                assert!(
                    buffering >= refs,
                    "{what}: a buffer of {buffering} below its {refs} references"
                )
            }
            other => panic!("{what}: the SPS bounds reordering at {other:?}, not zero"),
        }
    }

    /// `nal` rebuilt declaring a reorder depth of `reorder` in a buffer of `buffering`.
    pub fn with_depth(nal: &[u8], reorder: u32, buffering: u32) -> Vec<u8> {
        let rbsp = unescape(&nal[1..]);
        let bound = restriction(&rbsp, &locate(&rbsp).expect("located")).expect("read");
        let ((_, at, _), _) = bound.declared.expect("a restriction to rewrite");
        let mut w = Writer::new();
        w.copy_from(&rbsp, 0, at);
        w.ue(reorder);
        w.ue(buffering);
        w.trailing_bits();
        let mut out = vec![nal[0]];
        out.extend_from_slice(&escape(&w.bytes));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;
    use crate::RustCaptureSettings;

    #[test]
    fn a_stream_that_declares_nothing_reads_as_nothing() {
        assert_eq!(read_color(PI4_SPS), None);
    }

    #[test]
    fn what_is_written_reads_back() {
        for signal in [ColorSignal::BT601_FULL, ColorSignal::BT709_LIMITED] {
            let patched = write_color(PI4_SPS, signal).expect("the SPS takes a color");
            assert_eq!(
                read_color(&patched),
                Some(signal),
                "what was written did not read back"
            );
        }
    }

    /// The fields after the insertion point are what the encoder said about timing and delay, and
    /// a stream whose insertion trod on them decodes at the wrong rate rather than failing.
    #[test]
    fn the_bits_after_the_insertion_survive_it() {
        let patched = write_color(PI4_SPS, ColorSignal::BT601_FULL).expect("patched");
        let (before, after) = (unescape(&PI4_SPS[1..]), unescape(&patched[1..]));
        let at_before = locate(&before).expect("located");
        let at_after = locate(&after).expect("located");
        let tail_before = stop_bit(&before).expect("stop") - at_before.end;
        let tail_after = stop_bit(&after).expect("stop") - at_after.end;
        assert_eq!(tail_before, tail_after, "the tail changed length");
        for index in 0..tail_before {
            let bit = |data: &[u8], at: usize| (data[at >> 3] >> (7 - (at & 7))) & 1;
            assert_eq!(
                bit(&before, at_before.end + index),
                bit(&after, at_after.end + index),
                "bit {index} of the tail changed"
            );
        }
    }

    #[test]
    fn a_declared_color_reads_back_as_declared() {
        let patched = write_color(PI4_SPS, ColorSignal::BT709_LIMITED).expect("patched");
        assert_eq!(read_color(&patched), Some(ColorSignal::BT709_LIMITED));
        let again = write_color(&patched, ColorSignal::BT601_FULL).expect("rewritten");
        assert_eq!(
            read_color(&again),
            Some(ColorSignal::BT601_FULL),
            "a rewrite did not replace"
        );
        assert_eq!(
            again.len(),
            patched.len(),
            "replacing a color changed the length"
        );
    }

    /// Emulation prevention is not decoration: a byte pair of zeros followed by a small byte is
    /// a start code to a decoder unless it is escaped.
    #[test]
    fn escaping_round_trips() {
        let raw = vec![0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x03, 0xff, 0x00, 0x00];
        assert_eq!(unescape(&escape(&raw)), raw);
        assert!(escape(&raw).len() > raw.len(), "nothing was escaped");
    }

    /// Whether the first `bits` bits of two sets' payloads agree.
    fn same_prefix(a: &[u8], b: &[u8], bits: usize) -> bool {
        let (a, b) = (unescape(&a[1..]), unescape(&b[1..]));
        (0..bits).all(|i| (a[i >> 3] >> (7 - (i & 7))) & 1 == (b[i >> 3] >> (7 - (i & 7))) & 1)
    }

    #[test]
    fn exp_golomb_writes_what_it_reads() {
        let mut w = Writer::new();
        let values = [0, 1, 2, 3, 7, 15, 16, 255, 65_534, u32::MAX - 1];
        for v in values {
            w.ue(v);
        }
        w.trailing_bits();
        let mut r = Reader::new(&w.bytes);
        for v in values {
            assert_eq!(r.ue().expect("read"), v);
        }
    }

    /// A VUI without a restriction gets one saying no picture waits for a later one, and
    /// everything before it, the color and the timing included, is copied as it came.
    #[test]
    fn a_set_without_a_restriction_is_bounded_at_zero() {
        assert_eq!(
            reorder_of(VCE_SPS),
            None,
            "the fixture declares no restriction"
        );
        let bounded = bound_reorder(VCE_SPS, true)
            .expect("writable")
            .expect("a change");
        assert_eq!(
            reorder_of(&bounded),
            Some((0, 2)),
            "zero, in a buffer of the set's two references"
        );
        assert_eq!(read_color(&bounded), read_color(VCE_SPS), "the color moved");
        let flag = restriction(
            &unescape(&VCE_SPS[1..]),
            &locate(&unescape(&VCE_SPS[1..])).unwrap(),
        )
        .unwrap()
        .flag;
        assert!(
            same_prefix(VCE_SPS, &bounded, flag),
            "a bit before the restriction changed"
        );
        assert_eq!(
            bound_reorder(&bounded, true),
            Ok(None),
            "a bounded set is left as it came"
        );
    }

    /// A stream whose session cannot vouch for its order is bounded only where the set itself
    /// rules reordering out: picture order count type 2 has output order equal decoding order.
    #[test]
    fn only_an_order_the_stream_proves_is_bounded() {
        assert!(bound_reorder(VCE_SPS, false).is_err(), "type 0 may reorder");
        let pi4 = without(PI4_SPS, false);
        assert_eq!(locate(&unescape(&pi4[1..])).unwrap().pic_order_cnt_type, 2);
        let bounded = bound_reorder(&pi4, false)
            .expect("type 2 cannot reorder")
            .expect("a change");
        assert_eq!(reorder_of(&bounded), Some((0, 1)));
    }

    /// The Raspberry Pi 4 bounds its own stream, and the color written into it keeps the bound.
    #[test]
    fn a_bounded_set_is_left_as_it_came() {
        assert_eq!(reorder_of(PI4_SPS), Some((0, 1)));
        assert_eq!(bound_reorder(PI4_SPS, true), Ok(None));
        let colored = write_color(PI4_SPS, ColorSignal::BT601_FULL).expect("colored");
        assert_eq!(
            reorder_of(&colored),
            Some((0, 1)),
            "the color write lost the restriction"
        );
        assert_eq!(bound_reorder(&colored, false), Ok(None));
    }

    /// The restriction sits after both HRDs, so a set carrying them is read through them, and
    /// they come out bit for bit.
    #[test]
    fn hrd_parameters_are_read_through_and_kept() {
        let bare = without(PI4_SPS, false);
        assert_eq!(reorder_of(&bare), None);
        let bounded = bound_reorder(&bare, true)
            .expect("writable")
            .expect("a change");
        assert_eq!(reorder_of(&bounded), Some((0, 1)));
        let flag = restriction(
            &unescape(&bare[1..]),
            &locate(&unescape(&bare[1..])).unwrap(),
        )
        .unwrap()
        .flag;
        assert!(
            same_prefix(&bare, &bounded, flag),
            "the HRD parameters changed"
        );
    }

    /// A set without a VUI gets one carrying the restriction alone: no color is claimed.
    #[test]
    fn a_set_without_a_vui_takes_one_with_the_restriction_alone() {
        let bare = without(VCE_SPS, true);
        assert!(!locate(&unescape(&bare[1..])).unwrap().vui_present);
        let bounded = bound_reorder(&bare, true)
            .expect("writable")
            .expect("a change");
        assert_eq!(reorder_of(&bounded), Some((0, 2)));
        assert_eq!(read_color(&bounded), None, "a color nobody declared");
        let colored = write_color(&bounded, ColorSignal::BT709_LIMITED).expect("colored after");
        assert_eq!(
            reorder_of(&colored),
            Some((0, 2)),
            "a color written later keeps the bound"
        );
    }

    /// A set declaring a depth above zero is rewritten to zero, and a buffer below the
    /// reference count, which Chromium refuses the set for, is raised to it.
    #[test]
    fn a_declared_depth_is_rewritten() {
        let deep = with_depth(PI4_SPS, 3, 3);
        assert_eq!(reorder_of(&deep), Some((3, 3)));
        let bounded = bound_reorder(&deep, true)
            .expect("writable")
            .expect("a change");
        assert_eq!(
            reorder_of(&bounded),
            Some((0, 3)),
            "the declared buffer is kept"
        );
        let short = with_depth(&bound_reorder(VCE_SPS, true).unwrap().unwrap(), 0, 1);
        assert_eq!(reorder_of(&short), Some((0, 1)));
        let raised = bound_reorder(&short, true)
            .expect("writable")
            .expect("a change");
        assert_eq!(
            reorder_of(&raised),
            Some((0, 2)),
            "a buffer below the two references"
        );
    }

    /// Anything the reader cannot account for bit by bit leaves the set as it came.
    #[test]
    fn a_set_the_reader_doubts_is_refused() {
        let rbsp = unescape(&VCE_SPS[1..]);
        let last = stop_bit(&rbsp).unwrap();
        let mut w = Writer::new();
        w.copy_from(&rbsp, 0, last);
        w.bit(1);
        w.trailing_bits();
        let mut trailing = vec![VCE_SPS[0]];
        trailing.extend_from_slice(&escape(&w.bytes));
        assert!(
            bound_reorder(&trailing, true).is_err(),
            "a bit after the VUI went unnoticed"
        );
        let mut scaled = unescape(&PI4_SPS[1..]);
        scaled[3] |= 0x01;
        let mut scaled_nal = vec![PI4_SPS[0]];
        scaled_nal.extend_from_slice(&escape(&scaled));
        assert!(
            bound_reorder(&scaled_nal, true).is_err(),
            "scaling lists are refused"
        );
        assert!(
            bound_reorder(&PI4_SPS[..10], true).is_err(),
            "a truncated set"
        );
    }

    /// The sets of a real stream, stripped of their bound three ways and bounded again,
    /// decode to the same pictures as the stream as it came, through a decoder that is not
    /// ours; the unit's other NAL units pass through untouched.
    #[test]
    fn a_bounded_stream_decodes_as_it_came() {
        use crate::encoders::oh264::Openh264Encoder;
        use crate::webcam::decode::{Decoder as _, VideoDecoder};
        let (w, h) = (160usize, 96usize);
        let s = RustCaptureSettings {
            width: w as i32,
            height: h as i32,
            target_fps: 30.0,
            omit_stripe_headers: true,
            ..Default::default()
        };
        let mut enc = Openh264Encoder::new(&s).expect("openh264 init");
        let frames: Vec<Vec<u8>> = (0..6)
            .map(|t| {
                let pixels: Vec<u8> = (0..w * h)
                    .flat_map(|i| {
                        let v = ((i % w + 3 * t) * 5 + (i / w) * 3) as u8;
                        [v, v.wrapping_mul(3), v.wrapping_add(t as u8), 255]
                    })
                    .collect();
                enc.encode_host_argb(&pixels, w * 4, t as u64, t == 0, false)
                    .expect("encode")
            })
            .collect();
        let sets = sequence_parameter_sets(&frames[0]);
        assert_eq!(sets.len(), 1, "the key frame carries one set");
        let (start, end) = sets[0];
        let original = &frames[0][start..end];
        let refs = locate(&unescape(&original[1..]))
            .unwrap()
            .max_num_ref_frames;
        let decode = |stream: &[Vec<u8>]| -> Vec<Vec<u8>> {
            let mut dec = VideoDecoder::new(crate::encoders::codec::Codec::H264).expect("decoder");
            stream
                .iter()
                .map(|f| {
                    assert!(dec.decode(f).expect("decodes"), "no picture");
                    let p = dec.frame().expect("frame");
                    let rows =
                        |plane: &[u8], stride: usize, width: usize, height: usize| -> Vec<u8> {
                            (0..height)
                                .flat_map(|y| plane[y * stride..y * stride + width].to_vec())
                                .collect()
                        };
                    let mut out = rows(p.y, p.y_stride, p.width, p.height);
                    out.extend(rows(
                        p.u,
                        p.uv_stride,
                        p.width.div_ceil(2),
                        p.height.div_ceil(2),
                    ));
                    out.extend(rows(
                        p.v,
                        p.uv_stride,
                        p.width.div_ceil(2),
                        p.height.div_ceil(2),
                    ));
                    out
                })
                .collect()
        };
        let expected = decode(&frames);
        let mut variants = vec![
            (without(original, true), refs),
            (without(original, false), refs),
        ];
        if reorder_of(original).is_some() {
            variants.push((with_depth(original, 2, refs.max(2)), refs.max(2)));
        }
        for (variant, buffering) in variants {
            let mut stripped = frames.clone();
            stripped[0] = [&frames[0][..start], &variant[..], &frames[0][end..]].concat();
            let mut holder = NoReorder::new("The test encoder", true);
            let bounded = holder.apply(&stripped[0]).expect("a set to bound");
            assert_eq!(h264_reorder(&bounded), Some((0, buffering)));
            assert_eq!(
                holder.apply(&stripped[0]).as_ref(),
                Some(&bounded),
                "the second key frame differs"
            );
            assert_eq!(
                holder.apply(&frames[1]),
                None,
                "a unit without a set was touched"
            );
            assert!(
                bounded.starts_with(&frames[0][..start]),
                "what precedes the set changed"
            );
            assert!(
                bounded.ends_with(&frames[0][end..]),
                "the slices after the set changed"
            );
            let mut stream = stripped;
            stream[0] = bounded;
            assert_eq!(
                decode(&stream),
                expected,
                "the bounded stream decodes to other pictures"
            );
        }
    }
}
