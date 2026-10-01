/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! One NVENC session per monitor, fed packed BGRA rows from host memory.
//!
//! The session settings are pixelflux's (`src/encoders/nvenc.rs`, `NvencEncoder::new`): a P-preset
//! with ultra-low-latency tuning, CBR at the asked rate with a VBV of 1.5 frames and two-pass
//! quarter-resolution rate control, an infinite GOP with no B-frames and zero reorder delay, IDR
//! only when asked, SPS/PPS repeated on every key frame, four slices, H.264 CABAC and a
//! bitstream-restriction VUI, BT.709 limited range. The input path is pixelflux's host-ARGB path:
//! the capture buffer is pinned once (`cuMemHostRegister`), each frame is one `cuMemcpy2D` into a
//! pitched device surface registered with NVENC as ARGB, and the encoder converts to 4:2:0 itself.
//! The encode is synchronous: on Linux NVENC has no async mode, and `nvEncLockBitstream` returns
//! as soon as the frame is out.

use std::ffi::{c_void, CStr};
use std::ptr;
use std::sync::OnceLock;
use std::time::Instant;

use libloading::Library;
use nvcodec_sys::cuda::*;
use nvcodec_sys::*;

use crate::cuda::{ok, Context, Cuda, Current};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    H264,
    Hevc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H264Profile {
    /// Constrained baseline (42e0..): CAVLC, for viewers that only offer it.
    Baseline,
    /// Constrained high (640c..): CABAC, no B-frames.
    High,
}

#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub codec: Codec,
    pub h264_profile: H264Profile,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub bitrate_kbps: u32,
    /// NVENC preset 1 (fastest) to 7 (best). pixelflux uses 3.
    pub preset: u8,
    /// Two-pass at quarter resolution (pixelflux's default): steadier per-frame sizes for a
    /// fraction of a millisecond.
    pub two_pass: bool,
    pub slices: u32,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        EncoderConfig {
            codec: Codec::H264,
            h264_profile: H264Profile::High,
            width: 1920,
            height: 1080,
            fps: 60,
            bitrate_kbps: 20000,
            preset: 3,
            two_pass: true,
            slices: 4,
        }
    }
}

pub struct EncodedFrame {
    pub data: Vec<u8>,
    pub keyframe: bool,
    /// Copy into the encoder's surface plus the encode itself.
    pub encode_ms: f64,
}

struct Api {
    _lib: Library,
    fl: NV_ENCODE_API_FUNCTION_LIST,
    max_version: u32,
}
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

static API: OnceLock<Result<Api, String>> = OnceLock::new();

fn api() -> Result<&'static Api, String> {
    API.get_or_init(|| unsafe {
        let lib = Library::new(NVENC_DLL_NAME).map_err(|e| format!("cannot load {NVENC_DLL_NAME}: {e}"))?;
        let mut max_version = 0u32;
        if let Ok(f) = lib.get::<NvEncodeApiGetMaxSupportedVersionFn>(NV_ENCODE_API_GET_MAX_SUPPORTED_VERSION_FN_NAME) {
            f(&mut max_version);
        }
        // max_version packs (major << 4) | minor. This build speaks NVENC API 13.0 only.
        let need = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
        if max_version < need {
            return Err(format!(
                "the NVIDIA driver offers NVENC API {}.{}, this build needs {}.{} (driver 570 or newer)",
                max_version >> 4,
                max_version & 0xF,
                NVENCAPI_MAJOR_VERSION,
                NVENCAPI_MINOR_VERSION
            ));
        }
        let create: libloading::Symbol<NvEncodeApiCreateInstanceFn> = lib
            .get(NV_ENCODE_API_CREATE_INSTANCE_FN_NAME)
            .map_err(|e| format!("NvEncodeAPICreateInstance: {e}"))?;
        let mut fl: NV_ENCODE_API_FUNCTION_LIST = std::mem::zeroed();
        fl.version = NV_ENCODE_API_FUNCTION_LIST_VER;
        let st = create(&mut fl);
        if st != NVENCSTATUS::NV_ENC_SUCCESS {
            return Err(format!("NvEncodeAPICreateInstance: {st:?}"));
        }
        Ok(Api { _lib: lib, fl, max_version })
    })
    .as_ref()
    .map_err(|e| e.clone())
}

fn preset_guid(p: u8) -> GUID {
    match p {
        1 => NV_ENC_PRESET_P1_GUID,
        2 => NV_ENC_PRESET_P2_GUID,
        4 => NV_ENC_PRESET_P4_GUID,
        5 => NV_ENC_PRESET_P5_GUID,
        6 => NV_ENC_PRESET_P6_GUID,
        7 => NV_ENC_PRESET_P7_GUID,
        _ => NV_ENC_PRESET_P3_GUID,
    }
}

/// VBV of 1.5 frames at `bps`, pixelflux's `vbv_bits` for an infinite GOP.
fn vbv_bits(bps: u32, fps: u32) -> u32 {
    ((bps as f64 / fps.max(1) as f64) * 1.5).round().max(1.0) as u32
}

pub struct Encoder {
    cfg: EncoderConfig,
    fl: NV_ENCODE_API_FUNCTION_LIST,
    cu: &'static Cuda,
    ctx: &'static Context,
    session: *mut c_void,
    config: Box<NV_ENC_CONFIG>,
    init: NV_ENC_INITIALIZE_PARAMS,
    dev_ptr: CUdeviceptr,
    dev_pitch: usize,
    registered: NV_ENC_REGISTERED_PTR,
    bitstream: NV_ENC_OUTPUT_PTR,
    pinned: Option<(*mut c_void, usize)>,
    frame_idx: u32,
}

unsafe impl Send for Encoder {}

impl Encoder {
    pub fn device_name() -> Result<String, String> {
        Ok(Cuda::context()?.device_name.clone())
    }

    fn last_error(&self) -> String {
        unsafe {
            if let Some(f) = self.fl.nvEncGetLastErrorString {
                let p = f(self.session);
                if !p.is_null() {
                    return CStr::from_ptr(p).to_string_lossy().into_owned();
                }
            }
        }
        String::new()
    }

    pub fn new(cfg: EncoderConfig) -> Result<Encoder, String> {
        let a = api()?;
        let cu = Cuda::get()?;
        let ctx = Cuda::context()?;
        let _cur = Current::push(cu, ctx)?;
        let fl = a.fl;
        let _ = a.max_version;
        let mut enc = Encoder {
            cfg: cfg.clone(),
            fl,
            cu,
            ctx,
            session: ptr::null_mut(),
            config: Box::new(unsafe { std::mem::zeroed() }),
            init: unsafe { std::mem::zeroed() },
            dev_ptr: 0,
            dev_pitch: 0,
            registered: ptr::null_mut(),
            bitstream: ptr::null_mut(),
            pinned: None,
            frame_idx: 0,
        };
        unsafe { enc.open()? };
        Ok(enc)
    }

    unsafe fn open(&mut self) -> Result<(), String> {
        let cfg = self.cfg.clone();
        let fl = self.fl;
        let cu = self.cu;

        // Device surface first, so a failed session open has nothing of NVENC's to unwind.
        let mut pitch = 0usize;
        let r = (cu.cuMemAllocPitch_v2)(&mut self.dev_ptr, &mut pitch, cfg.width as usize * 4, cfg.height as usize, 16);
        if !ok(r) {
            return Err(format!("cuMemAllocPitch {}x{}: {}", cfg.width, cfg.height, cu.err(r)));
        }
        self.dev_pitch = pitch;

        let mut sp: NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS = std::mem::zeroed();
        sp.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
        sp.deviceType = NV_ENC_DEVICE_TYPE::NV_ENC_DEVICE_TYPE_CUDA;
        sp.device = self.ctx.ctx as *mut c_void;
        sp.apiVersion = NVENCAPI_VERSION;
        let st = (fl.nvEncOpenEncodeSessionEx.unwrap())(&mut sp, &mut self.session);
        if st != NVENCSTATUS::NV_ENC_SUCCESS {
            self.session = ptr::null_mut();
            return Err(format!(
                "nvEncOpenEncodeSessionEx: {st:?} (a GeForce card allows a limited number of sessions at once)"
            ));
        }

        let codec_guid = match cfg.codec {
            Codec::H264 => NV_ENC_CODEC_H264_GUID,
            Codec::Hevc => NV_ENC_CODEC_HEVC_GUID,
        };
        let preset = preset_guid(cfg.preset);
        let mut pc: NV_ENC_PRESET_CONFIG = std::mem::zeroed();
        pc.version = NV_ENC_PRESET_CONFIG_VER;
        pc.presetCfg.version = NV_ENC_CONFIG_VER;
        let st = (fl.nvEncGetEncodePresetConfigEx.unwrap())(
            self.session,
            codec_guid,
            preset,
            NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,
            &mut pc,
        );
        if st != NVENCSTATUS::NV_ENC_SUCCESS {
            return Err(format!("nvEncGetEncodePresetConfigEx: {st:?} {}", self.last_error()));
        }
        let mut c = pc.presetCfg;
        c.version = NV_ENC_CONFIG_VER;
        c.profileGUID = match (cfg.codec, cfg.h264_profile) {
            (Codec::H264, H264Profile::High) => NV_ENC_H264_PROFILE_HIGH_GUID,
            (Codec::H264, H264Profile::Baseline) => NV_ENC_H264_PROFILE_BASELINE_GUID,
            (Codec::Hevc, _) => NV_ENC_HEVC_PROFILE_MAIN_GUID,
        };
        let bps = cfg.bitrate_kbps.saturating_mul(1000);
        let vbv = vbv_bits(bps, cfg.fps);
        c.rcParams.rateControlMode = NV_ENC_PARAMS_RC_MODE::NV_ENC_PARAMS_RC_CBR;
        c.rcParams.multiPass = if cfg.two_pass {
            NV_ENC_MULTI_PASS::NV_ENC_TWO_PASS_QUARTER_RESOLUTION
        } else {
            NV_ENC_MULTI_PASS::NV_ENC_MULTI_PASS_DISABLED
        };
        c.rcParams.averageBitRate = bps;
        c.rcParams.maxBitRate = bps;
        c.rcParams.vbvBufferSize = vbv;
        c.rcParams.vbvInitialDelay = vbv;
        c.rcParams.set_enableAQ(0);
        c.frameIntervalP = 1;
        c.gopLength = 0xFFFF_FFFF;
        c.rcParams.set_zeroReorderDelay(1);
        c.rcParams.set_strictGOPTarget(1);
        c.rcParams.set_enableLookahead(0);
        c.rcParams.lookaheadDepth = 0;

        let set_vui = |vui: &mut NV_ENC_CONFIG_H264_VUI_PARAMETERS| {
            vui.videoSignalTypePresentFlag = 1;
            vui.videoFormat = NV_ENC_VUI_VIDEO_FORMAT::NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED;
            vui.colourDescriptionPresentFlag = 1;
            vui.colourPrimaries = NV_ENC_VUI_COLOR_PRIMARIES::NV_ENC_VUI_COLOR_PRIMARIES_BT709;
            vui.transferCharacteristics = NV_ENC_VUI_TRANSFER_CHARACTERISTIC::NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709;
            vui.colourMatrix = NV_ENC_VUI_MATRIX_COEFFS::NV_ENC_VUI_MATRIX_COEFFS_BT709;
            vui.videoFullRangeFlag = 0;
        };
        match cfg.codec {
            Codec::H264 => {
                let h = &mut c.encodeCodecConfig.h264Config;
                h.level = 0; // autoselect
                h.sliceMode = 3;
                h.sliceModeData = cfg.slices.max(1);
                h.idrPeriod = 0xFFFF_FFFF;
                h.chromaFormatIDC = 1;
                h.set_repeatSPSPPS(1);
                h.set_outputAUD(0);
                h.entropyCodingMode = match cfg.h264_profile {
                    H264Profile::High => NV_ENC_H264_ENTROPY_CODING_MODE::NV_ENC_H264_ENTROPY_CODING_MODE_CABAC,
                    H264Profile::Baseline => NV_ENC_H264_ENTROPY_CODING_MODE::NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC,
                };
                h.h264VUIParameters.bitstreamRestrictionFlag = 1;
                set_vui(&mut h.h264VUIParameters);
            }
            Codec::Hevc => {
                let h = &mut c.encodeCodecConfig.hevcConfig;
                h.level = 0;
                h.sliceMode = 3;
                h.sliceModeData = cfg.slices.max(1);
                h.idrPeriod = 0xFFFF_FFFF;
                h.set_chromaFormatIDC(1);
                h.inputBitDepth = NV_ENC_BIT_DEPTH::NV_ENC_BIT_DEPTH_8;
                h.outputBitDepth = NV_ENC_BIT_DEPTH::NV_ENC_BIT_DEPTH_8;
                h.set_repeatSPSPPS(1);
                h.set_outputAUD(0);
                set_vui(&mut h.hevcVUIParameters);
            }
        }
        *self.config = c;

        let mut ip: NV_ENC_INITIALIZE_PARAMS = std::mem::zeroed();
        ip.version = NV_ENC_INITIALIZE_PARAMS_VER;
        ip.encodeGUID = codec_guid;
        ip.presetGUID = preset;
        ip.tuningInfo = NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
        ip.encodeWidth = cfg.width;
        ip.encodeHeight = cfg.height;
        ip.darWidth = cfg.width;
        ip.darHeight = cfg.height;
        ip.frameRateNum = cfg.fps.max(1);
        ip.frameRateDen = 1;
        ip.enablePTD = 1;
        ip.enableEncodeAsync = 0;
        ip.encodeConfig = &mut *self.config;
        ip.maxEncodeWidth = cfg.width;
        ip.maxEncodeHeight = cfg.height;
        let st = (fl.nvEncInitializeEncoder.unwrap())(self.session, &mut ip);
        if st != NVENCSTATUS::NV_ENC_SUCCESS {
            return Err(format!(
                "nvEncInitializeEncoder {:?} {}x{}: {st:?} {}",
                cfg.codec,
                cfg.width,
                cfg.height,
                self.last_error()
            ));
        }
        self.init = ip;

        let mut rr: NV_ENC_REGISTER_RESOURCE = std::mem::zeroed();
        rr.version = NV_ENC_REGISTER_RESOURCE_VER;
        rr.resourceType = NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
        rr.width = cfg.width;
        rr.height = cfg.height;
        rr.pitch = self.dev_pitch as u32;
        rr.resourceToRegister = self.dev_ptr as *mut c_void;
        rr.bufferFormat = NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB;
        rr.bufferUsage = NV_ENC_BUFFER_USAGE::NV_ENC_INPUT_IMAGE;
        let st = (fl.nvEncRegisterResource.unwrap())(self.session, &mut rr);
        if st != NVENCSTATUS::NV_ENC_SUCCESS {
            return Err(format!("nvEncRegisterResource: {st:?} {}", self.last_error()));
        }
        self.registered = rr.registeredResource;

        let mut bb: NV_ENC_CREATE_BITSTREAM_BUFFER = std::mem::zeroed();
        bb.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
        let st = (fl.nvEncCreateBitstreamBuffer.unwrap())(self.session, &mut bb);
        if st != NVENCSTATUS::NV_ENC_SUCCESS {
            return Err(format!("nvEncCreateBitstreamBuffer: {st:?} {}", self.last_error()));
        }
        self.bitstream = bb.bitstreamBuffer;
        Ok(())
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.cfg
    }

    /// Pin the capture buffer so each upload is a DMA, not a staged copy. Optional: an unpinned
    /// buffer works, slower. Call again for a new buffer; the old one is unpinned.
    pub fn pin_host(&mut self, p: *mut u8, len: usize) -> Result<(), String> {
        let _cur = Current::push(self.cu, self.ctx)?;
        self.unpin();
        let r = unsafe { (self.cu.cuMemHostRegister_v2)(p as *mut c_void, len, 0) };
        if !ok(r) {
            return Err(format!("cuMemHostRegister: {}", self.cu.err(r)));
        }
        self.pinned = Some((p as *mut c_void, len));
        Ok(())
    }

    fn unpin(&mut self) {
        if let Some((p, _)) = self.pinned.take() {
            unsafe { (self.cu.cuMemHostUnregister)(p) };
        }
    }

    /// Change the CBR rate in place (no new key frame).
    pub fn set_bitrate(&mut self, kbps: u32) -> Result<(), String> {
        if kbps == self.cfg.bitrate_kbps {
            return Ok(());
        }
        let _cur = Current::push(self.cu, self.ctx)?;
        let bps = kbps.saturating_mul(1000);
        let vbv = vbv_bits(bps, self.cfg.fps);
        let mut c = *self.config;
        c.rcParams.averageBitRate = bps;
        c.rcParams.maxBitRate = bps;
        c.rcParams.vbvBufferSize = vbv;
        c.rcParams.vbvInitialDelay = vbv;
        unsafe {
            let mut rp: NV_ENC_RECONFIGURE_PARAMS = std::mem::zeroed();
            rp.version = NV_ENC_RECONFIGURE_PARAMS_VER;
            rp.reInitEncodeParams = self.init;
            rp.reInitEncodeParams.encodeConfig = &mut c;
            rp.set_resetEncoder(0);
            rp.set_forceIDR(0);
            let st = (self.fl.nvEncReconfigureEncoder.unwrap())(self.session, &mut rp);
            if st != NVENCSTATUS::NV_ENC_SUCCESS {
                return Err(format!("nvEncReconfigureEncoder: {st:?} {}", self.last_error()));
            }
        }
        *self.config = c;
        self.init.encodeConfig = &mut *self.config;
        self.cfg.bitrate_kbps = kbps;
        Ok(())
    }

    /// Encode one picture of packed BGRA rows (`pitch` bytes apart, the configured size).
    pub fn encode(&mut self, src: *const u8, pitch: usize, force_idr: bool) -> Result<EncodedFrame, String> {
        let t0 = Instant::now();
        let _cur = Current::push(self.cu, self.ctx)?;
        let fl = self.fl;
        unsafe {
            let mut cp: CUDA_MEMCPY2D = std::mem::zeroed();
            cp.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_HOST;
            cp.srcHost = src as *const c_void;
            cp.srcPitch = pitch;
            cp.dstMemoryType = CUmemorytype::CU_MEMORYTYPE_DEVICE;
            cp.dstDevice = self.dev_ptr;
            cp.dstPitch = self.dev_pitch;
            cp.WidthInBytes = self.cfg.width as usize * 4;
            cp.Height = self.cfg.height as usize;
            let r = (self.cu.cuMemcpy2D_v2)(&cp);
            if !ok(r) {
                return Err(format!("cuMemcpy2D: {}", self.cu.err(r)));
            }

            let mut mp: NV_ENC_MAP_INPUT_RESOURCE = std::mem::zeroed();
            mp.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
            mp.registeredResource = self.registered;
            let st = (fl.nvEncMapInputResource.unwrap())(self.session, &mut mp);
            if st != NVENCSTATUS::NV_ENC_SUCCESS {
                return Err(format!("nvEncMapInputResource: {st:?} {}", self.last_error()));
            }

            let mut pp: NV_ENC_PIC_PARAMS = std::mem::zeroed();
            pp.version = NV_ENC_PIC_PARAMS_VER;
            pp.inputWidth = self.cfg.width;
            pp.inputHeight = self.cfg.height;
            pp.inputPitch = self.dev_pitch as u32;
            pp.inputBuffer = mp.mappedResource;
            pp.outputBitstream = self.bitstream;
            pp.bufferFmt = mp.mappedBufferFmt;
            pp.pictureStruct = NV_ENC_PIC_STRUCT::NV_ENC_PIC_STRUCT_FRAME;
            pp.frameIdx = self.frame_idx;
            pp.inputTimeStamp = self.frame_idx as u64;
            if force_idr || self.frame_idx == 0 {
                pp.encodePicFlags =
                    NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_FORCEIDR as u32 | NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_OUTPUT_SPSPPS as u32;
            }
            self.frame_idx = self.frame_idx.wrapping_add(1);
            let st = (fl.nvEncEncodePicture.unwrap())(self.session, &mut pp);
            if st != NVENCSTATUS::NV_ENC_SUCCESS {
                (fl.nvEncUnmapInputResource.unwrap())(self.session, mp.mappedResource);
                return Err(format!("nvEncEncodePicture: {st:?} {}", self.last_error()));
            }

            let mut lb: NV_ENC_LOCK_BITSTREAM = std::mem::zeroed();
            lb.version = NV_ENC_LOCK_BITSTREAM_VER;
            lb.outputBitstream = self.bitstream;
            let st = (fl.nvEncLockBitstream.unwrap())(self.session, &mut lb);
            if st != NVENCSTATUS::NV_ENC_SUCCESS {
                (fl.nvEncUnmapInputResource.unwrap())(self.session, mp.mappedResource);
                return Err(format!("nvEncLockBitstream: {st:?} {}", self.last_error()));
            }
            let data =
                std::slice::from_raw_parts(lb.bitstreamBufferPtr as *const u8, lb.bitstreamSizeInBytes as usize).to_vec();
            let keyframe = lb.pictureType == NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_IDR
                || lb.pictureType == NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_I;
            (fl.nvEncUnlockBitstream.unwrap())(self.session, self.bitstream);
            (fl.nvEncUnmapInputResource.unwrap())(self.session, mp.mappedResource);
            Ok(EncodedFrame { data, keyframe, encode_ms: t0.elapsed().as_secs_f64() * 1000.0 })
        }
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        let Ok(_cur) = Current::push(self.cu, self.ctx) else { return };
        unsafe {
            let fl = self.fl;
            if !self.session.is_null() {
                if !self.bitstream.is_null() {
                    (fl.nvEncDestroyBitstreamBuffer.unwrap())(self.session, self.bitstream);
                }
                if !self.registered.is_null() {
                    (fl.nvEncUnregisterResource.unwrap())(self.session, self.registered);
                }
                (fl.nvEncDestroyEncoder.unwrap())(self.session);
            }
            if self.dev_ptr != 0 {
                (self.cu.cuMemFree_v2)(self.dev_ptr);
            }
        }
        self.unpin();
    }
}
