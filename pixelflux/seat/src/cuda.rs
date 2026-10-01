/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! The CUDA driver calls the encoder needs, loaded from `libcuda.so.1` at run time (no CUDA
//! toolkit at build time), and one primary context on device 0 shared by every encoder of the
//! process. Follows pixelflux's `CudaFunctions` loader.

#![allow(non_snake_case)]

use std::ffi::{c_char, c_void, CStr};
use std::sync::OnceLock;

use libloading::Library;
use nvcodec_sys::cuda::*;

pub struct Cuda {
    _lib: Library,
    pub cuInit: unsafe extern "C" fn(flags: u32) -> CUresult,
    pub cuDeviceGet: unsafe extern "C" fn(device: *mut CUdevice, ordinal: i32) -> CUresult,
    pub cuDeviceGetName: unsafe extern "C" fn(name: *mut c_char, len: i32, dev: CUdevice) -> CUresult,
    pub cuDevicePrimaryCtxRetain: unsafe extern "C" fn(pctx: *mut CUcontext, dev: CUdevice) -> CUresult,
    pub cuCtxPushCurrent_v2: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
    pub cuCtxPopCurrent_v2: unsafe extern "C" fn(pctx: *mut CUcontext) -> CUresult,
    pub cuMemAllocPitch_v2: unsafe extern "C" fn(
        dptr: *mut CUdeviceptr,
        pPitch: *mut usize,
        WidthInBytes: usize,
        Height: usize,
        ElementSizeBytes: u32,
    ) -> CUresult,
    pub cuMemFree_v2: unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult,
    pub cuMemcpy2D_v2: unsafe extern "C" fn(pCopy: *const CUDA_MEMCPY2D) -> CUresult,
    pub cuMemHostRegister_v2: unsafe extern "C" fn(p: *mut c_void, bytesize: usize, flags: u32) -> CUresult,
    pub cuMemHostUnregister: unsafe extern "C" fn(p: *mut c_void) -> CUresult,
    pub cuGetErrorName: unsafe extern "C" fn(error: CUresult, pStr: *mut *const c_char) -> CUresult,
}

// The function pointers are plain C entry points of a thread-safe driver API.
unsafe impl Send for Cuda {}
unsafe impl Sync for Cuda {}

pub struct Context {
    pub ctx: CUcontext,
    pub device_name: String,
}
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

static CUDA: OnceLock<Result<Cuda, String>> = OnceLock::new();
static CONTEXT: OnceLock<Result<Context, String>> = OnceLock::new();

pub fn ok(r: CUresult) -> bool {
    r == CUresult::CUDA_SUCCESS
}

impl Cuda {
    fn load() -> Result<Cuda, String> {
        unsafe {
            let lib = Library::new("libcuda.so.1").map_err(|e| format!("cannot load libcuda.so.1: {e}"))?;
            macro_rules! sym {
                ($name:literal) => {
                    *lib.get(concat!($name, "\0").as_bytes()).map_err(|e| format!("libcuda: no {}: {e}", $name))?
                };
            }
            Ok(Cuda {
                cuInit: sym!("cuInit"),
                cuDeviceGet: sym!("cuDeviceGet"),
                cuDeviceGetName: sym!("cuDeviceGetName"),
                cuDevicePrimaryCtxRetain: sym!("cuDevicePrimaryCtxRetain"),
                cuCtxPushCurrent_v2: sym!("cuCtxPushCurrent_v2"),
                cuCtxPopCurrent_v2: sym!("cuCtxPopCurrent_v2"),
                cuMemAllocPitch_v2: sym!("cuMemAllocPitch_v2"),
                cuMemFree_v2: sym!("cuMemFree_v2"),
                cuMemcpy2D_v2: sym!("cuMemcpy2D_v2"),
                cuMemHostRegister_v2: sym!("cuMemHostRegister_v2"),
                cuMemHostUnregister: sym!("cuMemHostUnregister"),
                cuGetErrorName: sym!("cuGetErrorName"),
                _lib: lib,
            })
        }
    }

    pub fn get() -> Result<&'static Cuda, String> {
        CUDA.get_or_init(Cuda::load).as_ref().map_err(|e| e.clone())
    }

    pub fn err(&self, r: CUresult) -> String {
        unsafe {
            let mut p: *const c_char = std::ptr::null();
            if ok((self.cuGetErrorName)(r, &mut p)) && !p.is_null() {
                return CStr::from_ptr(p).to_string_lossy().into_owned();
            }
        }
        format!("CUresult {}", r.0)
    }

    /// The process's one CUDA context: the primary context of device 0.
    pub fn context() -> Result<&'static Context, String> {
        CONTEXT
            .get_or_init(|| {
                let cu = Cuda::get()?;
                unsafe {
                    let r = (cu.cuInit)(0);
                    if !ok(r) {
                        return Err(format!("cuInit: {}", cu.err(r)));
                    }
                    let mut dev: CUdevice = 0;
                    let r = (cu.cuDeviceGet)(&mut dev, 0);
                    if !ok(r) {
                        return Err(format!("cuDeviceGet: {}", cu.err(r)));
                    }
                    let mut name = [0 as c_char; 128];
                    (cu.cuDeviceGetName)(name.as_mut_ptr(), name.len() as i32, dev);
                    let mut ctx: CUcontext = std::ptr::null_mut();
                    let r = (cu.cuDevicePrimaryCtxRetain)(&mut ctx, dev);
                    if !ok(r) {
                        return Err(format!("cuDevicePrimaryCtxRetain: {}", cu.err(r)));
                    }
                    Ok(Context { ctx, device_name: CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned() })
                }
            })
            .as_ref()
            .map_err(|e| e.clone())
    }
}

/// Make the shared context current on this thread for the guard's life.
pub struct Current<'a> {
    cu: &'a Cuda,
}

impl<'a> Current<'a> {
    pub fn push(cu: &'a Cuda, ctx: &Context) -> Result<Self, String> {
        let r = unsafe { (cu.cuCtxPushCurrent_v2)(ctx.ctx) };
        if !ok(r) {
            return Err(format!("cuCtxPushCurrent: {}", cu.err(r)));
        }
        Ok(Current { cu })
    }
}

impl Drop for Current<'_> {
    fn drop(&mut self) {
        unsafe {
            let mut old: CUcontext = std::ptr::null_mut();
            (self.cu.cuCtxPopCurrent_v2)(&mut old);
        }
    }
}
