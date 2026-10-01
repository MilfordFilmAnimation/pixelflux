/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! X11 capture of one rectangle of the root window (one monitor) through MIT-SHM.
//!
//! As in pixelflux's `x11` module: a shared segment both this process and the server map, one
//! `ShmGetImage` per frame, XDamage to tell whether the picture changed, and XFixes for the cursor
//! image, drawn into the picture. Differences: the segment is a memfd passed to the server
//! (`ShmAttachFd`), so it works whichever user the X server runs as; the connection takes an
//! explicit display and Xauthority file, so one process can follow several X servers.

use std::os::fd::{FromRawFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};

use x11rb::connection::{Connection, RequestConnection};
use x11rb::protocol::damage::{self, ConnectionExt as _, ReportLevel};
use x11rb::protocol::randr::{self, ConnectionExt as _};
use x11rb::protocol::shm::{self, ConnectionExt as _};
use x11rb::protocol::xfixes::{self, ConnectionExt as _, CursorNotifyMask};
use x11rb::protocol::xproto::{ConnectionExt as _, ImageFormat};
use x11rb::protocol::Event;
use x11rb::rust_connection::{DefaultStream, RustConnection};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XDisplay {
    /// ":0"
    pub display: String,
    pub xauthority: Option<PathBuf>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Monitor {
    pub id: u32,
    pub name: String,
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
    pub hz: f64,
    pub primary: bool,
}

fn display_number(display: &str) -> Result<u32, String> {
    let s = display.rsplit(':').next().unwrap_or("");
    let n = s.split('.').next().unwrap_or("");
    n.parse().map_err(|_| format!("cannot read a display number from {display:?}"))
}

/// The MIT-MAGIC-COOKIE-1 for display `num` from an Xauthority file.
pub fn read_xauthority(path: &Path, num: u32) -> Result<Option<(Vec<u8>, Vec<u8>)>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut i = 0usize;
    let mut first = None;
    let field = |i: &mut usize| -> Option<Vec<u8>> {
        if *i + 2 > b.len() {
            return None;
        }
        let n = u16::from_be_bytes([b[*i], b[*i + 1]]) as usize;
        *i += 2;
        if *i + n > b.len() {
            return None;
        }
        let v = b[*i..*i + n].to_vec();
        *i += n;
        Some(v)
    };
    while i + 2 <= b.len() {
        i += 2; // family
        let (Some(_addr), Some(number), Some(name), Some(data)) = (field(&mut i), field(&mut i), field(&mut i), field(&mut i))
        else {
            break;
        };
        if name != b"MIT-MAGIC-COOKIE-1" {
            continue;
        }
        if number.is_empty() || number == num.to_string().as_bytes() {
            return Ok(Some((name, data)));
        }
        if first.is_none() {
            first = Some((name, data));
        }
    }
    Ok(first)
}

/// Connect to a local display with its own cookie, never through the process environment.
pub fn connect(d: &XDisplay) -> Result<(RustConnection, usize), String> {
    let num = display_number(&d.display)?;
    let path = format!("/tmp/.X11-unix/X{num}");
    let sock = UnixStream::connect(&path).map_err(|e| format!("{path}: {e}"))?;
    let (stream, _peer) = DefaultStream::from_unix_stream(sock).map_err(|e| format!("{path}: {e}"))?;
    let (name, data) = match &d.xauthority {
        Some(p) => read_xauthority(p, num)?.unwrap_or_default(),
        None => Default::default(),
    };
    let conn = RustConnection::connect_to_stream_with_auth_info(stream, 0, name, data)
        .map_err(|e| format!("X connection to {}: {e}", d.display))?;
    Ok((conn, 0))
}

/// The RandR monitors of the screen, in RandR's order, with each one's refresh rate.
pub fn monitors(conn: &RustConnection, screen: usize) -> Result<Vec<Monitor>, String> {
    let root = conn.setup().roots[screen].root;
    conn.randr_query_version(1, 6).map_err(|e| e.to_string())?.reply().map_err(|e| format!("RandR: {e}"))?;
    let mons = conn
        .randr_get_monitors(root, true)
        .map_err(|e| e.to_string())?
        .reply()
        .map_err(|e| format!("RRGetMonitors: {e}"))?;
    let res = conn
        .randr_get_screen_resources_current(root)
        .map_err(|e| e.to_string())?
        .reply()
        .map_err(|e| format!("RRGetScreenResourcesCurrent: {e}"))?;
    let mut out = Vec::new();
    for (i, m) in mons.monitors.iter().enumerate() {
        let name = conn
            .get_atom_name(m.name)
            .ok()
            .and_then(|c| c.reply().ok())
            .map(|r| String::from_utf8_lossy(&r.name).into_owned())
            .unwrap_or_else(|| format!("monitor{i}"));
        let mut hz = 60.0;
        if let Some(&output) = m.outputs.first() {
            if let Ok(oi) = conn.randr_get_output_info(output, res.config_timestamp).map_err(|_| ()).and_then(|c| c.reply().map_err(|_| ())) {
                if oi.crtc != 0 {
                    if let Ok(ci) = conn.randr_get_crtc_info(oi.crtc, res.config_timestamp).map_err(|_| ()).and_then(|c| c.reply().map_err(|_| ())) {
                        if let Some(mode) = res.modes.iter().find(|md| md.id == ci.mode) {
                            let tot = mode.htotal as f64 * mode.vtotal as f64;
                            if tot > 0.0 {
                                hz = (mode.dot_clock as f64 / tot * 100.0).round() / 100.0;
                            }
                        }
                    }
                }
            }
        }
        out.push(Monitor {
            id: i as u32,
            name,
            x: m.x as i32,
            y: m.y as i32,
            w: m.width as u32,
            h: m.height as u32,
            hz,
            primary: m.primary,
        });
    }
    Ok(out)
}

struct CursorImg {
    w: u16,
    h: u16,
    xhot: u16,
    yhot: u16,
    pixels: Vec<u32>,
}

pub struct Capture {
    conn: RustConnection,
    root: u32,
    pub x: i16,
    pub y: i16,
    pub w: u16,
    pub h: u16,
    seg: u32,
    map: *mut u8,
    size: usize,
    damage: u32,
    pending: bool,
    cursor: Option<CursorImg>,
    cursor_dirty: bool,
    ptr: (i16, i16),
    drawn_ptr: Option<(i16, i16)>,
    pub draw_cursor: bool,
}

unsafe impl Send for Capture {}

fn xe<E: std::fmt::Display>(what: &'static str) -> impl Fn(E) -> String {
    move |e| format!("{what}: {e}")
}

impl Capture {
    pub fn new(d: &XDisplay, x: i32, y: i32, w: u32, h: u32) -> Result<Capture, String> {
        let (conn, screen) = connect(d)?;
        let setup_root = &conn.setup().roots[screen];
        let root = setup_root.root;
        let depth = setup_root.root_depth;
        let bpp = conn.setup().pixmap_formats.iter().find(|f| f.depth == depth).map(|f| f.bits_per_pixel).unwrap_or(32);
        if bpp != 32 {
            return Err(format!("root depth {depth} is {bpp} bits per pixel; only 32 is supported"));
        }
        if conn.extension_information(shm::X11_EXTENSION_NAME).map_err(xe("MIT-SHM"))?.is_none() {
            return Err("the X server has no MIT-SHM".into());
        }
        let v = conn.shm_query_version().map_err(xe("MIT-SHM"))?.reply().map_err(xe("MIT-SHM"))?;
        if (v.major_version, v.minor_version) < (1, 2) {
            return Err("MIT-SHM 1.2 (fd passing) needed".into());
        }
        conn.damage_query_version(1, 1).map_err(xe("DAMAGE"))?.reply().map_err(xe("DAMAGE"))?;
        conn.xfixes_query_version(5, 0).map_err(xe("XFIXES"))?.reply().map_err(xe("XFIXES"))?;

        let size = w as usize * h as usize * 4;
        let (map, fd) = unsafe {
            let fd = libc::memfd_create(c"mf-seat-capture".as_ptr(), libc::MFD_CLOEXEC);
            if fd < 0 {
                return Err(format!("memfd_create: {}", std::io::Error::last_os_error()));
            }
            let fd = OwnedFd::from_raw_fd(fd);
            use std::os::fd::AsRawFd;
            if libc::ftruncate(fd.as_raw_fd(), size as libc::off_t) != 0 {
                return Err(format!("ftruncate: {}", std::io::Error::last_os_error()));
            }
            let p = libc::mmap(std::ptr::null_mut(), size, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd.as_raw_fd(), 0);
            if p == libc::MAP_FAILED {
                return Err(format!("mmap: {}", std::io::Error::last_os_error()));
            }
            (p as *mut u8, fd)
        };
        let seg = conn.generate_id().map_err(xe("generate_id"))?;
        let r = conn.shm_attach_fd(seg, fd, false).map_err(xe("ShmAttachFd")).and_then(|c| c.check().map_err(xe("ShmAttachFd")));
        if let Err(e) = r {
            unsafe { libc::munmap(map as *mut libc::c_void, size) };
            return Err(e);
        }
        let damage = conn.generate_id().map_err(xe("generate_id"))?;
        conn.damage_create(damage, root, ReportLevel::RAW_RECTANGLES).map_err(xe("DamageCreate"))?;
        conn.xfixes_select_cursor_input(root, CursorNotifyMask::DISPLAY_CURSOR).map_err(xe("XFixesSelectCursorInput"))?;
        conn.flush().map_err(xe("flush"))?;
        Ok(Capture {
            conn,
            root,
            x: x as i16,
            y: y as i16,
            w: w as u16,
            h: h as u16,
            seg,
            map,
            size,
            damage,
            pending: true,
            cursor: None,
            cursor_dirty: true,
            ptr: (i16::MIN, i16::MIN),
            drawn_ptr: None,
            draw_cursor: true,
        })
    }

    pub fn data(&self) -> *mut u8 {
        self.map
    }
    pub fn stride(&self) -> usize {
        self.w as usize * 4
    }
    pub fn len(&self) -> usize {
        self.size
    }
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    fn intersects(&self, ax: i16, ay: i16, aw: u16, ah: u16) -> bool {
        let (ax, ay, aw, ah) = (ax as i32, ay as i32, aw as i32, ah as i32);
        let (bx, by, bw, bh) = (self.x as i32, self.y as i32, self.w as i32, self.h as i32);
        ax < bx + bw && bx < ax + aw && ay < by + bh && by < ay + ah
    }

    /// Read what happened since the last grab: true if the picture of this monitor or the cursor
    /// on it changed. An error means the X server is gone.
    pub fn poll(&mut self) -> Result<bool, String> {
        while let Some(ev) = self.conn.poll_for_event().map_err(xe("X connection"))? {
            match ev {
                Event::DamageNotify(d) => {
                    if !self.pending && self.intersects(d.area.x, d.area.y, d.area.width, d.area.height) {
                        self.pending = true;
                    }
                }
                Event::XfixesCursorNotify(_) => self.cursor_dirty = true,
                _ => {}
            }
        }
        if self.draw_cursor {
            let p = self.conn.query_pointer(self.root).map_err(xe("QueryPointer"))?.reply().map_err(xe("QueryPointer"))?;
            self.ptr = (p.root_x, p.root_y);
            let over = |pt: (i16, i16)| self.intersects(pt.0 - 64, pt.1 - 64, 128, 128);
            if (self.cursor_dirty && over(self.ptr)) || (Some(self.ptr) != self.drawn_ptr && (over(self.ptr) || self.drawn_ptr.is_some_and(over))) {
                return Ok(true);
            }
        }
        Ok(self.pending)
    }

    /// Ask for a new picture regardless of damage (key frame request).
    pub fn invalidate(&mut self) {
        self.pending = true;
    }

    /// Copy the monitor into the shared buffer and draw the cursor into it.
    pub fn grab(&mut self) -> Result<(), String> {
        self.pending = false;
        self.conn
            .shm_get_image(self.root, self.x, self.y, self.w, self.h, !0, ImageFormat::Z_PIXMAP.into(), self.seg, 0)
            .map_err(xe("ShmGetImage"))?
            .reply()
            .map_err(xe("ShmGetImage"))?;
        if !self.draw_cursor {
            return Ok(());
        }
        if self.cursor_dirty {
            self.cursor_dirty = false;
            match self.conn.xfixes_get_cursor_image().map_err(xe("XFixesGetCursorImage"))?.reply() {
                Ok(img) => {
                    self.ptr = (img.x, img.y);
                    self.cursor = Some(CursorImg {
                        w: img.width,
                        h: img.height,
                        xhot: img.xhot,
                        yhot: img.yhot,
                        pixels: img.cursor_image,
                    });
                }
                Err(_) => self.cursor = None,
            }
        }
        self.drawn_ptr = Some(self.ptr);
        if let Some(c) = &self.cursor {
            let ox = self.ptr.0 as i32 - c.xhot as i32 - self.x as i32;
            let oy = self.ptr.1 as i32 - c.yhot as i32 - self.y as i32;
            let (w, h) = (self.w as i32, self.h as i32);
            let buf = unsafe { std::slice::from_raw_parts_mut(self.map as *mut u32, self.size / 4) };
            for cy in 0..c.h as i32 {
                let y = oy + cy;
                if y < 0 || y >= h {
                    continue;
                }
                for cx in 0..c.w as i32 {
                    let x = ox + cx;
                    if x < 0 || x >= w {
                        continue;
                    }
                    let s = c.pixels[(cy * c.w as i32 + cx) as usize];
                    let a = s >> 24;
                    if a == 0 {
                        continue;
                    }
                    let d = &mut buf[(y * w + x) as usize];
                    if a == 255 {
                        *d = s | 0xFF00_0000;
                        continue;
                    }
                    // Premultiplied source over destination.
                    let inv = 255 - a;
                    let ch = |sh: u32| -> u32 {
                        let sc = (s >> sh) & 0xFF;
                        let dc = (*d >> sh) & 0xFF;
                        ((sc + (dc * inv + 127) / 255).min(255)) << sh
                    };
                    *d = 0xFF00_0000 | ch(16) | ch(8) | ch(0);
                }
            }
        }
        Ok(())
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        let _ = self.conn.damage_destroy(self.damage);
        let _ = self.conn.shm_detach(self.seg);
        let _ = self.conn.flush();
        unsafe { libc::munmap(self.map as *mut libc::c_void, self.size) };
    }
}

// Keep the module names in scope for readers grepping for what this uses.
#[allow(unused_imports)]
use {damage as _, randr as _, xfixes as _};
