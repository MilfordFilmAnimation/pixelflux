/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! How much of a monitor X reports as damaged per frame: whether grabbing only the changed part
//! would save copying. Usage: damage_probe <display> <xauthority> <monitor> <seconds>

use std::time::{Duration, Instant};

use pixelflux_seat::x11;
use x11rb::connection::Connection;
use x11rb::protocol::damage::{ConnectionExt as _, ReportLevel};
use x11rb::protocol::Event;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d = x11::XDisplay { display: a[1].clone(), xauthority: Some(a[2].clone().into()) };
    let mi: usize = a[3].parse().unwrap();
    let secs: f64 = a[4].parse().unwrap();
    let (conn, scr) = x11::connect(&d).unwrap();
    let m = x11::monitors(&conn, scr).unwrap()[mi].clone();
    let root = conn.setup().roots[scr].root;
    conn.damage_query_version(1, 1).unwrap().reply().unwrap();
    let dmg = conn.generate_id().unwrap();
    conn.damage_create(dmg, root, ReportLevel::RAW_RECTANGLES).unwrap();
    conn.flush().unwrap();
    let (mx0, my0, mx1, my1) = (m.x, m.y, m.x + m.w as i32, m.y + m.h as i32);
    let area = (m.w as f64) * (m.h as f64);
    let t0 = Instant::now();
    let mut frames = 0u32;
    let (mut sum_bbox, mut sum_rects, mut n_rects, mut full) = (0f64, 0f64, 0u64, 0u32);
    while t0.elapsed().as_secs_f64() < secs {
        std::thread::sleep(Duration::from_millis(16));
        let (mut bx0, mut by0, mut bx1, mut by1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        let mut rect_area = 0f64;
        let mut any = false;
        while let Some(ev) = conn.poll_for_event().unwrap() {
            if let Event::DamageNotify(e) = ev {
                let (x0, y0) = ((e.area.x as i32).max(mx0), (e.area.y as i32).max(my0));
                let (x1, y1) = ((e.area.x as i32 + e.area.width as i32).min(mx1), (e.area.y as i32 + e.area.height as i32).min(my1));
                if x1 <= x0 || y1 <= y0 {
                    continue;
                }
                any = true;
                n_rects += 1;
                rect_area += ((x1 - x0) * (y1 - y0)) as f64;
                bx0 = bx0.min(x0);
                by0 = by0.min(y0);
                bx1 = bx1.max(x1);
                by1 = by1.max(y1);
            }
        }
        if any {
            frames += 1;
            let b = ((bx1 - bx0) * (by1 - by0)) as f64 / area;
            sum_bbox += b;
            sum_rects += (rect_area / area).min(1.0);
            if b > 0.95 {
                full += 1;
            }
        }
    }
    let f = frames.max(1) as f64;
    println!(
        "RESULT monitor {} {}x{}: {} damaged frames in {:.0} s, {:.1} rects/frame, bounding box {:.1}% of the monitor on average, rect area {:.1}%, full-monitor frames {}",
        m.name, m.w, m.h, frames, secs, n_rects as f64 / f, 100.0 * sum_bbox / f, 100.0 * sum_rects / f, full
    );
}
