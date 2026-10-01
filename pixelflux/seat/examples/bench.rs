/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Time on the seat for one monitor: grab start to encoded frame, at a fixed frame rate.
//! Usage: bench <display> <xauthority> <monitor> <h264|hevc> <seconds> [always] [out.h264]

use std::io::Write;
use std::time::{Duration, Instant};

use pixelflux_seat::*;

fn pct(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d = XDisplay { display: a[1].clone(), xauthority: Some(a[2].clone().into()) };
    let mi: usize = a[3].parse().unwrap();
    let codec = if a[4] == "hevc" { Codec::Hevc } else { Codec::H264 };
    let secs: f64 = a[5].parse().unwrap();
    let always = a.get(6).map(|s| s == "always").unwrap_or(false);
    let mut out = a.get(7).map(|p| std::fs::File::create(p).unwrap());
    let (conn, scr) = x11::connect(&d).unwrap();
    let mons = x11::monitors(&conn, scr).unwrap();
    println!("monitors: {mons:?}");
    let m = &mons[mi];
    let mut cap = Capture::new(&d, m.x, m.y, m.w, m.h).unwrap();
    let mut enc = Encoder::new(EncoderConfig { codec, width: m.w, height: m.h, fps: 60, bitrate_kbps: 25000, ..Default::default() }).unwrap();
    println!("encoder on {}", Encoder::device_name().unwrap());
    enc.pin_host(cap.data(), cap.len()).unwrap();
    let period = Duration::from_nanos(1_000_000_000 / 60);
    let t_start = Instant::now();
    let mut next = t_start;
    let (mut grab, mut encd, mut tot) = (vec![], vec![], vec![]);
    let mut bytes = 0usize;
    let mut skipped = 0;
    while t_start.elapsed().as_secs_f64() < secs {
        next += period;
        let now = Instant::now();
        if next > now {
            std::thread::sleep(next - now);
        } else {
            next = now;
        }
        let t0 = Instant::now();
        let changed = cap.poll().unwrap();
        if !changed && !always {
            skipped += 1;
            continue;
        }
        cap.grab().unwrap();
        let t1 = Instant::now();
        let f = enc.encode(cap.data(), cap.stride(), false).unwrap();
        let t2 = Instant::now();
        grab.push((t1 - t0).as_secs_f64() * 1e3);
        encd.push((t2 - t1).as_secs_f64() * 1e3);
        tot.push((t2 - t0).as_secs_f64() * 1e3);
        bytes += f.data.len();
        if let Some(o) = out.as_mut() {
            o.write_all(&f.data).unwrap();
        }
    }
    let el = t_start.elapsed().as_secs_f64();
    let n = tot.len();
    println!(
        "RESULT frames={n} skipped={skipped} fps={:.2} mbit_s={:.1} grab_p50={:.2} grab_p99={:.2} enc_p50={:.2} enc_p99={:.2} seat_p50={:.2} seat_p99={:.2}",
        n as f64 / el,
        bytes as f64 * 8.0 / el / 1e6,
        pct(&mut grab, 0.5),
        pct(&mut grab, 0.99),
        pct(&mut encd, 0.5),
        pct(&mut encd, 0.99),
        pct(&mut tot, 0.5),
        pct(&mut tot, 0.99)
    );
}
