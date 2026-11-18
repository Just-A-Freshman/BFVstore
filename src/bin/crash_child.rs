//! 崩溃注入用的子进程：反复写，随时准备被父进程强杀。
//! 用法：crash_child <dir>
use vstore::{Config, OpenOptions, Repr, Store};

fn main() {
    let dir = std::env::args().nth(1).expect("usage: crash_child <dir>");
    const DIM: usize = 64;
    const BATCH: usize = 16;

    let store = match Store::open(&dir, OpenOptions::new("crash")) {
        Ok(s) => s,
        Err(_) => Store::create(&dir, Config::new(DIM as u32, "crash")).expect("create"),
    };

    // 派生表示走一遍，让崩溃注入覆盖到它
    let _ = store.convert(Repr::Sq8);

    let mut seed: u64 = std::process::id() as u64 | 1;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };

    loop {
        let mut buf = vec![0f32; BATCH * DIM];
        for v in buf.iter_mut() {
            *v = (rnd() % 2000) as f32 / 1000.0 - 1.0;
        }
        store.append(&buf).expect("append");

        let n = store.n();
        if n >= 8 {
            if rnd() % 3 == 0 {
                let slots: Vec<u32> = (0..4).map(|_| (rnd() % n as u64) as u32).collect();
                let _ = store.overwrite(&slots, &buf[..4 * DIM]);
            }
            if rnd() % 3 == 0 {
                let slots: Vec<u32> = (0..3).map(|_| (rnd() % n as u64) as u32).collect();
                let _ = store.delete(&slots);
            }
        }
    }
}
