//! 离线运维工具：查看 store 概况、切换表示、压实。
//!
//! vstore-convert <dir> [--info] [--to f32|sq8|sq4|bq] [--progress]
//!                      [--mem-budget <bytes>] [--compact-slab <i>] [--compact]
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

use vstore::{Config, Error, JobOptions, JobProgress, OpenOptions, Repr, Store};

fn usage() -> ExitCode {
    eprintln!(
        "用法: vstore-convert <dir> [选项]\n\
         \n\
         --info                 打印概况（dim/n/alive/capacity/片/表示/建议模式）\n\
         --to <repr>            切换表示: f32 | sq8\n\
         --progress             打印转换进度\n\
         --mem-budget <bytes>   --info 时用来算建议模式（默认 4 GiB）\n\
         --compact              逐片压实全部片\n\
         --compact-slab <i>     只压实第 i 片\n\
         --model-hash <h>       不填则从 meta.json 读"
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        return usage();
    }
    let dir = std::path::PathBuf::from(&args[0]);
    let mut info = false;
    let mut to: Option<Repr> = None;
    let mut progress = false;
    let mut compact = false;
    let mut compact_slab: Option<u32> = None;
    let mut model_hash: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--info" => info = true,
            "--progress" => progress = true,
            "--compact" => compact = true,
            "--to" => {
                i += 1;
                to = match args.get(i).map(|s| s.as_str()) {
                    Some("f32") => Some(Repr::F32),
                    Some("sq8") => Some(Repr::Sq8),
                    _ => return usage(),
                };
            }
            "--compact-slab" => {
                i += 1;
                match args.get(i).and_then(|s| s.parse().ok()) {
                    Some(v) => compact_slab = Some(v),
                    None => return usage(),
                }
            }
            "--model-hash" => {
                i += 1;
                match args.get(i) {
                    Some(v) => model_hash = Some(v.clone()),
                    None => return usage(),
                }
            }
            other => {
                eprintln!("未知参数: {other}");
                return usage();
            }
        }
        i += 1;
    }

    // 运维工具从 meta 里读指纹再传回去：它不消费模型，只要求 store 自洽。
    let hash = match model_hash {
        Some(h) => h,
        None => match std::fs::read(dir.join("meta.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .and_then(|v| v["model_hash"].as_str().map(|s| s.to_string()))
        {
            Some(h) => h,
            None => {
                eprintln!("读不到 {}/meta.json", dir.display());
                return ExitCode::from(1);
            }
        },
    };

    let store = match Store::open(&dir, OpenOptions::new(hash)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("打开失败: {e}");
            return ExitCode::from(1);
        }
    };

    if info {
        let st = store.stats();
        let (slabs, slab_rows) = store.slab_info();
        println!("dim        {}", st.dim);
        println!("n(水位)    {}", st.n);
        println!("alive      {}", st.alive);
        println!("capacity   {}", st.capacity);
        println!("slabs      {slabs} × {slab_rows} 行");
        println!("repr       {:?}", st.repr);
        println!("model      {}", store.model_hash());
        println!("suggest    {:?}", store.suggest_mode());
    }

    let cancel = AtomicBool::new(false);
    let cb = |p: JobProgress| {
        if progress {
            let pct = if p.total == 0 {
                100.0
            } else {
                p.done as f64 * 100.0 / p.total as f64
            };
            eprint!("\r{pct:5.1}%  {}/{} 行", p.done, p.total);
        }
    };
    let opts = JobOptions {
        progress: Some(&cb),
        cancel: Some(&cancel),
    };

    let mut fail: Option<Error> = None;
    if let Some(target) = to {
        if progress {
            eprintln!("convert → {target:?}");
        }
        if let Err(e) = store.convert_with(target, &opts) {
            fail = Some(e);
        } else if progress {
            eprintln!();
        }
    }
    if let Some(slab) = compact_slab {
        if let Err(e) = store.compact_slab_with(slab, &opts) {
            fail = Some(e);
        }
    }
    if compact {
        let (slabs, _) = store.slab_info();
        for slab in 0..slabs {
            if let Err(e) = store.compact_slab_with(slab, &opts) {
                fail = Some(e);
                break;
            }
        }
    }

    match fail {
        Some(e) => {
            eprintln!("\n失败: {e}");
            ExitCode::from(1)
        }
        None => ExitCode::SUCCESS,
    }
}

#[allow(dead_code)]
fn _unused(_: Config) {
    let _ = Ordering::Relaxed;
}
