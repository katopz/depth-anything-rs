//! A minimal CLI mirroring `da3-cli depth`.
//!
//! ```sh
//! cargo run --release --example da3 -- \
//!     depth --model models/depth-anything-base-f32.gguf \
//!     --input photo.jpg --png depth.png --pfm depth.pfm --pose pose.json
//! ```
//!
//! 3D exports (single-view, requires a camera-pose head):
//! ```sh
//! cargo run --release --example da3 -- \
//!     depth --model models/depth-anything-base-f32.gguf \
//!     --input photo.jpg --glb scene.glb --colmap sparse/
//! ```
//!
//! 3D-Gaussian reconstruction (requires a GSDPT head; DA3-Giant):
//! ```sh
//! cargo run --release --example da3 -- \
//!     reconstruct --model models/depth-anything-giant-f32.gguf \
//!     --input photo.jpg --ply gaussians.ply
//! ```

use std::process::ExitCode;

use depth_anything::{depth_export, write_gaussian_ply, Engine, ExportSpec};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> depth_anything::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    // Collect ALL values of a repeated flag (for multi-view --input a.jpg --input b.jpg).
    let get_all = |flag: &str| -> Vec<String> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < args.len() {
            if args[i] == flag {
                if let Some(v) = args.get(i + 1) {
                    out.push(v.clone());
                }
            }
            i += 1;
        }
        out
    };
    let cmd = args.get(1).cloned();

    if let Some(t) = get("-t").or_else(|| get("--threads")) {
        std::env::set_var("RAYON_NUM_THREADS", t);
    }

    let model = get("-m")
        .or_else(|| get("--model"))
        .ok_or_else(|| depth_anything::Error::Model("missing --model <path-to.gguf>".into()))?;
    let metric_model = get("--metric-model");
    let input = get("-i").or_else(|| get("--input"));
    let png = get("--png");
    let pfm = get("--pfm");
    let pose = get("--pose");
    let glb = get("--glb");
    let colmap = get("--colmap");
    let colmap_text = args.iter().any(|a| a == "--colmap-text");
    let ply = get("--ply");

    match cmd.as_deref() {
        Some("depth") => {
            let input = input
                .ok_or_else(|| depth_anything::Error::Model("missing --input <image>".into()))?;

            // --metric-model: nested metric-scale depth (anyview + metric branches).
            // Mirrors the C++ `--metric-model` routing in cmd_depth_metric.
            if let Some(metric_path) = metric_model.as_ref() {
                let engine = Engine::load_nested(&model, metric_path, None)?;
                eprintln!(
                    "[da3-rs] nested metric loaded; anyview={}, metric={}",
                    engine.config().checkpoint_name,
                    engine
                        .metric_config()
                        .map(|c| c.checkpoint_name.as_str())
                        .unwrap_or(""),
                );
                let out = engine.depth_metric_path(&input)?;
                let dmin = out.depth.iter().copied().fold(f32::INFINITY, f32::min);
                let dmax = out.depth.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                eprintln!(
                    "[da3-rs] nested metric depth: {}x{} min={:.4} max={:.4} scale_factor={:.6}",
                    out.w, out.h, dmin, dmax, out.scale_factor
                );
                if let Some(p) = pfm {
                    depth_export::write_pfm(p, &out.depth, out.w, out.h)?;
                }
                if let Some(p) = png {
                    depth_export::write_depth_png(p, &out.depth, out.w, out.h, true)?;
                }
                if let Some(p) = pose {
                    depth_export::write_pose_json(p, &out.extrinsics, &out.intrinsics)?;
                }
                return Ok(());
            }

            let engine = Engine::load(&model, None)?;
            eprintln!(
                "[da3-rs] {} loaded; pose head: {}",
                engine.config().checkpoint_name,
                engine.has_pose_head()
            );

            // --glb / --colmap: run the single-view export path (requires pose).
            if glb.is_some() || colmap.is_some() {
                if !engine.has_pose_head() {
                    return Err(depth_anything::Error::Unimplemented(
                        "--glb/--colmap require a camera-pose head (cam.*); \
                         this model has none — use depth_path instead",
                    ));
                }
                let spec = ExportSpec {
                    glb_path: glb.clone(),
                    colmap_dir: colmap.clone(),
                    colmap_binary: !colmap_text,
                    image_name: std::path::Path::new(&input)
                        .file_name()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_string()),
                    glb_opts: None,
                };
                let res = engine.depth_pose_export_path(&input, &spec)?;
                eprintln!(
                    "[da3-rs] export done: glb={}, colmap={}",
                    res.glb_written, res.colmap_written
                );
                return Ok(());
            }

            if pose.is_some() {
                let (d, p) = engine.depth_pose_path(&input)?;
                eprintln!(
                    "[da3-rs] depth+pose: {}x{}  ext trace = {:.4} {:.4} {:.4}",
                    d.h, d.w, p.ext[3], p.ext[7], p.ext[11]
                );
                if let Some(path) = pose {
                    depth_export::write_pose_json(path, &p.ext, &p.intr)?;
                }
                write_depth_outputs(&d, png.as_deref(), pfm.as_deref())?;
            } else {
                let d = engine.depth_path(&input)?;
                eprintln!("[da3-rs] depth: {}x{}", d.h, d.w);
                write_depth_outputs(&d, png.as_deref(), pfm.as_deref())?;
            }
            Ok(())
        }
        Some("multi") => {
            // multi-view: one backbone pass over all views, per-view depth + pose.
            // --input can be repeated (or use --inputs a.jpg b.jpg ...).
            let mut inputs = get_all("-i");
            inputs.extend(get_all("--input"));
            if inputs.is_empty() {
                return Err(depth_anything::Error::Model(
                    "missing --input <image> (repeat for multiple views)".into(),
                ));
            }
            let engine = Engine::load(&model, None)?;
            if !engine.has_pose_head() {
                return Err(depth_anything::Error::Unimplemented(
                    "multi-view requires a camera-pose head (cam.*); this model has none",
                ));
            }
            eprintln!(
                "[da3-rs] {} loaded; pose head: {}; views: {}",
                engine.config().checkpoint_name,
                engine.has_pose_head(),
                inputs.len()
            );
            let paths: Vec<&str> = inputs.iter().map(|s| s.as_str()).collect();
            let out = engine.depth_pose_multi_paths(&paths)?;
            eprintln!(
                "[da3-rs] multi-view: {}x{} ref_view={} views={}",
                out.w,
                out.h,
                out.ref_view,
                out.views.len()
            );
            for (v, view) in out.views.iter().enumerate() {
                let dmin = view.depth.iter().copied().fold(f32::INFINITY, f32::min);
                let dmax = view.depth.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                eprintln!(
                    "[da3-rs]   view {}: depth [{:.4}, {:.4}]  ext trace = {:.4} {:.4} {:.4}",
                    v, dmin, dmax, view.ext[3], view.ext[7], view.ext[11]
                );
                // Write per-view outputs if --png/--pfm/--pose are suffixed with a template.
                // For simplicity, suffix with _v{v}.
                if let Some(tpl) = &png {
                    let p = format_replace_v(tpl, v);
                    depth_export::write_depth_png(&p, &view.depth, out.w, out.h, true)?;
                }
                if let Some(tpl) = &pfm {
                    let p = format_replace_v(tpl, v);
                    depth_export::write_pfm(&p, &view.depth, out.w, out.h)?;
                }
                if let Some(tpl) = &pose {
                    let p = format_replace_v(tpl, v);
                    depth_export::write_pose_json(&p, &view.ext, &view.intr)?;
                }
            }
            Ok(())
        }
        Some("reconstruct") => {
            // 3D-Gaussian reconstruction (requires a GSDPT head; DA3-Giant).
            let input = input
                .ok_or_else(|| depth_anything::Error::Model("missing --input <image>".into()))?;
            let ply_path = ply
                .ok_or_else(|| depth_anything::Error::Model("missing --ply <output.ply>".into()))?;
            let engine = Engine::load(&model, None)?;
            eprintln!(
                "[da3-rs] {} loaded; pose head: {}; gs head: {}",
                engine.config().checkpoint_name,
                engine.has_pose_head(),
                engine.has_gs_head()
            );
            if !engine.has_gs_head() {
                return Err(depth_anything::Error::Unimplemented(
                    "reconstruct requires a GSDPT head (gs.*); this model has none. \
                     Load a DA3-Giant GGUF that includes the Gaussian head.",
                ));
            }
            let out = engine.reconstruct_path(&input)?;
            write_gaussian_ply(&ply_path, &out.gaussians)?;
            eprintln!(
                "[da3-rs] reconstruct: {}x{} -> {} gaussians, wrote {}",
                out.w, out.h, out.gaussians.n, ply_path
            );
            Ok(())
        }
        Some("info") => {
            // `info` only needs the header metadata; don't load weights.
            let file = depth_anything::GgufFile::open(&model)?;
            let c = depth_anything::Config::from_gguf(&file)?;
            let mut names: Vec<&String> = file.tensor_names().collect();
            names.sort();
            println!("checkpoint_name  : {}", c.checkpoint_name);
            println!("arch             : {:?}", c.arch);
            println!("patch_size       : {}", c.patch_size);
            println!("embed_dim        : {}", c.embed_dim);
            println!("depth (blocks)   : {}", c.depth);
            println!("num_heads        : {}", c.num_heads);
            println!("head_dim         : {}", c.head_dim);
            println!("mlp_hidden       : {}", c.mlp_hidden);
            println!("ffn_type         : {:?}", c.ffn_type);
            println!("cat_token        : {}", c.cat_token);
            println!("head_pos_embed   : {}", c.head_pos_embed);
            println!("out_layers       : {:?}", c.out_layers);
            println!("head_out_channels: {:?}", c.head_out_channels);
            println!("resize_target    : {}", c.img_resize_target);
            println!("resize_mode      : {:?}", c.img_resize_mode);
            println!("max_depth        : {}", c.head_max_depth);
            println!(
                "has_pose_head    : {}",
                names.iter().any(|n| n.starts_with("cam."))
            );
            println!("tensor_count     : {}", names.len());
            println!("-- first 20 tensor names --");
            for n in names.iter().take(20) {
                println!("  {n}");
            }
            Ok(())
        }
        Some(other) => Err(depth_anything::Error::Model(format!(
            "unknown command '{other}' (expected: depth | multi | reconstruct | info)"
        ))),
        None => Err(depth_anything::Error::Model(
            "usage: da3 <depth|multi|reconstruct|info> --model M --input I".into(),
        )),
    }
}

fn write_depth_outputs(
    d: &depth_anything::DepthOutput,
    png: Option<&str>,
    pfm: Option<&str>,
) -> depth_anything::Result<()> {
    if let Some(p) = png {
        depth_export::write_depth_png(p, &d.depth, d.w, d.h, true)?;
    }
    if let Some(p) = pfm {
        depth_export::write_pfm(p, &d.depth, d.w, d.h)?;
    }
    Ok(())
}

/// Replace a `{v}` placeholder in a path template with the view index, or
/// append `_v{v}` before the file extension if no placeholder is present.
/// e.g. `depth_{v}.pfm` -> `depth_0.pfm`, `out.png` -> `out_v0.png`.
fn format_replace_v(tpl: &str, v: usize) -> String {
    if tpl.contains("{v}") {
        tpl.replace("{v}", &v.to_string())
    } else {
        // Split at the last '.' to insert _v{v} before the extension.
        if let Some(dot) = tpl.rfind('.') {
            let (stem, ext) = tpl.split_at(dot);
            format!("{stem}_v{v}{ext}")
        } else {
            format!("{tpl}_v{v}")
        }
    }
}
