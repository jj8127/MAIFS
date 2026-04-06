#!/usr/bin/env python3
"""
Prepare additional SOTA benchmark datasets for MAIFS experiments.

What this script does:
1) Export OpenSDI samples (HF) into MAIFS 3-class folder layout.
2) Export AI-GenBench fake-part samples (HF).
3) Build an AI-GenBench proxy 3-class set by combining:
   - authentic: existing real-image directory
   - manipulated: existing tampered-image directory
   - ai_generated: AI-GenBench fake-part export
4) Check accessibility status for restricted datasets (Coverage/Columbia/NIST16).

Output layout:
  datasets/
    OpenSDID_subset/
      authentic/
      manipulated/
      ai_generated/
    AI-GenBench_fakepart_subset/
      ai_generated/
    AI-GenBench_proxy/
      authentic/
      manipulated/
      ai_generated/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from datasets import load_dataset
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"


@dataclass
class ExportStats:
    target_dir: str
    counts: Dict[str, int]
    total_saved: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare SOTA datasets for MAIFS")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--opensdi-max-per-class", type=int, default=300)
    p.add_argument("--aigen-fake-max", type=int, default=300)
    p.add_argument(
        "--proxy-authentic-src",
        type=str,
        default="datasets/GenImage_subset/BigGAN/val/nature",
        help="real-image source for AI-GenBench proxy authentic class",
    )
    p.add_argument(
        "--proxy-manipulated-src",
        type=str,
        default="datasets/IMD2020_subset/IMD2020_Generative_Image_Inpainting_yu2018_01/images",
        help="tampered-image source for AI-GenBench proxy manipulated class",
    )
    p.add_argument("--proxy-max-per-class", type=int, default=300)
    p.add_argument("--report-out", type=str, default="")
    p.add_argument("--skip-opensdi", action="store_true")
    p.add_argument("--skip-aigen-fake", action="store_true")
    p.add_argument("--skip-proxy-build", action="store_true")
    p.add_argument(
        "--clear-targets",
        action="store_true",
        help="remove previous exported subsets before writing",
    )
    p.add_argument(
        "--force-os-exit",
        action="store_true",
        help="workaround for occasional datasets streaming shutdown crash",
    )
    return p.parse_args()


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _iter_images(path: Path) -> Iterable[Path]:
    if not path.exists():
        return []
    if path.is_file() and _is_image_file(path):
        return [path]
    files = []
    for p in path.rglob("*"):
        if p.is_file() and _is_image_file(p):
            files.append(p)
    return files


def _ensure_clean_dir(path: Path, clear: bool) -> None:
    if clear and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _save_image(img: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        img.convert("RGB").save(out_path, quality=95)
    else:
        img.save(out_path)


def _map_opensdi_class(key: str, label: int) -> Optional[str]:
    """
    Map OpenSDI example to MAIFS 3 classes:
      - label 0 -> authentic
      - label 1 + partial -> manipulated
      - label 1 + entire/unknown -> ai_generated
    """
    if label == 0:
        return "authentic"
    if label != 1:
        return None
    key_l = (key or "").lower()
    if key_l.startswith("partial/") or "/partial/" in key_l:
        return "manipulated"
    return "ai_generated"


def _iter_targeted_opensdi_shards() -> List[Tuple[str, str]]:
    """
    Return targeted shard URLs to quickly cover each MAIFS class.
    """
    return [
        # manipulated (partial fake-rich shard)
        ("hf://datasets/nebula/OpenSDI_train/data/sd15-00000-of-00070.parquet", "manipulated"),
        # ai_generated (entire fake-rich shards)
        ("hf://datasets/nebula/OpenSDI_test/data/sd2-00000-of-00007.parquet", "ai_generated"),
        ("hf://datasets/nebula/OpenSDI_test/data/sdxl-00000-of-00009.parquet", "ai_generated"),
        ("hf://datasets/nebula/OpenSDI_test/data/sd3-00000-of-00010.parquet", "ai_generated"),
        ("hf://datasets/nebula/OpenSDI_test/data/flux-00000-of-00007.parquet", "ai_generated"),
        ("hf://datasets/nebula/OpenSDI_test/data/sd15-00000-of-00019.parquet", "ai_generated"),
        # authentic (real-rich tail shards)
        ("hf://datasets/nebula/OpenSDI_train/data/sd15-00069-of-00070.parquet", "authentic"),
        ("hf://datasets/nebula/OpenSDI_test/data/sd2-00006-of-00007.parquet", "authentic"),
        ("hf://datasets/nebula/OpenSDI_test/data/sdxl-00008-of-00009.parquet", "authentic"),
        ("hf://datasets/nebula/OpenSDI_test/data/sd3-00009-of-00010.parquet", "authentic"),
        ("hf://datasets/nebula/OpenSDI_test/data/flux-00006-of-00007.parquet", "authentic"),
        ("hf://datasets/nebula/OpenSDI_test/data/sd15-00018-of-00019.parquet", "authentic"),
    ]


def export_opensdi_subset(target_root: Path, max_per_class: int, clear_targets: bool) -> ExportStats:
    classes = ["authentic", "manipulated", "ai_generated"]
    counts = {c: 0 for c in classes}
    _ensure_clean_dir(target_root, clear_targets)
    for c in classes:
        (target_root / c).mkdir(parents=True, exist_ok=True)

    manifest_path = target_root / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    with manifest_path.open("w", encoding="utf-8") as mf:
        for shard_url, preferred_cls in _iter_targeted_opensdi_shards():
            if all(v >= max_per_class for v in counts.values()):
                break

            ds = load_dataset("parquet", data_files=shard_url, split="train", streaming=True)
            for idx, ex in enumerate(ds):
                if all(v >= max_per_class for v in counts.values()):
                    break

                key = str(ex.get("key", ""))
                label = int(ex.get("label", -1))
                cls = _map_opensdi_class(key, label)
                if cls is None:
                    continue
                # targeted shard should mostly map to one class; keep only desired class
                if cls != preferred_cls:
                    continue
                if counts[cls] >= max_per_class:
                    continue

                img = ex.get("image")
                if not isinstance(img, Image.Image):
                    continue

                base = f"opensdi__{preferred_cls}__{counts[cls]:06d}"
                out_path = target_root / cls / f"{base}.jpg"
                _save_image(img, out_path)
                counts[cls] += 1

                if counts[cls] % 50 == 0:
                    print(f"[OpenSDI] {cls}: {counts[cls]}/{max_per_class}")

                mf.write(
                    json.dumps(
                        {
                            "source": shard_url,
                            "key": key,
                            "label": label,
                            "class": cls,
                            "saved_path": str(out_path.relative_to(ROOT)),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    return ExportStats(
        target_dir=str(target_root.relative_to(ROOT)),
        counts=counts,
        total_saved=int(sum(counts.values())),
    )


def export_aigen_fake_subset(target_root: Path, max_samples: int, clear_targets: bool) -> ExportStats:
    _ensure_clean_dir(target_root, clear_targets)
    ai_dir = target_root / "ai_generated"
    ai_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    manifest_path = target_root / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    with manifest_path.open("w", encoding="utf-8") as mf:
        for split in ["train", "validation"]:
            if count >= max_samples:
                break
            ds = load_dataset("lrzpellegrini/AI-GenBench-fake_part", split=split, streaming=True)
            for ex in ds:
                if count >= max_samples:
                    break
                img = ex.get("image")
                if not isinstance(img, Image.Image):
                    continue
                base = f"aigenbench_fakepart__{split}__{count:06d}"
                out_path = ai_dir / f"{base}.jpg"
                _save_image(img, out_path)
                mf.write(
                    json.dumps(
                        {
                            "split": split,
                            "generator": ex.get("generator"),
                            "origin_dataset": ex.get("origin_dataset"),
                            "label": ex.get("label"),
                            "saved_path": str(out_path.relative_to(ROOT)),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                count += 1

    return ExportStats(
        target_dir=str(target_root.relative_to(ROOT)),
        counts={"ai_generated": int(count)},
        total_saved=int(count),
    )


def _copy_sampled_images(src: Path, dst: Path, n: int, seed: int) -> int:
    imgs = list(_iter_images(src))
    if not imgs:
        return 0
    rng = random.Random(seed)
    if len(imgs) > n:
        imgs = rng.sample(imgs, n)
    else:
        imgs = sorted(imgs)

    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for i, p in enumerate(imgs):
        ext = p.suffix.lower()
        out = dst / f"{i:06d}{ext if ext else '.jpg'}"
        shutil.copy2(p, out)
        copied += 1
    return copied


def build_aigen_proxy_dataset(
    target_root: Path,
    authentic_src: Path,
    manipulated_src: Path,
    aigen_fake_dir: Path,
    max_per_class: int,
    seed: int,
    clear_targets: bool,
) -> ExportStats:
    _ensure_clean_dir(target_root, clear_targets)
    auth_dir = target_root / "authentic"
    mani_dir = target_root / "manipulated"
    ai_dir = target_root / "ai_generated"

    auth_n = _copy_sampled_images(authentic_src, auth_dir, max_per_class, seed + 11)
    mani_n = _copy_sampled_images(manipulated_src, mani_dir, max_per_class, seed + 29)
    ai_n = _copy_sampled_images(aigen_fake_dir, ai_dir, max_per_class, seed + 53)

    meta = {
        "authentic_src": str(authentic_src),
        "manipulated_src": str(manipulated_src),
        "aigen_fake_src": str(aigen_fake_dir),
        "max_per_class": int(max_per_class),
        "counts": {
            "authentic": auth_n,
            "manipulated": mani_n,
            "ai_generated": ai_n,
        },
    }
    (target_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return ExportStats(
        target_dir=str(target_root.relative_to(ROOT)),
        counts={k: int(v) for k, v in meta["counts"].items()},
        total_saved=int(sum(meta["counts"].values())),
    )


def check_restricted_dataset_access() -> Dict[str, Dict[str, object]]:
    checks = {
        "coverage": {
            "url": "https://1drv.ms/f/s!AggVhXcCj1FLhUUyUrqSpV_yI_GH",
            "local_candidates": [
                "datasets/COVERAGE",
                "datasets/Coverage",
                "datasets/coverage",
            ],
        },
        "columbia": {
            "url": "https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/",
            "local_candidates": [
                "datasets/Columbia",
                "datasets/COLUMBIA",
                "datasets/columbia",
            ],
        },
        "nist16": {
            "url": "https://mig.nist.gov/MFC/PubData/Resources.html",
            "local_candidates": [
                "datasets/NIST16",
                "datasets/nist16",
            ],
        },
    }

    out: Dict[str, Dict[str, object]] = {}
    for name, cfg in checks.items():
        url = str(cfg["url"])
        status_code = None
        error_msg = None
        try:
            resp = requests.head(url, allow_redirects=True, timeout=20)
            status_code = int(resp.status_code)
        except Exception as e:  # pragma: no cover
            error_msg = str(e)

        locals_found = []
        for p in cfg["local_candidates"]:
            ap = ROOT / p
            if ap.exists():
                locals_found.append(str(ap.relative_to(ROOT)))

        out[name] = {
            "url": url,
            "http_status": status_code,
            "error": error_msg,
            "local_found": locals_found,
            "local_ready": bool(locals_found),
        }
    return out


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    report: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "seed": int(args.seed),
        "exports": {},
        "restricted_dataset_checks": {},
    }

    if not args.skip_opensdi:
        opensdi_root = DATASETS_DIR / "OpenSDID_subset"
        stats = export_opensdi_subset(
            target_root=opensdi_root,
            max_per_class=int(args.opensdi_max_per_class),
            clear_targets=bool(args.clear_targets),
        )
        report["exports"]["opensdi"] = asdict(stats)
        print(f"[OpenSDI] {stats.counts} -> {stats.target_dir}")

    if not args.skip_aigen_fake:
        aigen_fake_root = DATASETS_DIR / "AI-GenBench_fakepart_subset"
        stats = export_aigen_fake_subset(
            target_root=aigen_fake_root,
            max_samples=int(args.aigen_fake_max),
            clear_targets=bool(args.clear_targets),
        )
        report["exports"]["aigen_fake_part"] = asdict(stats)
        print(f"[AI-GenBench fake-part] {stats.counts} -> {stats.target_dir}")

    if not args.skip_proxy_build:
        proxy_root = DATASETS_DIR / "AI-GenBench_proxy"
        auth_src = (ROOT / args.proxy_authentic_src).resolve()
        mani_src = (ROOT / args.proxy_manipulated_src).resolve()
        fake_src = (DATASETS_DIR / "AI-GenBench_fakepart_subset" / "ai_generated").resolve()
        stats = build_aigen_proxy_dataset(
            target_root=proxy_root,
            authentic_src=auth_src,
            manipulated_src=mani_src,
            aigen_fake_dir=fake_src,
            max_per_class=int(args.proxy_max_per_class),
            seed=int(args.seed),
            clear_targets=bool(args.clear_targets),
        )
        report["exports"]["aigen_proxy"] = asdict(stats)
        print(f"[AI-GenBench proxy] {stats.counts} -> {stats.target_dir}")

    checks = check_restricted_dataset_access()
    report["restricted_dataset_checks"] = checks
    print("[restricted checks]")
    for name, info in checks.items():
        print(
            f"  - {name}: status={info.get('http_status')} local_ready={info.get('local_ready')} "
            f"local={info.get('local_found')}"
        )

    if args.report_out:
        out_path = (ROOT / args.report_out).resolve()
    else:
        out_path = ROOT / "experiments" / "results" / "sota_dataset_prep_report_20260304.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")

    return 0


if __name__ == "__main__":
    code = main()
    # Workaround for occasional PyArrow/Datasets shutdown crash in streaming mode.
    if "--force-os-exit" in sys.argv:
        os._exit(code)
    raise SystemExit(code)
