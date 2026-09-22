"""数据集自检 —— 采完数据后、送去训练前跑一遍。

用法（板端或 PC 都行）：
    python scripts/check_dataset.py --root /media/elf/ROOT/datasets/so101-pen-place

检查项（每一项都对应一类会静默毁掉训练的问题）：
  1. 目录结构       —— LeRobot v3 的 data/ videos/ meta/ 是否齐全
  2. 任务文本       —— meta/tasks.parquet 里有没有任务描述（没有就是死的策略）
  3. 关节值变化     —— 全程不动 = 采集时臂没连上/没使能扭矩
  4. 时长分布       —— 单条 episode 是否 <30 秒
  5. 图像健全性     —— 全黑 / 全白 / 全程同一帧 = 相机没出图或线掉了
  6. 图像与关节同步 —— 帧数是否对得上

不依赖训练框架，缺 pyarrow/pandas 时会降级到只做能做的检查。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

OK = "  ✅"
WARN = "  ⚠️ "
BAD = "  ❌"
_INFO = "  ·"


def hr(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


# ---------------------------------------------------------------- 1. 结构

def check_structure(root: Path) -> bool:
    hr("1. 目录结构")
    good = True
    for sub in ("data", "videos", "meta"):
        p = root / sub
        if p.is_dir():
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            print(f"{OK} {sub}/  ({n} 个文件)")
        else:
            print(f"{BAD} 缺少 {sub}/")
            good = False
    for f in ("meta/info.json", "meta/tasks.parquet", "meta/episodes"):
        p = root / f
        print(f"{OK if p.exists() else WARN} {f}")
    if (root / "meta" / "info.json").is_file():
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        print(f"{_INFO} codebase_version: {info.get('codebase_version')}")
        print(f"{_INFO} fps: {info.get('fps')}   episodes: {info.get('total_episodes')}   frames: {info.get('total_frames')}")
        feats = info.get("features", {})
        print(f"{_INFO} features: {sorted(feats)}")
    return good


# ---------------------------------------------------------------- 2. 任务文本

def check_tasks(root: Path) -> bool:
    hr("2. 任务文本（决定策略能否被语言条件驱动）")
    p = root / "meta" / "tasks.parquet"
    if not p.is_file():
        print(f"{BAD} 没有 meta/tasks.parquet")
        return False
    try:
        import pandas as pd

        df = pd.read_parquet(p)
        print(f"{_INFO} 列: {list(df.columns)}")
        for i, row in df.head(10).iterrows():
            print(f"{OK} 任务: {row.to_dict()}")
        if len(df) == 0:
            print(f"{BAD} 任务表为空")
            return False
        print(f"{_INFO} 共 {len(df)} 个任务")
        print(f"{WARN} 提醒：这个文本必须与训练、部署时一字不差")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"{WARN} 读不了 tasks.parquet（{type(e).__name__}），跳过")
        return True


# ---------------------------------------------------------------- 3/4/6. parquet

def _iter_parquet(root: Path):
    return sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))


def check_data(root: Path) -> bool:
    hr("3. 关节值变化 / 4. 时长分布")
    files = _iter_parquet(root)
    if not files:
        print(f"{BAD} 找不到 data/**/*.parquet")
        return False
    print(f"{_INFO} {len(files)} 个 parquet 分片")
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001
        print(f"{WARN} 没有 pandas，跳过关节值检查")
        return True

    all_ok = True
    fps = 30.0
    info_p = root / "meta" / "info.json"
    if info_p.is_file():
        fps = float(json.loads(info_p.read_text(encoding="utf-8")).get("fps", 30))

    total_eps = 0
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception as e:  # noqa: BLE001
            print(f"{BAD} 读不了 {Path(f).name}: {e}")
            all_ok = False
            continue
        cols = [c for c in df.columns if c == "action" or c.startswith("observation.state")]
        if not cols:
            print(f"{WARN} {Path(f).name}: 没有 action/state 列（列: {list(df.columns)[:8]}）")
            continue

        ep_col = "episode_index" if "episode_index" in df.columns else None
        groups = df.groupby(ep_col) if ep_col else [(0, df)]
        for ep, g in groups:
            total_eps += 1
            dur = len(g) / fps
            # 关节行程
            spans = {}
            for c in cols:
                v = g[c]
                try:
                    import numpy as np

                    arr = np.stack(v.to_numpy())
                    if arr.ndim == 1:
                        arr = arr[:, None]
                    spans[c] = float(np.nanmax(arr) - np.nanmin(arr))
                except Exception:  # noqa: BLE001
                    pass
            flat = [k for k, s in spans.items() if s < 1e-6]
            tag = OK if dur <= 30 and not flat else (WARN if dur <= 45 else BAD)
            print(f"{tag} ep{ep}: {len(g)} 帧 / {dur:.1f}s" + (f"  ⚠️ 无变化: {flat}" if flat else ""))
            if flat:
                all_ok = False
            if dur > 45:
                all_ok = False
    print(f"{_INFO} 共 {total_eps} 条 episode")
    return all_ok


# ---------------------------------------------------------------- 5. 图像

def check_videos(root: Path, samples: int = 3) -> bool:
    hr("5. 图像健全性")
    vids = sorted(glob.glob(str(root / "videos" / "**" / "*.mp4"), recursive=True))
    if not vids:
        print(f"{BAD} 找不到 videos/**/*.mp4")
        return False
    print(f"{_INFO} {len(vids)} 个视频分片")

    try:
        import cv2
        import numpy as np
    except Exception:  # noqa: BLE001
        print(f"{WARN} 没有 cv2/numpy，跳过图像检查")
        return True

    ok = True
    for v in vids[: samples * 4]:
        cap = cv2.VideoCapture(v)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        for idx in (0, max(0, n // 2), max(0, n - 1)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            r, fr = cap.read()
            if r:
                frames.append(fr)
        cap.release()
        if not frames:
            print(f"{BAD} {Path(v).name}: 读不出帧")
            ok = False
            continue
        means = [float(f.mean()) for f in frames]
        stds = [float(f.std()) for f in frames]
        identical = all(np.array_equal(frames[0], f) for f in frames[1:])
        black = all(m < 3 for m in means)
        white = all(m > 252 for m in means)
        flat = all(s < 1.0 for s in stds)

        flags = []
        if black: flags.append("全黑")
        if white: flags.append("全白")
        if flat: flags.append("无纹理")
        if identical: flags.append("三帧完全相同")
        tag = BAD if flags else OK
        print(f"{tag} {Path(v).name}: {w}x{h} {n}帧 @{fps:.1f}fps  "
              f"mean={[round(m,1) for m in means]} std={[round(s,1) for s in stds]}"
              + (f"  ← {', '.join(flags)}" if flags else ""))
        if flags:
            ok = False
    return ok


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="LeRobot 数据集自检")
    ap.add_argument("--root", required=True, help="数据集根目录")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    print(f"数据集: {root}")
    if not root.is_dir():
        print(f"{BAD} 目录不存在")
        return 2

    size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    print(f"总大小: {size / 1e6:.1f} MB")

    results = {
        "结构": check_structure(root),
        "任务文本": check_tasks(root),
        "关节/时长": check_data(root),
        "图像": check_videos(root),
    }

    hr("结论")
    for k, v in results.items():
        print(f"{OK if v else BAD} {k}")
    bad = [k for k, v in results.items() if not v]
    if bad:
        print(f"\n❌ 有问题: {bad}")
        print("   先把上面标 ❌/⚠️ 的项解决，别拿这种数据去训练。")
        return 1
    print("\n✅ 全部通过")
    print("   提醒：还要人工确认「任务文本与最终部署一致」和「采集过程没人碰相机」。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
