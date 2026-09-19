"""ONNX → RKNN 转换，并**自动确认输入顺序**。

为什么需要这个脚本
------------------
实测：ACT 的 ONNX 输入顺序是 `[cam_high, cam_left, state]`，转换后的 RKNN
期望 `[state, cam_high, cam_left]` —— **RKNN 编译器会重排输入**。

后果分两种：

* 顺序错得"明显"：报 `input[0] need 2dims input, but 4dims`（state 是 2 维、图像 4 维）
* 顺序错得"隐蔽"：两个图像输入形状相同，互换**不报错**，机械臂动作全错

所以本脚本的做法是：

1. 造一组**互相可区分**的参考输入（两路图像用不同数值填充）
2. 用 ONNX Runtime 跑出参考输出
3. 暴力枚举图像输入的**所有排列**，逐个跑 RKNN，与参考输出比对
4. 取误差最小的排列；若最小误差仍超过容差 → **报错退出**，不生成 manifest
5. 把确认结果写进 `*.rknn.manifest.json`

板端 `rk3588/act_rknn.py` 在缺少 manifest 时**拒绝运行**。

注意：rknn-toolkit2 要求 torch<=2.4.0 / numpy<=1.26.4，与 LeRobot 环境冲突。
**必须在独立的 .venv-rknn 里跑本脚本。**

用法
----
    python3 -m venv .venv-rknn && source .venv-rknn/bin/activate
    pip install rknn-toolkit2==2.3.2 onnx onnxruntime numpy
    python pc/convert/convert_to_rknn.py \
        --onnx models/act.onnx \
        --out  models/act.rknn
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger("convert")

# fp16 下 RKNN 与 ORT 的允许差异。超过它说明输入顺序错了或算子回落到了 CPU。
DEFAULT_TOL = 1e-2


def _onnx_inputs(onnx_path: str) -> list[tuple[str, tuple]]:
    import onnx

    m = onnx.load(onnx_path)
    out = []
    for inp in m.graph.input:
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else -1)
        out.append((inp.name, tuple(dims)))
    return out


def _make_reference_inputs(inputs: list[tuple[str, tuple]], seed: int = 0) -> list[np.ndarray]:
    """造参考输入 —— 关键是**每路图像数值不同**，否则无法区分是否被互换。"""
    rng = np.random.default_rng(seed)
    arrs = []
    img_idx = 0
    for name, shape in inputs:
        shape = tuple(1 if d <= 0 else d for d in shape)
        if len(shape) == 2:
            # state：用中位附近的值，接近真实
            a = rng.normal(2047.0, 300.0, size=shape).astype(np.float32)
        elif len(shape) == 4:
            # 图像：每路一个明显不同的常数基底 + 噪声
            base = 0.1 + 0.35 * img_idx
            a = (base + 0.05 * rng.random(size=shape)).astype(np.float32)
            img_idx += 1
        else:
            a = rng.random(size=shape).astype(np.float32)
        arrs.append(a)
    return arrs


def _ort_run(onnx_path: str, feeds: dict[str, np.ndarray]) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    res = sess.run(out_names, feeds)
    return np.asarray(res[0], dtype=np.float32)


def _rknn_run(rknn, feeds_ordered: list[np.ndarray]) -> np.ndarray:
    outs = rknn.inference(inputs=feeds_ordered)
    return np.asarray(outs[0], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="导出的 ACT ONNX")
    ap.add_argument("--out", required=True, help="输出的 .rknn 路径")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    onnx_path = str(Path(args.onnx).resolve())
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 读 ONNX 输入 ----------
    inputs = _onnx_inputs(onnx_path)
    log.info("ONNX 输入:")
    for n, s in inputs:
        log.info("  %-40s %s", n, s)

    state_idx = [i for i, (_, s) in enumerate(inputs) if len(s) == 2]
    image_idx = [i for i, (_, s) in enumerate(inputs) if len(s) == 4]
    if len(state_idx) != 1:
        log.error("期望恰好 1 个 2 维 state 输入，实际 %d 个", len(state_idx))
        return 2
    if not image_idx:
        log.error("未找到 4 维图像输入")
        return 2
    log.info("state 输入: %s | 图像输入: %s", inputs[state_idx[0]][0], [inputs[i][0] for i in image_idx])

    # ---------- 2. 语义槽位命名 ----------
    # state -> "state"；图像按 ONNX 名推断，单相机统一叫 "front"
    def image_slot(name: str, k: int) -> str:
        low = name.lower()
        for key, slot in (
            ("wrist", "wrist"), ("left", "camera_left"), ("right", "camera_right"),
            ("front", "front"), ("top", "front"), ("high", "front"),
        ):
            if key in low:
                return slot
        if len(image_idx) == 1:
            return "front"
        return f"camera{k + 1}"

    slot_of: dict[int, str] = {state_idx[0]: "state"}
    for k, i in enumerate(image_idx):
        slot_of[i] = image_slot(inputs[i][0], k)

    # ---------- 3. 参考输出（ORT） ----------
    ref_arrs = _make_reference_inputs(inputs, seed=args.seed)
    # ORT 需要按 ONNX 图声明的名字喂
    feeds = {inputs[i][0]: ref_arrs[i] for i in range(len(inputs))}
    # 图像输入一般固定 batch=1
    for i in image_idx:
        feeds[inputs[i][0]] = ref_arrs[i]

    log.info("用 ONNX Runtime 生成参考输出……")
    ref = _ort_run(onnx_path, feeds)
    log.info("参考输出 shape=%s  均值=%.4f  标准差=%.4f", ref.shape, float(ref.mean()), float(ref.std()))
    if not np.isfinite(ref).all():
        log.error("参考输出含 NaN/Inf，ONNX 本身有问题，先查导出")
        return 2
    if float(np.abs(ref).max()) < 1e-8:
        log.error("参考输出全零，ONNX 有问题，先查导出")
        return 2

    # ---------- 4. 转 RKNN ----------
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    log.info("rknn.config(target_platform='rk3588', float_dtype='float16')")
    ret = rknn.config(
        target_platform="rk3588",
        float_dtype="float16",
        optimization_level=3,
        single_core_mode=False,
    )
    if ret != 0:
        log.error("rknn.config 失败 ret=%d", ret)
        return 3

    log.info("加载 ONNX 并构建（不做量化：transformer 图走 fp16）……")
    if rknn.load_onnx(model=onnx_path) != 0:
        log.error("load_onnx 失败")
        return 3
    # 故意不传 dataset —— do_quantization=False
    if rknn.build(do_quantization=False) != 0:
        log.error(
            "build 失败。请检查转换日志里是否出现 `unsupport cpu <Op> op` —— "
            "那是硬性阻断，不要指望 CPU fallback 兜底。\n"
            "对策：回到 exporter 层改 PyTorch 源码消掉该算子，**不要做 ONNX 事后手术**"
            "（插入的 Constant 会被 fold_constant/fuse_ops 剥掉）。"
        )
        return 3
    if rknn.export_rknn(str(out_path)) != 0:
        log.error("export_rknn 失败")
        return 3
    log.info("已导出 %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    # ---------- 5. 暴力确认输入顺序 ----------
    log.info("开始确认输入顺序（枚举 %d! = %d 种图像排列）……",
             len(image_idx), len(list(itertools.permutations(image_idx))))

    best: tuple[float, tuple[int, ...]] | None = None
    results: list[dict] = []
    for perm in itertools.permutations(image_idx):
        order = [state_idx[0], *perm]  # state 必须放最前（它决定了 2 维/4 维的分界）
        ordered = [ref_arrs[i] for i in order]
        try:
            got = _rknn_run(rknn, ordered)
            if got.shape != ref.shape:
                diff = float("inf")
                note = f"shape 不匹配 {got.shape} vs {ref.shape}"
            else:
                diff = float(np.abs(got - ref).max())
                note = ""
        except Exception as e:  # noqa: BLE001
            diff = float("inf")
            note = f"异常: {e}"

        names = [slot_of[i] for i in order]
        results.append({"order_slots": names, "max_abs_diff": diff, "note": note})
        log.info("  %-45s max|Δ|=%-12s %s", names, f"{diff:.3e}" if np.isfinite(diff) else "inf", note)

        if best is None or diff < best[0]:
            best = (diff, order)

    assert best is not None
    best_diff, best_order = best
    sorted_diffs = sorted(r["max_abs_diff"] for r in results)
    log.info("最佳排列: %s  max|Δ|=%.3e", [slot_of[i] for i in best_order], best_diff)

    if not np.isfinite(best_diff) or best_diff > args.tol:
        log.error("=" * 70)
        log.error("输入顺序确认失败：最佳排列误差 %.3e 仍超过容差 %.3e", best_diff, args.tol)
        log.error("可能原因：")
        log.error("  1) 有算子回落到 CPU（查转换日志 `unsupport cpu <Op> op`）")
        log.error("  2) ONNX 导出本身与 RKNN 数值不一致（先在 PC 端验 RT vs ONNX）")
        log.error("  3) RKNN / librknnrt 版本不匹配（`Invalid RKNN model version6`）")
        log.error("**不生成 manifest**。板端会因此拒绝运行 —— 这是刻意的。")
        log.error("=" * 70)
        rknn.release()
        return 4

    # 次优与最优差距太小说明两路图像不可区分，无法确认顺序
    if len(sorted_diffs) > 1 and sorted_diffs[1] - best_diff < 1e-6:
        log.error("最优与次优排列误差几乎相同 —— 参考输入不足以区分图像槽位，无法确认顺序")
        rknn.release()
        return 4

    # ---------- 6. 写 manifest ----------
    try:
        import rknn
        toolkit_ver = getattr(rknn, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        toolkit_ver = "unknown"

    img_shapes = {slot_of[i]: list(ref_arrs[i].shape) for i in image_idx}
    H = ref_arrs[image_idx[0]].shape[2]
    W = ref_arrs[image_idx[0]].shape[3]

    manifest = {
        "format": "rkrobot.rknn.manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "onnx": Path(onnx_path).name,
        "rknn": out_path.name,
        "target_platform": "rk3588",
        "float_dtype": "float16",
        "do_quantization": False,
        # ⚠️ 板端按这个顺序传参
        "input_order": [slot_of[i] for i in best_order],
        "image_slots": [slot_of[i] for i in image_idx],
        "image_shapes": img_shapes,
        "image_size": [int(H), int(W)],
        "onnx_input_order": [inputs[i][0] for i in range(len(inputs))],
        "output_shape": list(ref.shape),
        "verified_max_abs_diff": best_diff,
        "tolerance": args.tol,
        "all_permutations": results,
        "versions": {
            "rknn_toolkit2": toolkit_ver,
            "python": platform.python_version(),
        },
        "note": "input_order 由暴力枚举 + ORT 比对确认，禁止手改",
    }
    mpath = out_path.with_suffix(out_path.suffix + ".manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("已写 manifest: %s", mpath)

    rknn.release()

    log.info("=" * 70)
    log.info("转换完成。下一步：")
    log.info("  1) 把 %s 与 %s 拷到板端", out_path.name, mpath.name)
    log.info("  2) 板端 python rk3588/main.py --once      # 单次推理，不碰舵机")
    log.info("  3) 板端 python rk3588/main.py --dry-run   # 闭环但不写舵机")
    log.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
