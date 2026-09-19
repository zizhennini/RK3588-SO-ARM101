"""把 LeRobot 的 ACT 策略导出为 ONNX（供 RKNN 转换使用）。

来源与归属
----------
本文件的做法借鉴自 openEuler **IB_Robot** 项目的
`src/model_utils/model_utils/export_onnx_rknn.py`（该实现已在 RK3588 上实测
跑通 ACT：float16、114MB、NPU 推理约 **121 ms**，输出 `(1,100,6)`）。

IB_Robot 的原始实现是 Apache-2.0 许可的 OpenEuler 项目的一部分。
这里只保留核心导出逻辑（wrapper + 输出裁剪 + onnxsim），
去掉其自有的 manifest/backend 框架，改为输出本仓库
`rk3588/act_rknn.py` 能直接消费的格式。

为什么不能直接用 LeRobot 官方导出
---------------------------------
LeRobot 官方**不提供** ONNX 导出工具，相关 issue 长期无响应，
Jetson 支持甚至被以 "Not Planned" 关闭。所以必须自己写 —— 但不是从零写，
而是照搬已被验证的写法。

关键实现要点（都来自 IB_Robot 的实测经验）
------------------------------------------
1. 用 `ACTPolicy.from_pretrained()` 载入，包一层 wrapper 直接驱动 `policy.model`
2. wrapper 把各个输入张量组装成 batch dict，并把所有 `observation.images.*`
   聚成 `batch["observation.images"] = [img1, img2, ...]`
3. 输出只保留 `action`：显式裁剪 ONNX 的 graph.output，
   **减少 NPU 访存**（IB_Robot 的 `strip_extra_outputs`）
4. `opset_version=13`（不是我们最初以为的 17 —— 这是 IB_Robot 实测用的值）
5. 再过一遍 `onnxsim.simplify`

用法
----
    # 在 lerobot 环境里跑（需要 torch + lerobot，不是 rknn 环境）
    python pc/convert/export_act_onnx.py \
        --policy_path outputs/act_pen_place/checkpoints/last/pretrained_model \
        --output models/act_pen_place.onnx

    # 再去 rknn 环境做转换 + 输入顺序确认
    python pc/convert/convert_to_rknn.py \
        --onnx models/act_pen_place.onnx --out models/act_pen_place.rknn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IMAGE_PREFIX = "observation.images."
STATE_KEY = "observation.state"


def log(msg: str) -> None:
    print(f"[export_act_onnx] {msg}")


def read_policy_config(policy_path: Path) -> dict:
    cfg_path = policy_path / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"找不到 {cfg_path}。\n"
            "ACT 的 ONNX 导出必须有一份完整的 policy 目录（config.json + 权重），"
            "而不是单个 .safetensors 文件。"
        )
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def collect_input_specs(config: dict) -> tuple[list[str], list[str], list[tuple]]:
    """从 config.input_features 推导输入名、图像键与形状。

    返回 (input_names, image_keys, shapes)。顺序即 config 里 input_features 的顺序，
    **必须保持**，因为后面的输入顺序验证以 ONNX 声明的顺序为基准。
    """
    features = config.get("input_features")
    if not features:
        raise ValueError("config.json 里没有 input_features，无法推导输入签名")

    input_names: list[str] = []
    image_keys: list[str] = []
    shapes: list[tuple] = []

    for key, spec in features.items():
        if key != STATE_KEY and not key.startswith(IMAGE_PREFIX):
            continue
        shape = spec.get("shape")
        if not shape:
            raise ValueError(f"input_features[{key!r}] 缺少 shape")
        input_names.append(key)
        shapes.append(tuple(shape))
        if key.startswith(IMAGE_PREFIX):
            image_keys.append(key)

    if not image_keys:
        raise ValueError(
            f"没有找到任何 {IMAGE_PREFIX}* 图像输入。"
            "训练时至少要有一路相机，否则 ACT 无法工作。"
        )
    if STATE_KEY not in input_names:
        raise ValueError(f"缺少 {STATE_KEY} 输入。ACT 依赖机器人状态。")
    return input_names, image_keys, shapes


def strip_extra_outputs(src: Path, dst: Path, keep: tuple[str, ...] = ("action",)) -> None:
    """只保留指定输出名。

    RKNN 会把每个输出都算出来并搬回主机，多余的输出白白占用 NPU 访存。
    ACT 的策略输出里往往还带一些调试/辅助输出，必须裁掉。

    做法照搬 IB_Robot：直接重建 graph.output 列表。
    """
    import onnx

    model = onnx.load(str(src))
    kept = [o for o in model.graph.output if o.name in keep]
    if not kept:
        raise ValueError(
            f"ONNX 里找不到输出 {keep}，实际输出: "
            f"{[o.name for o in model.graph.output]}"
        )
    if len(kept) == len(model.graph.output):
        log(f"输出已经是 {keep}，无需裁剪")
        src.replace(dst)
        return

    dropped = [o.name for o in model.graph.output if o.name not in keep]
    log(f"裁剪多余输出: {dropped}")
    while len(model.graph.output) > 0:
        model.graph.output.pop()
    for o in kept:
        model.graph.output.append(o)
    onnx.save(model, str(dst))


def simplify_onnx(src: Path, dst: Path) -> None:
    import onnx
    from onnxsim import simplify

    model = onnx.load(str(src))
    model_simp, ok = simplify(model)
    if not ok:
        raise ValueError(
            "onnxsim 校验失败。常见原因：图的 shape 推断不自洽，"
            "或者存在动态 shape。ACT 必须是全静态形状。"
        )
    onnx.save(model_simp, str(dst))


def export_from_policy(policy_path: Path, output: Path, device: str, opset: int) -> Path:
    import torch

    try:
        import lerobot
    except ImportError as e:  # noqa: BLE001
        raise RuntimeError(
            "当前环境没有 lerobot。ONNX 导出必须在 **lerobot 环境**里跑"
            "（conda activate lerobot），而不是 rknn 环境。"
        ) from e

    log(f"lerobot: {lerobot.__file__}")

    from lerobot.policies.act.modeling_act import ACTPolicy

    config = read_policy_config(policy_path)
    input_names, image_keys, shapes = collect_input_specs(config)
    log(f"策略类型: {config.get('type')}  chunk={config.get('chunk_size')}  n_action_steps={config.get('n_action_steps')}")
    for n, s in zip(input_names, shapes, strict=True):
        log(f"  输入 {n:38s} {[1, *s]}")

    dummy = [
        torch.randn(1, *s, dtype=torch.float32, device=device) for s in shapes
    ]

    class ACTONNXWrapper(torch.nn.Module):
        """把「多个张量」形式的 ONNX 输入还原成 ACT 需要的 batch dict。"""

        def __init__(self, model, input_names, image_keys):
            super().__init__()
            self.model = model
            self.input_names = list(input_names)
            self.image_keys = list(image_keys)

        def forward(self, *args):
            batch = {n: t for n, t in zip(self.input_names, args, strict=False)}
            # ACT 的 forward 期望把所有相机聚成一个列表
            batch["observation.images"] = [batch[k] for k in self.image_keys]
            out = self.model(batch)
            if isinstance(out, dict):
                return out["action"]
            if isinstance(out, tuple):
                return out[0]
            return out

    log(f"载入策略 {policy_path}")
    policy = ACTPolicy.from_pretrained(str(policy_path))
    policy.model = policy.model.to(device)
    policy.model.eval()

    wrapped = ACTONNXWrapper(policy.model, input_names, image_keys)
    wrapped.eval()

    output.parent.mkdir(parents=True, exist_ok=True)
    raw = output.with_name(f"{output.stem}_raw.onnx")
    stripped = output.with_name(f"{output.stem}_stripped.onnx")

    log(f"导出 ONNX（opset={opset}，只保留 action 输出）……")
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            tuple(dummy),
            str(raw),
            input_names=input_names,
            opset_version=opset,
            output_names=["action"],
            do_constant_folding=True,
            verbose=False,
        )

    strip_extra_outputs(raw, stripped)
    simplify_onnx(stripped, output)

    for p in (raw, stripped):
        p.unlink(missing_ok=True)

    size_mb = output.stat().st_size / (1024 * 1024)
    log(f"完成: {output}  ({size_mb:.1f} MB)")
    return output


def verify_onnx(onnx_path: Path, policy_path: Path, device: str) -> bool:
    """用 ONNX Runtime 跑一遍，并与 PyTorch 比对。

    这一步是「转换前」的守门人：如果 ONNX 本身就和 PyTorch 不一致，
    后面 RKNN 的问题根本无从排查。
    """
    import numpy as np
    import onnxruntime as ort
    import torch

    from lerobot.policies.act.modeling_act import ACTPolicy

    config = read_policy_config(policy_path)
    input_names, image_keys, shapes = collect_input_specs(config)

    rng = np.random.default_rng(0)
    feeds = {n: rng.normal(0, 0.5, (1, *s)).astype(np.float32) for n, s in zip(input_names, shapes, strict=True)}

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, feeds)[0]

    policy = ACTPolicy.from_pretrained(str(policy_path))
    policy.model = policy.model.to(device).eval()
    batch = {n: torch.from_numpy(v).to(device) for n, v in feeds.items()}
    batch["observation.images"] = [batch[k] for k in image_keys]
    with torch.no_grad():
        ref = policy.model(batch)
    ref = (ref["action"] if isinstance(ref, dict) else (ref[0] if isinstance(ref, tuple) else ref))
    ref = ref.detach().cpu().numpy()

    log(f"ONNX 输出 shape={got.shape}   PyTorch 输出 shape={ref.shape}")
    if got.shape != ref.shape:
        log("❌ 形状不一致")
        return False
    diff = float(np.abs(got - ref).max())
    log(f"max|Δ| = {diff:.3e}")
    ok = diff < 1e-3
    log("✅ ONNX 与 PyTorch 一致" if ok else "❌ ONNX 与 PyTorch 不一致，先查导出而不是查 RKNN")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="LeRobot ACT -> ONNX（供 RKNN 转换）")
    ap.add_argument("--policy_path", required=True, help="pretrained_model 目录（含 config.json）")
    ap.add_argument("--output", required=True, help="输出的 ONNX 路径")
    ap.add_argument("--device", default="cpu", help="导出设备，cpu 或 cuda")
    ap.add_argument("--opset", type=int, default=13, help="ONNX opset（IB_Robot 实测用 13）")
    ap.add_argument("--skip-verify", action="store_true", help="跳过与 PyTorch 的数值比对")
    args = ap.parse_args()

    policy_path = Path(args.policy_path).resolve()
    output = Path(args.output).resolve()

    try:
        export_from_policy(policy_path, output, args.device, args.opset)
    except Exception as e:  # noqa: BLE001
        log(f"❌ 导出失败: {type(e).__name__}: {e}")
        return 2

    if not args.skip_verify:
        try:
            if not verify_onnx(output, policy_path, args.device):
                return 3
        except Exception as e:  # noqa: BLE001
            log(f"⚠️ 数值验证无法完成: {type(e).__name__}: {e}")
            log("   导出文件仍然生成，但请自行确认正确性后再转换。")

    log("")
    log("下一步（在 rknn 环境里）:")
    log(f"  python pc/convert/convert_to_rknn.py --onnx {output} --out <同名>.rknn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
