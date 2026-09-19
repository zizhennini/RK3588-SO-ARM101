"""生成一个「假 ACT 策略」目录，用于在采集真实数据前验证导出链路。

为什么需要
----------
真实的 `pretrained_model` 目录要等数据采集 + 训练之后才有。
但 `pc/convert/export_act_onnx.py` 依赖的只是 **config.json + 权重**，
用**真实的 LeRobot ACT 架构**随机初始化一份即可把导出链路验证到底：

  * 输入签名（observation.state / observation.images.*）与真实训练完全一致
  * 走的是真正的 ACTPolicy / ACT 模块，不是手搓的假图
  * 导出、onnxsim、与 PyTorch 数值比对、RKNN 转换、输入顺序确认 全部能测

默认用很小的图像尺寸以加快测试（导出路径与分辨率无关）。
要复现真实尺寸就传 `--height 480 --width 640`。

用法（在 **lerobot 环境**里跑）
    python pc/convert/make_dummy_act_policy.py --out /tmp/dummy_act
    python pc/convert/export_act_onnx.py --policy_path /tmp/dummy_act --output /tmp/dummy_act.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build(out_dir: Path, height: int, width: int, state_dim: int,
          action_dim: int, chunk: int, camera: str, seed: int) -> int:
    import torch

    try:
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
    except ImportError as e:  # noqa: BLE001
        print(f"❌ 需要 lerobot 环境（conda activate lerobot）: {e}")
        return 2

    torch.manual_seed(seed)

    image_key = f"observation.images.{camera}"
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, height, width)),
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }

    print(f"构造 ACTConfig: state={state_dim} action={action_dim} chunk={chunk}")
    print(f"  图像 {image_key} -> (3, {height}, {width})")

    cfg = ACTConfig(
        n_obs_steps=1,
        chunk_size=chunk,
        n_action_steps=chunk,
        input_features=input_features,
        output_features=output_features,
    )

    policy = ACTPolicy(cfg)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  参数量: {n_params/1e6:.1f} M  （真实 ACT 约 52M）")

    out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(out_dir))

    # save_pretrained 应该已经写了 config.json 与权重，这里确认并打印签名
    cfg_path = out_dir / "config.json"
    if not cfg_path.is_file():
        print(f"❌ save_pretrained 没有生成 {cfg_path}")
        return 3

    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    feats = saved.get("input_features", {})
    print(f"  config.json 里的 input_features: {sorted(feats)}")
    if image_key not in feats:
        print(f"❌ config.json 里没有 {image_key}")
        return 3

    weights = [p for p in out_dir.iterdir() if p.suffix in (".safetensors", ".bin")]
    print(f"  权重文件: {[p.name for p in weights]}")
    if not weights:
        print("❌ 没有权重文件")
        return 3

    total_mb = sum(p.stat().st_size for p in out_dir.iterdir() if p.is_file()) / 1e6
    print(f"✅ 已生成 {out_dir}  ({total_mb:.1f} MB)")
    print("   注意：权重是随机初始化的，输出无意义，只用于验证链路。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="生成随机初始化的 ACT 策略（链路测试用）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--height", type=int, default=120, help="图像高（默认用小的加快测试）")
    ap.add_argument("--width", type=int, default=160, help="图像宽（默认用小的加快测试）")
    ap.add_argument("--state-dim", type=int, default=6)
    ap.add_argument("--action-dim", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--camera", default="front")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    return build(Path(args.out).resolve(), args.height, args.width,
                 args.state_dim, args.action_dim, args.chunk, args.camera, args.seed)


if __name__ == "__main__":
    sys.exit(main())
