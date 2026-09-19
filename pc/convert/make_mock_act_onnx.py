"""生成一个"假 ACT"ONNX，用于在采集真实数据之前把整条链路测通。

用途
----
真实 ACT 的 ONNX 要等数据采集 + 微调之后才有。在那之前，本脚本生成一个
**输入输出签名与 ACT 一致**的假模型，用来验证：

  * `convert_to_rknn.py` 的转换、输入顺序确认、manifest 生成
  * 板端 `act_rknn.py` 的加载、输入顺序断言、输出校验、反归一化
  * `main.py --once / --dry-run` 的整条闭环（用假模型即可跑通调度逻辑）

签名刻意与真实 ACT 对齐（单相机）：
    输入  observation.state          [1, 6]        float32   （2 维）
    输入  observation.images.front   [1, 3, 480, 640] float32 （4 维）
    输出  action                     [1, 100, 6]   float32

forward 里**故意保留 LayerNormalization** —— 它会触发 rknn-toolkit2 2.3.2 的
`convert_layernorm_to_exnorm` 缺陷，正好用来验证转换脚本的回退是否生效。

用法
----
    python pc/convert/make_mock_act_onnx.py --out /tmp/mock_act.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build(out_path: Path, chunk: int = 100, action_dim: int = 6,
          state_dim: int = 6, h: int = 480, w: int = 640) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)

    def init(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name)

    nodes = [
        # 图像 -> 全局平均池化 -> 特征（避免大卷积，模拟器才跑得动）
        helper.make_node("ReduceMean", ["img"], ["img_pool"], axes=[2, 3], keepdims=1),
        helper.make_node("Reshape", ["img_pool", "shape_1x3"], ["img_flat"]),
        helper.make_node("MatMul", ["img_flat", "W_img"], ["img_feat"]),
        helper.make_node("Add", ["img_feat", "b_img"], ["img_feat_b"]),
        # 拼接 state
        helper.make_node("Concat", ["img_feat_b", "state"], ["fused"], axis=1),
        # ⚠️ 故意保留 LayerNormalization：触发 rknn-toolkit2 的融合规则缺陷
        helper.make_node("LayerNormalization", ["fused", "ln_scale", "ln_bias"], ["norm"], axis=-1),
        helper.make_node("Relu", ["norm"], ["act1"]),
        # 映射到动作块
        helper.make_node("MatMul", ["act1", "W_out"], ["out_flat"]),
        helper.make_node("Add", ["out_flat", "b_out"], ["out_flat_b"]),
        helper.make_node("Reshape", ["out_flat_b", "shape_out"], ["action"]),
    ]

    inits = [
        init("W_img", rng.normal(0, 0.05, (3, 32))),
        init("b_img", rng.normal(0, 0.05, (32,))),
        init("ln_scale", np.ones((32 + state_dim,))),
        init("ln_bias", np.zeros((32 + state_dim,))),
        init("W_out", rng.normal(0, 0.05, (32 + state_dim, chunk * action_dim))),
        init("b_out", rng.normal(0, 0.05, (chunk * action_dim,))),
        numpy_helper.from_array(np.array([1, 3], dtype=np.int64), "shape_1x3"),
        numpy_helper.from_array(np.array([1, chunk, action_dim], dtype=np.int64), "shape_out"),
    ]

    graph = helper.make_graph(
        nodes,
        "mock_act",
        [
            helper.make_tensor_value_info("state", TensorProto.FLOAT, [1, state_dim]),
            helper.make_tensor_value_info("img", TensorProto.FLOAT, [1, 3, h, w]),
        ],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, chunk, action_dim])],
        initializer=inits,
    )
    # opset 17：与已验证的 SmolVLA/ACT 转换流水线一致
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    print(f"已生成 {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")
    print(f"  输入: state[1,{state_dim}]  img[1,3,{h},{w}]")
    print(f"  输出: action[1,{chunk},{action_dim}]")
    print("  含 LayerNormalization —— 用于验证转换脚本的融合规则回退")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/mock_act.onnx")
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--action-dim", type=int, default=6)
    ap.add_argument("--state-dim", type=int, default=6)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    try:
        build(Path(args.out), args.chunk, args.action_dim, args.state_dim,
              args.height, args.width)
    except ImportError as e:
        print(f"缺少依赖（需要 onnx / numpy）: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
