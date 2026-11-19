from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np


ROOT = Path(__file__).resolve().parent


def save_model(name: str, graph: onnx.GraphProto) -> None:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 11
    onnx.save(model, ROOT / name)


def identity_f32() -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 4])
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "identity_f32", [x], [y])
    save_model("identity_f32.onnx", graph)


def linear_4in_1out() -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    w = numpy_helper.from_array(np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name="B")
    matmul = helper.make_node("MatMul", ["input", "W"], ["matmul"])
    add = helper.make_node("Add", ["matmul", "B"], ["output"])
    graph = helper.make_graph([matmul, add], "linear_4in_1out", [x], [y], [w, b])
    save_model("linear_4in_1out.onnx", graph)


def two_input_two_output() -> None:
    lhs = helper.make_tensor_value_info("lhs", TensorProto.FLOAT, [2])
    rhs = helper.make_tensor_value_info("rhs", TensorProto.FLOAT, [2])
    sum_out = helper.make_tensor_value_info("sum", TensorProto.FLOAT, [2])
    diff_out = helper.make_tensor_value_info("diff", TensorProto.FLOAT, [2])
    add = helper.make_node("Add", ["lhs", "rhs"], ["sum"])
    sub = helper.make_node("Sub", ["lhs", "rhs"], ["diff"])
    graph = helper.make_graph([add, sub], "two_input_two_output", [lhs, rhs], [sum_out, diff_out])
    save_model("two_input_two_output.onnx", graph)


def dynamic_batch_linear() -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 1])
    w = numpy_helper.from_array(np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name="B")
    matmul = helper.make_node("MatMul", ["input", "W"], ["matmul"])
    add = helper.make_node("Add", ["matmul", "B"], ["output"])
    graph = helper.make_graph([matmul, add], "dynamic_batch_linear", [x], [y], [w, b])
    save_model("dynamic_batch_linear.onnx", graph)


def static_batch_8() -> None:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [8, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [8, 1])
    w = numpy_helper.from_array(np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name="B")
    matmul = helper.make_node("MatMul", ["input", "W"], ["matmul"])
    add = helper.make_node("Add", ["matmul", "B"], ["output"])
    graph = helper.make_graph([matmul, add], "static_batch_8", [x], [y], [w, b])
    save_model("static_batch_8.onnx", graph)


def ctc_fixed_batch(batch: int) -> None:
    """A CTC text recognizer whose graph fixes the batch at `batch`.

    Input `x [batch, 3, 48, width]`; output `[batch, 4, 4]` over four classes
    (blank, `a`, `b`, and a trailing space class past a two-entry dictionary).
    Every row decodes to "a b" whatever the pixels, which lets a test check
    that lines reach a fixed-batch graph `batch` at a time and come back in
    order. The input still feeds the output (times zero) so ORT checks its shape.
    """
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, 3, 48, "width"])
    y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [batch, 4, 4])
    # Per time step, the argmax class: a, space, b, blank -> "a b".
    steps = [1, 3, 2, 0]
    one_hot = np.zeros((batch, 4, 4), dtype=np.float32)
    for t, c in enumerate(steps):
        one_hot[:, t, c] = 1.0
    table = numpy_helper.from_array(one_hot, name="table")
    shape = numpy_helper.from_array(np.array([1, 1, 1], dtype=np.int64), name="shape")
    zero = numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="zero")
    nodes = [
        helper.make_node("ReduceMean", ["x"], ["mean"], keepdims=1),
        helper.make_node("Reshape", ["mean", "shape"], ["mean3"]),
        helper.make_node("Mul", ["mean3", "zero"], ["nothing"]),
        helper.make_node("Add", ["table", "nothing"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes, f"ctc_fixed_batch_{batch}", [x], [y], [table, shape, zero]
    )
    save_model(f"ctc_fixed_batch_{batch}.onnx", graph)


def malformed() -> None:
    (ROOT / "malformed.onnx").write_text("not a valid onnx model\n", encoding="utf-8")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    identity_f32()
    linear_4in_1out()
    two_input_two_output()
    dynamic_batch_linear()
    static_batch_8()
    ctc_fixed_batch(1)
    ctc_fixed_batch(2)
    malformed()


if __name__ == "__main__":
    main()
