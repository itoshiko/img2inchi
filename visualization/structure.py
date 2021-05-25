import torch
from torchviz import make_dot, make_dot_from_trace


def generate_structure(model, config):
    x = torch.rand(config.batch_size, 3, 512, 256).cuda()
    y = torch.rand(config.batch_size, 100).cuda()
    vis_graph = make_dot(model(x, y), params=dict(model.named_parameters()))
    vis_graph.view()  # 会在当前目录下保存一个“Digraph.gv.pdf”文件，并在默认浏览器中打开

    with torch.onnx.select_model_mode_for_export(model, False):
        trace, _ = torch.jit._get_trace_graph(model, args=(x, y))
    make_dot_from_trace(trace)
