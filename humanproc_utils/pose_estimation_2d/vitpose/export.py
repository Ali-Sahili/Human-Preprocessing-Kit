
import torch
import urllib
from pathlib import Path


#-------------------------------------------------------------------------------
def url2file(url):
    url = str(Path(url)).replace(':/', '://')
    return Path(urllib.parse.unquote(url)).name.split('?')[0]

#-------------------------------------------------------------------------------
def export_onnx(model, im, file):
    import onnx

    f = file.with_suffix('.onnx')

    output_names = ['output0'] 
    torch.onnx.export(model, im, f, verbose=False, opset_version=12, do_constant_folding=True,
                        input_names=['images'], output_names=output_names, dynamic_axes=None)

    # Checks
    model_onnx = onnx.load(f)             # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    onnx.save(model_onnx, f)
    return f, model_onnx

#-------------------------------------------------------------------------------
def export_engine(model, im, file, workspace):
    assert im.device.type != 'cpu', 'export running on CPU but must be on GPU, i.e. `python export.py --device 0`'
    import tensorrt as trt

    export_onnx(model, im, file)
    onnx = file.with_suffix('.onnx')

    print(f'\n starting export with TensorRT {trt.__version__}...')
    assert onnx.exists(), f'failed to export ONNX file: {onnx}'
    f = file.with_suffix('.engine')  # TensorRT engine file
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    # config.max_workspace_size = workspace * 1 << 30
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        print(f'TensorRT: input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        print(f'TensorRT: output "{out.name}" with shape{out.shape} {out.dtype}')

    print(f'TensorRT: building FP{16 if builder.platform_has_fast_fp16  else 32} engine as {f}')
    if builder.platform_has_fast_fp16 :
        config.set_flag(trt.BuilderFlag.FP16)

    serialized_engine = builder.build_serialized_network(network, config)
    with open(f, 'wb') as t:
        t.write(serialized_engine)


#-------------------------------------------------------------------------------
def run(weights='pose_higher_hrnet_w32_512', imgsz=(640, 640), batch_size=1, device='cpu',
        workspace=4 ):  # TensorRT: workspace size (GB)
    
    file = Path(url2file(weights) if str(weights).startswith(('http:/', 'https:/')) else weights)  # PyTorch weights
    from .model_builder import build_model

    model = build_model('ViTPose_base_coco_256x192',weights)
    model.cuda()
    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand

    im = torch.ones(batch_size, 3, 256,192).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Update model
    model.eval()
    for _ in range(2):
        y = model(im)  # dry runs

    export_engine(model, im, file, workspace)


#-------------------------------------------------------------------------------
if __name__ == "__main__":
    run(weights='weights/vitpose-b-multi-coco.pth', device="cuda")