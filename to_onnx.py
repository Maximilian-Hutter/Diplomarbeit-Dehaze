import torch.onnx
import torch
from models import Dehaze
from params import hparams

PATH = "./weights/9Dehaze.pth"
Net = Dehaze(hparams["mhac_filter"], hparams["mha_filter"], hparams["num_mhablock"], hparams["num_mhac"], hparams["num_parallel_conv"],hparams["kernel_list"], hparams["pad_list"])

Net.load_state_dict(torch.load(PATH, torch.device("cuda")))

# set the model to inference mode
Net.eval()

x = torch.randn(1, 3, hparams["height"], hparams["width"], requires_grad=True)
x = x.to(torch.device("cuda"))
torch_out = Net(x)


torch.onnx.export(Net,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "Dehaze.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})