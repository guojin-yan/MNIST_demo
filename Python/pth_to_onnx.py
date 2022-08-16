import torch
import torch.nn
import onnx

model = torch.load('best_model.pth')
model.eval()

input_names = ['input']
output_names = ['output']

x = torch.randn(1,1,28,28,requires_grad=True).cuda()

torch.onnx.export(model, x, 'best_model.onnx', input_names=input_names, output_names=output_names, verbose='True')

