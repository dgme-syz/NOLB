import gradio as gr
import argparse

import gradio.themes

from train import main
import matplotlib.pyplot as plt

inputs = [
    gr.inputs.Dropdown(choices=['cifar10', 'cifar100', 'imagenet', 'inat', 'place365'], label='Dataset', default='cifar10'),
    gr.inputs.Textbox(label='Data Path', default='./dataset/data_img'),
    gr.inputs.Textbox(label='Image Path', default='/home/datasets/Places365'),
    gr.inputs.Dropdown(choices=['exp', 'step'], label='Imbalance Type', default='exp'),
    gr.inputs.Dropdown(choices=['BalancedRS','None','EffectNumRS','ClassAware','EffectNumRW','BalancedRW'],label='Train Rule', default='None'),
    gr.inputs.Checkbox(label='Mix-up', default=False),
    gr.inputs.Number(label='Print Frequency', default=10),
    gr.inputs.Textbox(label='Resume Path', default=None),
    gr.inputs.Checkbox(label='Pretrained', default=False),
    gr.inputs.Number(label='Seed', default=123),
    gr.inputs.Number(label='GPU', default=0),
    gr.inputs.Textbox(label='Root Log', default='./log'),
    gr.inputs.Number(label='Start Epoch', default=0),
    gr.inputs.Number(label='Momentum', default=0.9),
    gr.inputs.Dropdown(choices=['resnet20', 'resnet32','resnet44'], label='Model Architecture', default='resnet32'),
    gr.inputs.Number(label='Epochs', default=100),
    gr.inputs.Number(label='Workers', default=0),
    gr.inputs.Number(label='Weight Decay', default=2e-4),
    gr.inputs.Number(label='Random Number', default=0),
    gr.inputs.Textbox(label='Experiment String', default='bs512_lr002_110'),
    gr.inputs.Dropdown(choices=['BSCE', 'LDAM', 'CE', 'Focal', 'FeaBal', 'GML', 'Lade'], label='Loss Type', default='CE'),
    gr.inputs.Number(label='Imbalance Factor', default=0.01),
    gr.inputs.Number(label='Batch Size', default=64),
    gr.inputs.Number(label='Learning Rate', default=0.1),
    gr.inputs.Number(label='Lambda', default=60),
]

def analyze_arguments(*input_values):
    # 将输入值与参数名称对应
    args = argparse.Namespace()
    input_values = list(input_values)
    argument_values = {
        'dataset': input_values[0],
        'data_path': input_values[1],
        'img_path': input_values[2],
        'imb_type': input_values[3],
        'train_rule': input_values[4],
        'mixup': input_values[5],
        'print_freq': input_values[6],
        'resume': input_values[7],
        'pretrained': input_values[8],
        'seed': int(input_values[9]),
        'gpu': int(input_values[10]),
        'root_log': input_values[11],
        'start_epoch': int(input_values[12]),
        'momentum': input_values[13],
        'arch': input_values[14],
        'epochs': int(input_values[15]),
        'workers': int(input_values[16]),
        'weight_decay': input_values[17],
        'rand_number': int(input_values[18]),
        'exp_str': input_values[19],
        'loss_type': input_values[20],
        'imb_factor': input_values[21],
        'batch_size': int(input_values[22]),
        'lr': input_values[23],
        'lambda_': input_values[24],
    }

    # 更新args对象的属性值
    for arg_name, arg_value in argument_values.items():
        setattr(args, arg_name, arg_value)

    main(args)
    # 返回更新后的args对象
    return vars(args),"./Prec@1.png","./GM.png","./HM.png","./LR.png"

if __name__ == '__main__':
    outputs = [
        gr.outputs.Textbox(label="args"),
        gr.outputs.Image(type='filepath',label="Pretop1"),
        gr.outputs.Image(type='filepath', label="GM"),
        gr.outputs.Image(type='filepath', label="HM"),
        gr.outputs.Image(type='filepath', label="LR")
    ]
    interface = gr.Interface(fn=analyze_arguments, inputs=inputs, outputs=outputs, title='WebUI',theme=gr.themes.Soft(primary_hue="orange", secondary_hue="blue",neutral_hue = "green",font="GoogleFont"))
    interface.launch()