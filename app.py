import os

import gradio as gr
import argparse

from train import main

folder_path = '.\\models\\pretrained'
model_file = os.listdir(folder_path)

def analyze_arguments(*input_values):
    args = argparse.Namespace()
    input_values = list(input_values)
    argument_values = {
        'dataset': input_values[0],
        'data_path': input_values[2],
        'img_path': '/home/datasets/Places365',
        'imb_type': input_values[3],
        'train_rule': input_values[4],
        'mixup': False,
        'print_freq': 10,
        'resume': None,
        'pretrained': input_values[5],
        'pretrained_path': os.path.join(folder_path,input_values[7]),
        'ensemble': input_values[6],
        't1': input_values[8],
        't2': input_values[9],
        'seed': int(input_values[10]),
        'gpu': 0,
        'root_log': input_values[11],
        'start_epoch': int(input_values[12]),
        'momentum': input_values[16],
        'arch': input_values[17],
        'epochs': int(input_values[13]),
        'workers': int(input_values[14]),
        'weight_decay': input_values[18],
        'rand_number': int(input_values[22]),
        'exp_str': input_values[23],
        'loss_type': input_values[19],
        'imb_factor': input_values[1],
        'batch_size': int(input_values[15]),
        'lr': input_values[20],
        'lambda_': input_values[21],
    }

    # 更新args对象的属性值
    for arg_name, arg_value in argument_values.items():
        setattr(args, arg_name, arg_value)

    main(args)
    # 返回更新后的args对象
    return "./results/Prec@1.png","./results/GM.png","./results/HM.png","./results/LR.png"

app = gr.Blocks(title='WebUI',
                theme=gr.themes.Soft(primary_hue="orange",
                                    secondary_hue="blue",
                                    neutral_hue = "green",))
with app:
    gr.Markdown(value="""# Machine Learning 课设
                        **Author**:[DGMEFG](https://github.com/DGMEFG) [Garden-Unicorn](https://github.com/Garden-Unicorn) [dukexh](https://github.com/dukexh) """)
    with gr.Tabs():
        res = []
        with gr.TabItem("训练"):
            gr.Markdown('### 数据集')
            with gr.Row():
                Dataset = gr.Dropdown(choices=['cifar10', 'cifar100'], label='Dataset', value='cifar10')
                IF = gr.Number(label='Imblance Factor', value=0.01)
                Data_Path = gr.Textbox(label='Data Path', value='./dataset/data_img')
                res.extend([Dataset,IF,Data_Path])
            gr.Markdown("### IF值 & reweight_resample策略")
            with gr.Row():
                IT = gr.Dropdown(choices=['exp', 'step'], label='Imbalance Type', value='exp')
                TR = gr.Dropdown(choices=['BalancedRS','None','EffectNumRS','ClassAware','EffectNumRW','BalancedRW'],label='Train Rule', value='None')
                res.extend([IT,TR])
            with gr.Row():
                pretrained = gr.Checkbox(label='Pretrained', value=False)
                ensemble = gr.Checkbox(label='Ensemble',value=False)
                res.extend([pretrained,ensemble])
            gr.Markdown("### 预训练 & 集成模型参数设置")
            with gr.Row():
                pretrained_path = gr.Dropdown(choices=model_file,label='Select Pretrained Model')
                t1 = gr.Number(label='Temperature for New Model', value=1)
                t2 = gr.Number(label='Temperature for Old Model', value=1)
                res.extend([pretrained_path,t1,t2])
            seed = gr.Number(label='Seed', value=123)
            root_log = gr.Textbox(label='Root Log', value='./log')
            res.extend([seed,root_log])
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 训练参数")
                    start_epoch = gr.Number(label='Start Epoch', value=0)
                    end_epoch = gr.Number(label='Epochs', value=100)
                    workers = gr.Number(label='Workers', value=0)
                    batch_size = gr.Number(label='Batch Size', value=64)
                    res.extend([start_epoch,end_epoch,workers,batch_size])
                with gr.Column():
                    gr.Markdown("### 模型参数")
                    mome = gr.Number(label='Momentum', value=0.9)
                    arch = gr.Dropdown(choices=['resnet20', 'resnet32','resnet44'], label='Model Architecture', value='resnet32')
                    weight_decay = gr.Number(label='Weight Decay', value=2e-4)
                    loss = gr.Dropdown(choices=['BSCE', 'LDAM', 'CE', 'Focal', 'FeaBal', 'GML', 'Lade'], label='Loss Type', value='CE')
                    lr = gr.Number(label='Learning Rate', value=0.1)
                    Lam_ = gr.Number(label='Lambda', value=60)
                    res.extend([mome,arch,weight_decay,loss,lr,Lam_])
            Random_Num = gr.Number(label='Random Number', value=0)
            Expstr = gr.Textbox(label='Experiment String', value='bs512_lr002_110')
            res.extend([Random_Num,Expstr])
            submit = gr.Button("Train and Load", variant="primary")
            gr.Markdown("### precision&1 and GM")
            with gr.Row():
                p1 = gr.Image(type='filepath',label="Pretop1")
                GM = gr.Image(type='filepath', label="GM")
            gr.Markdown("### HM & LR")
            with gr.Row():
                HM = gr.Image(type='filepath', label="HM")
                LR = gr.Image(type='filepath', label="LR")
        submit.click(fn=analyze_arguments, inputs=res, outputs=[p1, GM, HM, LR])
app.launch()



