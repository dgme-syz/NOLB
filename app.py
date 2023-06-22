import os

import gradio as gr
import argparse

import torch
import torchvision

from models.resnet import *

from train import main

if os.path.exists('.\\models\\pretrained') == False:
    os.makedirs(name='.\\models\\pretrained', exist_ok=True)

folder_path = '.\\models\\pretrained'
model_file = os.listdir(folder_path)

model_path = '.\\Trained_Model'
existed_models = os.listdir(model_path)

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

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
        'pretrained_freezing': input_values[5],
        'pretrained_keep': input_values[6],
        'pretrained_path': os.path.join(folder_path,input_values[8]),
        'ensemble': input_values[7],
        't1': input_values[9],
        't2': input_values[10],
        'seed': int(input_values[11]),
        'gpu': 0,
        'root_log': input_values[12],
        'start_epoch': int(input_values[13]),
        'momentum': input_values[17],
        'arch': input_values[18],
        'epochs': int(input_values[14]),
        'workers': int(input_values[15]),
        'weight_decay': input_values[19],
        'rand_number': int(input_values[23]),
        'exp_str': input_values[24],
        'loss_type': input_values[20],
        'imb_factor': input_values[1],
        'batch_size': int(input_values[16]),
        'lr': input_values[21],
        'lambda_': input_values[22],
    }

    # 更新args对象的属性值
    for arg_name, arg_value in argument_values.items():
        setattr(args, arg_name, arg_value)

    model = main(args)


    if input_values[-1] == True:
        if os.path.exists('./Trained_Model') == False:
            os.makedirs(name='./Trained_Model',exist_ok=True)
        torch.save(model.state_dict(),'./Trained_Model/model.pt')
    # 返回更新后的args对象
    return "./results/Prec@1.png","./results/GM.png","./results/HM.png","./results/LR.png"

def predict(img,model_para,arch):
    img = torchvision.transforms.ToTensor()(img).unsqueeze(0)
    if arch == 'resnet20':
        model = resnet20()
    elif arch == 'resnet32':
        model = resnet32()
    else:
        model = resnet44()
    choose_model_path = os.path.join(model_path,model_para)
    device = 'cpu' if torch.cuda.is_available() == False else 'cuda'
    model.load_state_dict(torch.load(choose_model_path,map_location=device))
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(img)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(len(labels))}
    return confidences


app = gr.Blocks(title='WebUI',
                theme=gr.themes.Soft(primary_hue="orange",
                                    secondary_hue="blue",))
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
                pretrained_freezing = gr.Checkbox(label='Pretrained(Freezing feature extraction)', value=False)
                pretrained_keep = gr.Checkbox(label='Pretrained(Freezing all parameters)', value=False)
                ensemble = gr.Checkbox(label='Ensemble',value=False)
                res.extend([pretrained_freezing,pretrained_keep,ensemble])
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
            save = gr.Checkbox(label="save model(You can find it in 'Trained_model' folder)",value=False)
            res.extend([Random_Num, Expstr, save])
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
        with gr.TabItem("推理"):
            gr.Markdown("注：由于本课设基于cifar10，故推理图片将会自动处理为32x32")
            input_image = gr.Image(type='pil',image_mode='RGB',shape=(32,32))
            input_arch = gr.Dropdown(choices=['resnet20', 'resnet32','resnet44'], label='Model Architecture', value='resnet32')
            input_model = gr.Dropdown(choices=existed_models,label='Load Model')
            output_class = gr.Label(num_top_classes=10)
            interface = gr.Button("Inference", variant="primary")
        interface.click(fn=predict,inputs=[input_image,input_model,input_arch],outputs=output_class)
parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--share', default=False, type=bool, help='share your webui')
args = parser.parse_args()
app.launch(share=args.share)



