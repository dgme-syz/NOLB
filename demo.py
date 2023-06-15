import gradio as gr

def greet(mixed, gpu):
    greeting = f" "
    return greeting, gpu


def change_textbox(choice):
    if choice == "None":
        return gr.update(lines=2, visible=True, value="Short story: ")

with gr.Blocks() as demo:
    radio1 = gr.Radio(
        ["None", "BalancedRS", "EffectNumRS","ClassAware","EffectNumRW","BalancedRW"], label="Train Rule"
    )
    C1 = gr.Checkbox(label='mixed')
    C2 = gr.Checkbox(label='pre_trained')
    gr.Slider(0,4,label='GPU')
    text = gr.Textbox(lines=2, interactive=True)
    radio1.change(fn=change_textbox, inputs=radio1, outputs=text)

demo.launch()