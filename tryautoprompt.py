import gradio as gr

def my_function(text):
    # Process the text here
    return "Processed text:", text 

def inject_and_submit():
    iface.inputs[0].value = "Injected text"  # Access using index (0)
    iface.submit()
    
iface = gr.Interface(
    my_function,
    [gr.Textbox()],  # Remove "name" argument
    live=True,
)

iface.launch()
