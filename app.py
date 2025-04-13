import os
from smolagents import CodeAgent
from dotenv import load_dotenv
from smolagents import HfApiModel
from smolagents import Tool
import gradio as gr

load_dotenv()
model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"

def get_tools():
    hf_speech2text_tool = Tool.from_hub(
    "GTimothee/hf_text2speech_tool",
    token=os.getenv('HF_TOKEN'),
    trust_remote_code=True
    )

    hf_text2speech_tool = Tool.from_hub(
    "GTimothee/kokoro_text2speech_tool",
    token=os.getenv('HF_TOKEN'),
    trust_remote_code=True
    )
    add_base_tools = True
    tools_list = [hf_speech2text_tool, hf_text2speech_tool]
    return tools_list, add_base_tools


def agent_response(text_input):
    response = agent.run(text_input)
    return response


if __name__ == "__main__":
    tools_list, add_base_tools = get_tools()
    model = HfApiModel(model_id)
    agent = CodeAgent(tools=tools_list, model=model, add_base_tools=add_base_tools, additional_authorized_imports=['web_search'])

    demo = gr.Interface(
        fn=agent_response,
        inputs=["text"],
        outputs=["text"],
    )

    demo.launch()