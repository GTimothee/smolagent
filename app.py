import os
from smolagents import CodeAgent
from dotenv import load_dotenv
from smolagents import HfApiModel
from smolagents import Tool
import gradio as gr
from smolagents import GradioUI

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


if __name__ == "__main__":
    tools_list, add_base_tools = get_tools()
    model = HfApiModel(model_id, provider=None)
    agent = CodeAgent(tools=tools_list, model=model, add_base_tools=add_base_tools, additional_authorized_imports=['web_search'])
    GradioUI(agent).launch()