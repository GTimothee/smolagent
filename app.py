import io
import os

import soundfile as sf
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from smolagents import CodeAgent, GradioUI, HfApiModel

load_dotenv()


def convert_data_to_audio_filelike(your_input_tuple):
    """Convert (sample_rate, np.ndarray) to a BytesIO WAV file"""
    sample_rate, audio_data = your_input_tuple
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer


def speech2text_func(data, model: str = "openai/whisper-small.en") -> str:
    if isinstance(data, tuple):
        buffer = convert_data_to_audio_filelike(data)
        data = buffer.read()
    client = InferenceClient(
        provider="hf-inference",
        api_key=os.getenv("HF_TOKEN"),
    )
    return client.automatic_speech_recognition(data, model=model).text


def get_tools():
    add_base_tools = True
    tools_list = []
    return tools_list, add_base_tools


if __name__ == "__main__":
    tools_list, add_base_tools = get_tools()
    model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct", provider=None)
    agent = CodeAgent(
        tools=tools_list,
        model=model,
        add_base_tools=add_base_tools,
        additional_authorized_imports=["web_search"],
    )
    GradioUI(agent).launch(speech2text_func=speech2text_func)
