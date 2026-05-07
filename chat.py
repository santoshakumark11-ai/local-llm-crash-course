from ctransformers import AutoModelForCausalLM
from typing import List
import chainlit as cl


def get_prompt(instructions: str, history: List[str] = None) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the question in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if history is not None:
        prompt += f"This is the converstation history with user:{''.join(history)}, Now converse with user."
    prompt += f"{instructions}\n\n### Response:\n"
    #print(prompt)
    return prompt


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("history")
    response = ""
    response += "User Question: " + message.content + " and answer you gave - "
    msg = cl.Message(content="")
    await msg.send()

    prompt = get_prompt(message.content, message_history)

    for word in llm(prompt, stream=True):
        await msg.stream_token(word)
        response += word

    await msg.update()
    message_history.append(response + " - end of your answer. ")


@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("history", [])
    global llm
    llm = AutoModelForCausalLM.mro(
        "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF", model_file="Mistral-7B-Instruct-v0.3.Q6_K.gguf", **{'threads': 10, 'batch_size': 512}
    )
    print("A new chat session has started!")


''' 
history = []
answer = ""
query = "What is the name of the capital city of India ?"

for word in llm(get_prompt(query), stream=True):
    print(word, end="", flush=True)
    answer += word
print()
history.append(answer)
query2 = "and which is of Nepal ?"

for word in llm(get_prompt(query2, history), stream=True):
    print(word, end="", flush=True)
    answer += word
print()
print()
history.append(answer)
'''
