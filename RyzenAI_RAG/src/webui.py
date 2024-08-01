import gradio as gr
from model import Zhipuai_model
import time

model=Zhipuai_model()

def echo(message, history):
    result=model.chat(message)
    for i in range(len(result)):
        time.sleep(0.02)
        yield result[: i+1]

        #自定义的流式输出



demo = gr.ChatInterface(fn=echo, 
                        examples=["中华人民共和国消费者权益保护法什么时候,在哪个会议上通过的？", "中华人民共和国消费者权益保护的目录是什么？","RinyRAG的项目结构是怎么样的"], 
                        title="Echo Bot",
                        theme="soft")
demo.launch()