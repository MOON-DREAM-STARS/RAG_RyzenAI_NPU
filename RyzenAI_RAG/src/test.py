import torch
import argparse 
import psutil
from transformers import set_seed
from transformers import AutoTokenizer

import os
import builtins

import qlinear 

from utils import Utils

import gc 
import smooth
import psutil

import gradio as gr
from model import Zhipuai_model
import time

 #WebUI
import gradio as gr
import time
# HF-langchain
from langchain.llms import huggingface_pipeline
from langchain import LLMChain, PromptTemplate
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline
from langchain import LLMChain, PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory

set_seed(123)

# 导入模型
def load_model(args):
    print(f"Loading model ...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/" +  args.model_name)
    if args.load:
        if  (args.quant_mode == "w8a8") and \
            (args.flash_attention == False) and\
            (args.smoothquant == True):
            model = torch.load("./quantized_%s_%s.pth"%(args.model_name, args.dtype))
        else:
            print(f" *** MODE NOT SUPPORTED *** : rerun without --load")
            raise SystemExit
    else:
        if args.amdopt:
            from modeling_opt_amd import OPTForCausalLM, OPTAttention
        else:
            from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTAttention # type: ignore
        class OPTForCausalLMT(OPTForCausalLM):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.tokenizer = None  
        
        if args.dtype == "bfloat16":
            model = OPTForCausalLMT.from_pretrained("facebook/" + args.model_name, torch_dtype=torch.bfloat16)        
        else:
            model = OPTForCausalLMT.from_pretrained("facebook/" + args.model_name) 
                
        model.tokenizer = tokenizer 
        # print(model)
        if (args.smoothquant == True):
            act_scales = torch.load(os.getenv("PYTORCH_AIE_PATH") + "/ext/smoothquant/act_scales/" + "%s.pt"%args.model_name)
            smooth.smooth_lm(model, act_scales, 0.5)
            print(f"SmoothQuant enabled ...")
        if args.dtype == "bfloat16":
            model = model.to(torch.bfloat16)
        
        # print(model)
        if (args.quant_mode == "w8a8"):
            torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True )
        # print(model)
        if (args.flash_attention == True):
            from opt_flash_attention import OPTFlashAttention # type: ignore
            node_args = ()
            if args.quant_mode == "w8a8":
                quant = 1
            else:
                quant = None
            node_kwargs = {
                'quant_mode': quant,
                'device' : args.target,
                'dtype': args.dtype, 
                'impl': args.impl,
                'embed_dim': model.config.hidden_size,
                'num_heads': model.config.num_attention_heads,
                'opt_name': "facebook/" + args.model_name, }
            Utils.replace_node( model,
                                OPTAttention,
                                OPTFlashAttention,
                                node_args, node_kwargs   )
    collected = gc.collect()
    model.eval()
    # print(model)
    print(f"Model loaded ...")
    return model, tokenizer 

# 模型转换
def model_transform(model, args):
    print(f"Transforming model ...")
    if args.quant_mode == "w8a8":
        if (args.target == "aie") :
            node_args = ()
            quant_mode = 1
            node_kwargs = {'device': 'aie', 'quant_mode':args.quant_mode, 'profiler':args.profile, 'dtype':args.dtype, 'impl':args.impl}
            if (args.no_accelerate == False):
                Utils.replace_node(    model, 
                                        torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                                        qlinear.QLinear, 
                                        node_args, node_kwargs )
            else:
                Utils.replace_node(     model, 
                                        torch.ao.nn.quantized.dynamic.modules.linear.Linear, 
                                        qlinear_experimental.QLinearExperimentalPyTiling,  # type: ignore
                                        node_args, node_kwargs )
        elif (args.target == "customcpu"):
            node_args = ()
            node_kwargs = {'quant_mode':1, 'requantize_out_scale':128}
            Utils.replace_node( model, 
                                torch.ao.nn.quantized.dynamic.modules.linear.Linear, 
                                qlinear_experimental.QLinearExperimentalCPU,  # type: ignore
                                node_args, node_kwargs )
        else: #target == "cpu":
            pass 
    else: # quant_mode == None 
        if (args.target == "aie") :
            print(f"*** FP32 MODEL ON AIE NOT SUPPORTED ***")
            raise SystemExit 
        elif (args.target == "customcpu"):
            quant_mode = None
            node_args = ()
            node_kwargs = {'quant_mode':quant_mode}
            Utils.replace_node( model, 
                                torch.nn.Linear, 
                                qlinear_experimental.QLinearExperimentalCPU,  # type: ignore
                                node_args, node_kwargs )
        else: #target == "cpu":
            pass 
        """
        if args.quant_mode == ptsq: - for aie and customcpu
            scales = torch.load(".\calibration_%s\quantized_%s_act_scales.pt"%(args.model_name, args.model_name))
            for name, module in model.named_modules():
                if name in scales.keys():
                    print(name, module.x_scale, scales[name])
                    module.x_scale = scales[name]
                    print(name, module.x_scale, scales[name])
        """
    # print(model)
    print(f"Model transformed ...")
    return model


    def __init__(self,model_name:str='opt-1.3b',temperature:float=0.9) -> None:
        
        #初始化大模型
        self.model_name=model_name
        self.temperature=temperature
        self.model=OpenAI(model=model_name,temperature=temperature)

        #加载向量数据库，embedding模型
        self.db=Vectordatabase()
        self.db.load_vector()
        self.embedding_model=Zhipuembedding()
        
    #定义chat方法
    def chat(self,question:str):

        #这里利用输入的问题与向量数据库里的相似度来匹配最相关的信息，填充到输入的提示词中
        template="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {info}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:"""


        info=self.db.query(question,self.embedding_model,1)

        prompt=PromptTemplate(template=template,input_variables=["question","info"]).format(question=question,info=info)

        res=self.model.invoke(prompt)


        return  res

prompts = ["What is AI?"]

def my_decode_prompts(model, tokenizer, prompt, input_ids=None, max_new_tokens=100):
    if input_ids is None:
        print(f"prompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt")

    if prompt is None:
         generate_ids = model.generate(input_ids)
    else:
        generate_ids = model.generate(inputs.input_ids,max_length=100,repetition_penalty=1.2)

    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]

    print(f"response: {response}")

def RAG_chat(question:str,model):
    #这里利用输入的问题与向量数据库里的相似度来匹配最相关的信息，填充到输入的提示词中
    template="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
    问题: {question}
    可参考的上下文：
    ···
    {info}
    ···
    如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
    有用的回答:"""

    info=model.db.query(question,model.embedding_model,1)

    prompt=PromptTemplate(template=template,input_variables=["question","info"]).format(question=question,info=info)

    res=model.invoke(prompt)

    return  res

def echo(message, history, RAG_model):
    result = RAG_chat(message,RAG_model)
    for i in range(len(result)):
        time.sleep(0.02)
        yield result[: i+1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="Different OPT model sizes", type=str, default="opt-1.3b", choices=["opt-125m", "opt-350m", "opt-1.3b", "opt-2.7b", "opt-6.7b", "opt-13b", "opt-30b"])
    parser.add_argument("--target", help="cpu, custumcpu, aie", type=str, default="aie", choices=["cpu", "customcpu", "aie"])
    parser.add_argument('--dtype', help="All ops other than linear ops in bfloat16 or float32", type=str, default="float32", choices=["bfloat16", "float32"])
    parser.add_argument('--quant_mode', help="Quantization mode - w8a8", type=str, default="w8a8", choices=["w8a8", "none"]) # ptsq not suported
    parser.add_argument('--smoothquant', help="Enable smoothquant", action='store_true')
    parser.add_argument('--amdopt', help="Use OPT from local folder - with profile instrumentation: use without --load", action='store_true')
    parser.add_argument('--profile', help="Log matmul times for prompt and token phases - supported only for AIE target", action='store_true')
    parser.add_argument('--no-accelerate', help="Use tiling/padding in c++ or python with AIE target", action='store_true')
    parser.add_argument('--dataset', help="Dataset - wikitext2-raw-v1, wikitext2-v1", type=str, default="raw", choices=["non-raw", "raw"])
    parser.add_argument('--load', help="Load quantized weights from checkpoint", action='store_true')
    parser.add_argument('--flash_attention', help="Enable flash attention", action='store_true')
    parser.add_argument('--num_torch_threads', help="Number of torch threads", type=int, default=2, choices=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--task', help="decode: Decode set of prompts;" , type=str, default="decode", choices=["decode","RAG"] )
    parser.add_argument('--impl', help="Choose between different implementations for aie target", type=str, default="v0", choices=["v0", "v1"])
    parser.add_argument('--fuse_mlp', help="Enable MLP fusion", action='store_true')
    args = parser.parse_args()
    print(f"{args}")

    dev = os.getenv("DEVICE")
    if dev == "stx":
        p = psutil.Process()
        p.cpu_affinity([0, 1, 2, 3])
    torch.set_num_threads(args.num_torch_threads)
    
    # Set amdopt option for flash attention import
    builtins.amdopt = args.amdopt
    builtins.fuse_mlp = args.fuse_mlp
    builtins.impl = args.impl
    builtins.model_name = args.model_name

    # Step 1 - Load model
    model, tokenizer = load_model(args)

    # Step 2 - Model transformation to use target device
    model = model_transform(model, args) 
    collected = gc.collect()

    # Step 3 - Run
    if (args.task == "decode"):
        my_decode_prompts(model, tokenizer, prompts[0])
    elif (args.task == "RAG"):
        # 加载OPT模型和生成管道
        model_name = "facebook/opt-1.3b"
        generator = pipeline('text-generation', model=model_name, tokenizer=model_name)
        llm = huggingface_pipeline(pipeline=generator)

        # 设置文档加载器和分割器
        loader = TextLoader('path_to_your_documents')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # 创建向量数据库
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(texts, embedding_model)

        # 设置检索QA链
        retriever = vectorstore.as_retriever()
        rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        # 使用检索增强生成
        query = "请解释一下量子计算的基本概念。"
        result = rag_chain.run(query)
        print(result)

        