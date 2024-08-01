## **How to install this demo?(如何安装这个demo)**

请将该文件移动到RyzenAI-SW\example\transformers\models\opt\目录下，并且确保按照教程能够成功运行（1.1分支，不是main分支）

然后运行

```
pip install -r requirements.txt
```

之后运行

```
cd RyzenAI_RAG/src
```

输入

```
python test.py --task RAG --target aie --model_name opt-1.3b
```

等待命令提示符运行完毕出现 http://127.0.0.1:7860 链接时，在浏览器里面输入 http://127.0.0.1:7860 ，进入webui

## 原理

先将opt-1.3b模型按照demo进行量化

然后利用量化之后的模型进行推理

利用智普清言，OPENAI的在线API或者本地embedding模型对本地知识图谱进行向量化

利用langchain将知识库中的内容嵌入对话中

RAG功能大体上可以实现了
