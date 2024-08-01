请将该文件移动到RyzenAI-SW\example\transformers\models\opt\目录下，并且确保按照教程能够成功运行

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

等待命令提示符运行完毕出现http://127.0.0.1:7860链接时，在浏览器里面输入http://127.0.0.1:7860，进入webui
