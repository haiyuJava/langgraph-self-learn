# LangGraph 学习示例
# API 地址 https://docs.langchain.com/oss/python/langgraph/quickstart#full-code-example

文件名已将 `-` 改为 `_`，方便 Python 模块之间正常导入。

- `HelloWorld.py`：独立的 hello world 示例，无需 API Key。
- `state_self_defined.py`：消息状态和模型调用次数。
- `tool_self_defined.py`：加法、乘法、除法工具及名称索引。
- `model_node_self_defined.py`：模型节点。
- `tool_node.py`：执行模型请求的工具，将结果转换为工具消息。
- `end_logic.py`：判断继续调用工具还是结束。
- `build_and_compile_agent.py`：初始化模型、绑定工具、构建图和执行入口。

依赖方向：入口引用各节点；模型节点引用状态；工具节点引用状态和工具；结束判断引用状态。导入模块不会调用模型或联网绘图。

## 运行

以下 PowerShell 命令在仓库根目录执行。首次配置环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r learn/requirements.txt
```

无需密钥的示例和离线验证：

```powershell
.\.venv\Scripts\python.exe -m learn.HelloWorld
.\.venv\Scripts\python.exe -m unittest learn.test_agent -v
```

运行真实 Claude Agent，需要在运行进程的环境中设置自己的 Anthropic API Key：

```powershell
$env:ANTHROPIC_API_KEY = "你的 API Key"
.\.venv\Scripts\python.exe -m learn.build_and_compile_agent
.\.venv\Scripts\python.exe -m learn.build_and_compile_agent "Multiply 6 and 7."
```

默认问题为 `Add 3 and 4.`。也支持直接运行 `learn/build_and_compile_agent.py`，以及在 `learn` 目录执行 `python build_and_compile_agent.py`。

PyCharm 请将项目解释器设置为仓库下的 `.venv\Scripts\python.exe`，在运行配置的环境变量中设置 `ANTHROPIC_API_KEY`。

可选添加 `--draw learn/agent.png`，在执行后通过在线 Mermaid 服务生成图。默认执行不绘图，也不需要 IPython。

离线测试使用预设模型回复，验证真实图的路由、工具执行、消息累积和调用计数；不验证 Claude 服务的连通性或回答质量。
