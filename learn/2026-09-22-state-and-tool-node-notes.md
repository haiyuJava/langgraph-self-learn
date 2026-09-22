# LangGraph 学习笔记：状态定义与工具调用

日期：2026-09-22

本文整理今天的学习内容，结合项目中的代码说明：状态保存什么、工具怎样执行，以及如何理解 `for tool_call in state["messages"][-1].tool_calls`。

## 1. 先理解整个执行过程

以用户提问“3 加 4 等于多少？”为例，假设模型选择调用加法工具：

```text
用户输入问题
    ↓
模型节点：提出 add(a=3, b=4) 的调用请求
    ↓
工具节点：执行 add，得到结果 7，返回工具结果消息
    ↓
模型节点：读取工具结果，组织最终回复
    ↓
没有新的工具调用，流程结束
```

两个节点的职责不同：

- 模型节点决定调用哪个工具、传入什么参数。
- 工具节点真正执行工具，并记录执行结果。

下面的模型消息是教学示例，真实模型返回的文字和调用编号可能不同。

## 2. `state_self_defined.py`：定义状态

源码：[state_self_defined.py](./state_self_defined.py)

```python
import operator
from typing import Annotated

from langchain_core.messages import AnyMessage
from typing_extensions import NotRequired, TypedDict


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: NotRequired[int]
```

可以把 state 理解为各个节点共用的一本记录本。

| 字段 | 保存什么 | 收到节点更新时如何处理 |
| --- | --- | --- |
| `messages` | 用户、模型、工具等消息 | 把新消息列表拼接到原列表后面 |
| `llm_calls` | 模型调用次数 | 使用节点返回的新数值 |

这个文件只声明状态结构，本身不调用模型或工具。

### 2.1 `TypedDict`：描述字典结构

```python
class MessagesState(TypedDict):
```

这里虽然使用 `class`，但实际状态数据仍然是字典，例如：

```python
from langchain_core.messages import HumanMessage

state = {
    "messages": [HumanMessage(content="3 加 4 等于多少？")],
    "llm_calls": 0,
}
```

使用字典的方式读取：

```python
state["messages"]
state["llm_calls"]
```

`TypedDict` 帮助编辑器和类型检查工具理解字段，不会自动创建初始数据，也不会自动对普通字典进行运行时类型校验。

### 2.2 `list[AnyMessage]`：消息列表

```python
messages: list[AnyMessage]
```

表示 `messages` 是列表，里面保存支持的消息对象，例如：

- `HumanMessage`：用户消息。
- `AIMessage`：模型消息，可以包含工具调用请求。
- `ToolMessage`：工具执行结果。

`AnyMessage` 表示支持的各种消息类型，不是任意 Python 对象。

### 2.3 `Annotated` 和 `operator.add`：指定合并方式

```python
messages: Annotated[list[AnyMessage], operator.add]
```

`Annotated` 的结构是：

```python
Annotated[数据类型, 附加信息]
```

这里 `list[AnyMessage]` 描述类型，`operator.add` 为 LangGraph 指定合并函数，也叫 reducer。

`operator.add(a, b)` 相当于 `a + b`。对列表来说，就是拼接：

```python
[旧消息1, 旧消息2] + [新消息3]
# 得到 [旧消息1, 旧消息2, 新消息3]
```

所以节点只需返回本次新增的消息：

```python
return {"messages": [新消息]}
```

LangGraph 会负责合并。不要在这个规则下每次都返回全部历史消息，否则旧消息也会再次拼接进去。

这个行为由 LangGraph 读取注解后实现，普通 Python 字典不会自动合并。

### 2.4 `NotRequired[int]`：字段可以缺省

```python
llm_calls: NotRequired[int]
```

表示值应当是整数，但初始字典可以没有这个键，例如：

```python
{"messages": [HumanMessage(content="你好")]}
```

`NotRequired` 不会自动设置默认值 `0`。默认值来自模型节点中的代码：

```python
"llm_calls": state.get("llm_calls", 0) + 1
```

意思是：读取已有次数，如果不存在就取 `0`，然后加一。

`llm_calls` 没有配置累加合并规则。节点返回 `2` 时，这个字段就更新为 `2`，不会再与旧值相加。节点不返回这个字段时，已有值保留。

### 2.5 状态定义如何交给 LangGraph

在 [build_and_compile_agent.py](./build_and_compile_agent.py) 中：

```python
agent_builder = StateGraph(MessagesState)
```

这行让 LangGraph 知道状态有哪些字段，以及如何处理节点返回的更新。

## 3. `tool_node.py`：执行工具调用

源码：[tool_node.py](./tool_node.py)

核心代码：

```python
def tool_node(state: MessagesState):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )
    return {"messages": result}
```

### 3.1 模型先返回调用请求

模型通过 `model.bind_tools(tools)` 得知有哪些工具可用。假设它选择调用加法工具，返回的模型消息如下：

```python
AIMessage(
    content="",
    tool_calls=[
        {
            "name": "add",
            "args": {"a": 3, "b": 4},
            "id": "call_1",
        }
    ],
)
```

| 字段 | 含义 |
| --- | --- |
| `name` | 工具名称，这里是 `add` |
| `args` | 工具参数，这里是 `a=3`、`b=4` |
| `id` | 本次调用编号，用于关联结果 |

这时只是提出请求，加法工具还没有执行。

这条消息加入状态后，[end_logic.py](./end_logic.py) 中的 `should_continue()` 检查到工具调用请求，将流程导向工具节点。

### 3.2 准备结果列表

```python
result = []
```

用来收集本次节点执行生成的所有工具结果消息。

### 3.3 根据工具名称查找工具对象

```python
tool = tools_by_name[tool_call["name"]]
```

[tool_self_defined.py](./tool_self_defined.py) 中定义了：

```python
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
```

相当于建立一个“名称 → 工具对象”的查找表：

```text
"add"      → 加法工具
"multiply" → 乘法工具
"divide"   → 除法工具
```

本次 `tool_call["name"]` 为 `"add"`，所以找到加法工具。

### 3.4 调用工具

```python
observation = tool.invoke(tool_call["args"])
```

代入本次参数，相当于：

```python
observation = add.invoke({"a": 3, "b": 4})
```

加法工具的底层函数为：

```python
@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`."""
    return a + b
```

`@tool` 将函数包装成工具对象。通过 `.invoke()` 调用时，参数字典中的值会传给底层函数，最终得到整数 `7`。

`observation` 是保存工具执行结果的变量名。

### 3.5 将结果包装成消息

```python
ToolMessage(
    content=str(observation),
    tool_call_id=tool_call["id"],
)
```

本次得到：

```python
ToolMessage(content="7", tool_call_id="call_1")
```

- `str(observation)` 将数字 `7` 转为文本 `"7"`。
- `tool_call_id` 指明这是哪一次调用的结果。

请求与结果的对应关系为：

```text
请求：call_1 → add(a=3, b=4)
结果：call_1 → "7"
```

随后 `result.append(...)` 将这条消息加入结果列表。

### 3.6 返回状态更新

```python
return {"messages": result}
```

这里返回的是新增消息。LangGraph 按照 `operator.add` 规则将它们追加到已有消息后面。

工具节点不直接生成最终的用户回复。图中还有这条边：

```python
agent_builder.add_edge("tool_node", "llm_call")
```

因此工具结果会再次交给模型，由模型继续决定下一步或组织回复。

## 4. 重点拆解 `for tool_call in state["messages"][-1].tool_calls`

完整代码：

```python
for tool_call in state["messages"][-1].tool_calls:
```

意思是：找到最后一条消息，把其中的工具调用请求逐个取出来。

### 4.1 示例状态

```python
from langchain_core.messages import AIMessage, HumanMessage

state = {
    "messages": [
        HumanMessage(content="帮我算 3 加 4"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "add",
                    "args": {"a": 3, "b": 4},
                    "id": "call_1",
                }
            ],
        ),
    ]
}
```

### 4.2 `state["messages"]`：取出消息列表

得到的是包含用户消息和模型消息的列表。

```python
messages = state["messages"]
```

### 4.3 `[-1]`：取出最后一个元素

Python 列表索引的含义：

```python
items = ["第一项", "第二项", "第三项"]

items[0]   # "第一项"
items[-1]  # "第三项"
```

所以：

```python
last_message = state["messages"][-1]
```

取到的是模型刚返回的 `AIMessage`。

这里能按模型消息处理，是因为当前图的流程保证工具节点接在提出工具请求的模型节点之后。

### 4.4 `.tool_calls`：读取对象属性

```python
tool_calls = last_message.tool_calls
```

取出模型消息中的工具调用列表。本例的核心内容为：

```python
[
    {
        "name": "add",
        "args": {"a": 3, "b": 4},
        "id": "call_1",
    }
]
```

外面是列表 `[]`，里面每个请求是字典 `{}`。框架可能为请求补充类型等字段，不影响这里的理解。

为什么有时用方括号，有时用点？

| 写法 | 操作的数据 | 含义 |
| --- | --- | --- |
| `state["messages"]` | 字典 | 按键取值 |
| `messages[-1]` | 列表 | 按位置取元素 |
| `last_message.tool_calls` | 消息对象 | 读取属性 |
| `tool_call["name"]` | 字典 | 读取某个调用请求的名称 |

### 4.5 `for tool_call in ...`：每次取一个请求

先看普通循环：

```python
for number in [10, 20]:
    print(number)
```

第一次 `number` 为 `10`，第二次为 `20`。

工具调用循环也是一样的：

```python
for tool_call in tool_calls:
    print(tool_call["name"])
```

本例只有一个请求，因此循环一次。循环中的 `tool_call` 就是那个包含 `name`、`args`、`id` 的字典。

`tool_calls` 是列表，`tool_call` 是循环中代表单个元素的变量名。

如果列表有两个请求，就循环两次；如果是空列表，就一次也不执行。这份工具节点代码会按循环顺序执行各个工具。

### 4.6 展开成容易阅读的写法

原来的一行等价于：

```python
messages = state["messages"]          # 所有消息
last_message = messages[-1]           # 最后一条模型消息
tool_calls = last_message.tool_calls  # 模型提出的工具调用列表

for tool_call in tool_calls:          # 每次取一个调用请求
    tool = tools_by_name[tool_call["name"]]
    observation = tool.invoke(tool_call["args"])
```

这里遍历的是最后一条模型消息中的工具调用请求，不是全部聊天消息。取出请求本身也不会执行工具，真正执行发生在 `.invoke()` 那一行。

## 5. 用消息记录串起全过程

在模型调用一次加法工具、随后给出最终回复的例子中：

| 时刻 | 新增消息 | 消息总数 | `llm_calls` |
| --- | --- | --- | --- |
| 用户输入 | `HumanMessage`：3 加 4 等于多少？ | 1 | 可以尚未提供 |
| 第一次调用模型后 | `AIMessage`：请求调用 `add` | 2 | 1 |
| 执行工具后 | `ToolMessage`：结果为 `"7"` | 3 | 1 |
| 第二次调用模型后 | `AIMessage`：3 加 4 等于 7 | 4 | 2 |

因此，这个例子中模型调用两次，工具执行一次。模型调用次数和工具调用次数不是同一个概念。

项目的 [test_agent.py](./test_agent.py) 中，`test_arithmetic_tool_loop` 使用预设模型消息验证了这一类流程，也检查了工具结果及调用编号的对应关系。

## 6. 阅读代码时可以自问的几个问题

1. 当前拿到的是字典、列表，还是消息对象？这决定了如何取值。
2. 当前数据是模型提出的调用请求，还是工具已经执行完的结果？
3. 节点返回的是本次新增消息，还是完整历史？本项目的拼接规则要求返回新增消息。
4. `tool_call_id` 是否对应模型请求中的 `id`？
5. 工具节点执行后为什么还要调用模型？因为模型需要读取工具结果，再继续处理或给出回复。

## 7. 相关文件

- [state_self_defined.py](./state_self_defined.py)：状态字段及消息合并规则。
- [model_node_self_defined.py](./model_node_self_defined.py)：调用模型并更新模型调用次数。
- [tool_self_defined.py](./tool_self_defined.py)：加法、乘法、除法工具及名称查找表。
- [tool_node.py](./tool_node.py)：执行工具请求并返回结果消息。
- [end_logic.py](./end_logic.py)：判断进入工具节点还是结束。
- [build_and_compile_agent.py](./build_and_compile_agent.py)：连接节点、构建运行流程。
