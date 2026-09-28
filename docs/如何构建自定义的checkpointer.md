LangGraph 的 **Build a custom checkpointer（构建自定义检查点器）** 这一节内容比较多、相对偏底层，涉及到如何让 LangGraph 与你自己的数据库（比如 Redis、MongoDB、或者是公司内部的某种自定义存储服务）进行对接。

我们可以从**核心目标、核心方法（必须要实现的接口）和设计模式**三部分来轻松理解。

---

### 一、 为什么要自定义 Checkpointer？

LangGraph 官方自带了很多内存或常见数据库的 Checkpointer（比如 `MemorySaver`、`SqliteSaver`、`PostgresSaver` 等）。但如果你的项目在用一些冷门数据库、云原生键值存储、或者有非常特殊的加密/序列化需求，官方的满足不了你，你就需要继承并实现一个自定义的 Checkpointer。

---

### 二、 核心骨架：你需要实现哪些方法？

要实现一个自定义的 Checkpointer，你需要继承基础的 `BaseCheckpointSaver` 类（通常支持同步和异步），并实现以下 **5 个核心契约方法（Contract）**。
* **5个核心方法代码如下**：
```
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

class MyCheckpointer(BaseCheckpointSaver):
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        ...

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        ...

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        ...

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        ...
        yield  # make this an async generator

    async def adelete_thread(self, thread_id: str) -> None:
   ```
每一个方法都有同步版本（如 `put`）和异步版本（如 `aput`）：

1. **`put / aput`（保存检查点）**
* **作用**：当图（Graph）完成一个超步（Super-step）时，LangGraph 会调用它把当前的 `checkpoint` 数据保存下来。
* **参数**：包含 `config`（有 `thread_id` 等信息）、`checkpoint`（当前状态快照）以及 `metadata`（元数据）。
* **aput具体代码实现与难点讲解**：


2. **`put_writes / aput_writes`（保存节点任务写入）**
* **作用**：用来保存单个节点在执行中产生的中间写入（Pending Writes）。这对于容错（Fault-tolerance）非常重要——如果某个节点中途崩溃，恢复时不需要重新执行已经成功的节点。


3. **`get_tuple / aget_tuple`（读取单个检查点）**
* **作用**：根据配置（比如给定了 `thread_id` 和可选的 `checkpoint_id`）去数据库里捞出对应的检查点。
* **返回值**：需要返回一个包含 `(config, checkpoint, metadata, parent_config)` 的元组（Tuple）。


4. **`list / alist`（列出历史检查点）**
* **作用**：用于“时间旅行（Time travel）”或者查看历史记录（`get_state_history`）。它会根据 `thread_id` 返回一连串的检查点列表，支持分页或按条件过滤（如 `before`、`limit`）。


5. **`delete_thread / adelete_thread`（删除线程）**
* **作用**：可选或按需实现，用于清理某个 `thread_id` 对应的所有历史状态数据。


* **aput方法讲解如下**：
```
async def aput(self, config, checkpoint, metadata, new_versions):
    thread_id = config["configurable"]["thread_id"]
    checkpoint_ns = config["configurable"]["checkpoint_ns"]
    checkpoint_id = checkpoint["id"]
    parent_id = config["configurable"].get("checkpoint_id")

    type_, blob = self.serde.dumps_typed(checkpoint)
    serialized_metadata = self.serde.dumps_typed(metadata)

    await self.db.execute(
        "INSERT INTO checkpoints (...) VALUES (...)",
        thread_id, checkpoint_ns, checkpoint_id, parent_id,
        type_, blob, *serialized_metadata,
    )
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
    }
```
这段代码是一个自定义 **LangGraph Checkpointer**（状态持久化/保存器）的核心方法 `aput`。它的作用是**异步地将 LangGraph 执行过程中的状态快照（Checkpoint）及其元数据保存到数据库中**。

下面逐行拆解它的工作机制和各个参数的作用：

---

### 1. 提取配置信息 (Config Extraction)

```python
thread_id = config["configurable"]["thread_id"]
checkpoint_ns = config["configurable"]["checkpoint_ns"]
checkpoint_id = checkpoint["id"]
parent_id = config["configurable"].get("checkpoint_id")

```

* **`thread_id`**：会话/线程 ID。用于区分不同的用户或不同的对话流程。
* **`checkpoint_ns`**：命名空间（Namespace）。LangGraph 支持子图（Subgraphs），命名空间用来隔离不同层级图的状态。
* **`checkpoint_id`**：当前状态快照的唯一标识符（通常是一个时间戳或 UUID）。
* **`parent_id`**：上一个状态的标识符（即父节点）。通过记录 `parent_id`，LangGraph 可以把所有状态连成一条历史链条，实现“时间旅行”（Time Travel/状态回滚）功能。

---

### 2. 序列化 (Serialization)

```python
type_, blob = self.serde.dumps_typed(checkpoint)
serialized_metadata = self.serde.dumps_typed(metadata)

```

* LangGraph 中的 `checkpoint` 和 `metadata` 通常是 Python 字典或复杂的 Python 对象。
* **`self.serde`** 是序列化器（Serializer/Deserializer，通常是 `SerializerProtocol`，默认用 JSON 或 Pickle）。
* **`dumps_typed(...)`** 将复杂的数据结构转换成可以存储在数据库中的格式（比如 `type_` 标识数据格式是 JSON 还是二进制，`blob` 是实际序列化后的内容）。

---

### 3. 写入数据库 (Database Insertion)

```python
await self.db.execute(
    "INSERT INTO checkpoints (...)",
    thread_id, checkpoint_ns, checkpoint_id, parent_id,
    type_, blob, *serialized_metadata,
)

```

* **`await self.db.execute(...)`**：以异步方式将上述信息插入到名为 `checkpoints` 的数据库表中。
* 包含了控制流信息（`thread_id`, `checkpoint_ns`, `checkpoint_id`, `parent_id`）以及实际的图状态数据（`type_`, `blob`）和元数据。

---

### 4. 返回更新后的配置 (Return Updated Config)

```python
return {
    "configurable": {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint_id": checkpoint_id,
    }
}

```

* 保存成功后，返回一个包含最新状态指针的 `config` 字典。
* 这样后续的 Step 或节点在读取状态时，就能知道当前最新的 `checkpoint_id` 是什么。

---

### 💡 需要注意的小细节

1. **`new_versions` 未被处理**：方法签名中有 `new_versions` 参数，但在 SQL 写入中没有出现。在更完善的 LangGraph Saver 实现中，通常还需要处理通道版本（`writes` / `writes_versions`）或写操作历史表，以支持并发控制和细粒度的版本比对。
2. **`*serialized_metadata`**：这里使用了解包符号 `*`，具体取决于 `dumps_typed` 返回的是 `(type, blob)` 元组还是其他数据结构，传入数据库驱动时需确保与 SQL 中的参数占位符（`?` 或 `$1`）数量一致。

### 💡 实际开发中如何使用？

以上的返回结果，可以理解为一个用来“指向”或“定位”数据库中特定快照（Snapshot）的复合主键**。

就是把 `config` 字典想象成去寄存处拿行李的 **“凭证卡”**：

```
[ 图的下一次运行 / 外部交互 ]
            │
            │  持凭证 config 去查数据库：
            │  "请给我找 thread_id='t1', checkpoint_ns='', checkpoint_id='chk_123' 的记录"
            ▼
┌──────────────────────────────────────────────────┐
│                   数据库 (DB)                    │
│ ──────────────────────────────────────────────── │
│ thread_id │ checkpoint_ns │ checkpoint_id │ blob │
│ ───────── │ ───────────── │ ───────────── │ ──── │
│ t1        │ ""            │ chk_123       │ ...  │ ◄── 找到了对应快照并读取状态
└──────────────────────────────────────────────────┘

```

---

### 为什么要返回这个字典？

在 LangGraph 中，**图（Graph）本身是无状态的**。它的运行完全依赖外部传入的 `config` 参数。

当你调用 `aput` 保存完当前状态后，需要告诉 LangGraph 或调用方：**“我已经把状态存好了，它的最新位置是这个。”**

#### 1. 给 LangGraph 引擎使用

保存完当前的 Checkpoint 之后，图如果还要继续往下走下一个节点，引擎需要拿着这个最新的 `checkpoint_id` 去作为下一个节点的 **`parent_id`**（上一个状态标识），从而在数据库里把这些状态一环扣一环地连成一条链表。

#### 2. 给开发者/前端使用

在实际开发中，你通常会这样运行图：

```
python
# 1. 第一次运行，只指定 thread_id
config = {"configurable": {"thread_id": "user_123"}}
checkpoint_config = await app.ainvoke(input_data, config)

# 保存完后返回的 checkpoint_config 会形如：
# {
#     "configurable": {
#         "thread_id": "user_123",
#         "checkpoint_ns": "",
#         "checkpoint_id": "1ef633d0-..."  <-- 自动生成的最新快照ID
#     }
# }

# 2. 如果你想从【最新状态】继续对话，或者实现【状态回滚（时间旅行）】：
# 只需要把这个 config 传回给 LangGraph，它就能从该 checkpoint_id 恢复状态
state = await app.aget_state(checkpoint_config)

```

---

### 💡💡 结构拆解（重点！！！）

`config["configurable"]` 里的三个字段加在一起，在数据库中构成了一个**唯一联合主键**：

* **`thread_id`**：定位是**哪一次对话**（例如：用户 A 的会话）。
* **`checkpoint_ns`**：定位是**哪一层图**（主图还是某个子图）。
* **`checkpoint_id`**：定位是**该对话中的哪一秒/哪一步**（具体的某一次快照）。

**总结一句话：**
返回这个字典，本质上就是返回了一套“坐标”（`thread_id` + `checkpoint_ns` + `checkpoint_id`），以便后续无论是 LangGraph 引擎还是你自己，都能凭这套坐标精准找到刚存进去的那条状态记录。


---

### 三、 编写自定义 Checkpointer 的“核心思路与雷区”

如果你真的去写一个自定义 Checkpointer，有几个关键的设计点官方文档反复强调：

1. **Row Key / 索引设计（怎么在数据库里存）**
* 你的数据库表或键值设计必须能够通过 **`thread_id`** 和 **`checkpoint_id`** 联合检索。
* 同时，因为支持“时间旅行”和历史回溯，同一个 `thread_id` 下会有很多个不同的 `checkpoint_id`。


2. **序列化（Serialization）**
* 检查点里包含 Python 的各种对象、状态字典甚至复杂的类。你的 Checkpointer 通常需要借用 LangGraph 自带的 `SerializerProtocol`（比如用 `pickle` 或 JSON）把 Python 对象转成二进制或字符串再存入数据库。


3. **并发与多租户 / 异步支持**
* 现代应用大多是异步的（Async）。虽然你可以只写同步方法，但为了配合高并发的 FastAPI 等框架，强烈建议同步和异步方法（`put` / `aput`, `get_tuple` / `aget_tuple` 等）都实现。



---

### 四、 如何测试你的自定义 Checkpointer？

LangGraph 官方非常贴心地提供了一个叫 **Conformance Suite（一致性测试套件）** 的工具。
你不需要自己写一堆单元测试去验证你的 Checkpointer 有没有漏掉什么边界条件。LangGraph 自带了一套标准的测试用例，你只需要把你的自定义 Checkpointer 实例传进去，跑一下官方的测试集，就能知道你的实现是否符合 LangGraph 运行时的所有标准。

---

### 总结建议

* **如果只是日常业务开发**：你**完全不需要**看这部分内容，直接用官方的 `MemorySaver` 或 `SqliteSaver` / `AsyncPostgresSaver` 即可。
* **如果公司有强烈的基建定制需求**：再回来看这部分文档，它的本质就是实现一套 **CRUD（增删改查）** 接口，把 LangGraph 内部的 `checkpoint` 和 `writes` 数据结构存到你们指定的数据库里。