# 透明的并行流式处理设计文档

## 问题陈述

### 用户需求
用户希望在使用并行工具调用时，只需添加 `stream=True` 参数即可自动获得流式支持，而不需要手动创建 `ParallelBase` 实例。

### 当前限制
1. **非流式工作正常**：使用 `Iterable[Union[A, B]]` 可以正常工作
2. **流式需要手动转换**：要使用流式，用户必须显式创建 `ParallelBase` 实例
3. **用户体验不一致**：其他模式（如 `Partial`、`Iterable`）支持透明流式，但并行模式不支持

### 代码分析

#### 当前的并行处理流程（非流式）
```python
# 用户代码
response_model = Iterable[Union[User, Task, Organization]]
results = client.chat.completions.create(
    model="gpt-4",
    response_model=response_model,
    mode=instructor.Mode.PARALLEL_TOOLS,
    messages=[...],
)

# 内部流程
1. handle_response_model() 检测到 PARALLEL_TOOLS 模式
2. 调用 handle_parallel_tools(response_model, new_kwargs)
3. handle_parallel_tools() 返回:
   - response_model: ParallelModel(typehint=response_model)
   - new_kwargs: 包含 tools 和 tool_choice
4. 最终调用 ParallelBase.from_response() 解析完整响应
```

#### 流式处理的当前状态
```python
# 已实现但需要手动配置
from instructor.dsl.parallel import OpenAIParallelModel

# 用户必须手动创建 ParallelBase
parallel_base = OpenAIParallelModel(Iterable[Union[User, Task, Organization]])

stream = client.chat.completions.create(
    response_model=parallel_base,  # 需要 ParallelBase 实例
    stream=True,  # 流式参数
    mode=instructor.Mode.PARALLEL_TOOLS,
    messages=[...],
)

# 内部流程
1. response_model 已经是 ParallelBase 实例
2. handle_response_model() 直接返回它（因为是 isinstance 检查）
3. process_response() 检测到 ParallelBase + stream=True
4. 调用 from_streaming_response() 返回 Generator[ParallelResult]
```

### 关键代码位置

1. **instructor/processing/response.py::handle_response_model()**
   - 当前：在 PARALLEL_MODES 调用 handler，返回转换后的 ParallelBase
   - 需要：在流式模式下自动检测并转换

2. **instructor/providers/openai/utils.py::handle_parallel_tools()**
   - 当前：总是返回 ParallelModel()
   - 保持不变：继续处理非流式情况

3. **instructor/processing/response.py::process_response()**
   - 当前：检测 `isinstance(response_model, ParallelBase) and stream`
   - 优化：统一处理，减少重复检查

## 设计方案

### 核心原则
1. **透明性**：用户只需添加 `stream=True`，无需感知 ParallelBase
2. **向后兼容**：现有的非流式用法完全不变
3. **一致性**：与 Partial、Iterable 的流式体验一致

### 实现方案

#### 方案 1：在 handle_response_model 中自动转换（推荐）

**修改位置**：`instructor/processing/response.py::handle_response_model()`

**逻辑**：
```python
def handle_response_model(
    response_model: type[T] | None, 
    mode: Mode = Mode.TOOLS, 
    **kwargs: Any
) -> tuple[type[T] | None, dict[str, Any]]:
    new_kwargs = kwargs.copy()
    
    PARALLEL_MODES = {
        Mode.PARALLEL_TOOLS: handle_parallel_tools,
        Mode.VERTEXAI_PARALLEL_TOOLS: handle_vertexai_parallel_tools,
        Mode.ANTHROPIC_PARALLEL_TOOLS: handle_anthropic_parallel_tools,
    }
    
    if mode in PARALLEL_MODES:
        # 流式模式检测
        is_streaming = new_kwargs.get("stream", False)
        
        if is_streaming:
            # 流式场景：需要确保 response_model 是 ParallelBase 实例
            if isinstance(response_model, type) and get_origin(response_model) is Iterable:
                # 用户传入的是 Iterable[Union[...]]，需要转为 ParallelBase
                if mode == Mode.PARALLEL_TOOLS:
                    response_model = OpenAIParallelModel(response_model)  # 实例
                elif mode == Mode.VERTEXAI_PARALLEL_TOOLS:
                    response_model = VertexAIParallelModel(response_model)  # 实例
                elif mode == Mode.ANTHROPIC_PARALLEL_TOOLS:
                    response_model = AnthropicParallelModel(response_model)  # 实例
            # 如果已经是 ParallelBase 实例，直接使用
        
        # 调用原有的 handler（非流式）
        response_model, new_kwargs = PARALLEL_MODES[mode](
            response_model, new_kwargs
        )
        
        return response_model, new_kwargs
```

**优点**：
- 集中在一处处理，逻辑清晰
- 最小化代码修改
- 易于维护和测试

**缺点**：
- 需要在 handle_response_model 中导入 provider 特定的 ParallelModel 类

#### 方案 2：在 provider handler 中流式感知（备选）

**修改位置**：`instructor/providers/openai/utils.py::handle_parallel_tools()`

**逻辑**：
```python
def handle_parallel_tools(
    response_model: type[Any], 
    new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """Handle OpenAI parallel tools mode."""
    
    # 检查是否为流式模式
    is_streaming = new_kwargs.get("stream", False)
    
    if is_streaming:
        # 流式模式：直接返回 ParallelBase 实例
        return OpenAIParallelModel(response_model), new_kwargs
    
    # 非流式模式：原有逻辑保持不变
    new_kwargs["tools"] = handle_parallel_model(response_model)
    new_kwargs["tool_choice"] = "auto"
    return cast(type[Any], ParallelModel(typehint=response_model)), new_kwargs
```

**优点**：
- 逻辑放在合适的位置
- 每个 provider 可以有自己的实现
- 遵循单一职责原则

**缺点**：
- 需要在每个 provider 的 handler 中都实现
- 代码重复度高

### 推荐方案：方案 1

**理由**：
1. 所有并行模式的集中处理
2. 一处修改，全局生效
3. 更容易添加新的并行模式
4. 减少代码重复

## 实现细节

### 需要修改的文件

1. **instructor/processing/response.py**
   - 修改 `handle_response_model()` 函数
   - 添加流式检测逻辑
   - 自动转换 `Iterable[Union[...]]` 为对应的 `ParallelBase` 实例

2. **instructor/processing/response.py::process_response()**
   - 优化 ParallelBase 流式检测逻辑
   - 移除重复的 isinstance 检查

### 具体修改

#### 修改 1：handle_response_model 自动转换

```python
# 在 PARALLEL_MODES 检查前添加导入
from ..dsl.parallel import (
    OpenAIParallelModel,
    AnthropicParallelModel,
    VertexAIParallelModel,
)
from typing import get_origin
from collections.abc import Iterable

# 修改 PARALLEL_MODES 处理块
if mode in PARALLEL_MODES:
    # 检测流式模式
    is_streaming = new_kwargs.get("stream", False)
    
    if is_streaming:
        # 流式模式：确保 response_model 是 ParallelBase 实例
        from ..dsl.parallel import ParallelBase
        
        if not isinstance(response_model, ParallelBase):
            # 检查是否为 Iterable[Union[...]] 类型
            is_iterable = (
                isinstance(response_model, type) 
                and get_origin(response_model) is Iterable
            )
            
            if is_iterable:
                # 根据模式创建对应的 ParallelBase 实例
                if mode == Mode.PARALLEL_TOOLS:
                    response_model = OpenAIParallelModel(response_model)
                elif mode == Mode.VERTEXAI_PARALLEL_TOOLS:
                    response_model = VertexAIParallelModel(response_model)
                elif mode == Mode.ANTHROPIC_PARALLEL_TOOLS:
                    response_model = AnthropicParallelModel(response_model)
                # 其他 provider 可以在这里添加
    
    # 调用原有的 handler（处理 tools 参数等）
    # 注意：流式模式下我们已经在上面转换了 response_model
    # 这里只需要处理 new_kwargs 中的 tools/tool_choice
    if not is_streaming:
        response_model, new_kwargs = PARALLEL_MODES[mode](response_model, new_kwargs)
    else:
        # 流式模式：只处理 kwargs，不转换 response_model
        _, new_kwargs = PARALLEL_MODES[mode](None, new_kwargs)
        
    logger.debug(f"Instructor Request: {mode.value=}, {response_model=}, {new_kwargs=}")
    return response_model, new_kwargs
```

#### 修改 2：优化 process_response 流式检测

```python
def process_response(
    response: T_Model,
    *,
    response_model: type[OpenAISchema | BaseModel] | None = None,
    stream: bool,
    validation_context: dict[str, Any] | None = None,
    strict=None,
    mode: Mode = Mode.TOOLS,
) -> T_Model | list[T_Model] | None:
    # ... 前面的代码保持不变 ...
    
    # 统一的流式检测和处理
    if isinstance(response_model, ParallelBase) and stream:
        # 流式模式：调用 from_streaming_response
        return response_model.from_streaming_response(
            response,
            mode=mode,
            validation_context=validation_context,
            strict=strict,
        )
    
    # ... 后面的代码保持不变 ...
```

### 使用示例

#### 示例 1：OpenAI 并行流式

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel
from typing import Iterable, Union

class User(BaseModel):
    name: str
    email: str

class Task(BaseModel):
    title: str
    status: str

client = instructor.from_openai(OpenAI(), mode=instructor.Mode.PARALLEL_TOOLS)

# 使用 Iterable[Union[...]] 类型，只需添加 stream=True
stream = client.chat.completions.create(
    model="gpt-4",
    response_model=Iterable[Union[User, Task]],  # 标准类型
    stream=True,  # 启用流式
    messages=[{
        "role": "user",
        "content": "Extract user and task information"
    }],
)

# 流式响应：ParallelResult 对象
for result in stream:
    print(f"Completed: {len(result.completed)}, Partial: {len(result.partial)}")
    for model in result.get_all_models():
        print(f"  {model}")
```

#### 示例 2：VertexAI 并行流式

```python
import instructor
from pydantic import BaseModel
from typing import Iterable, Union

class Weather(BaseModel):
    location: str
    temperature: float

class SearchQuery(BaseModel):
    query: str

client = instructor.from_provider(
    "vertexai/gemini-2.5-flash",
    mode=instructor.Mode.VERTEXAI_PARALLEL_TOOLS,
)

# 同样的接口，使用 stream=True
stream = client.chat.completions.create(
    response_model=Iterable[Union[Weather, SearchQuery]],
    stream=True,  # 流式支持
    messages=[{
        "role": "user",
        "content": "What's the weather and what should I search for?"
    }],
)

for result in stream:
    # 处理流式结果
    pass
```

#### 示例 3：Anthropic 并行流式

```python
import instructor
from pydantic import BaseModel
from typing import Iterable, Union

class Article(BaseModel):
    title: str
    content: str

class Image(BaseModel):
    url: str
    alt_text: str

client = instructor.from_provider(
    "anthropic/claude-3-7-sonnet-latest",
    mode=instructor.Mode.ANTHROPIC_PARALLEL_TOOLS,
)

# 流式并行处理
stream = client.chat.completions.create(
    response_model=Iterable[Union[Article, Image]],
    stream=True,  # 自动检测
    messages=[{
        "role": "user",
        "content": "Generate content for my blog post"
    }],
)

for result in stream:
    # 处理并行流式结果
    pass
```

## 兼容性保证

### 向后兼容

1. **非流式代码完全不变**
   ```python
   # 这段代码继续正常工作
   results = client.chat.completions.create(
       response_model=Iterable[Union[A, B]],
       mode=instructor.Mode.PARALLEL_TOOLS,
       messages=[...],
   )
   ```

2. **手动 ParallelBase 继续工作**
   ```python
   # 这段代码也继续正常工作
   parallel_base = OpenAIParallelModel(Iterable[Union[A, B]])
   stream = client.chat.completions.create(
       response_model=parallel_base,
       stream=True,
       messages=[...],
   )
   ```

3. **其他模式不受影响**
   - `Partial` 模式流式继续工作
   - `Iterable` 模式流式继续工作
   - 普通模式不受影响

### 边界情况处理

1. **非 Iterable 类型**
   - 如果 response_model 不是 Iterable 类型，不做转换
   - 让原有逻辑处理

2. **非流式模式**
   - 保持原有逻辑不变
   - 不进行自动转换

3. **非并行模式**
   - 不受任何影响
   - 完全不触及这些代码路径

## 测试计划

### 单元测试

1. **自动转换测试**
   - 测试 `stream=True` + `Iterable[Union[...]]` 自动转换为 `ParallelBase`
   - 测试 `stream=False` 保持原有行为

2. **Provider 测试**
   - 测试 OpenAI、VertexAI、Anthropic 流式转换
   - 验证每个 provider 使用正确的 ParallelBase 子类

3. **向后兼容测试**
   - 测试现有非流式代码
   - 测试手动 ParallelBase 用法

### 集成测试

1. **端到端流式测试**
   - 测试完整的流式响应流程
   - 验证 ParallelResult 的正确性

2. **多-provider 测试**
   - 在不同 provider 上验证流式支持
   - 确保 API 兼容性

## 性能影响

### 预期影响

1. **非流式场景**
   - 几乎无性能影响
   - 只添加一个条件检查

2. **流式场景**
   - 减少用户代码复杂度
   - 自动转换的开销可以忽略

3. **内存使用**
   - 无额外内存分配
   - 并行模式原有内存需求

## 结论

本方案通过在 `handle_response_model()` 中添加智能的流式检测和自动转换逻辑，实现了透明的并行流式处理：

1. **用户友好**：只需添加 `stream=True`
2. **向后兼容**：现有代码完全不变
3. **维护性好**：集中在一处处理
4. **扩展性强**：易于支持新的并行模式

这个设计完全满足用户需求，并且与 instructor 的现有架构完美集成。