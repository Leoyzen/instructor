# RFC: Streaming Support for PARALLEL_TOOL Mode

**Status:** Draft | **Authors:** Instructor Community | **Created:** 2025-12-25 | **Discussion:** [Link to Issue/PR]

---

## Summary

This RFC proposes adding streaming support for the `PARALLEL_TOOL` mode in Instructor. Currently, users can only receive parallel tool call results after the complete response is generated. This enhancement will enable real-time streaming of partial results for each parallel tool call, similar to how `Partial` mode enables streaming for single tool calls.

---

## Motivation

### Current Limitations

The `PARALLEL_TOOL` mode currently only supports non-streaming responses:

```python
# Current behavior - waits for complete response
response_model = Iterable[Union[User, Task, Organization]]
results = client.chat.completions.create(
    model="gpt-4",
    response_model=response_model,
    response_format={"type": "json_object"},
    messages=[...],
    mode=instructor.Mode.PARALLEL_TOOLS,
)
# Returns: [User(...), Task(...), Organization(...)] (all at once)
```

This has several drawbacks:

1. **Poor user experience**: Users must wait for all tool calls to complete before seeing any results
2. **No progress indication**: Cannot show incremental progress for each tool call
3. **Inconsistent with other modes**: `Partial` mode supports streaming but `Parallel` does not

### Use Cases

1. **Real-time multi-task processing**: Generate and display multiple JSON objects as they're being created
2. **Progressive UI updates**: Update UI progressively as each parallel tool call completes
3. **Early validation**: Start validating results as soon as partial data is available
4. **Performance optimization**: Begin processing completed tasks while others are still streaming

---

## Proposed Solution

### Architecture Overview

The solution introduces streaming support for `ParallelBase` classes by implementing `from_streaming_response` methods similar to `PartialBase`. The design uses a state tracking helper to manage multiple parallel tool calls simultaneously.

```mermaid
flowchart TD
    A[User: create_partial with Parallel mode] --> B[stream=True]
    B --> C[LLM API streaming response]
    C --> D[ParallelBase.from_streaming_response]
    D --> E[ParallelStreamHelper tracks state]
    E --> F[Extract tool_call deltas]
    F --> G{New or existing tool_call?}
    G -->|New| H[Create Partial[Model] instance]
    G -->|Existing| I[Update existing Partial state]
    H --> J[Parse incremental JSON]
    I --> J
    J --> K[Yield ParallelResult]
```

### Key Components

#### 1. `ParallelStreamHelper` Class

A helper class to track the streaming state of multiple tool calls:

```python
class ParallelStreamHelper:
    """
    Tracks streaming state for multiple parallel tool calls.
    
    Maintains a mapping of tool_call_id to its current state:
    - tool_call_id: Local identifier for the tool call
    - model_class: The Pydantic model class to instantiate  
    - partial_state: Accumulated JSON string for this tool call
    - partial_model: Partial[Model] instance for incremental validation
    - is_complete: Whether this tool call has finished streaming
    """
    
    def __init__(self, parallel_base: ParallelBase):
        self.parallel_base = parallel_base
        self.tool_call_states: dict[str, ToolCallState] = {}
    
    def process_delta(self, delta: ToolCallDelta) -> ParallelResult:
        """Process a streaming delta and return updated partial results."""
```

#### 2. `ParallelResult` Data Structure

Streaming results will be wrapped in a structured format:

```python
@dataclass
class ParallelResult:
    """
    Represents the state of parallel tool calls during streaming.
    
    Attributes:
        completed: Dict of tool_call_id to fully completed BaseModel instances
        partial: Dict of tool_call_id to Partial[Model] instances still streaming
        tool_call_id: The tool_call_id that was updated in this iteration
    
    Example:
        >>> result = ParallelResult(
        ...     completed={"call_123": User(name="Alice")},
        ...     partial={"call_456": Partial[Task](title="Create", description="...")},
        ...     tool_call_id="call_456"
        ... )
    """
    completed: dict[str, BaseModel]
    partial: dict[str, BaseModel]
    tool_call_id: str
    
    def get_all_models(self) -> list[BaseModel]:
        """Get all available models (completed + partial)."""
        return list(self.completed.values()) + list(self.partial.values())
```

#### 3. `ToolCallDelta` Data Structure

```python
@dataclass
class ToolCallDelta:
    """Represents a streaming delta for a single tool call."""
    id: str | None
    index: int
    name: str | None
    arguments: str
    
    @classmethod
    def from_openai_chunk(cls, chunk) -> list["ToolCallDelta"]:
        """Extract deltas from OpenAI streaming chunk."""
```

### API Changes

#### 4.1 ParallelBase Enhancement

Add streaming support methods to `ParallelBase` (both sync and async):

```python
class ParallelBase:
    # Existing from_response() method remains unchanged
    
    @classmethod
    def from_streaming_response(
        cls,
        completion: Iterable[Any],
        mode: Mode,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> Generator[ParallelResult, None, None]:
        """
        Process streaming parallel tool calls (synchronous).
        
        Yields ParallelResult objects as each tool call delta arrives,
        providing real-time access to partial and completed models.
        
        Args:
            completion: Streaming response from LLM API
            mode: The Mode for this parallel operation
            validation_context: Context for Pydantic validation
            strict: Strict JSON parsing mode
            
        Yields:
            ParallelResult with current state of all tool calls
            
        Example:
            >>> for result in ParallelBase.from_streaming_response(
            ...     stream,
            ...     mode=Mode.PARALLEL_TOOLS
            ... ):
            ...     print(f"Completed: {len(result.completed)}")
            ...     print(f"Partial: {len(result.partial)}")
            ...     for model in result.get_all_models():
            ...         print(model)
        """
        helper = ParallelStreamHelper(cls)
        
        for chunk in completion:
            deltas = cls._extract_tool_deltas(chunk, mode)
            for delta in deltas:
                yield helper.process_delta(delta, validation_context, strict)
    
    @classmethod
    async def from_streaming_response_async(
        cls,
        completion: AsyncGenerator[Any, None],
        mode: Mode,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> AsyncGenerator[ParallelResult, None]:
        """
        Process streaming parallel tool calls (asynchronous).
        
        Async version of from_streaming_response for async clients.
        
        Yields:
            ParallelResult with current state of all tool calls
        """
        helper = ParallelStreamHelper(cls)
        
        async for chunk in completion:
            deltas = cls._extract_tool_deltas(chunk, mode)
            for delta in deltas:
                yield helper.process_delta(delta, validation_context, strict)
```

#### 4.2 Provider-Specific Delta Extraction

Add static methods to extract streaming deltas for each provider:

**OpenAI/Generic:**

```python
@staticmethod
def _extract_tool_deltas(
    chunk: Any, 
    mode: Mode
) -> list[ToolCallDelta]:
    """Extract tool call deltas from streaming chunk."""
    if mode == Mode.PARALLEL_TOOLS:
        if not chunk.choices or not chunk.choices[0].delta.tool_calls:
            return []
        
        deltas = []
        for tc in chunk.choices[0].delta.tool_calls:
            deltas.append(ToolCallDelta(
                id=tc.id,
                index=tc.index,
                name=tc.function.name if tc.function else None,
                arguments=tc.function.arguments or "",
            ))
        return deltas
    # ... other modes
```

**Anthropic:**

```python
# AnthropicParallelBase override
@staticmethod
def _extract_tool_deltas(chunk: Any, mode: Mode) -> list[ToolCallDelta]:
    """Extract tool call deltas from Anthropic streaming chunk."""
    if mode != Mode.ANTHROPIC_PARALLEL_TOOLS:
        return []
    
    deltas = []
    if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content_block'):
        content = chunk.delta.content_block
        if hasattr(content, 'tool_use'):
            deltas.append(ToolCallDelta(
                id=getattr(content.tool_use, 'id', None),
                index=getattr(content, 'index', 0),
                name=getattr(content.tool_use, 'name', None),
                arguments=json.dumps(getattr(chunk.delta, 'partial_json', {})),
            ))
    return deltas
```

**VertexAI:**

```python
# VertexAIParallelBase override
@staticmethod
def _extract_tool_deltas(chunk: Any, mode: Mode) -> list[ToolCallDelta]:
    """Extract tool call deltas from VertexAI streaming chunk."""
    # VertexAI-specific implementation
```

**litellm:**

```python
# litellm support for PARALLEL_TOOLS mode
@staticmethod
def _extract_tool_deltas(chunk: Any, mode: Mode) -> list[ToolCallDelta]:
    """Extract tool call deltas from litellm streaming chunk."""
    if mode != Mode.PARALLEL_TOOLS:
        return []
    
    # litellm exposes tool_calls in chunk.delta
    deltas = []
    if hasattr(chunk, 'choices') and chunk.choices:
        if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'tool_calls'):
            for tc in chunk.choices[0].delta.tool_calls:
                deltas.append(ToolCallDelta(
                    id=getattr(tc, 'id', None),
                    index=getattr(tc, 'index', 0),
                    name=getattr(tc['function'], 'name', None) if 'function' in tc else None,
                    arguments=getattr(tc['function'], 'arguments', "") if 'function' in tc else "",
                ))
    return deltas
```

**Note**: litellm supports multiple providers (Ollama, Groq, Together, etc.) through a unified API. The implementation above is compatible with all litellm providers that support streaming with tool calls.

#### 4.3 Process Response Integration

Update both `process_response()` and `process_response_async()` to handle parallel streaming:

```python
# instructor/processing/response.py

def process_response(
    response: T_Model,
    *,
    response_model: type[OpenAISchema | BaseModel] | None = None,
    stream: bool,
    validation_context: dict[str, Any] | None = None,
    strict: bool | None = None,
    mode: Mode = Mode.TOOLS,
) -> T_Model | list[T_Model] | Generator[ParallelResult, None, None] | None:
    """Process and transform LLM responses (synchronous)."""
    # ... existing code ...
    
    # NEW: Handle parallel streaming
    if (
        inspect.isclass(response_model)
        and issubclass(response_model, ParallelBase)
        and stream
    ):
        return response_model.from_streaming_response(
            response,
            mode=mode,
            validation_context=validation_context,
            strict=strict,
        )
    
    # ... rest of function ...

async def process_response_async(
    response: ChatCompletion,
    *,
    response_model: type[T_Model | OpenAISchema | BaseModel] | None,
    stream: bool = False,
    validation_context: dict[str, Any] | None = None,
    strict: bool | None = None,
    mode: Mode = Mode.TOOLS,
) -> T_Model | ChatCompletion:
    """Process and transform LLM responses (asynchronous)."""
    # ... existing code ...
    
    # NEW: Handle parallel streaming
    if (
        inspect.isclass(response_model)
        and issubclass(response_model, ParallelBase)
        and stream
    ):
        return response_model.from_streaming_response_async(
            cast(AsyncGenerator[Any, None], response),
            mode=mode,
            validation_context=validation_context,
            strict=strict,
        )
    
    # ... rest of function ...
```

#### 4.4 AsyncInstructor.create Support

Modify `AsyncInstructor.create()` to allow parallel streaming:

```python
# instructor/core/client.py

async def create(
    self,
    response_model: type[T] | None,
    # ... other params ...
) -> T | Any:
    # ... existing code ...
    
    # NEW: Allow streaming for ParallelBase when stream=True
    if response_model and stream:
        if isinstance(response_model, type) and issubclass(response_model, ParallelBase):
            return await self.create_fn(
                response_model=response_model,
                messages=messages,
                stream=stream,
                # ... other params ...
            )
    
    # ... rest of function ...
```

### Usage Examples

#### Example 1: Basic Parallel Streaming

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel
from collections.abc import Iterable
from typing import Union

class User(BaseModel):
    name: str
    email: str

class Task(BaseModel):
    title: str
    status: str

class Organization(BaseModel):
    name: str
    domain: str

client = instructor.from_openai(OpenAI(), mode=instructor.Mode.PARALLEL_TOOLS)

response_model = Iterable[Union[User, Task, Organization]]

# Stream parallel tool calls
stream = client.chat.completions.create_partial(
    model="gpt-4",
    response_model=response_model,
    messages=[{
        "role": "user",
        "content": "Extract user, task, and organization info"
    }],
)

for result in stream:
    print(f"Completed: {len(result.completed)}, Partial: {len(result.partial)}")
    for model in result.get_all_models():
        print(f"  {model}")
```

#### Example 2: Real-time UI Updates

```python
async def display_parallel_results():
    """Display parallel results as they stream in."""
    completed = set()
    
    for result in stream:
        # Display newly completed models
        for tc_id, model in result.completed.items():
            if tc_id not in completed:
                print(f"✓ Completed: {model}")
                completed.add(tc_id)
        
        # Display partial models
        for tc_id, model in result.partial.items():
            if tc_id not in result.completed:
                print(f"⏳ Streaming: {model}")
```

#### Example 3: Filtering and Processing

```python
# Process results as they become available
for result in stream:
    # Only process completed models
    for model in result.completed.values():
        if isinstance(model, User):
            send_to_database(model)
        elif isinstance(model, Task):
            create_ticket(model)
```

---

## Implementation Details

This RFC proposes a **two-phase implementation** approach:

#### **Phase 1: Core Infrastructure + OpenAI Support** (Priority)

**Goal**: Establish a foundation and enable streaming for the most commonly used provider.

**Tasks**:
1. Implement core data structures:
   - `ParallelStreamHelper` class (with sync + async support)
   - `ToolCallDelta` dataclass
   - `ParallelResult` dataclass
   - `ToolCallState` dataclass

2. Implement streaming methods in `ParallelBase`:
   - `from_streaming_response()` (synchronous)
   - `from_streaming_response_async()` (asynchronous)

3. Implement OpenAI-specific delta extraction:
   - `_extract_tool_deltas()` for OpenAI `PARALLEL_TOOLS` mode
   - Handle `ChatCompletionChunk.delta.tool_calls` parsing

4. Integrate with core processing:
   - Update `process_response()` to handle parallel streaming (sync)
   - Update `process_response_async()` to handle parallel streaming (async)
   - Update `AsyncInstructor.create()` to allow parallel streaming

5. Add tests and examples:
   - Unit tests for OpenAI parallel streaming
   - Example: `examples/parallel_streaming/run.py`
   - Document usage in `docs/examples/parallel_streaming.md`

**Deliverables**:
- Working parallel streaming for OpenAI provider (both sync and async)
- Comprehensive test coverage
- Documentation and examples
- Foundation ready for additional providers

---

#### **Phase 2: Additional Provider Support**

**Goal**: Extend streaming support to other commonly used parallel modes.

**Tasks**:
1. Implement Anthropic streaming:
   - Override `_extract_tool_deltas()` in `AnthropicParallelBase`
   - Handle `ANTHROPIC_PARALLEL_TOOLS` mode
   - Extract data from `MessageDeltaEvent` with `partial_json`

2. Implement VertexAI streaming:
   - Override `_extract_tool_deltas()` in `VertexAIParallelBase`
   - Handle `VERTEXAI_PARALLEL_TOOLS` mode
   - Extract data from VertexAI streaming chunks

3. Implement litellm streaming support:
   - Add generic `_extract_tool_deltas()` for `PARALLEL_TOOLS` mode when using litellm
   - Handle litellm chunk structure with tool_calls
   - Ensure compatibility with multiple litellm-supported providers (via Ollama, Groq, etc.)

4. Add provider-specific tests:
   - Anthropic parallel streaming tests
   - VertexAI parallel streaming tests
   - litellm parallel streaming tests with multiple providers

5. Update documentation:
   - Add provider-specific examples
   - Update API reference
   - Document litellm compatibility

**Deliverables**:
- Full parallel streaming support for OpenAI, Anthropic, VertexAI, and litellm
- Provider-specific test coverage
- Complete documentation
- litellm multi-provider compatibility

---

**Note**: Both sync and async support will be implemented simultaneously in each phase.

### State Management

The `ParallelStreamHelper` maintains the following state:

```python
@dataclass
class ToolCallState:
    """State for a single tool call during streaming."""
    tool_call_id: str
    model_class: type[BaseModel]
    accumulated_json: str = ""
    partial_model: BaseModel | None = None
    is_complete: bool = False
```

**State Transitions:**

1. **Initialization**: First delta with tool_call_id creates new state
2. **Accumulation**: Subsequent deltas append to `accumulated_json`
3. **Validation**: Each chunk attempts to parse with `jiter.from_json()`
4. **Completion**: Tool call finishes, validate final JSON, mark complete

### Error Handling

1. **JSON Parsing Errors**: Use `jiter.from_json()` with `partial_mode="trailing-strings"` for graceful degradation
2. **Missing tool_call_id**: Generate local ID based on index if not provided
3. **Unknown model name**: Skip tool calls not in registry (log warning)
4. **Stream interruption**: Return partially completed results for all tracked tool calls

### Backward Compatibility

- **Non-streaming behavior unchanged**: Existing `from_response()` methods continue to work
- **Opt-in**: Streaming requires explicit `stream=True` parameter
- **No breaking changes**: All existing code continues to work without modification

---

## Alternatives Considered

### Alternative 1: Separate API Methods

Add dedicated methods like `create_parallel_stream()`:

```python
# Alternative approach
client.create_parallel_stream(response_model=Iterable[Union[...]], ...)
```

**Pros**: Clear separation of concerns
**Cons**: More verbose API, inconsistent with existing patterns

**Decision**: Rejected. Use existing `create_partial()` with parallel support for consistency.

### Alternative 2: Yield Completed Models Only

Only yield models when fully complete, skip partial states:

```python
# Yields one model at a time as it completes
for model in stream:
    print(model)  # Only completed models
```

**Pros**: Simpler API
**Cons**: Less information, no insight into streaming progress

**Decision**: Rejected. Providing both completed and partial states gives users more control.

### Alternative 3: Separate Stream Modes

Introduce separate modes like `PARALLEL_TOOLS_STREAMING`:

```python
mode=instructor.Mode.PARALLEL_TOOLS_STREAMING
```

**Pros**: Explicit streaming intent
**Cons**: Mode proliferation, redundant with `stream` parameter

**Decision**: Rejected. Use existing `stream` parameter for consistency.

---

## Open Questions

1. **Tool Call ID Generation**: How should we handle cases where `tool_call_id` is not provided in early chunks? (Proposal: Use index as fallback)

2. **Memory Management**: Should we impose limits on the number of concurrent tool calls? (Proposal: No limit, but add warning for >10 concurrent)

3. **Partial vs Complete Priority**: Should completed models be yielded immediately or batched with other updates? (Proposal: Yield on every delta for maximum responsiveness)

4. **Error Recovery**: If one tool call fails JSON parsing, should we:
   - Skip that tool call and continue?
   - Fail the entire stream?
   - (Proposal: Log warning and skip, allow others to continue)

5. **Async Support**: Should we prioritize sync or async implementation? (Proposal: Implement both simultaneously)

---

## Success Criteria

- [ ] `ParallelBase.from_streaming_response()` implemented
- [ ] All three parallel modes support streaming (OpenAI, Anthropic, VertexAI)
- [ ] `process_response()` handles parallel streaming
- [ ] `AsyncInstructor.create()` allows parallel streaming  
- [ ] Unit tests covering:
  - Single tool call streaming
  - Multiple concurrent tool calls
  - Tool call ID generation
  - JSON parsing errors
  - Provider-specific behaviors
- [ ] Documentation updated with examples
- [ ] No breaking changes to existing code
- [ ] Performance comparable to non-streaming for simple cases

---

## References

- [`Partial` streaming implementation](../instructor/dsl/partial.py)
- [`ParallelBase` current implementation](../instructor/dsl/parallel.py)
- [`process_response` dispatcher](../instructor/processing/response.py)
- [OpenAI streaming documentation](https://platform.openai.com/docs/api-reference/streaming)
- [Anthropic streaming documentation](https://docs.anthropic.com/claude/reference/streaming)

---

## Appendix: Migration Guide

### For Existing Users

No migration required - this is a purely additive feature.

### For New Users

Add streaming support to parallel tool calls:

```python
# Before (non-streaming)
results = client.chat.completions.create(
    response_model=Iterable[Union[A, B, C]],
    mode=instructor.Mode.PARALLEL_TOOLS,
    messages=[...],
)
for result in results:
    print(result)

# After (streaming)
for parallel_result in client.chat.completions.create_partial(
    response_model=Iterable[Union[A, B, C]],
    mode=instructor.Mode.PARALLEL_TOOLS,
    messages=[...],
):
    for model in parallel_result.get_all_models():
        print(model)
```

---

**EOF**
