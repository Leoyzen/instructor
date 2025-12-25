# Parallel Streaming Examples

This directory contains comprehensive examples demonstrating parallel streaming with Instructor.

## Overview

Parallel streaming allows you to extract multiple structured objects (of different types) in real-time as they are being generated, rather than waiting for all objects to complete. This provides:

- **Real-time progress**: See results as they're generated
- **Better UX**: Users see immediate feedback
- **Early validation**: Start validating data as soon as it's available
- **Progressive updates**: Update UI incrementally as sections complete

## Quick Start

### Basic Usage

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel
from instructor.dsl.parallel import OpenAIParallelBase

class User(BaseModel):
    name: str
    age: int

class Task(BaseModel):
    title: str
    status: str

# Create instructor client with parallel tools mode
client = instructor.from_openai(OpenAI(), mode=instructor.Mode.PARALLEL_TOOLS)

# Create parallel base with multiple models
parallel_base = OpenAIParallelBase(User, Task)

# Stream parallel tool calls
for result in client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=parallel_base,
    messages=[
        {
            "role": "user",
            "content": "Extract user Alice (age 25) and her task 'Complete documentation' (status: in-progress)",
        },
    ],
    stream=True,
):
    # result.completed: Dict of tool_call_id to fully completed models
    # result.partial: Dict of tool_call_id to partial models still streaming
    
    for tool_call_id, model in result.completed.items():
        print(f"✓ Completed: {model}")
    
    for tool_call_id, model in result.partial.items():
        print(f"⏳ Streaming: {model}")
```

## Running the Examples

### Run All Examples

```bash
python run.py
```

### Run a Specific Example

```bash
# Example 1: Basic Single Model
python run.py --example 1

# Example 2: Multiple Models
python run.py --example 2

# Example 7: Async Streaming
python run.py --example 7
```

### Run Async Examples Only

```bash
python run.py --async
```

### Use a Different Model

```bash
python run.py --model gpt-4
```

## Examples Overview

### Example 1: Basic Parallel Streaming (Single Model)

- Demonstrates simplest use case with a single model type
- Shows how to stream a User model extraction
- Displays progress updates in real-time

### Example 2: Multiple Models Streaming

- Extracts User and Task information simultaneously
- Shows true power of parallel streaming with different model types
- Demonstrates tracking and displaying multiple concurrent extractions

### Example 3: Complex Multi-Model Extraction

- Extracts User, Task, Product, and Address information
- Shows handling of complex nested data
- Demonstrates real-time progress tracking for multiple model types

### Example 4: Real-Time Progress Display

- Enhanced progress display with detailed streaming information
- Visual progress bar showing completion status
- Detailed breakdown of completed and partial models

### Example 5: Streaming with Validation Context

- Passes validation context to influence model validation
- Demonstrates how validation rules affect extraction
- Shows proper use of `validation_context` parameter

### Example 6: Strict Mode Streaming

- Demonstrates strict vs non-strict JSON parsing
- Shows how strict mode affects streaming behavior
- Compares results with `strict=True` and `strict=False`

### Example 7: Asynchronous Parallel Streaming

- Same functionality using async/await
- Ideal for applications handling concurrent requests
- Shows `AsyncInstructor` usage

### Example 8: Error Handling and Best Practices

- Proper error handling for parallel streaming
- Validation and defensive programming techniques
- Robust implementation patterns

### Example 9: Filtering and Processing Results

- Real-time filtering and processing of streamed results
- Model-specific processing logic
- Demonstrates transforming data as it arrives

### Example 10: Progressive UI Updates Simulation

- Simulates UI updates during streaming
- Shows how to build progressive interfaces
- Component-based rendering as models complete

## Key Concepts

### ParallelResult

The `ParallelResult` object represents the state of parallel tool calls during streaming:

```python
@dataclass
class ParallelResult:
    completed: dict[str, BaseModel]  # Fully completed models
    partial: dict[str, BaseModel]     # Partial models still streaming
    tool_call_id: str                # ID updated in this iteration
    
    def get_all_models(self) -> list[BaseModel]:
        """Get all available models (completed + partial)."""
```

### OpenAIParallelBase

Wrapper for OpenAI provider with streaming support:

```python
parallel_base = OpenAIParallelBase(User, Task, Product)
```

### The Streaming Flow

1. Initialize `ParallelBase` with your model types
2. Call `create()` with `stream=True`
3. Receive `ParallelResult` objects as they stream in
4. Process completed models immediately
5. Display progress for partial models
6. Continue until all models complete

## Use Cases

### Real-Time Multi-Task Processing

Generate and display multiple JSON objects as they're being created:

```python
for result in parallel_stream:
    for model in result.completed.values():
        # Process completed models immediately
        process_model(model)
```

### Progressive UI Updates

Update UI progressively as each parallel tool call completes:

```python
completed = set()
for result in parallel_stream:
    for tc_id, model in result.completed.items():
        if tc_id not in completed:
            update_ui_component(model)
            completed.add(tc_id)
```

### Early Validation

Start validating results as soon as partial data is available:

```python
for result in parallel_stream:
    for model in result.get_all_models():
        validate_model(model)
```

## Best Practices

### 1. Track Completed Models

```python
completed_tool_calls = set()
for result in parallel_stream:
    for tool_call_id, model in result.completed.items():
        if tool_call_id not in completed_tool_calls:
            process_model(model)
            completed_tool_calls.add(tool_call_id)
```

### 2. Handle Errors Gracefully

```python
try:
    for result in parallel_stream:
        # Process results
except Exception as e:
    logger.error(f"Streaming error: {e}")
    # Handle partial results
```

### 3. Use Validation Context

```python
validation_context = {
    "min_age": 18,
    "allowed_statuses": ["todo", "in-progress", "completed"],
}

# Pass validation context
for result in client.chat.completions.create(
    response_model=parallel_base,
    validation_context=validation_context,
    stream=True,
):
    # ...
```

### 4. Understand Partial vs Completed

- **Partial**: Models still being generated, may have missing fields
- **Completed**: Fully validated models with all required fields

### 5. Use Async for Concurrent Operations

```python
async def process_multiple_requests():
    # Run multiple parallel extractions concurrently
    tasks = [
        extract_parallel_streaming(data1),
        extract_parallel_streaming(data2),
        extract_parallel_streaming(data3),
    ]
    await asyncio.gather(*tasks)
```

## API Reference

### ParallelResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `completed` | `dict[str, BaseModel]` | Fully completed models indexed by tool_call_id |
| `partial` | `dict[str, BaseModel]` | Partial models still streaming |
| `tool_call_id` | `str` | ID of the tool call updated in this iteration |

| Method | Return | Description |
|--------|--------|-------------|
| `get_all_models()` | `list[BaseModel]` | Get all available models (completed + partial) |

### OpenAIParallelBase

| Method | Description |
|--------|-------------|
| `__init__(*models)` | Initialize with model classes |
| `from_streaming_response(completion, mode, ...)` | Process streaming parallel tool calls (sync) |
| `from_streaming_response_async(completion, mode, ...)` | Process streaming parallel tool calls (async) |

## Troubleshooting

### No Results Streaming

**Problem**: No results are being streamed.

**Solutions**:

- Verify `stream=True` is set
- Check model supports parallel tools (e.g., `gpt-4o-mini`)
- Ensure response_model is an `OpenAIParallelBase` instance

### Empty Completed Models

**Problem**: `result.completed` is always empty.

**Solutions**:

- Wait for streaming to complete (final chunks contain completed models)
- Check if response actually contains tool calls
- Verify model classes match expected structure

### Validation Errors

**Problem**: Models fail validation during streaming.

**Solutions**:

- Use `strict=False` for more lenient parsing
- Check model definitions match expected JSON structure
- Use `validation_context` for custom validation rules

### Too Many Partial Updates

**Problem**: Too many partial updates causing performance issues.

**Solutions**:

- Debounce partial updates (process every Nth chunk)
- Filter partial updates to display only meaningful progress
- Consider displaying completed models only if updates are too frequent

## Requirements

- Python 3.8+
- `openai` package
- `pydantic` package
- `instructor` package

Install dependencies:

```bash
pip install openai pydantic instructor
```

## Additional Resources

- [Instructor Documentation](https://jxnl.github.io/instructor/)
- [Parallel Streaming RFC](../../plan/parallel_streaming_rfc.md)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference/streaming)

## License

These examples are part of the Instructor project and follow the same license.
