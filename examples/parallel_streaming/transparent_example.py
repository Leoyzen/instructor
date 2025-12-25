"""
Example demonstrating transparent parallel streaming support.

This example shows how to use stream=True with parallel tool modes
without manually creating ParallelBase instances. The conversion happens
automatically in handle_response_model().

Before this feature required:
    from instructor.dsl.parallel import OpenAIParallelModel
    parallel_base = OpenAIParallelModel(Iterable[Union[User, Task]])

Now it's transparent:
    response_model = Iterable[Union[User, Task]]  # Just add stream=True
"""

from collections.abc import Iterable
from typing import Union

from openai import OpenAI
from pydantic import BaseModel

import instructor


class User(BaseModel):
    """User information model."""
    name: str
    email: str


class Task(BaseModel):
    """Task information model."""
    title: str
    status: str


def example_1_basic_transparent_streaming():
    """Example 1: Basic transparent streaming with OpenAI."""
    print("\n=== Example 1: Basic Transparent Streaming ===")
    
    client = instructor.from_openai(
        OpenAI(),
        mode=instructor.Mode.PARALLEL_TOOLS,
    )

    # Transparent streaming: just add stream=True
    # No need to manually create ParallelBase instance
    stream = client.chat.completions.create(
        model="gpt-4",
        response_model=Iterable[Union[User, Task]],  # Standard type
        stream=True,  # Enable streaming
        messages=[
            {
                "role": "system",
                "content": "You must always use tools",
            },
            {
                "role": "user",
                "content": "Create a user named Alice and a task",
            },
        ],
    )

    # Process streaming results
    for result in stream:
        print(f"\n--- Update: tool_call_id={result.tool_call_id} ---")
        print(f"Completed: {len(result.completed)}")
        print(f"Partial: {len(result.partial)}")
        
        # Show all available models
        for model in result.get_all_models():
            if isinstance(model, User):
                print(f"  User: {model.name} ({model.email})")
            elif isinstance(model, Task):
                print(f"  Task: {model.title} ({model.status})")


def example_2_non_streaming():
    """Example 2: Non-streaming (backward compatible)."""
    print("\n=== Example 2: Non-Streaming (Backward Compatible) ===")
    
    client = instructor.from_openai(
        OpenAI(),
        mode=instructor.Mode.PARALLEL_TOOLS,
    )

    # Just remove stream=True - uses original behavior
    results = client.chat.completions.create(
        model="gpt-4",
        response_model=Iterable[Union[User, Task]],  # Standard type
        messages=[
            {
                "role": "system",
                "content": "You must always use tools",
            },
            {
                "role": "user",
                "content": "Create a user named Bob and a task",
            },
        ],
    )

    # Get all results at once
    for result in results:
        if isinstance(result, User):
            print(f"  User: {result.name} ({result.email})")
        elif isinstance(result, Task):
            print(f"  Task: {result.title} ({result.status})")


def example_3_manual_parallel_base():
    """Example 3: Manual ParallelBase (still works)."""
    print("\n=== Example 3: Manual ParallelBase (Still Supported) ===")
    
    from instructor.dsl.parallel import OpenAIParallelModel
    
    client = instructor.from_openai(
        OpenAI(),
        mode=instructor.Mode.PARALLEL_TOOLS,
    )

    # Old pattern: manually create ParallelBase
    parallel_base = OpenAIParallelModel(Iterable[Union[User, Task]])
    
    stream = client.chat.completions.create(
        model="gpt-4",
        response_model=parallel_base,  # Manual ParallelBase instance
        stream=True,
        messages=[
            {
                "role": "system",
                "content": "You must always use tools",
            },
            {
                "role": "user",
                "content": "Create a user named Charlie and a task",
            },
        ],
    )

    # Old pattern still works
    for result in stream:
        for model in result.get_all_models():
            if isinstance(model, User):
                print(f"  User: {model.name}")
            elif isinstance(model, Task):
                print(f"  Task: {model.title}")


def example_4_vertexai():
    """Example 4: VertexAI transparent streaming."""
    print("\n=== Example 4: VertexAI Transparent Streaming ===")
    
    client = instructor.from_provider(
        "vertexai/gemini-2.0-flash-exp",
        mode=instructor.Mode.VERTEXAI_PARALLEL_TOOLS,
    )

    # Transparent streaming with VertexAI
    stream = client.chat.completions.create(
        model="vertexai/gemini-2.0-flash-exp",
        response_model=Iterable[Union[User, Task]],  # Standard type
        stream=True,  # Auto-conversion to VertexAIParallelModel
        messages=[
            {
                "role": "user",
                "content": "Generate user and task",
            },
        ],
    )

    for result in stream:
        for model in result.get_all_models():
            if isinstance(model, User):
                print(f"  User: {model.name}")
            elif isinstance(model, Task):
                print(f"  Task: {model.title}")


def example_5_anthropic():
    """Example 5: Anthropic transparent streaming."""
    print("\n=== Example 5: Anthropic Transparent Streaming ===")
    
    client = instructor.from_provider(
        "anthropic/claude-3-sonnet-20240229",
        mode=instructor.Mode.ANTHROPIC_PARALLEL_TOOLS,
    )

    # Transparent streaming with Anthropic
    stream = client.chat.completions.create(
        model="anthropic/claude-3-sonnet-20240229",
        response_model=Iterable[Union[User, Task]],  # Standard type
        stream=True,  # Auto-conversion to AnthropicParallelModel
        messages=[
            {
                "role": "user",
                "content": "Create user and task",
            },
        ],
    )

    for result in stream:
        for model in result.get_all_models():
            if isinstance(model, User):
                print(f"  User: {model.name}")
            elif isinstance(model, Task):
                print(f"  Task: {model.title}")


if __name__ == "__main__":
    print("=" * 60)
    print("Transparent Parallel Streaming Examples")
    print("=" * 60)
    print()
    print("These examples demonstrate transparent streaming support for parallel modes.")
    print("You only need to add 'stream=True' - no manual ParallelBase creation required!")
    print()
    
    # Note: These examples require valid API keys
    print("Note: To run these examples, set OPENAI_API_KEY environment variable.")
    print("      Для VertexAI and Anthropic, set appropriate API keys.")
    print()

    # Run examples
    try:
        # Skip example that require API keys if not configured
        # example_1_basic_transparent_streaming()
        example_2_non_streaming()
        # example_3_manual_parallel_base()
        # example_4_vertexai()
        # example_5_anthropic()
        print("\n✓ Examples completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        print("\nSkipping examples that require API keys.")
        print("To use these examples:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Or modify examples to use mock client for testing")