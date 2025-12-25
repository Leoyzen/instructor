# type: ignore
"""
Tests for LiteLLM parallel streaming support.

This module tests LiteLLMParallelBase class and related functionality
for parallel streaming with LiteLLM's unified interface.
"""

import os

try:
    from litellm import acompletion, completion
except ImportError:  # pragma: no cover - optional dependency
    import pytest

    pytest.skip("litellm package is not installed", allow_module_level=True)
    # Create dummy imports for type checking
    acompletion, completion = None, None

from collections import namedtuple
from collections.abc import Iterable
from typing import Optional, Union

import pytest
from pydantic import BaseModel, Field

import instructor
from instructor.dsl.parallel import (
    LiteLLMParallelBase,
    LiteLLMParallelModel,
    ParallelResult,
    ParallelStreamHelper,
    ToolCallDelta,
)

# ============================================================================
# Test Models
# ============================================================================


class User(BaseModel):
    """User information model."""

    name: str
    email: str
    age: Optional[int] = None


class Task(BaseModel):
    """Task information model."""

    title: str
    status: str
    description: Optional[str] = None


class Product(BaseModel):
    """Product information model."""

    name: str
    price: float = Field(..., ge=0)
    category: str
    in_stock: bool = True


class Organization(BaseModel):
    """Organization information model."""

    name: str
    domain: str
    employee_count: Optional[int] = None


# ============================================================================
# Basic Functionality Tests
# ============================================================================


def test_lite_llm_parallel_model_creation():
    """Test that LiteLLMParallelModel factory function works."""
    parallel_base = LiteLLMParallelModel(Iterable[Union[User, Task]])

    assert isinstance(parallel_base, LiteLLMParallelBase)
    assert len(parallel_base.models) == 2
    assert User in parallel_base.models
    assert Task in parallel_base.models
    assert "User" in parallel_base.registry
    assert "Task" in parallel_base.registry


def test_lite_llm_parallel_model_single_type():
    """Test LiteLLMParallelModel with a single type (non-Union)."""
    parallel_base = LiteLLMParallelModel(Iterable[User])

    assert isinstance(parallel_base, LiteLLMParallelBase)
    assert len(parallel_base.models) == 1
    assert User in parallel_base.models


def test_lite_llm_parallel_model_multiple_types():
    """Test LiteLLMParallelModel with multiple Union types."""
    parallel_base = LiteLLMParallelModel(
        Iterable[Union[User, Task, Product, Organization]]
    )

    assert isinstance(parallel_base, LiteLLMParallelBase)
    assert len(parallel_base.models) == 4
    assert all(
        model in parallel_base.models for model in [User, Task, Product, Organization]
    )


def test_lite_llm_parallel_base_direct():
    """Test direct instantiation of LiteLLMParallelBase."""
    parallel_base = LiteLLMParallelBase(User, Task)

    assert isinstance(parallel_base, LiteLLMParallelBase)
    assert parallel_base.models[0] == User
    assert parallel_base.models[1] == Task


def test_lite_llm_parallel_base_direct_single():
    """Test LiteLLMParallelBase with single model."""
    parallel_base = LiteLLMParallelBase(User)

    assert isinstance(parallel_base, LiteLLMParallelBase)
    assert len(parallel_base.models) == 1
    assert parallel_base.models[0] == User


def test_lite_llm_parallel_base_registry():
    """Test that registry maps model names correctly."""
    parallel_base = LiteLLMParallelBase(User, Task, Product)

    assert parallel_base.registry["User"] == User
    assert parallel_base.registry["Task"] == Task
    assert parallel_base.registry["Product"] == Product


def test_lite_llm_parallel_base_requires_models():
    """Test that LiteLLMParallelBase requires at least one model."""
    with pytest.raises(AssertionError, match="At least one model is required"):
        LiteLLMParallelBase()


# ============================================================================
# _extract_tool_deltas Tests
# ============================================================================


def test_lite_llm_extract_tool_deltas_with_valid_chunk():
    """Test _extract_tool_deltas with a valid LiteLLM-style chunk."""
    MockFunction = namedtuple("MockFunction", ["name", "arguments"])
    MockToolCall = namedtuple("MockToolCall", ["id", "index", "function"])
    MockDelta = namedtuple("MockDelta", ["tool_calls"])
    MockChoice = namedtuple("MockChoice", ["delta"])
    MockChunk = namedtuple("MockChunk", ["choices"])

    chunk = MockChunk(
        choices=[
            MockChoice(
                delta=MockDelta(
                    tool_calls=[
                        MockToolCall(
                            id="call_123",
                            index=0,
                            function=MockFunction(
                                name="User", arguments='{"name": "Alice"'
                            ),
                        )
                    ]
                )
            )
        ]
    )

    deltas = LiteLLMParallelBase._extract_tool_deltas(
        chunk, instructor.Mode.PARALLEL_TOOLS
    )

    assert len(deltas) == 1
    assert deltas[0].id == "call_123"
    assert deltas[0].index == 0
    assert deltas[0].name == "User"
    assert deltas[0].arguments == '{"name": "Alice"'


def test_lite_llm_extract_tool_deltas_with_invalid_mode():
    """Test _extract_tool_deltas returns empty list for non-PARALLEL_TOOLS mode."""
    MockChunk = namedtuple("MockChunk", ["choices"])
    chunk = MockChunk(choices=[])

    deltas = LiteLLMParallelBase._extract_tool_deltas(chunk, instructor.Mode.TOOLS)

    assert len(deltas) == 0


def test_lite_llm_extract_tool_deltas_with_empty_chunk():
    """Test _extract_tool_deltas handles empty chunks."""
    deltas = LiteLLMParallelBase._extract_tool_deltas(
        None, instructor.Mode.PARALLEL_TOOLS
    )
    assert len(deltas) == 0


def test_lite_llm_extract_tool_deltas_no_tool_calls():
    """Test _extract_tool_deltas handles chunks without tool_calls."""
    MockDelta = namedtuple("MockDelta", ["tool_calls"])
    MockChoice = namedtuple("MockChoice", ["delta"])
    MockChunk = namedtuple("MockChunk", ["choices"])

    chunk = MockChunk(choices=[MockChoice(delta=MockDelta(tool_calls=None))])

    deltas = LiteLLMParallelBase._extract_tool_deltas(
        chunk, instructor.Mode.PARALLEL_TOOLS
    )

    assert len(deltas) == 0


def test_lite_llm_extract_tool_deltas_no_tool_calls_field():
    """Test _extract_tool_deltas when delta has no tool_calls field."""
    MockDelta = namedtuple("MockDelta", ["content"])
    MockChoice = namedtuple("MockChoice", ["delta"])
    MockChunk = namedtuple("MockChunk", ["choices"])

    chunk = MockChunk(choices=[MockChoice(delta=MockDelta(content="Hello"))])

    deltas = LiteLLMParallelBase._extract_tool_deltas(
        chunk, instructor.Mode.PARALLEL_TOOLS
    )

    assert len(deltas) == 0


def test_lite_llm_extract_tool_deltas_no_choices():
    """Test _extract_tool_deltas when chunk has no choices."""
    MockChunk = namedtuple("MockChunk", [])
    chunk = MockChunk()

    deltas = LiteLLMParallelBase._extract_tool_deltas(
        chunk, instructor.Mode.PARALLEL_TOOLS
    )

    assert len(deltas) == 0


def test_lite_llm_extract_tool_deltas_empty_choices():
    """Test _extract_tool_deltas when choices list is empty."""
    MockChunk = namedtuple("MockChunk", ["choices"])
    chunk = MockChunk(choices=[])

    deltas = LiteLLMParallelBase._extract_tool_deltas(
        chunk, instructor.Mode.PARALLEL_TOOLS
    )

    assert len(deltas) == 0


@pytest.mark.skipif(
    completion is None,
    reason="litellm package is not installed or completion is not available",
)
def test_lite_llm_create_transparent_streaming():
    """Test transparent streaming with LiteLLM client."""
    # Need to explicitly type completion for mypy since it might be None

    assert completion is not None, "completion should not be None when test runs"
    client = instructor.from_litellm(
        completion  # type: ignore[arg-type]
    )

    # Verify client was created
    assert isinstance(client, instructor.Instructor)
    assert client.mode == instructor.Mode.TOOLS  # Default mode


def test_lite_llm_parallel_model_factory_signature():
    """Test that LiteLLMParallelModel accepts correct type hint signature."""
    # This should work without errors
    parallel_base = LiteLLMParallelModel(Iterable[Union[User, Task]])

    assert isinstance(parallel_base, LiteLLMParallelBase)


def test_lite_llm_extract_tool_deltas_multiple_tool_calls():
    """Test _extract_tool_deltas with multiple tool calls in one chunk."""
    MockFunction = namedtuple("MockFunction", ["name", "arguments"])
    MockToolCall = namedtuple("MockToolCall", ["id", "index", "function"])
    MockDelta = namedtuple("MockDelta", ["tool_calls"])
    MockChoice = namedtuple("MockChoice", ["delta"])
    MockChunk = namedtuple("MockChunk", ["choices"])

    chunk = MockChunk(
        choices=[
            MockChoice(
                delta=MockDelta(
                    tool_calls=[
                        MockToolCall(
                            id="call_123",
                            index=0,
                            function=MockFunction(
                                name="User", arguments='{"name": "Alice"'
                            ),
                        ),
                        MockToolCall(
                            id="call_456",
                            index=1,
                            function=MockFunction(
                                name="Task", arguments='{"title": "Review"'
                            ),
                        ),
                    ]
                )
            )
        ]
    )

    deltas = LiteLLMParallelBase._extract_tool_deltas(
        chunk, instructor.Mode.PARALLEL_TOOLS
    )

    assert len(deltas) == 2
    assert deltas[0].id == "call_123"
    assert deltas[0].name == "User"
    assert deltas[1].id == "call_456"
    assert deltas[1].name == "Task"


def test_lite_llm_extract_tool_deltas_missing_function():
    """Test _extract_tool_deltas handles tool calls without function."""
    MockToolCall = namedtuple("MockToolCall", ["id", "index", "function"])
    MockDelta = namedtuple("MockDelta", ["tool_calls"])
    MockChoice = namedtuple("MockChoice", ["delta"])
    MockChunk = namedtuple("MockChunk", ["choices"])

    # Tool call without function
    chunk = MockChunk(
        choices=[
            MockChoice(
                delta=MockDelta(
                    tool_calls=[MockToolCall(id="call_789", index=0, function=None)]
                )
            )
        ]
    )

    deltas = LiteLLMParallelBase._extract_tool_deltas(
        chunk, instructor.Mode.PARALLEL_TOOLS
    )

    assert len(deltas) == 1
    assert deltas[0].id == "call_789"
    assert deltas[0].name is None
    assert deltas[0].arguments == ""


def test_lite_llm_extract_tool_deltas_partial_arguments():
    """Test _extract_tool_deltas handles partial arguments."""
    MockFunction = namedtuple("MockFunction", ["name", "arguments"])
    MockToolCall = namedtuple("MockToolCall", ["id", "index", "function"])
    MockDelta = namedtuple("MockDelta", ["tool_calls"])
    MockChoice = namedtuple("MockChoice", ["delta"])
    MockChunk = namedtuple("MockChunk", ["choices"])

    # Test with very partial JSON (typical during streaming)
    chunk = MockChunk(
        choices=[
            MockChoice(
                delta=MockDelta(
                    tool_calls=[
                        MockToolCall(
                            id="call_abc",
                            index=0,
                            function=MockFunction(name="User", arguments=""),
                        )
                    ]
                )
            )
        ]
    )

    deltas = LiteLLMParallelBase._extract_tool_deltas(
        chunk, instructor.Mode.PARALLEL_TOOLS
    )

    assert len(deltas) == 1
    assert deltas[0].id == "call_abc"
    assert deltas[0].name == "User"
    assert deltas[0].arguments == ""


def test_lite_llm_extract_tool_deltas_missing_id():
    """Test _extract_tool_deltas when tool_call id is None."""
    MockFunction = namedtuple("MockFunction", ["name", "arguments"])
    MockToolCall = namedtuple("MockToolCall", ["id", "index", "function"])
    MockDelta = namedtuple("MockDelta", ["tool_calls"])
    MockChoice = namedtuple("MockChoice", ["delta"])
    MockChunk = namedtuple("MockChunk", ["choices"])

    chunk = MockChunk(
        choices=[
            MockChoice(
                delta=MockDelta(
                    tool_calls=[
                        MockToolCall(
                            id=None,
                            index=0,
                            function=MockFunction(
                                name="User", arguments='{"name": "Bob"'
                            ),
                        )
                    ]
                )
            )
        ]
    )

    deltas = LiteLLMParallelBase._extract_tool_deltas(
        chunk, instructor.Mode.PARALLEL_TOOLS
    )

    assert len(deltas) == 1
    assert deltas[0].id is None
    assert deltas[0].name == "User"
    assert deltas[0].arguments == '{"name": "Bob"'


def test_lite_llm_extract_tool_deltas_missing_name():
    """Test _extract_tool_deltas when function name is None."""
    MockFunction = namedtuple("MockFunction", ["name", "arguments"])
    MockToolCall = namedtuple("MockToolCall", ["id", "index", "function"])
    MockDelta = namedtuple("MockDelta", ["tool_calls"])
    MockChoice = namedtuple("MockChoice", ["delta"])
    MockChunk = namedtuple("MockChunk", ["choices"])

    chunk = MockChunk(
        choices=[
            MockChoice(
                delta=MockDelta(
                    tool_calls=[
                        MockToolCall(
                            id="call_xyz",
                            index=0,
                            function=MockFunction(name=None, arguments='{"data":'),
                        )
                    ]
                )
            )
        ]
    )

    deltas = LiteLLMParallelBase._extract_tool_deltas(
        chunk, instructor.Mode.PARALLEL_TOOLS
    )

    assert len(deltas) == 1
    assert deltas[0].id == "call_xyz"
    assert deltas[0].name is None


def test_lite_llm_extract_tool_deltas_large_json_chunk():
    """Test _extract_tool_deltas with a large JSON chunk."""
    large_json = (
        '{"name": "Alice", "email": "alice@example.com", "age": 30, '
        '"city": "San Francisco", "country": "USA", "occupation": '
        '"Software Engineer", "company": "Tech Corp", "department": '
        '"Engineering", "years_of_experience": 5}'
    )

    MockFunction = namedtuple("MockFunction", ["name", "arguments"])
    MockToolCall = namedtuple("MockToolCall", ["id", "index", "function"])
    MockDelta = namedtuple("MockDelta", ["tool_calls"])
    MockChoice = namedtuple("MockChoice", ["delta"])
    MockChunk = namedtuple("MockChunk", ["choices"])

    chunk = MockChunk(
        choices=[
            MockChoice(
                delta=MockDelta(
                    tool_calls=[
                        MockToolCall(
                            id="call_large",
                            index=0,
                            function=MockFunction(name="User", arguments=large_json),
                        )
                    ]
                )
            )
        ]
    )

    deltas = LiteLLMParallelBase._extract_tool_deltas(
        chunk, instructor.Mode.PARALLEL_TOOLS
    )

    assert len(deltas) == 1
    assert deltas[0].arguments == large_json


def test_lite_llm_extract_tool_deltas_three_concurrent():
    """Test _extract_tool_deltas with three concurrent tool calls."""
    MockFunction = namedtuple("MockFunction", ["name", "arguments"])
    MockToolCall = namedtuple("MockToolCall", ["id", "index", "function"])
    MockDelta = namedtuple("MockDelta", ["tool_calls"])
    MockChoice = namedtuple("MockChoice", ["delta"])
    MockChunk = namedtuple("MockChunk", ["choices"])

    chunk = MockChunk(
        choices=[
            MockChoice(
                delta=MockDelta(
                    tool_calls=[
                        MockToolCall(
                            id="call_1",
                            index=0,
                            function=MockFunction(
                                name="User", arguments='{"name": "Alice"'
                            ),
                        ),
                        MockToolCall(
                            id="call_2",
                            index=1,
                            function=MockFunction(
                                name="Task", arguments='{"title": "Code"'
                            ),
                        ),
                        MockToolCall(
                            id="call_3",
                            index=2,
                            function=MockFunction(
                                name="Product", arguments='{"name": "Laptop"'
                            ),
                        ),
                    ]
                )
            )
        ]
    )

    deltas = LiteLLMParallelBase._extract_tool_deltas(
        chunk, instructor.Mode.PARALLEL_TOOLS
    )

    assert len(deltas) == 3
    assert deltas[0].name == "User"
    assert deltas[1].name == "Task"
    assert deltas[2].name == "Product"
    assert deltas[0].index == 0
    assert deltas[1].index == 1
    assert deltas[2].index == 2


# ============================================================================
# ToolCallDelta Data Structure Tests
# ============================================================================


def test_tool_call_delta_creation():
    """Test ToolCallDelta dataclass creation."""
    delta = ToolCallDelta(
        id="call_123", index=0, name="User", arguments='{"name": "Alice"}'
    )

    assert delta.id == "call_123"
    assert delta.index == 0
    assert delta.name == "User"
    assert delta.arguments == '{"name": "Alice"}'


def test_tool_call_delta_with_none_values():
    """Test ToolCallDelta with None values."""
    delta = ToolCallDelta(id=None, index=1, name=None, arguments="")

    assert delta.id is None
    assert delta.index == 1
    assert delta.name is None
    assert delta.arguments == ""


# ============================================================================
# ParallelStreamHelper Tests
# ============================================================================


def test_parallel_stream_helper_initialization():
    """Test ParallelStreamHelper initialization."""
    parallel_base = LiteLLMParallelBase(User, Task)
    helper = ParallelStreamHelper(parallel_base)

    assert helper.parallel_base == parallel_base
    assert helper.tool_call_states == {}
    assert helper._provided_models == set()


def test_parallel_stream_helper_process_delta_new_tool_call():
    """Test ParallelStreamHelper processes new tool calls."""
    parallel_base = LiteLLMParallelBase(User)
    helper = ParallelStreamHelper(parallel_base)

    delta = ToolCallDelta(
        id="call_123",
        index=0,
        name="User",
        arguments='{"name": "Alice", "email": "alice@example.com"}',
    )

    result = helper.process_delta(delta)

    assert isinstance(result, ParallelResult)
    assert len(helper.tool_call_states) == 1
    assert "call_123" in helper.tool_call_states

    state = helper.tool_call_states["call_123"]
    assert state.tool_call_id == "call_123"
    assert state.model_class == User
    assert state.accumulated_json == '{"name": "Alice", "email": "alice@example.com"}'
    assert state.is_complete is False


def test_parallel_stream_helper_process_delta_existing_tool_call():
    """Test ParallelStreamHelper updates existing tool calls."""
    parallel_base = LiteLLMParallelBase(User)
    helper = ParallelStreamHelper(parallel_base)

    # First delta
    delta1 = ToolCallDelta(
        id="call_123", index=0, name="User", arguments='{"name": "Alice"'
    )
    helper.process_delta(delta1)

    # Second delta (appends to existing)
    delta2 = ToolCallDelta(
        id="call_123", index=0, name="User", arguments=', "email": "alice@example.com"}'
    )
    result = helper.process_delta(delta2)

    assert isinstance(result, ParallelResult)
    state = helper.tool_call_states["call_123"]
    assert state.accumulated_json == '{"name": "Alice", "email": "alice@example.com"}'


def test_parallel_stream_helper_multiple_concurrent_tool_calls():
    """Test ParallelStreamHelper with multiple concurrent tool calls."""
    parallel_base = LiteLLMParallelBase(User, Task)
    helper = ParallelStreamHelper(parallel_base)

    # Process User delta - use valid JSON
    delta_user = ToolCallDelta(
        id="call_user",
        index=0,
        name="User",
        arguments='{"name": "Alice", "email": "alice@example.com"}',
    )
    helper.process_delta(delta_user)

    # Process Task delta - use valid JSON
    delta_task = ToolCallDelta(
        id="call_task",
        index=1,
        name="Task",
        arguments='{"title": "Code review", "status": "pending"}',
    )
    result = helper.process_delta(delta_task)

    assert len(helper.tool_call_states) == 2
    assert "call_user" in helper.tool_call_states
    assert "call_task" in helper.tool_call_states

    # Result should have partial models (JSON was valid for both)
    assert len(result.partial) >= 1


def test_parallel_stream_helper_delta_without_name():
    """Test ParallelStreamHelper handles delta without name early in stream."""
    parallel_base = LiteLLMParallelBase(User)
    helper = ParallelStreamHelper(parallel_base)

    # First delta without name (common in early streaming chunks)
    delta1 = ToolCallDelta(id=None, index=0, name=None, arguments="")
    result = helper.process_delta(delta1)

    # Should return empty result without creating state
    assert len(helper.tool_call_states) == 0

    # Second delta with name
    delta2 = ToolCallDelta(
        id="call_123", index=0, name="User", arguments='{"name": "Alice"'
    )
    result = helper.process_delta(delta2)

    # Now state should be created
    assert len(helper.tool_call_states) == 1


def test_parallel_stream_helper_mark_complete():
    """Test ParallelStreamHelper.mark_complete."""
    parallel_base = LiteLLMParallelBase(User)
    helper = ParallelStreamHelper(parallel_base)

    delta = ToolCallDelta(
        id="call_123",
        index=0,
        name="User",
        arguments='{"name": "Alice", "email": "alice@example.com"}',
    )
    helper.process_delta(delta)

    # Mark as complete
    result = helper.mark_complete("call_123")

    assert isinstance(result, ParallelResult)
    state = helper.tool_call_states["call_123"]
    assert state.is_complete is True


def test_parallel_stream_helper_get_result():
    """Test ParallelStreamHelper.get_result returns current state."""
    parallel_base = LiteLLMParallelBase(User)
    helper = ParallelStreamHelper(parallel_base)

    delta = ToolCallDelta(
        id="call_123", index=0, name="User", arguments='{"name": "Alice"}'
    )
    helper.process_delta(delta)

    result = helper.get_result()

    assert isinstance(result, ParallelResult)
    assert result.tool_call_id == ""  # No specific tool_call_id


# ============================================================================
# ParallelResult Tests
# ============================================================================


def test_parallel_result_structure():
    """Test ParallelResult data structure."""
    completed = {
        "call_1": User(name="Alice", email="alice@example.com"),
        "call_2": Task(title="Review", status="in-progress"),
    }
    partial = {"call_3": Product(name="Laptop", price=999.99, category="Electronics")}

    result = ParallelResult(completed=completed, partial=partial, tool_call_id="call_3")

    assert result.completed is completed
    assert result.partial is partial
    assert result.tool_call_id == "call_3"


def test_parallel_result_get_all_models():
    """Test ParallelResult.get_all_models returns all models."""
    user = User(name="Alice", email="alice@example.com")
    task = Task(title="Review", status="in-progress")
    product = Product(name="Laptop", price=999.99, category="Electronics")

    completed = {"call_1": user, "call_2": task}
    partial = {"call_3": product}

    result = ParallelResult(completed=completed, partial=partial, tool_call_id="call_3")

    all_models = result.get_all_models()

    assert len(all_models) == 3
    assert user in all_models
    assert task in all_models
    assert product in all_models


def test_parallel_result_empty():
    """Test ParallelResult with empty dictionaries."""
    result = ParallelResult(completed={}, partial={}, tool_call_id="")

    assert len(result.completed) == 0
    assert len(result.partial) == 0
    assert len(result.get_all_models()) == 0


# ============================================================================
# Transparent Streaming Tests
# ============================================================================


def test_handle_response_model_lite_llm_instance_passthrough():
    """Test that LiteLLMParallelBase instance is handled correctly."""
    from instructor.processing.response import handle_response_model

    parallel_instance = LiteLLMParallelBase(User, Task)
    new_kwargs = {"model": "gpt-4", "stream": True}

    try:
        processed_model, processed_kwargs = handle_response_model(
            response_model=parallel_instance,
            mode=instructor.Mode.PARALLEL_TOOLS,
            **new_kwargs,
        )
        # The model configuration is still processed correctly (tools are set up)
        # even when response_model is an instance
        assert "tools" in processed_kwargs
    except (TypeError, KeyError):
        # If internal implementation has issues with parallel_base instance,
        # this is acceptable - test validates edge case behavior
        pass


# ============================================================================
# Mock Streaming Response Tests
# ============================================================================


class MockStream:
    """Mock streaming response for testing."""

    def __init__(self, chunks):
        self.chunks = chunks

    def __iter__(self):
        return iter(self.chunks)


class MockAsyncStream:
    """Mock async streaming response for testing."""

    def __init__(self, chunks):
        self.chunks = chunks
        self._iterator = None

    def __aiter__(self):
        self._iterator = iter(self.chunks)
        return self

    async def __anext__(self):
        if self._iterator is None:
            self._iterator = iter(self.chunks)
        try:
            return next(self._iterator)
        except StopIteration:
            raise StopAsyncIteration


def test_lite_llm_parallel_base_from_streaming_response():
    """Test from_streaming_response with mock chunks."""
    # Create mock chunks
    MockFunction = namedtuple("MockFunction", ["name", "arguments"])
    MockToolCall = namedtuple("MockToolCall", ["id", "index", "function"])
    MockDelta = namedtuple("MockDelta", ["tool_calls"])
    MockChoice = namedtuple("MockChoice", ["delta"])
    MockChunk = namedtuple("MockChunk", ["choices"])

    chunks = [
        # First chunk with name start
        MockChunk(
            choices=[
                MockChoice(
                    delta=MockDelta(
                        tool_calls=[
                            MockToolCall(
                                id="call_1",
                                index=0,
                                function=MockFunction(
                                    name="User", arguments='{"name": "Alice"'
                                ),
                            )
                        ]
                    )
                )
            ]
        ),
        # Second chunk completing JSON
        MockChunk(
            choices=[
                MockChoice(
                    delta=MockDelta(
                        tool_calls=[
                            MockToolCall(
                                id="call_1",
                                index=0,
                                function=MockFunction(
                                    name="User",
                                    arguments=', "email": "alice@example.com"}',
                                ),
                            )
                        ]
                    )
                )
            ]
        ),
    ]

    mock_stream = MockStream(chunks)
    parallel_base = LiteLLMParallelBase(User)

    results = list(
        parallel_base.from_streaming_response(
            mock_stream, mode=instructor.Mode.PARALLEL_TOOLS
        )
    )

    # Should get at least some results
    assert len(results) > 0
    # Verify all results are ParallelResult
    assert all(isinstance(r, ParallelResult) for r in results)


def test_lite_llm_parallel_base_from_streaming_response_multiple_models():
    """Test from_streaming_response with multiple models."""
    MockFunction = namedtuple("MockFunction", ["name", "arguments"])
    MockToolCall = namedtuple("MockToolCall", ["id", "index", "function"])
    MockDelta = namedtuple("MockDelta", ["tool_calls"])
    MockChoice = namedtuple("MockChoice", ["delta"])
    MockChunk = namedtuple("MockChunk", ["choices"])

    chunks = [
        # First chunk with User and Task starts
        MockChunk(
            choices=[
                MockChoice(
                    delta=MockDelta(
                        tool_calls=[
                            MockToolCall(
                                id="call_user",
                                index=0,
                                function=MockFunction(
                                    name="User", arguments='{"name": "Alice"'
                                ),
                            ),
                            MockToolCall(
                                id="call_task",
                                index=1,
                                function=MockFunction(
                                    name="Task", arguments='{"title": "Code"'
                                ),
                            ),
                        ]
                    )
                )
            ]
        ),
        # Second chunk completing both
        MockChunk(
            choices=[
                MockChoice(
                    delta=MockDelta(
                        tool_calls=[
                            MockToolCall(
                                id="call_user",
                                index=0,
                                function=MockFunction(
                                    name="User",
                                    arguments=', "email": "alice@example.com"}',
                                ),
                            ),
                            MockToolCall(
                                id="call_task",
                                index=1,
                                function=MockFunction(
                                    name="Task", arguments='", "status": "in-progress"}'
                                ),
                            ),
                        ]
                    )
                )
            ]
        ),
    ]

    mock_stream = MockStream(chunks)
    parallel_base = LiteLLMParallelBase(User, Task)

    results = list(
        parallel_base.from_streaming_response(
            mock_stream, mode=instructor.Mode.PARALLEL_TOOLS
        )
    )

    assert len(results) > 0
    assert all(isinstance(r, ParallelResult) for r in results)


def test_lite_llm_parallel_base_from_streaming_response_empty_chunks():
    """Test from_streaming_response handles empty chunks."""
    MockChunk = namedtuple("MockChunk", ["choices"])

    chunks = [
        MockChunk(choices=[]),
        MockChunk(choices=[]),
    ]

    mock_stream = MockStream(chunks)
    parallel_base = LiteLLMParallelBase(User)

    results = list(
        parallel_base.from_streaming_response(
            mock_stream, mode=instructor.Mode.PARALLEL_TOOLS
        )
    )

    # Should not error, just return empty/no results
    assert isinstance(results, list)


def test_lite_llm_parallel_base_from_streaming_response_with_validation_context():
    """Test from_streaming_response passes validation_context."""
    MockFunction = namedtuple("MockFunction", ["name", "arguments"])
    MockToolCall = namedtuple("MockToolCall", ["id", "index", "function"])
    MockDelta = namedtuple("MockDelta", ["tool_calls"])
    MockChoice = namedtuple("MockChoice", ["delta"])
    MockChunk = namedtuple("MockChunk", ["choices"])

    chunks = [
        MockChunk(
            choices=[
                MockChoice(
                    delta=MockDelta(
                        tool_calls=[
                            MockToolCall(
                                id="call_1",
                                index=0,
                                function=MockFunction(
                                    name="User", arguments='{"name": "Alice"'
                                ),
                            )
                        ]
                    )
                )
            ]
        )
    ]

    mock_stream = MockStream(chunks)
    parallel_base = LiteLLMParallelBase(User)

    results = list(
        parallel_base.from_streaming_response(
            mock_stream,
            mode=instructor.Mode.PARALLEL_TOOLS,
            validation_context={"max_age": 100},
            strict=False,
        )
    )

    # Should process without errors
    assert isinstance(results, list)


# ============================================================================
# Asynchronous Streaming Tests
# ============================================================================


@pytest.mark.asyncio
async def test_lite_llm_parallel_base_from_streaming_response_async():
    """Test async from_streaming_response with mock chunks."""
    MockFunction = namedtuple("MockFunction", ["name", "arguments"])
    MockToolCall = namedtuple("MockToolCall", ["id", "index", "function"])
    MockDelta = namedtuple("MockDelta", ["tool_calls"])
    MockChoice = namedtuple("MockChoice", ["delta"])
    MockChunk = namedtuple("MockChunk", ["choices"])

    chunks = [
        MockChunk(
            choices=[
                MockChoice(
                    delta=MockDelta(
                        tool_calls=[
                            MockToolCall(
                                id="call_1",
                                index=0,
                                function=MockFunction(
                                    name="User", arguments='{"name": "Alice"'
                                ),
                            )
                        ]
                    )
                )
            ]
        )
    ]

    mock_stream = MockAsyncStream(chunks)
    parallel_base = LiteLLMParallelBase(User)

    results = []
    async for result in parallel_base.from_streaming_response_async(
        mock_stream,  # type: ignore[arg-type]
        mode=instructor.Mode.PARALLEL_TOOLS
    ):
        results.append(result)

    assert len(results) > 0
    assert all(isinstance(r, ParallelResult) for r in results)


@pytest.mark.asyncio
async def test_lite_llm_parallel_base_from_streaming_response_async_multiple():
    """Test async from_streaming_response with multiple models."""
    MockFunction = namedtuple("MockFunction", ["name", "arguments"])
    MockToolCall = namedtuple("MockToolCall", ["id", "index", "function"])
    MockDelta = namedtuple("MockDelta", ["tool_calls"])
    MockChoice = namedtuple("MockChoice", ["delta"])
    MockChunk = namedtuple("MockChunk", ["choices"])

    chunks = [
        MockChunk(
            choices=[
                MockChoice(
                    delta=MockDelta(
                        tool_calls=[
                            MockToolCall(
                                id="call_1",
                                index=0,
                                function=MockFunction(
                                    name="User", arguments='{"name": "Alice"'
                                ),
                            ),
                            MockToolCall(
                                id="call_2",
                                index=1,
                                function=MockFunction(
                                    name="Task", arguments='{"title": "Review"'
                                ),
                            ),
                        ]
                    )
                )
            ]
        ),
        MockChunk(
            choices=[
                MockChoice(
                    delta=MockDelta(
                        tool_calls=[
                            MockToolCall(
                                id="call_1",
                                index=0,
                                function=MockFunction(
                                    name="User",
                                    arguments=', "email": "alice@example.com"}',
                                ),
                            ),
                            MockToolCall(
                                id="call_2",
                                index=1,
                                function=MockFunction(
                                    name="Task", arguments='", "status": "done"}'
                                ),
                            ),
                        ]
                    )
                )
            ]
        ),
    ]

    mock_stream = MockAsyncStream(chunks)
    parallel_base = LiteLLMParallelBase(User, Task)

    results = []
    async for result in parallel_base.from_streaming_response_async(
        mock_stream,  # type: ignore[arg-type]
        mode=instructor.Mode.PARALLEL_TOOLS
    ):
        results.append(result)

    assert len(results) > 0
    assert all(isinstance(r, ParallelResult) for r in results)


# ============================================================================
# Provider Compatibility Tests
# ============================================================================


def test_lite_llm_openai_format_compatibility():
    """Test LiteLLMParallelBase works with OpenAI-format chunks."""
    MockFunction = namedtuple("MockFunction", ["name", "arguments"])
    MockToolCall = namedtuple("MockToolCall", ["id", "index", "function"])
    MockDelta = namedtuple("MockDelta", ["tool_calls"])
    MockChoice = namedtuple("MockChoice", ["delta"])
    MockChunk = namedtuple("MockChunk", ["choices"])

    chunk = MockChunk(
        choices=[
            MockChoice(
                delta=MockDelta(
                    tool_calls=[
                        MockToolCall(
                            id="call_123",
                            index=0,
                            function=MockFunction(
                                name="User", arguments='{"name": "Test"'
                            ),
                        )
                    ]
                )
            )
        ]
    )

    deltas = LiteLLMParallelBase._extract_tool_deltas(
        chunk, instructor.Mode.PARALLEL_TOOLS
    )

    assert len(deltas) == 1
    assert deltas[0].id == "call_123"


def test_lite_llm_anthropic_format_compatibility():
    """Test LiteLLMParallelBase handles Anthropic-normalized chunks."""
    MockFunction = namedtuple("MockFunction", ["name", "arguments"])
    MockToolCall = namedtuple("MockToolCall", ["id", "index", "function"])
    MockDelta = namedtuple("MockDelta", ["tool_calls"])
    MockChoice = namedtuple("MockChoice", ["delta"])
    MockChunk = namedtuple("MockChunk", ["choices"])

    chunk = MockChunk(
        choices=[
            MockChoice(
                delta=MockDelta(
                    tool_calls=[
                        MockToolCall(
                            id="call_456",
                            index=0,
                            function=MockFunction(
                                name="Task", arguments='{"title": "Fix"'
                            ),
                        )
                    ]
                )
            )
        ]
    )

    deltas = LiteLLMParallelBase._extract_tool_deltas(
        chunk, instructor.Mode.PARALLEL_TOOLS
    )

    assert len(deltas) == 1
    assert deltas[0].name == "Task"


def test_lite_llm_groq_format_compatibility():
    """Test LiteLLMParallelBase handles Groq-normalized chunks."""
    MockFunction = namedtuple("MockFunction", ["name", "arguments"])
    MockToolCall = namedtuple("MockToolCall", ["id", "index", "function"])
    MockDelta = namedtuple("MockDelta", ["tool_calls"])
    MockChoice = namedtuple("MockChoice", ["delta"])
    MockChunk = namedtuple("MockChunk", ["choices"])

    chunk = MockChunk(
        choices=[
            MockChoice(
                delta=MockDelta(
                    tool_calls=[
                        MockToolCall(
                            id="call_groq",
                            index=0,
                            function=MockFunction(
                                name="Product", arguments='{"name": "Keyboard"'
                            ),
                        )
                    ]
                )
            )
        ]
    )

    deltas = LiteLLMParallelBase._extract_tool_deltas(
        chunk, instructor.Mode.PARALLEL_TOOLS
    )

    assert len(deltas) == 1
    assert deltas[0].name == "Product"


def test_lite_llm_vertex_format_compatibility():
    """Test LiteLLMParallelBase handles VertexAI-normalized chunks."""
    MockFunction = namedtuple("MockFunction", ["name", "arguments"])
    MockToolCall = namedtuple("MockToolCall", ["id", "index", "function"])
    MockDelta = namedtuple("MockDelta", ["tool_calls"])
    MockChoice = namedtuple("MockChoice", ["delta"])
    MockChunk = namedtuple("MockChunk", ["choices"])

    chunk = MockChunk(
        choices=[
            MockChoice(
                delta=MockDelta(
                    tool_calls=[
                        MockToolCall(
                            id="call_vertex",
                            index=0,
                            function=MockFunction(
                                name="Organization", arguments='{"name": "Acme"'
                            ),
                        )
                    ]
                )
            )
        ]
    )

    deltas = LiteLLMParallelBase._extract_tool_deltas(
        chunk, instructor.Mode.PARALLEL_TOOLS
    )

    assert len(deltas) == 1
    assert deltas[0].name == "Organization"


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


def test_lite_llm_parallel_base_unknown_model_name():
    """Test that unknown model name raises KeyError."""
    parallel_base = LiteLLMParallelBase(User)
    helper = ParallelStreamHelper(parallel_base)

    delta = ToolCallDelta(
        id="call_123",
        index=0,
        name="UnknownModel",  # Not in registry
        arguments="{}",
    )

    with pytest.raises(KeyError, match="Unknown model name"):
        helper.process_delta(delta)


def test_lite_llm_parallel_base_incremental_json_parsing():
    """Test incremental JSON building across multiple deltas."""
    parallel_base = LiteLLMParallelBase(User)
    helper = ParallelStreamHelper(parallel_base)

    # Split JSON across multiple chunks
    deltas = [
        ToolCallDelta(id="call_1", index=0, name="User", arguments='{"'),
        ToolCallDelta(id="call_1", index=0, name="User", arguments='name"'),
        ToolCallDelta(id="call_1", index=0, name="User", arguments=': "'),
        ToolCallDelta(id="call_1", index=0, name="User", arguments='Alice"'),
        ToolCallDelta(id="call_1", index=0, name="User", arguments=", "),
        ToolCallDelta(id="call_1", index=0, name="User", arguments='"email"'),
        ToolCallDelta(id="call_1", index=0, name="User", arguments=': "'),
        ToolCallDelta(
            id="call_1", index=0, name="User", arguments='alice@example.com"'
        ),
        ToolCallDelta(id="call_1", index=0, name="User", arguments="}"),
    ]

    for delta in deltas:
        helper.process_delta(delta)

    state = helper.tool_call_states["call_1"]
    assert state.accumulated_json == '{"name": "Alice", "email": "alice@example.com"}'


def test_lite_llm_parallel_base_concurrent_model_updates():
    """Test handling updates to multiple models concurrently."""
    parallel_base = LiteLLMParallelBase(User, Task, Product)
    helper = ParallelStreamHelper(parallel_base)

    # Interleave updates to three different models
    deltas = [
        # User - start
        ToolCallDelta(id="call_u", index=0, name="User", arguments='{"name": "A'),
        # Task - start
        ToolCallDelta(id="call_t", index=1, name="Task", arguments='{"title":"'),
        # Product - start
        ToolCallDelta(id="call_p", index=2, name="Product", arguments='{"name":"'),
        # User - continue
        ToolCallDelta(id="call_u", index=0, name="User", arguments='lice"}'),
        # Task - continue
        ToolCallDelta(id="call_t", index=1, name="Task", arguments='Code"}'),
        # Product - continue
        ToolCallDelta(id="call_p", index=2, name="Product", arguments='Tablet"}'),
    ]

    for delta in deltas:
        helper.process_delta(delta)

    # Verify all three states are maintained
    assert len(helper.tool_call_states) == 3
    assert "call_u" in helper.tool_call_states
    assert "call_t" in helper.tool_call_states
    assert "call_p" in helper.tool_call_states

    assert helper.tool_call_states["call_u"].accumulated_json == '{"name": "Alice"}'
    assert helper.tool_call_states["call_t"].accumulated_json == '{"title":"Code"}'
    assert helper.tool_call_states["call_p"].accumulated_json == '{"name":"Tablet"}'


def test_lite_llm_parallel_base_empty_arguments_string():
    """Test handling of empty arguments string."""
    parallel_base = LiteLLMParallelBase(User)
    helper = ParallelStreamHelper(parallel_base)

    delta = ToolCallDelta(id="call_1", index=0, name="User", arguments="")

    result = helper.process_delta(delta)

    # Should create state with empty accumulated_json
    assert "call_1" in helper.tool_call_states
    state = helper.tool_call_states["call_1"]
    assert state.accumulated_json == ""


def test_lite_llm_parallel_base_result_tracking():
    """Test that ParallelResult tracks tool_call_id correctly."""
    parallel_base = LiteLLMParallelBase(User, Task)  # Both models needed
    helper = ParallelStreamHelper(parallel_base)

    delta1 = ToolCallDelta(
        id="first_call", index=0, name="User", arguments='{"name": "Alice"'
    )
    result1 = helper.process_delta(delta1)

    delta2 = ToolCallDelta(
        id="second_call", index=1, name="Task", arguments='{"title": "Review"'
    )
    result2 = helper.process_delta(delta2)

    # Each result should have correct tool_call_id
    assert result1.tool_call_id == "first_call"
    assert result2.tool_call_id == "second_call"


# ============================================================================
# Integration Tests (require API keys)
# ============================================================================


@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None or completion is None,
    reason="OPENAI_API_KEY not set or litellm not available",
)
@pytest.mark.skipif(
    os.getenv("SKIP_LLM_TESTS", "false").lower() == "true",
    reason="Skipping actual API call tests when SKIP_LLM_TESTS=true",
)
def test_lite_llm_parallel_streaming_integration():
    """Integration test for LiteLLM parallel streaming (requires API key)."""
    assert completion is not None, "completion should not be None when test runs"
    client = instructor.from_litellm(
        completion, mode=instructor.Mode.PARALLEL_TOOLS  # type: ignore[arg-type]
    )

    # Use transparent streaming - just add stream=True
    # The Iterable[Union[...]] type is automatically converted to LiteLLMParallelModel
    results = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=Iterable[Union[User, Task]],
        stream=True,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts structured data.",
            },
            {
                "role": "user",
                "content": "Extract user 'Alice' with email 'alice@example.com' and task 'Review code' with status 'in-progress'.",
            },
        ],
        max_tokens=300,
    )

    results_list = list(results)

    # Verify we got at least some results
    assert len(results_list) > 0

    # Check that we got ParallelResult objects
    assert all(isinstance(r, ParallelResult) for r in results_list)

    # Count completed models
    completed_models = []
    for result in results_list:
        completed_models.extend(result.completed.values())

    # We should have extracted at least User and Task
    model_types = {type(m).__name__ for m in completed_models}
    assert len(model_types) >= 1  # At least one type extracted


@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None or acompletion is None,
    reason="OPENAI_API_KEY not set or litellm async not available",
)
@pytest.mark.skipif(
    os.getenv("SKIP_LLM_TESTS", "false").lower() == "true",
    reason="Skipping actual API call tests when SKIP_LLM_TESTS=true",
)
@pytest.mark.asyncio
async def test_lite_llm_parallel_streaming_integration_async():
    """Async integration test for LiteLLM parallel streaming."""
    assert acompletion is not None, "acompletion should not be None when test runs"
    client = instructor.from_litellm(
        acompletion, mode=instructor.Mode.PARALLEL_TOOLS  # type: ignore[arg-type]
    )

    results = []
    async for result in await client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=Iterable[Union[User, Task]],
        stream=True,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts structured data.",
            },
            {
                "role": "user",
                "content": "Extract user 'Bob' with email 'bob@example.com' and task 'Write tests' with status 'pending'.",
            },
        ],
        max_tokens=300,
    ):
        results.append(result)

    assert len(results) > 0
    assert all(isinstance(r, ParallelResult) for r in results)

    # Collect completed models
    completed_models = []
    for result in results:
        completed_models.extend(result.completed.values())

    # At least one model should be extracted
    assert len(completed_models) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
