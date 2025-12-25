"""
Comprehensive unit tests for parallel streaming data structures.

This module tests the core data structures from instructor.dsl.parallel:
- ToolCallDelta
- ParallelResult
- ToolCallState
- ParallelStreamHelper
- ParallelBase (streaming methods)

Tests focus on streaming scenarios, edge cases, error handling, and coverage.
"""

from collections.abc import AsyncGenerator
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field, field_validator

from instructor.dsl.parallel import (
    Mode,
    ParallelBase,
    ParallelResult,
    ParallelStreamHelper,
    ToolCallDelta,
    ToolCallState,
)

# ============================================================================
# Test Models
# ============================================================================


class TestModel(BaseModel):
    """Simple test model for unit tests."""

    name: str
    age: int
    email: Optional[str] = None


class NestedModel(BaseModel):
    """Nested model for testing complex structures."""

    title: str
    description: Optional[str] = None

    @field_validator("description")
    def validate_description(cls, v):
        if v and len(v) < 5:
            raise ValueError("Description must be at least 5 characters")
        return v


class ComplexModel(BaseModel):
    """Complex model with multiple field types."""

    number: int = Field(..., ge=0, le=100)
    text: str = Field(..., min_length=1, max_length=50)
    optional_text: Optional[str] = None
    nested: Optional[NestedModel] = None


class WeirdlyNamedModel(BaseModel):
    """Model with a quirky name for testing."""

    value: str


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tool_call_delta():
    """Basic ToolCallDelta fixture."""
    return ToolCallDelta(
        id="call_abc123", index=0, name="TestModel", arguments='{"name": "Alice"'
    )


@pytest.fixture
def tool_call_delta_full():
    """ToolCallDelta with complete JSON arguments."""
    return ToolCallDelta(
        id="call_xyz789",
        index=1,
        name="TestModel",
        arguments='{"name": "Bob", "age": 25}',
    )


@pytest.fixture
def tool_call_delta_no_id():
    """ToolCallDelta without ID (using index-based generation)."""
    return ToolCallDelta(
        id=None, index=2, name="TestModel", arguments='{"name": "Charlie"'
    )


@pytest.fixture
def tool_call_delta_no_name():
    """ToolCallDelta without name (early chunk)."""
    return ToolCallDelta(id="call_early", index=0, name=None, arguments="")


@pytest.fixture
def tool_call_state():
    """Basic ToolCallState fixture."""
    return ToolCallState(
        tool_call_id="call_123",
        model_class=TestModel,
        accumulated_json='{"name": "Alice"',
        partial_model=None,
        is_complete=False,
    )


@pytest.fixture
def tool_call_state_with_partial():
    """ToolCallState with partial model."""
    partial = TestModel.model_validate({"name": "Alice", "age": 30})
    return ToolCallState(
        tool_call_id="call_456",
        model_class=TestModel,
        accumulated_json='{"name": "Alice", "age": 30}',
        partial_model=partial,
        is_complete=False,
    )


@pytest.fixture
def parallel_result():
    """Basic ParallelResult fixture."""
    completed = {
        "call_done": TestModel(name="Complete", age=100, email="done@test.com")
    }
    partial = {"call_partial": TestModel.model_validate({"name": "Partial", "age": 50})}
    return ParallelResult(
        completed=completed, partial=partial, tool_call_id="call_partial"
    )


@pytest.fixture
def parallel_base():
    """ParallelBase with test models."""
    return ParallelBase(TestModel, NestedModel, ComplexModel)


@pytest.fixture
def mock_response():
    """Mock response for stream_chunk testing."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]

    # Create mock message with tool_calls
    mock_message = MagicMock()
    mock_tool_call1 = MagicMock()
    mock_tool_call1.function.name = "TestModel"
    mock_tool_call1.function.arguments = '{"name": "Alice", "age": 30}'

    mock_tool_call2 = MagicMock()
    mock_tool_call2.function.name = "NestedModel"
    mock_tool_call2.function.arguments = (
        '{"title": "Task 1", "description": "A description"}'
    )

    mock_message.tool_calls = [mock_tool_call1, mock_tool_call2]
    mock_response.choices[0].message = mock_message

    return mock_response


# ============================================================================
# ToolCallDelta Tests
# ============================================================================


class TestToolCallDelta:
    """Test cases for ToolCallDelta dataclass."""

    def test_tool_call_delta_creation(self, tool_call_delta):
        """Test basic creation of ToolCallDelta."""
        assert tool_call_delta.id == "call_abc123"
        assert tool_call_delta.index == 0
        assert tool_call_delta.name == "TestModel"
        assert tool_call_delta.arguments == '{"name": "Alice"'

    def test_tool_call_delta_full_arguments(self, tool_call_delta_full):
        """Test ToolCallDelta with complete JSON."""
        assert tool_call_delta_full.arguments == '{"name": "Bob", "age": 25}'
        assert tool_call_delta_full.name == "TestModel"

    def test_tool_call_delta_with_none_id(self, tool_call_delta_no_id):
        """Test ToolCallDelta when id is None (early chunks)."""
        assert tool_call_delta_no_id.id is None
        assert tool_call_delta_no_id.index == 2
        assert tool_call_delta_no_id.name == "TestModel"

    def test_tool_call_delta_none_name(self, tool_call_delta_no_name):
        """Test ToolCallDelta when name is None."""
        assert tool_call_delta_no_name.name is None
        assert tool_call_delta_no_name.id == "call_early"
        assert tool_call_delta_no_name.arguments == ""

    def test_tool_call_delta_all_fields_present(self):
        """Test ToolCallDelta with all fields populated."""
        delta = ToolCallDelta(
            id="test_id",
            index=5,
            name="ComplexModel",
            arguments='{"number": 42, "text": "hello"}',
        )
        assert delta.id == "test_id"
        assert delta.index == 5
        assert delta.name == "ComplexModel"
        assert delta.arguments == '{"number": 42, "text": "hello"}'

    @pytest.mark.parametrize(
        "delta_id,expected_id",
        [
            ("call_1", "call_1"),
            (None, None),
            ("", ""),
        ],
    )
    def test_tool_call_delta_various_ids(self, delta_id, expected_id):
        """Test ToolCallDelta with various ID values."""
        delta = ToolCallDelta(
            id=delta_id, index=0, name="TestModel", arguments='{"name": "test"}'
        )
        assert delta.id == expected_id


# ============================================================================
# ToolCallState Tests
# ============================================================================


class TestToolCallState:
    """Test cases for ToolCallState dataclass."""

    def test_tool_call_state_creation(self, tool_call_state):
        """Test basic creation of ToolCallState."""
        assert tool_call_state.tool_call_id == "call_123"
        assert tool_call_state.model_class == TestModel
        assert tool_call_state.accumulated_json == '{"name": "Alice"'
        assert tool_call_state.partial_model is None
        assert tool_call_state.is_complete is False

    def test_tool_call_state_with_partial_model(self, tool_call_state_with_partial):
        """Test ToolCallState with partial model populated."""
        assert tool_call_state_with_partial.tool_call_id == "call_456"
        assert tool_call_state_with_partial.partial_model is not None
        assert tool_call_state_with_partial.partial_model.name == "Alice"
        assert tool_call_state_with_partial.partial_model.age == 30
        assert tool_call_state_with_partial.is_complete is False

    def test_tool_call_state_default_values(self):
        """Test ToolCallState with default values."""
        state = ToolCallState(tool_call_id="call_new", model_class=NestedModel)
        assert state.accumulated_json == ""
        assert state.partial_model is None
        assert state.is_complete is False

    def test_tool_call_state_complete(self):
        """Test ToolCallState marked as complete."""
        state = ToolCallState(
            tool_call_id="call_complete",
            model_class=TestModel,
            accumulated_json='{"name": "Complete", "age": 50}',
            partial_model=TestModel(name="Complete", age=50),
            is_complete=True,
        )
        assert state.is_complete is True
        assert state.partial_model.name == "Complete"

    def test_tool_call_state_different_models(self):
        """Test ToolCallState with different model classes."""
        test_state = [
            (TestModel, '{"name": "Alice", "age": 30}'),
            (NestedModel, '{"title": "Task", "description": "A valid description"}'),
            (ComplexModel, '{"number": 42, "text": "hello"}'),
        ]

        for model_class, json_str in test_state:
            state = ToolCallState(
                tool_call_id=f"call_{model_class.__name__}",
                model_class=model_class,
                accumulated_json=json_str,
            )
            assert state.model_class == model_class
            assert state.accumulated_json == json_str

    def test_tool_call_state_empty_accumulated_json(self):
        """Test ToolCallState with empty accumulated JSON."""
        state = ToolCallState(
            tool_call_id="call_empty", model_class=TestModel, accumulated_json=""
        )
        assert state.accumulated_json == ""
        assert state.partial_model is None


# ============================================================================
# ParallelResult Tests
# ============================================================================


class TestParallelResult:
    """Test cases for ParallelResult dataclass."""

    def test_parallel_result_creation(self, parallel_result):
        """Test basic creation of ParallelResult."""
        assert len(parallel_result.completed) == 1
        assert len(parallel_result.partial) == 1
        assert parallel_result.tool_call_id == "call_partial"
        assert "call_done" in parallel_result.completed
        assert "call_partial" in parallel_result.partial

    def test_parallel_result_get_all_models(self, parallel_result):
        """Test get_all_models returns both completed and partial models."""
        all_models = parallel_result.get_all_models()
        assert len(all_models) == 2
        assert all_models[0].name == "Complete"
        assert all_models[1].name == "Partial"

    def test_parallel_result_empty(self):
        """Test ParallelResult with no models."""
        result = ParallelResult(completed={}, partial={}, tool_call_id="")
        assert result.get_all_models() == []

    def test_parallel_result_only_completed(self):
        """Test ParallelResult with only completed models."""
        completed = {
            "call_1": TestModel(name="Alice", age=30),
            "call_2": TestModel(name="Bob", age=25),
        }
        result = ParallelResult(completed=completed, partial={}, tool_call_id="call_2")
        all_models = result.get_all_models()
        assert len(all_models) == 2
        assert all(m.name in ["Alice", "Bob"] for m in all_models)

    def test_parallel_result_only_partial(self):
        """Test ParallelResult with only partial models."""
        partial = {
            "call_1": TestModel.model_validate({"name": "Charlie", "age": 25}),
            "call_2": NestedModel.model_validate({"title": "Task"}),
        }
        result = ParallelResult(completed={}, partial=partial, tool_call_id="call_1")
        all_models = result.get_all_models()
        assert len(all_models) == 2

    def test_parallel_result_mixed_model_types(self):
        """Test ParallelResult with different model types."""
        completed = {"call_test": TestModel(name="Test", age=20)}
        partial = {
            "call_nested": NestedModel.model_validate(
                {"title": "Nested", "description": "Description"}
            ),
            "call_complex": ComplexModel.model_validate(
                {"number": 42, "text": "Complex"}
            ),
        }
        result = ParallelResult(
            completed=completed, partial=partial, tool_call_id="call_complex"
        )
        all_models = result.get_all_models()
        assert len(all_models) == 3
        assert any(m.name == "Test" for m in all_models)
        assert any(hasattr(m, "title") for m in all_models)

    def test_parallel_result_order_preserved(self):
        """Test that get_all_models preserves order (completed then partial)."""
        completed = {
            "call_1": TestModel(name="C1", age=1),
        }
        partial = {
            "call_2": TestModel(name="P1", age=2),
            "call_3": TestModel(name="P2", age=3),
        }
        result = ParallelResult(
            completed=completed, partial=partial, tool_call_id="call_2"
        )
        all_models = result.get_all_models()
        # First should be completed, then partials
        assert all_models[0].name == "C1"
        assert all_models[1].name == "P1"
        assert all_models[2].name == "P2"

    def test_parallel_result_get_all_models_various_models(self):
        """Test get_all_models with various edge cases."""
        # Case 1: All complete
        result1 = ParallelResult(
            completed={
                "call_a": TestModel(name="A", age=1),
                "call_b": NestedModel(title="B", description="Description B"),
            },
            partial={},
            tool_call_id="call_a",
        )
        models1 = result1.get_all_models()
        assert len(models1) == 2

        # Case 2: All partial
        result2 = ParallelResult(
            completed={},
            partial={
                "call_c": TestModel(name="C", age=35, email=None),
                "call_d": ComplexModel(
                    number=42, text="D", optional_text=None, nested=None
                ),
            },
            tool_call_id="call_c",
        )
        models2 = result2.get_all_models()
        assert len(models2) == 2


# ============================================================================
# ParallelStreamHelper Tests
# ============================================================================


class TestParallelStreamHelper:
    """Test cases for ParallelStreamHelper class."""

    def test_parallel_stream_helper_initialization(self, parallel_base):
        """Test initialization of ParallelStreamHelper."""
        helper = ParallelStreamHelper(parallel_base)
        assert helper.parallel_base == parallel_base
        assert helper.tool_call_states == {}
        assert helper._provided_models == set()

    def test_parallel_stream_helper_tool_call_states_created(self, parallel_base):
        """Test that tool_call_states dictionary is properly initialized."""
        helper = ParallelStreamHelper(parallel_base)
        assert isinstance(helper.tool_call_states, dict)
        assert len(helper.tool_call_states) == 0

    def test_parallel_stream_helper_states_dict_mutable(self, parallel_base):
        """Test that tool_call_states can be modified after initialization."""
        helper = ParallelStreamHelper(parallel_base)
        helper.tool_call_states["test_key"] = MagicMock()
        assert "test_key" in helper.tool_call_states

    # process_delta tests

    def test_process_delta_first_chunk_creates_state(
        self, parallel_base, tool_call_delta
    ):
        """Test that first chunk creates new ToolCallState."""
        helper = ParallelStreamHelper(parallel_base)
        result = helper.process_delta(tool_call_delta)

        assert "call_abc123" in helper.tool_call_states
        state = helper.tool_call_states["call_abc123"]
        assert state.tool_call_id == "call_abc123"
        assert state.model_class == TestModel
        assert state.is_complete is False

    def test_process_delta_accumulates_json(self, parallel_base, tool_call_delta):
        """Test that process_delta accumulates JSON chunks."""
        helper = ParallelStreamHelper(parallel_base)

        # First chunk
        helper.process_delta(tool_call_delta)

        # Second chunk
        delta2 = ToolCallDelta(
            id="call_abc123", index=0, name="TestModel", arguments=', "age": 30}'
        )
        helper.process_delta(delta2)

        assert helper.tool_call_states["call_abc123"].accumulated_json == (
            '{"name": "Alice", "age": 30}'
        )

    def test_process_delta_partial_json_parsing(self, parallel_base):
        """Test that process_delta handles partial JSON."""
        helper = ParallelStreamHelper(parallel_base)

        # Send partial JSON chunks
        delta1 = ToolCallDelta(id="call_1", index=0, name="TestModel", arguments='{"na')
        result1 = helper.process_delta(delta1)

        delta2 = ToolCallDelta(
            id="call_1", index=0, name="TestModel", arguments='me": "Alice", "age": 30}'
        )
        result2 = helper.process_delta(delta2)

        state = helper.tool_call_states["call_1"]
        assert state.accumulated_json == '{"name": "Alice", "age": 30}'

    def test_process_delta_complete_json(self, parallel_base):
        """Test that process_delta handles complete JSON."""
        helper = ParallelStreamHelper(parallel_base)

        delta = ToolCallDelta(
            id="call_1",
            index=0,
            name="TestModel",
            arguments='{"name": "Alice", "age": 30, "email": "alice@example.com"}',
        )
        result = helper.process_delta(delta)

        state = helper.tool_call_states["call_1"]
        assert state.partial_model is not None
        assert state.partial_model.name == "Alice"
        assert state.partial_model.age == 30
        assert state.partial_model.email == "alice@example.com"

    def test_process_delta_return_result_structure(self, parallel_base):
        """Test that process_delta returns proper ParallelResult structure."""
        helper = ParallelStreamHelper(parallel_base)

        delta = ToolCallDelta(
            id="call_1",
            index=0,
            name="TestModel",
            arguments='{"name": "Alice", "age": 30}',
        )
        result = helper.process_delta(delta)

        assert isinstance(result, ParallelResult)
        assert result.tool_call_id == "call_1"

    def test_process_delta_no_name_returns_early(
        self, parallel_base, tool_call_delta_no_name
    ):
        """Test that delta without name returns early without creating state."""
        helper = ParallelStreamHelper(parallel_base)
        result = helper.process_delta(tool_call_delta_no_name)

        assert "call_early" not in helper.tool_call_states

    def test_process_delta_unknown_model_raises_error(self, parallel_base):
        """Test that unknown model name raises KeyError."""
        helper = ParallelStreamHelper(parallel_base)

        delta = ToolCallDelta(
            id="call_1",
            index=0,
            name="InvalidModel",  # Not in registry
            arguments='{"test": "data"}',
        )

        with pytest.raises(KeyError) as exc_info:
            helper.process_delta(delta)

        assert "Unknown model name 'InvalidModel'" in str(exc_info.value)

    def test_process_delta_tool_call_id_missing_uses_index(
        self, parallel_base, tool_call_delta_no_id
    ):
        """Test that missing tool_call_id uses index-based ID generation."""
        helper = ParallelStreamHelper(parallel_base)
        result = helper.process_delta(tool_call_delta_no_id)

        # Should create state with ID based on index
        expected_id = "partial_call_2"
        assert expected_id in helper.tool_call_states

    def test_process_delta_multiple_concurrent_calls(self, parallel_base):
        """Test processing multiple concurrent tool calls."""
        helper = ParallelStreamHelper(parallel_base)

        # First tool call
        delta1 = ToolCallDelta(
            id="call_1", index=0, name="TestModel", arguments='{"name": "Alice"'
        )
        helper.process_delta(delta1)

        # Second tool call
        delta2 = ToolCallDelta(
            id="call_2", index=1, name="NestedModel", arguments='{"title": "Task}'
        )
        helper.process_delta(delta2)

        # Both states should exist
        assert "call_1" in helper.tool_call_states
        assert "call_2" in helper.tool_call_states
        assert helper.tool_call_states["call_1"].model_class == TestModel
        assert helper.tool_call_states["call_2"].model_class == NestedModel

    def test_process_delta_validation_context_passed(self, parallel_base):
        """Test that validation_context is passed to model_validate."""
        helper = ParallelStreamHelper(parallel_base)

        delta = ToolCallDelta(
            id="call_1",
            index=0,
            name="TestModel",
            arguments='{"name": "Alice", "age": 30}',
        )

        validation_context = {"user_id": "123"}
        result = helper.process_delta(delta, validation_context=validation_context)

        # Validation context should be used in model_validate
        assert result.tool_call_id == "call_1"

    def test_process_delta_strict_mode(self, parallel_base):
        """Test that strict parameter is passed to parser."""
        helper = ParallelStreamHelper(parallel_base)

        delta = ToolCallDelta(
            id="call_1",
            index=0,
            name="TestModel",
            arguments='{"name": "Alice", "age": 30}',
        )

        # Test strict mode
        result = helper.process_delta(delta, strict=False)
        assert result.tool_call_id == "call_1"

    def test_process_delta_invalid_json_error_handling(self, parallel_base):
        """Test that invalid JSON errors are handled gracefully."""
        helper = ParallelStreamHelper(parallel_base)

        delta = ToolCallDelta(
            id="call_1",
            index=0,
            name="TestModel",
            arguments='{"name": "Alice", "age": invalid',  # Invalid JSON
        )

        # Should not raise error, just return result
        result = helper.process_delta(delta)
        assert result.tool_call_id == "call_1"

    def test_process_delta_trailing_strings_json(self, parallel_base):
        """Test that trailing-strings extension works for incomplete JSON."""
        helper = ParallelStreamHelper(parallel_base)

        # Send incomplete JSON with trailing string - use ComplexModel from registry
        delta = ToolCallDelta(
            id="call_1",
            index=0,
            name="ComplexModel",
            arguments='{"text": "partial text',
        )

        result = helper.process_delta(delta)
        assert "call_1" in helper.tool_call_states
        assert (
            helper.tool_call_states["call_1"].accumulated_json
            == '{"text": "partial text'
        )

    def test_process_delta_empty_arguments(self, parallel_base):
        """Test that empty arguments are handled correctly."""
        helper = ParallelStreamHelper(parallel_base)

        delta = ToolCallDelta(id="call_1", index=0, name="TestModel", arguments="")

        result = helper.process_delta(delta)
        state = helper.tool_call_states["call_1"]
        assert state.accumulated_json == ""

    # mark_complete tests

    def test_mark_complete_marks_state_as_complete(self, parallel_base):
        """Test that mark_complete updates state.is_complete."""
        helper = ParallelStreamHelper(parallel_base)

        # First add a tool call
        delta = ToolCallDelta(
            id="call_1",
            index=0,
            name="TestModel",
            arguments='{"name": "Alice", "age": 30}',
        )
        helper.process_delta(delta)

        # Mark as complete
        result = helper.mark_complete("call_1")

        assert helper.tool_call_states["call_1"].is_complete is True
        assert "call_1" in result.completed

    def test_mark_complete_nonexistent_call(self, parallel_base):
        """Test marking complete on non-existent tool call."""
        helper = ParallelStreamHelper(parallel_base)
        result = helper.mark_complete("nonexistent")

        # Should return result without error
        assert isinstance(result, ParallelResult)

    def test_mark_complete_multiple_calls(self, parallel_base):
        """Test completing multiple tool calls."""
        helper = ParallelStreamHelper(parallel_base)

        # Add two tool calls
        delta1 = ToolCallDelta(
            id="call_1",
            index=0,
            name="TestModel",
            arguments='{"name": "Alice", "age": 30}',
        )
        helper.process_delta(delta1)

        delta2 = ToolCallDelta(
            id="call_2",
            index=1,
            name="TestModel",
            arguments='{"name": "Bob", "age": 25}',
        )
        helper.process_delta(delta2)

        # Complete both
        helper.mark_complete("call_1")
        helper.mark_complete("call_2")

        assert helper.tool_call_states["call_1"].is_complete is True
        assert helper.tool_call_states["call_2"].is_complete is True

    def test_mark_complete_result_structure(self, parallel_base):
        """Test that mark_complete returns proper ParallelResult."""
        helper = ParallelStreamHelper(parallel_base)

        delta = ToolCallDelta(
            id="call_1",
            index=0,
            name="TestModel",
            arguments='{"name": "Alice", "age": 30}',
        )
        helper.process_delta(delta)

        result = helper.mark_complete("call_1")
        assert isinstance(result, ParallelResult)
        assert "call_1" in result.completed

    # get_result tests

    def test_get_result_returns_current_state(self, parallel_base):
        """Test that get_result returns state without processing delta."""
        helper = ParallelStreamHelper(parallel_base)

        # Add a tool call
        delta = ToolCallDelta(
            id="call_1",
            index=0,
            name="TestModel",
            arguments='{"name": "Alice", "age": 30}',
        )
        helper.process_delta(delta)

        # Get result without processing delta
        result = helper.get_result()

        assert isinstance(result, ParallelResult)
        assert "call_1" in result.partial

    def test_get_result_empty_helper(self, parallel_base):
        """Test get_result when no tool calls have been added."""
        helper = ParallelStreamHelper(parallel_base)
        result = helper.get_result()

        assert isinstance(result, ParallelResult)
        assert len(result.completed) == 0
        assert len(result.partial) == 0
        assert result.tool_call_id == ""

    def test_get_result_after_multiple_deltas(self, parallel_base):
        """Test get_result after processing multiple deltas."""
        helper = ParallelStreamHelper(parallel_base)

        # Process multiple deltas
        deltas = [
            ToolCallDelta(
                id="call_1", index=0, name="TestModel", arguments='{"name": "Alice"'
            ),
            ToolCallDelta(
                id="call_1", index=0, name="TestModel", arguments=', "age": 30}'
            ),
            ToolCallDelta(
                id="call_2", index=1, name="TestModel", arguments='{"name": "Bob"'
            ),
            ToolCallDelta(
                id="call_2", index=1, name="TestModel", arguments=', "age": 25}'
            ),
        ]

        for delta in deltas:
            helper.process_delta(delta)

        result = helper.get_result()
        assert len(result.partial) == 2

    def test_get_result_tool_call_id_none(self, parallel_base):
        """Test that get_result returns empty tool_call_id."""
        helper = ParallelStreamHelper(parallel_base)
        result = helper.get_result()

        assert result.tool_call_id == ""

    # Testing state management

    def test_state_management_initialization(self, parallel_base):
        """Test initialization of state management structures."""
        helper = ParallelStreamHelper(parallel_base)

        assert hasattr(helper, "tool_call_states")
        assert hasattr(helper, "_provided_models")
        assert isinstance(helper._provided_models, set)

    def test_state_dictionary_mutation(self, parallel_base):
        """Test that tool_call_states can be mutated correctly."""
        helper = ParallelStreamHelper(parallel_base)

        initial_count = len(helper.tool_call_states)
        helper.tool_call_states["test"] = MagicMock()

        assert len(helper.tool_call_states) == initial_count + 1
        del helper.tool_call_states["test"]
        assert len(helper.tool_call_states) == initial_count

    def test_provided_models_set_operations(self, parallel_base):
        """Test operations on _provided_models set."""
        helper = ParallelStreamHelper(parallel_base)

        # Test add
        helper._provided_models.add("model_1")
        assert "model_1" in helper._provided_models

        # Test remove
        helper._provided_models.remove("model_1")
        assert "model_1" not in helper._provided_models

    # Complex integration tests

    def test_integration_complete_streaming_workflow(self, parallel_base):
        """Test complete streaming workflow from start to end."""
        helper = ParallelStreamHelper(parallel_base)

        # Simulate streaming deltas
        streaming_sequence = [
            ToolCallDelta(
                id="call_1", index=0, name="TestModel", arguments='{"name": "Alice"'
            ),
            ToolCallDelta(
                id="call_1", index=0, name="TestModel", arguments=', "age": 30}'
            ),
            ToolCallDelta(
                id="call_2", index=1, name="NestedModel", arguments='{"title": "Task 1"'
            ),
            ToolCallDelta(
                id="call_2",
                index=1,
                name="NestedModel",
                arguments=', "description": "A valid description"}',
            ),
        ]

        partial_results = []
        for delta in streaming_sequence:
            result = helper.process_delta(delta)
            partial_results.append(result)

        # Mark calls as complete
        helper.mark_complete("call_1")
        helper.mark_complete("call_2")

        # Final result
        final_result = helper.get_result()

        assert len(final_result.completed) == 2
        assert "call_1" in final_result.completed
        assert "call_2" in final_result.completed

    def test_integration_multiple_concurrent_partial_calls(self, parallel_base):
        """Test multiple concurrent calls that remain partial."""
        helper = ParallelStreamHelper(parallel_base)

        # Add 3 concurrent calls, none complete
        for i in range(3):
            delta = ToolCallDelta(
                id=f"call_{i}",
                index=i,
                name="TestModel",
                arguments=f'{{"name": "User{i}", "age": {20 + i}}}',
            )
            helper.process_delta(delta)

        result = helper.get_result()
        assert len(result.partial) == 3

    def test_integration_mixed_complete_and_partial_calls(self, parallel_base):
        """Test scenario with mix of complete and partial calls."""
        helper = ParallelStreamHelper(parallel_base)

        # Add and complete first call
        delta1 = ToolCallDelta(
            id="call_1",
            index=0,
            name="TestModel",
            arguments='{"name": "Alice", "age": 30}',
        )
        helper.process_delta(delta1)
        helper.mark_complete("call_1")

        # Add second call, keep partial - need more to create a valid partial model
        delta2 = ToolCallDelta(
            id="call_2", index=1, name="TestModel", arguments='{"name": "Bob",'
        )
        helper.process_delta(delta2)

        # Add enough to create a parseable partial model
        delta2b = ToolCallDelta(
            id="call_2", index=1, name="TestModel", arguments=' "age": 25'
        )
        helper.process_delta(delta2b)

        result = helper.get_result()
        assert len(result.completed) == 1
        assert len(result.partial) == 1
        assert "call_1" in result.completed
        assert "call_2" in result.partial

    def test_integration_model_registry_lookup(self, parallel_base):
        """Test that model registry lookup works correctly."""
        helper = ParallelStreamHelper(parallel_base)

        # Test with each model in registry
        models_to_test = [
            ("TestModel", TestModel),
            ("NestedModel", NestedModel),
            ("ComplexModel", ComplexModel),
        ]

        for model_name, model_class in models_to_test:
            delta = ToolCallDelta(
                id=f"call_{model_name}", index=0, name=model_name, arguments="{}"
            )
            helper.process_delta(delta)

            state = helper.tool_call_states[f"call_{model_name}"]
            assert state.model_class == model_class

    # Edge case tests

    def test_edge_case_whitespace_only_arguments(self, parallel_base):
        """Test handling of whitespace-only arguments."""
        helper = ParallelStreamHelper(parallel_base)

        delta = ToolCallDelta(id="call_1", index=0, name="TestModel", arguments="   ")
        result = helper.process_delta(delta)

        state = helper.tool_call_states["call_1"]
        assert state.accumulated_json == "   "

    def test_edge_case_empty_string_after_strip(self, parallel_base):
        """Test handling of empty string after strip."""
        helper = ParallelStreamHelper(parallel_base)

        delta = ToolCallDelta(id="call_1", index=0, name="TestModel", arguments="")

        # First delta sets state
        helper.process_delta(delta)
        state = helper.tool_call_states["call_1"]
        assert state.accumulated_json == ""

    def test_edge_case_special_characters_in_json(self, parallel_base):
        """Test handling of special characters in JSON."""
        helper = ParallelStreamHelper(parallel_base)

        delta = ToolCallDelta(
            id="call_1",
            index=0,
            name="TestModel",
            arguments='{"name": "Test\\"Quote\\nNewline", "age": 30}',
        )
        result = helper.process_delta(delta)

        state = helper.tool_call_states["call_1"]
        assert state.partial_model is not None
        assert "Quote" in state.partial_model.name

    def test_edge_case_unicode_characters(self, parallel_base):
        """Test handling of unicode characters in JSON."""
        helper = ParallelStreamHelper(parallel_base)

        delta = ToolCallDelta(
            id="call_1",
            index=0,
            name="TestModel",
            arguments='{"name": "中文日本語한글", "age": 30}',
        )
        result = helper.process_delta(delta)

        state = helper.tool_call_states["call_1"]
        assert state.partial_model is not None

    def test_edge_case_very_long_json(self, parallel_base):
        """Test handling of very long JSON strings."""
        helper = ParallelStreamHelper(parallel_base)

        long_str = "x" * 10000
        delta = ToolCallDelta(
            id="call_1",
            index=0,
            name="TestModel",
            arguments=f'{{"name": "{long_str}"}}',
        )
        result = helper.process_delta(delta)

        state = helper.tool_call_states["call_1"]
        assert len(state.accumulated_json) > 10000


# ============================================================================
# Parameterized Tests
# ============================================================================


class TestParallelStreamingParameterized:
    """Parameterized tests for parallel streaming scenarios."""

    @pytest.mark.parametrize(
        "model_class,valid_json",
        [
            (TestModel, '{"name": "Alice", "age": 30}'),
            (NestedModel, '{"title": "Task", "description": "Valid description"}'),
            (ComplexModel, '{"number": 42, "text": "Hello", "optional_text": null}'),
        ],
    )
    def test_parameterized_valid_json_parsing(
        self, parallel_base, model_class, valid_json
    ):
        """Test that different model types handle valid JSON correctly."""
        helper = ParallelStreamHelper(parallel_base)

        delta = ToolCallDelta(
            id="call_1", index=0, name=model_class.__name__, arguments=valid_json
        )

        result = helper.process_delta(delta)
        assert result.tool_call_id == "call_1"

    @pytest.mark.parametrize(
        "partial_json,expected_accumulated",
        [
            ('{"name":', '{"name":'),
            ('{"name": "Alice",', '{"name": "Alice",'),
            ('{"name": "Alice", "age":', '{"name": "Alice", "age":'),
            ('{"name": "Alice", "age": 3', '{"name": "Alice", "age": 3'),
        ],
    )
    def test_parameterized_partial_json_accumulation(
        self, parallel_base, partial_json, expected_accumulated
    ):
        """Test that partial JSON chunks accumulate correctly."""
        helper = ParallelStreamHelper(parallel_base)

        delta = ToolCallDelta(
            id="call_1", index=0, name="TestModel", arguments=partial_json
        )

        helper.process_delta(delta)
        assert (
            helper.tool_call_states["call_1"].accumulated_json == expected_accumulated
        )

    @pytest.mark.parametrize(
        "invalid_json",
        [
            '{"name": "Alice", "age": invalid}',
            '{"name": "Alice", "age": }',
            '{"name": "Alice", "age":',
            "not json at all",
            '{"incomplete"',
        ],
    )
    def test_parameterized_invalid_json_handling(self, parallel_base, invalid_json):
        """Test that various invalid JSON formats are handled gracefully."""
        helper = ParallelStreamHelper(parallel_base)

        delta = ToolCallDelta(
            id="call_1", index=0, name="TestModel", arguments=invalid_json
        )

        # Should not raise error
        result = helper.process_delta(delta)
        assert result.tool_call_id == "call_1"

    @pytest.mark.parametrize(
        "tool_call_id,index",
        [
            ("call_123", 0),
            ("call_456", 1),
            ("call_789", 2),
        ],
    )
    def test_parameterized_various_tool_call_ids(
        self, parallel_base, tool_call_id, index
    ):
        """Test that various tool call IDs work correctly."""
        helper = ParallelStreamHelper(parallel_base)

        delta = ToolCallDelta(
            id=tool_call_id,
            index=index,
            name="TestModel",
            arguments='{"name": "Test", "age": 30}',
        )

        result = helper.process_delta(delta)
        assert tool_call_id in helper.tool_call_states

    @pytest.mark.parametrize(
        "delta_id,index,expected_state_id",
        [
            (None, 0, "partial_call_0"),
            (None, 5, "partial_call_5"),
            (None, 10, "partial_call_10"),
        ],
    )
    def test_parameterized_id_generation_from_index(
        self, parallel_base, delta_id, index, expected_state_id
    ):
        """Test that ID generation from index works correctly."""
        helper = ParallelStreamHelper(parallel_base)

        delta = ToolCallDelta(
            id=delta_id, index=index, name="TestModel", arguments='{"name": "Test"}'
        )

        helper.process_delta(delta)
        assert expected_state_id in helper.tool_call_states

    @pytest.mark.parametrize("num_calls", [1, 2, 5, 10])
    def test_parameterized_multiple_concurrent_calls(self, parallel_base, num_calls):
        """Test that multiple concurrent tool calls are handled correctly."""
        helper = ParallelStreamHelper(parallel_base)

        for i in range(num_calls):
            delta = ToolCallDelta(
                id=f"call_{i}",
                index=i,
                name="TestModel",
                arguments=f'{{"name": "User{i}", "age": {20 + i}}}',
            )
            helper.process_delta(delta)

        assert len(helper.tool_call_states) == num_calls

    @pytest.mark.parametrize(
        "chunk_sequence,expected_final",
        [
            (
                [
                    ("{", ""),
                    ('"name"', ""),
                    (": ", ""),
                    ('"Alice"', ""),
                    (", ", ""),
                    ('"age"', ""),
                    (": ", ""),
                    ("30", ""),
                    ("}", ""),
                ],
                '{"name": "Alice", "age": 30}',
            ),
            (
                [
                    ('{"name": "', ""),
                    ("Alice", ""),
                    ('", "age": ', ""),
                    ("30", ""),
                    ("}", ""),
                ],
                '{"name": "Alice", "age": 30}',
            ),
        ],
    )
    def test_parameterized_chunk_accumulation_sequences(
        self, parallel_base, chunk_sequence, expected_final
    ):
        """Test various chunk accumulation sequences."""
        helper = ParallelStreamHelper(parallel_base)

        for chunk, _ in chunk_sequence:
            delta = ToolCallDelta(
                id="call_1", index=0, name="TestModel", arguments=chunk
            )
            helper.process_delta(delta)

        assert helper.tool_call_states["call_1"].accumulated_json == expected_final

    @pytest.mark.parametrize("complete_after_n_chunks", [1, 2, 5, 10])
    def test_parameterized_completion_after_chunks(
        self, parallel_base, complete_after_n_chunks
    ):
        """Test completion after different numbers of chunks."""
        helper = ParallelStreamHelper(parallel_base)

        full_json = '{"name": "Alice", "age": 30, "email": "alice@example.com"}'

        # Split into chunks
        chunk_size = max(1, len(full_json) // complete_after_n_chunks)
        chunks = [
            full_json[i : i + chunk_size] for i in range(0, len(full_json), chunk_size)
        ]

        for i, chunk in enumerate(chunks):
            delta = ToolCallDelta(
                id="call_1", index=0, name="TestModel", arguments=chunk
            )
            helper.process_delta(delta)

            if i == len(chunks) - 1:
                helper.mark_complete("call_1")

        assert helper.tool_call_states["call_1"].is_complete is True


# ============================================================================
# Test ParallelBase Subclass for Testing
# ============================================================================


class TestParallelBase(ParallelBase):
    """Test subclass of ParallelBase with custom _extract_tool_deltas implementation."""

    @staticmethod
    def _extract_tool_deltas(
        chunk: Any,  # noqa: ARG004
        mode: Mode,  # noqa: ARG004
    ) -> list[ToolCallDelta]:
        """
        Custom implementation for testing purposes.

        This implementation handles mock chunks with a specific structure.
        Mock chunks should have:
        - chunk.deltas: list of ToolCallDelta objects (optional)
        - chunk.mode: Mode value (optional, defaults to Mode.PARALLEL_TOOLS)
        """
        # Base implementation returns empty list
        return []


class MockStreamingChunk:
    """Mock streaming chunk for testing."""

    def __init__(
        self,
        deltas: Optional[list[ToolCallDelta]] = None,
        mode: Mode = Mode.PARALLEL_TOOLS,
    ):
        self.deltas = deltas or []
        self.mode = mode


class MockAsyncStreamingGenerator:
    """Mock async streaming generator for testing."""

    def __init__(self, chunks: list[Any]):
        self.chunks = chunks
        self.index = 0

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


class AsyncParallelBase(ParallelBase):
    """Async-enabled ParallelBase subclass for testing async methods."""

    @staticmethod
    def _extract_tool_deltas(
        chunk: Any,
        mode: Mode,  # noqa: ARG004
    ) -> list[ToolCallDelta]:
        """Extract deltas from mock chunks."""
        if hasattr(chunk, "deltas"):
            return chunk.deltas
        if isinstance(chunk, ToolCallDelta):
            return [chunk]
        return []


# ============================================================================
# ParallelBase Tests
# ============================================================================


class TestParallelBaseClass:
    """Test cases for ParallelBase class initialization and attributes."""

    def test_parallel_base_initialization_single_model(self):
        """Test ParallelBase initialization with single model."""
        parallel_base = TestParallelBase(TestModel)
        assert len(parallel_base.models) == 1
        assert parallel_base.models[0] == TestModel
        assert "TestModel" in parallel_base.registry
        assert parallel_base.registry["TestModel"] == TestModel

    def test_parallel_base_initialization_multiple_models(self):
        """Test ParallelBase initialization with multiple models."""
        parallel_base = TestParallelBase(TestModel, NestedModel, ComplexModel)
        assert len(parallel_base.models) == 3
        assert len(parallel_base.registry) == 3
        assert "TestModel" in parallel_base.registry
        assert "NestedModel" in parallel_base.registry
        assert "ComplexModel" in parallel_base.registry

    def test_parallel_base_requires_at_least_one_model(self):
        """Test that ParallelBase requires at least one model."""
        with pytest.raises(AssertionError) as exc_info:
            TestParallelBase()
        assert "At least one model is required" in str(exc_info.value)

    def test_parallel_base_registry_creation(self):
        """Test that registry is properly created from model names."""
        parallel_base = TestParallelBase(TestModel, NestedModel)
        assert isinstance(parallel_base.registry, dict)
        assert parallel_base.registry["TestModel"] == TestModel
        assert parallel_base.registry["NestedModel"] == NestedModel

    def test_parallel_base_models_attribute(self):
        """Test that models attribute stores all provided models."""
        models = [TestModel, NestedModel, ComplexModel]
        parallel_base = TestParallelBase(*models)
        assert list(parallel_base.models) == models

    def test_parallel_base_with_unnamed_model(self):
        """Test ParallelBase behavior with unnamed model (edge case)."""

        # Create a model instance instead of class
        class TempModel(BaseModel):
            value: str

        temp_model_instance = TempModel(value="test")
        # The registry will use the string representation of the object
        parallel_base = TestParallelBase(TempModel)
        assert len(parallel_base.registry) == 1


# ============================================================================
# ParallelBase from_streaming_response Tests (Synchronous)
# ============================================================================


class TestParallelBaseStreamingSync:
    """Test cases for ParallelBase.from_streaming_response (synchronous)."""

    def test_from_streaming_response_basic_streaming(self):
        """Test basic streaming functionality."""
        parallel_base = AsyncParallelBase(TestModel)

        # Create mock streaming chunks
        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1",
                        index=0,
                        name="TestModel",
                        arguments='{"name": "Alice"',
                    )
                ]
            ),
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments=', "age": 30}'
                    )
                ]
            ),
        ]

        results = list(
            parallel_base.from_streaming_response(chunks, mode=Mode.PARALLEL_TOOLS)
        )

        assert len(results) == 2
        assert "call_1" in results[0].partial or "call_1" in results[1].partial

    def test_from_streaming_response_multiple_tools(self):
        """Test streaming with multiple concurrent tool calls."""
        parallel_base = AsyncParallelBase(TestModel, NestedModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1",
                        index=0,
                        name="TestModel",
                        arguments='{"name": "Alice"',
                    )
                ]
            ),
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_2",
                        index=1,
                        name="NestedModel",
                        arguments='{"title": "Task"',
                    )
                ]
            ),
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments=', "age": 30}'
                    ),
                    ToolCallDelta(
                        id="call_2",
                        index=1,
                        name="NestedModel",
                        arguments=', "description": "Valid description"}',
                    ),
                ]
            ),
        ]

        results = list(
            parallel_base.from_streaming_response(chunks, mode=Mode.PARALLEL_TOOLS)
        )

        assert len(results) > 0
        final_result = results[-1]
        # Should have both tool calls tracked
        assert len(final_result.completed) + len(final_result.partial) >= 1

    def test_from_streaming_response_empty_chunks(self):
        """Test streaming with empty chunks."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk([]),
            MockStreamingChunk([]),
        ]

        results = list(
            parallel_base.from_streaming_response(chunks, mode=Mode.PARALLEL_TOOLS)
        )

        # Empty chunks won't yield results
        assert len(results) == 0

    def test_from_streaming_response_single_chunk_complete_json(self):
        """Test streaming with complete JSON in single chunk."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1",
                        index=0,
                        name="TestModel",
                        arguments='{"name": "Alice", "age": 30, "email": "alice@example.com"}',
                    )
                ]
            ),
        ]

        results = list(
            parallel_base.from_streaming_response(chunks, mode=Mode.PARALLEL_TOOLS)
        )

        assert len(results) == 1
        assert "call_1" in results[0].partial
        model = results[0].partial["call_1"]
        assert model.name == "Alice"
        assert model.age == 30

    def test_from_streaming_response_with_validation_context(self):
        """Test streaming with validation context."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1",
                        index=0,
                        name="TestModel",
                        arguments='{"name": "Alice", "age": 30}',
                    )
                ]
            ),
        ]

        validation_context = {"user_id": "123", "request_id": "abc"}
        results = list(
            parallel_base.from_streaming_response(
                chunks, mode=Mode.PARALLEL_TOOLS, validation_context=validation_context
            )
        )

        assert len(results) == 1
        # Context should be passed through (validation happens internally)
        assert results[0].tool_call_id == "call_1"

    def test_from_streaming_response_with_strict_mode_true(self):
        """Test streaming with strict mode enabled."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1",
                        index=0,
                        name="TestModel",
                        arguments='{"name": "Alice", "age": 30}',
                    )
                ]
            ),
        ]

        results = list(
            parallel_base.from_streaming_response(
                chunks, mode=Mode.PARALLEL_TOOLS, strict=True
            )
        )

        assert len(results) == 1
        assert results[0].tool_call_id == "call_1"

    def test_from_streaming_response_with_strict_mode_false(self):
        """Test streaming with strict mode disabled."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1",
                        index=0,
                        name="TestModel",
                        arguments='{"name": "Alice", "age": 30}',
                    )
                ]
            ),
        ]

        results = list(
            parallel_base.from_streaming_response(
                chunks, mode=Mode.PARALLEL_TOOLS, strict=False
            )
        )

        assert len(results) == 1

    def test_from_streaming_response_partial_json_accumulation(self):
        """Test that partial JSON accumulates correctly across chunks."""
        parallel_base = AsyncParallelBase(TestModel)

        # Send JSON character by character
        json_str = '{"name": "Alice", "age": 30}'
        chunks = []
        for char in json_str:
            chunk = MockStreamingChunk(
                [ToolCallDelta(id="call_1", index=0, name="TestModel", arguments=char)]
            )
            chunks.append(chunk)

        results = list(
            parallel_base.from_streaming_response(chunks, mode=Mode.PARALLEL_TOOLS)
        )

        # Should process all chunks
        assert len(results) == len(chunks)

    def test_from_streaming_response_multiple_updates_same_call(self):
        """Test multiple updates to the same tool call."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments='{"name": '
                    ),
                ]
            ),
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments='"Alice"'
                    ),
                ]
            ),
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments=', "age": 30'
                    ),
                ]
            ),
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments="}"
                    ),
                ]
            ),
        ]

        results = list(
            parallel_base.from_streaming_response(chunks, mode=Mode.PARALLEL_TOOLS)
        )

        # Should produce 4 results
        assert len(results) == 4
        # All results should reference the same tool call
        for result in results:
            assert result.tool_call_id == "call_1"

    def test_from_streaming_response_parallel_calls_interleaved(self):
        """Test interleaved updates to multiple parallel calls."""
        parallel_base = AsyncParallelBase(TestModel, NestedModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments='{"name":'
                    ),
                ]
            ),
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_2", index=1, name="NestedModel", arguments='{"title":'
                    ),
                ]
            ),
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1",
                        index=0,
                        name="TestModel",
                        arguments=' "Alice", "age": 30}',
                    ),
                    ToolCallDelta(
                        id="call_2", index=1, name="NestedModel", arguments=' "Task"}'
                    ),
                ]
            ),
        ]

        results = list(
            parallel_base.from_streaming_response(chunks, mode=Mode.PARALLEL_TOOLS)
        )

        # Should have results for both calls (includes empty partial results too)
        # First two chunks with incomplete JSON return empty results,
        # last chunk results in valid models
        assert len(results) == 4

    def test_from_streaming_response_error_handling_invalid_model(self):
        """Test error handling when model name is not in registry."""
        parallel_base = AsyncParallelBase(TestModel)

        # This should not raise an error during streaming
        # The error will be caught internally and handled
        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1",
                        index=0,
                        name="InvalidModel",
                        arguments='{"test": "data"}',
                    )
                ]
            ),
        ]

        with pytest.raises(KeyError) as exc_info:
            list(
                parallel_base.from_streaming_response(chunks, mode=Mode.PARALLEL_TOOLS)
            )

        assert "Unknown model name 'InvalidModel'" in str(exc_info.value)

    def test_from_streaming_response_generator_exhaustion(self):
        """Test that the generator can be fully exhausted."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments='{"name":'
                    ),
                ]
            ),
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments=' "Alice"'
                    ),
                ]
            ),
        ]

        generator = parallel_base.from_streaming_response(
            chunks, mode=Mode.PARALLEL_TOOLS
        )

        results = []
        for result in generator:
            results.append(result)

        assert len(results) == 2


# ============================================================================
# ParallelBase from_streaming_response_async Tests (Asynchronous)
# ============================================================================


@pytest.mark.asyncio
class TestParallelBaseStreamingAsync:
    """Test cases for ParallelBase.from_streaming_response_async (asynchronous)."""

    async def test_from_streaming_response_async_basic(self):
        """Test basic async streaming functionality."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1",
                        index=0,
                        name="TestModel",
                        arguments='{"name": "Alice"',
                    )
                ]
            ),
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments=', "age": 30}'
                    )
                ]
            ),
        ]

        results = []
        async for result in parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks), mode=Mode.PARALLEL_TOOLS
        ):
            results.append(result)

        assert len(results) == 2
        assert "call_1" in results[0].partial or "call_1" in results[1].partial

    async def test_from_streaming_response_async_multiple_tools(self):
        """Test async streaming with multiple concurrent tool calls."""
        parallel_base = AsyncParallelBase(TestModel, NestedModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1",
                        index=0,
                        name="TestModel",
                        arguments='{"name": "Alice"',
                    )
                ]
            ),
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_2",
                        index=1,
                        name="NestedModel",
                        arguments='{"title": "Task"',
                    )
                ]
            ),
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments=', "age": 30}'
                    ),
                    ToolCallDelta(
                        id="call_2",
                        index=1,
                        name="NestedModel",
                        arguments=', "description": "Valid description"}',
                    ),
                ]
            ),
        ]

        results = []
        async for result in parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks), mode=Mode.PARALLEL_TOOLS
        ):
            results.append(result)

        assert len(results) > 0
        final_result = results[-1]
        assert len(final_result.completed) + len(final_result.partial) >= 1

    async def test_from_streaming_response_async_empty_chunks(self):
        """Test async streaming with empty chunks."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk([]),
            MockStreamingChunk([]),
        ]

        results = []
        async for result in parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks), mode=Mode.PARALLEL_TOOLS
        ):
            results.append(result)

        # Empty chunks won't yield results
        assert len(results) == 0

    async def test_from_streaming_response_async_single_chunk_complete_json(self):
        """Test async streaming with complete JSON in single chunk."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1",
                        index=0,
                        name="TestModel",
                        arguments='{"name": "Alice", "age": 30, "email": "alice@example.com"}',
                    )
                ]
            ),
        ]

        results = []
        async for result in parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks), mode=Mode.PARALLEL_TOOLS
        ):
            results.append(result)

        assert len(results) == 1
        assert "call_1" in results[0].partial
        model = results[0].partial["call_1"]
        assert model.name == "Alice"
        assert model.age == 30

    async def test_from_streaming_response_async_with_validation_context(self):
        """Test async streaming with validation context."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1",
                        index=0,
                        name="TestModel",
                        arguments='{"name": "Alice", "age": 30}',
                    )
                ]
            ),
        ]

        validation_context = {"user_id": "123", "request_id": "abc"}
        results = []
        async for result in parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks),
            mode=Mode.PARALLEL_TOOLS,
            validation_context=validation_context,
        ):
            results.append(result)

        assert len(results) == 1
        assert results[0].tool_call_id == "call_1"

    async def test_from_streaming_response_async_with_strict_mode_true(self):
        """Test async streaming with strict mode enabled."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1",
                        index=0,
                        name="TestModel",
                        arguments='{"name": "Alice", "age": 30}',
                    )
                ]
            ),
        ]

        results = []
        async for result in parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks), mode=Mode.PARALLEL_TOOLS, strict=True
        ):
            results.append(result)

        assert len(results) == 1
        assert results[0].tool_call_id == "call_1"

    async def test_from_streaming_response_async_with_strict_mode_false(self):
        """Test async streaming with strict mode disabled."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1",
                        index=0,
                        name="TestModel",
                        arguments='{"name": "Alice", "age": 30}',
                    )
                ]
            ),
        ]

        results = []
        async for result in parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks), mode=Mode.PARALLEL_TOOLS, strict=False
        ):
            results.append(result)

        assert len(results) == 1

    async def test_from_streaming_response_async_partial_json_accumulation(self):
        """Test that partial JSON accumulates correctly in async streaming."""
        parallel_base = AsyncParallelBase(TestModel)

        json_str = '{"name": "Alice", "age": 30}'
        chunks = []
        for char in json_str:
            chunk = MockStreamingChunk(
                [ToolCallDelta(id="call_1", index=0, name="TestModel", arguments=char)]
            )
            chunks.append(chunk)

        results = []
        async for result in parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks), mode=Mode.PARALLEL_TOOLS
        ):
            results.append(result)

        assert len(results) == len(chunks)

    async def test_from_streaming_response_async_multiple_updates_same_call(self):
        """Test multiple updates to the same tool call in async streaming."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments='{"name": '
                    ),
                ]
            ),
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments='"Alice"'
                    ),
                ]
            ),
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments=', "age": 30'
                    ),
                ]
            ),
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments="}"
                    ),
                ]
            ),
        ]

        results = []
        async for result in parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks), mode=Mode.PARALLEL_TOOLS
        ):
            results.append(result)

        assert len(results) == 4
        for result in results:
            assert result.tool_call_id == "call_1"

    async def test_from_streaming_response_async_parallel_calls_interleaved(self):
        """Test interleaved updates to multiple calls in async streaming."""
        parallel_base = AsyncParallelBase(TestModel, NestedModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments='{"name":'
                    ),
                ]
            ),
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_2", index=1, name="NestedModel", arguments='{"title":'
                    ),
                ]
            ),
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1",
                        index=0,
                        name="TestModel",
                        arguments=' "Alice", "age": 30}',
                    ),
                    ToolCallDelta(
                        id="call_2", index=1, name="NestedModel", arguments=' "Task"}'
                    ),
                ]
            ),
        ]

        results = []
        async for result in parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks), mode=Mode.PARALLEL_TOOLS
        ):
            results.append(result)

        assert len(results) == 4

    async def test_from_streaming_response_async_error_handling_invalid_model(self):
        """Test async error handling when model name is not in registry."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1",
                        index=0,
                        name="InvalidModel",
                        arguments='{"test": "data"}',
                    )
                ]
            ),
        ]

        with pytest.raises(KeyError) as exc_info:
            results = []
            async for result in parallel_base.from_streaming_response_async(
                MockAsyncStreamingGenerator(chunks), mode=Mode.PARALLEL_TOOLS
            ):
                results.append(result)

        assert "Unknown model name 'InvalidModel'" in str(exc_info.value)

    async def test_from_streaming_response_async_generator_exhaustion(self):
        """Test that the async generator can be fully exhausted."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments='{"name":'
                    ),
                ]
            ),
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments=' "Alice"'
                    ),
                ]
            ),
        ]

        results = []
        async for result in parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks), mode=Mode.PARALLEL_TOOLS
        ):
            results.append(result)

        assert len(results) == 2

    async def test_from_streaming_response_async_async_generator_type(self):
        """Test that from_streaming_response_async returns an AsyncGenerator."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1",
                        index=0,
                        name="TestModel",
                        arguments='{"name": "Alice"}',
                    )
                ]
            ),
        ]

        async_gen = parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks), mode=Mode.PARALLEL_TOOLS
        )

        assert isinstance(async_gen, AsyncGenerator)

        # Consume it to verify it works
        results = []
        async for result in async_gen:
            results.append(result)

        assert len(results) == 1


# ============================================================================
# ParallelBase _extract_tool_deltas Tests
# ============================================================================


class TestParallelBaseExtractToolDeltas:
    """Test cases for ParallelBase._extract_tool_deltas method."""

    def test_extract_tool_deltas_base_implementation(self):
        """Test base implementation returns empty list."""
        parallel_base = TestParallelBase(TestModel)

        # Base implementation should return empty list
        base_chunk = MagicMock()
        deltas = parallel_base._extract_tool_deltas(base_chunk, Mode.PARALLEL_TOOLS)

        assert isinstance(deltas, list)
        assert len(deltas) == 0

    def test_extract_tool_deltas_custom_implementation(self):
        """Test custom implementation in AsyncParallelBase."""
        parallel_base = AsyncParallelBase(TestModel)

        chunk = MockStreamingChunk(
            [
                ToolCallDelta(
                    id="call_1",
                    index=0,
                    name="TestModel",
                    arguments='{"name": "Alice"}',
                )
            ]
        )

        deltas = parallel_base._extract_tool_deltas(chunk, Mode.PARALLEL_TOOLS)

        assert len(deltas) == 1
        assert deltas[0].id == "call_1"
        assert deltas[0].name == "TestModel"

    def test_extract_tool_deltas_empty_chunk(self):
        """Test extraction from chunk with no deltas."""
        parallel_base = AsyncParallelBase(TestModel)

        chunk = MockStreamingChunk([])
        deltas = parallel_base._extract_tool_deltas(chunk, Mode.PARALLEL_TOOLS)

        assert len(deltas) == 0

    def test_extract_tool_deltas_multiple_deltas(self):
        """Test extraction with multiple deltas in single chunk."""
        parallel_base = AsyncParallelBase(TestModel, NestedModel)

        chunk = MockStreamingChunk(
            [
                ToolCallDelta(
                    id="call_1",
                    index=0,
                    name="TestModel",
                    arguments='{"name": "Alice"}',
                ),
                ToolCallDelta(
                    id="call_2",
                    index=1,
                    name="NestedModel",
                    arguments='{"title": "Task"}',
                ),
            ]
        )

        deltas = parallel_base._extract_tool_deltas(chunk, Mode.PARALLEL_TOOLS)

        assert len(deltas) == 2
        assert deltas[0].name == "TestModel"
        assert deltas[1].name == "NestedModel"

    def test_extract_tool_deltas_direct_delta_input(self):
        """Test extraction when chunk is a ToolCallDelta itself."""
        parallel_base = AsyncParallelBase(TestModel)

        delta = ToolCallDelta(
            id="call_1", index=0, name="TestModel", arguments='{"name": "Alice"}'
        )
        deltas = parallel_base._extract_tool_deltas(delta, Mode.PARALLEL_TOOLS)

        # AsyncParallelBase implementation handles ToolCallDelta directly
        assert len(deltas) == 1
        assert deltas[0] == delta

    def test_extract_tool_deltas_different_modes(self):
        """Test extraction with different mode values."""
        parallel_base = AsyncParallelBase(TestModel)

        chunk = MockStreamingChunk(
            [
                ToolCallDelta(
                    id="call_1",
                    index=0,
                    name="TestModel",
                    arguments='{"name": "Alice"}',
                )
            ]
        )

        # Test with PARALLEL_TOOLS mode
        deltas1 = parallel_base._extract_tool_deltas(chunk, Mode.PARALLEL_TOOLS)
        assert len(deltas1) == 1

        # Test with different mode (implementation should still work)
        deltas2 = parallel_base._extract_tool_deltas(chunk, Mode.MD_JSON)
        assert len(deltas2) == 1


# ============================================================================
# ParallelBase Parameterized Tests
# ============================================================================


class TestParallelBaseParameterized:
    """Parameterized tests for ParallelBase streaming methods."""

    @pytest.mark.parametrize(
        "num_chunks,chunk_size",
        [
            (1, 100),
            (2, 50),
            (5, 20),
            (10, 10),
        ],
    )
    def test_parameterized_sync_streaming_chunk_sizes(self, num_chunks, chunk_size):
        """Test synchronous streaming with various chunk sizes."""
        parallel_base = AsyncParallelBase(TestModel)

        full_json = '{"name": "Alice", "age": 30, "email": "alice@example.com"}'
        chunks = []

        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(full_json))
            chunk_str = full_json[start:end]

            chunk = MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments=chunk_str
                    )
                ]
            )
            chunks.append(chunk)

        results = list(
            parallel_base.from_streaming_response(chunks, mode=Mode.PARALLEL_TOOLS)
        )

        assert len(results) == num_chunks

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "num_chunks,chunk_size",
        [
            (1, 100),
            (2, 50),
            (5, 20),
        ],
    )
    async def test_parameterized_async_streaming_chunk_sizes(
        self, num_chunks, chunk_size
    ):
        """Test asynchronous streaming with various chunk sizes."""
        parallel_base = AsyncParallelBase(TestModel)

        full_json = '{"name": "Alice", "age": 30, "email": "alice@example.com"}'
        chunks = []

        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(full_json))
            chunk_str = full_json[start:end]

            chunk = MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments=chunk_str
                    )
                ]
            )
            chunks.append(chunk)

        results = []
        async for result in parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks), mode=Mode.PARALLEL_TOOLS
        ):
            results.append(result)

        assert len(results) == num_chunks

    @pytest.mark.parametrize(
        "model_class,json_data",
        [
            (TestModel, '{"name": "Alice", "age": 30}'),
            (NestedModel, '{"title": "Task", "description": "Valid description"}'),
            (ComplexModel, '{"number": 42, "text": "Hello"}'),
        ],
    )
    def test_parameterized_sync_different_models(self, model_class, json_data):
        """Test synchronous streaming with different model types."""
        parallel_base = AsyncParallelBase(model_class)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1",
                        index=0,
                        name=model_class.__name__,
                        arguments=json_data,
                    )
                ]
            ),
        ]

        results = list(
            parallel_base.from_streaming_response(chunks, mode=Mode.PARALLEL_TOOLS)
        )

        assert len(results) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model_class,json_data",
        [
            (TestModel, '{"name": "Alice", "age": 30}'),
            (NestedModel, '{"title": "Task", "description": "Valid description"}'),
        ],
    )
    async def test_parameterized_async_different_models(self, model_class, json_data):
        """Test asynchronous streaming with different model types."""
        parallel_base = AsyncParallelBase(model_class)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1",
                        index=0,
                        name=model_class.__name__,
                        arguments=json_data,
                    )
                ]
            ),
        ]

        results = []
        async for result in parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks), mode=Mode.PARALLEL_TOOLS
        ):
            results.append(result)

        assert len(results) == 1

    @pytest.mark.parametrize(
        "validation_context,strict",
        [
            (None, None),
            ({"key": "value"}, None),
            (None, True),
            (None, False),
            ({"user_id": "123"}, True),
            ({"user_id": "456"}, False),
        ],
    )
    def test_parameterized_sync_validation_options(self, validation_context, strict):
        """Test synchronous streaming with various validation options."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1",
                        index=0,
                        name="TestModel",
                        arguments='{"name": "Alice", "age": 30}',
                    )
                ]
            ),
        ]

        results = list(
            parallel_base.from_streaming_response(
                chunks,
                mode=Mode.PARALLEL_TOOLS,
                validation_context=validation_context,
                strict=strict,
            )
        )

        assert len(results) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "validation_context,strict",
        [
            (None, None),
            ({"key": "value"}, None),
            (None, True),
            (None, False),
        ],
    )
    async def test_parameterized_async_validation_options(
        self, validation_context, strict
    ):
        """Test asynchronous streaming with various validation options."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1",
                        index=0,
                        name="TestModel",
                        arguments='{"name": "Alice", "age": 30}',
                    )
                ]
            ),
        ]

        results = []
        async for result in parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks),
            mode=Mode.PARALLEL_TOOLS,
            validation_context=validation_context,
            strict=strict,
        ):
            results.append(result)

        assert len(results) == 1

    @pytest.mark.parametrize("num_parallel_calls", [1, 2, 3, 5])
    def test_parameterized_sync_parallel_calls(self, num_parallel_calls):
        """Test synchronous streaming with various numbers of parallel calls."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = []
        for i in range(num_parallel_calls):
            chunk = MockStreamingChunk(
                [
                    ToolCallDelta(
                        id=f"call_{i}",
                        index=i,
                        name="TestModel",
                        arguments=f'{{"name": "User{i}", "age": {20 + i}}}',
                    )
                ]
            )
            chunks.append(chunk)

        results = list(
            parallel_base.from_streaming_response(chunks, mode=Mode.PARALLEL_TOOLS)
        )

        assert len(results) == num_parallel_calls

    @pytest.mark.asyncio
    @pytest.mark.parametrize("num_parallel_calls", [1, 2, 3])
    async def test_parameterized_async_parallel_calls(self, num_parallel_calls):
        """Test asynchronous streaming with various numbers of parallel calls."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = []
        for i in range(num_parallel_calls):
            chunk = MockStreamingChunk(
                [
                    ToolCallDelta(
                        id=f"call_{i}",
                        index=i,
                        name="TestModel",
                        arguments=f'{{"name": "User{i}", "age": {20 + i}}}',
                    )
                ]
            )
            chunks.append(chunk)

        results = []
        async for result in parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks), mode=Mode.PARALLEL_TOOLS
        ):
            results.append(result)

        assert len(results) == num_parallel_calls


# ============================================================================
# ParallelBase Edge Case and Error Handling Tests
# ============================================================================


class TestParallelBaseEdgeCases:
    """Edge case and error handling tests for ParallelBase streaming."""

    def test_edge_case_sync_empty_json_braces(self):
        """Test handling of empty JSON braces in sync streaming."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk(
                [ToolCallDelta(id="call_1", index=0, name="TestModel", arguments="{}")]
            ),
        ]

        results = list(
            parallel_base.from_streaming_response(chunks, mode=Mode.PARALLEL_TOOLS)
        )

        assert len(results) == 1

    def test_edge_case_sync_whitespace_only_chunks(self):
        """Test handling of whitespace-only chunks in sync streaming."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk(
                [ToolCallDelta(id="call_1", index=0, name="TestModel", arguments="  ")]
            ),
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments="\t\n"
                    )
                ]
            ),
        ]

        # Should not raise errors
        results = list(
            parallel_base.from_streaming_response(chunks, mode=Mode.PARALLEL_TOOLS)
        )

        assert len(results) == 2

    def test_edge_case_sync_none_id_tool_call(self):
        """Test handling of tool calls with None ID in sync streaming."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id=None,
                        index=0,
                        name="TestModel",
                        arguments='{"name": "Alice"}',
                    )
                ]
            ),
        ]

        results = list(
            parallel_base.from_streaming_response(chunks, mode=Mode.PARALLEL_TOOLS)
        )

        # Should use index-based ID generation
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_edge_case_async_empty_json_braces(self):
        """Test handling of empty JSON braces in async streaming."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk(
                [ToolCallDelta(id="call_1", index=0, name="TestModel", arguments="{}")]
            ),
        ]

        results = []
        async for result in parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks), mode=Mode.PARALLEL_TOOLS
        ):
            results.append(result)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_edge_case_async_none_id_tool_call(self):
        """Test handling of tool calls with None ID in async streaming."""
        parallel_base = AsyncParallelBase(TestModel)

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id=None,
                        index=0,
                        name="TestModel",
                        arguments='{"name": "Alice"}',
                    )
                ]
            ),
        ]

        results = []
        async for result in parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks), mode=Mode.PARALLEL_TOOLS
        ):
            results.append(result)

        assert len(results) == 1

    def test_edge_case_sync_very_long_single_chunk(self):
        """Test handling of very long single chunk in sync streaming."""
        parallel_base = AsyncParallelBase(TestModel)

        long_text = "A" * 10000
        json_data = f'{{"name": "{long_text}", "age": 30}}'

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments=json_data
                    )
                ]
            ),
        ]

        results = list(
            parallel_base.from_streaming_response(chunks, mode=Mode.PARALLEL_TOOLS)
        )

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_edge_case_async_very_long_single_chunk(self):
        """Test handling of very long single chunk in async streaming."""
        parallel_base = AsyncParallelBase(TestModel)

        long_text = "A" * 5000
        json_data = f'{{"name": "{long_text}", "age": 30}}'

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments=json_data
                    )
                ]
            ),
        ]

        results = []
        async for result in parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks), mode=Mode.PARALLEL_TOOLS
        ):
            results.append(result)

        assert len(results) == 1

    def test_edge_case_sync_special_characters_json(self):
        """Test handling of special characters in JSON in sync streaming."""
        parallel_base = AsyncParallelBase(TestModel)

        json_data = '{"name": "Test\\"Quote\\nNewline\\tTab", "age": 30}'

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments=json_data
                    )
                ]
            ),
        ]

        results = list(
            parallel_base.from_streaming_response(chunks, mode=Mode.PARALLEL_TOOLS)
        )

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_edge_case_async_special_characters_json(self):
        """Test handling of special characters in JSON in async streaming."""
        parallel_base = AsyncParallelBase(TestModel)

        json_data = '{"name": "Test\\"Quote\\nNewline", "age": 30}'

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments=json_data
                    )
                ]
            ),
        ]

        results = []
        async for result in parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks), mode=Mode.PARALLEL_TOOLS
        ):
            results.append(result)

        assert len(results) == 1

    def test_edge_case_sync_unicode_json(self):
        """Test handling of unicode characters in JSON in sync streaming."""
        parallel_base = AsyncParallelBase(TestModel)

        json_data = '{"name": "中文日本語한글", "age": 30}'

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments=json_data
                    )
                ]
            ),
        ]

        results = list(
            parallel_base.from_streaming_response(chunks, mode=Mode.PARALLEL_TOOLS)
        )

        assert len(results) == 1


# ============================================================================
# OpenAI-Specific Integration Tests
# ============================================================================


class MockOpenAIFunction:
    """Mock OpenAI Function object."""

    def __init__(self, name: Optional[str] = None, arguments: Optional[str] = ""):
        self.name = name
        self.arguments = arguments


class MockOpenAIToolCall:
    """Mock OpenAI ToolCall object for streaming."""

    def __init__(
        self,
        id: Optional[str] = None,
        index: int = 0,
        function: Optional[MockOpenAIFunction] = None,
    ):
        self.id = id
        self.index = index
        self.function = function or MockOpenAIFunction()


class MockOpenAIDelta:
    """Mock OpenAI ChatCompletionChunk delta."""

    def __init__(self, tool_calls: Optional[list] = None):
        self.tool_calls = tool_calls


class MockOpenAIChoice:
    """Mock OpenAI Choice object."""

    def __init__(self, delta: Optional[MockOpenAIDelta] = None):
        self.delta = delta or MockOpenAIDelta()


class MockOpenAIChunk:
    """Mock OpenAI ChatCompletionChunk."""

    def __init__(self, choices: Optional[list] = None):
        self.choices = choices or []


# ============================================================================
# TestOpenAIParallelBase Tests
# ============================================================================


class TestOpenAIParallelBase:
    """Test cases for OpenAIParallelBase initialization and attributes."""

    @pytest.fixture
    def openai_parallel_base(self):
        """OpenAI ParallelBase fixture."""
        from instructor.dsl.parallel import OpenAIParallelBase

        return OpenAIParallelBase(TestModel, NestedModel, ComplexModel)

    def test_openai_parallel_base_initialization(self, openai_parallel_base):
        """Test OpenAI-specific initialization."""
        assert len(openai_parallel_base.models) == 3
        assert "TestModel" in openai_parallel_base.registry
        assert "NestedModel" in openai_parallel_base.registry
        assert "ComplexModel" in openai_parallel_base.registry

    def test_openai_parallel_base_model_registration(self, openai_parallel_base):
        """Test model registration in OpenAI ParallelBase."""
        assert openai_parallel_base.registry["TestModel"] == TestModel
        assert openai_parallel_base.registry["NestedModel"] == NestedModel
        assert openai_parallel_base.registry["ComplexModel"] == ComplexModel

    def test_openai_parallel_base_inheritance(self, openai_parallel_base):
        """Test that OpenAIParallelBase inherits from ParallelBase."""
        from instructor.dsl.parallel import ParallelBase

        assert isinstance(openai_parallel_base, ParallelBase)

    def test_openai_parallel_base_requires_models(self):
        """Test that OpenAI ParallelBase requires at least one model."""
        from instructor.dsl.parallel import OpenAIParallelBase

        with pytest.raises(AssertionError) as exc_info:
            OpenAIParallelBase()
        assert "At least one model is required" in str(exc_info.value)

    def test_openai_parallel_base_single_model(self):
        """Test OpenAIParallelBase with single model."""
        from instructor.dsl.parallel import OpenAIParallelBase

        parallel_base = OpenAIParallelBase(TestModel)
        assert len(parallel_base.models) == 1
        assert "TestModel" in parallel_base.registry


# ============================================================================
# TestOpenAIExtractToolDeltas Tests
# ============================================================================


class TestOpenAIExtractToolDeltas:
    """Test cases for OpenAI-specific _extract_tool_deltas method."""

    @pytest.fixture
    def openai_parallel_base(self):
        """OpenAI ParallelBase fixture."""
        from instructor.dsl.parallel import OpenAIParallelBase

        return OpenAIParallelBase(TestModel, NestedModel)

    def test_extract_tool_deltas_with_valid_openai_chunk(self, openai_parallel_base):
        """Test extraction with valid OpenAI chunk containing tool_calls."""
        chunk = MockOpenAIChunk(
            choices=[
                MockOpenAIChoice(
                    delta=MockOpenAIDelta(
                        tool_calls=[
                            MockOpenAIToolCall(
                                id="call_abc123",
                                index=0,
                                function=MockOpenAIFunction(
                                    name="TestModel", arguments='{"name": "Alice"'
                                ),
                            )
                        ]
                    )
                )
            ]
        )

        deltas = openai_parallel_base._extract_tool_deltas(chunk, Mode.PARALLEL_TOOLS)

        assert len(deltas) == 1
        assert deltas[0].id == "call_abc123"
        assert deltas[0].index == 0
        assert deltas[0].name == "TestModel"
        assert deltas[0].arguments == '{"name": "Alice"'

    def test_extract_tool_deltas_with_multiple_tool_calls(self, openai_parallel_base):
        """Test extraction with multiple tool calls in single chunk."""
        chunk = MockOpenAIChunk(
            choices=[
                MockOpenAIChoice(
                    delta=MockOpenAIDelta(
                        tool_calls=[
                            MockOpenAIToolCall(
                                id="call_1",
                                index=0,
                                function=MockOpenAIFunction(
                                    name="TestModel", arguments='{"name": "Alice"'
                                ),
                            ),
                            MockOpenAIToolCall(
                                id="call_2",
                                index=1,
                                function=MockOpenAIFunction(
                                    name="NestedModel", arguments='{"title": "Task"'
                                ),
                            ),
                        ]
                    )
                )
            ]
        )

        deltas = openai_parallel_base._extract_tool_deltas(chunk, Mode.PARALLEL_TOOLS)

        assert len(deltas) == 2
        assert deltas[0].id == "call_1"
        assert deltas[0].name == "TestModel"
        assert deltas[1].id == "call_2"
        assert deltas[1].name == "NestedModel"

    def test_extract_tool_deltas_with_missing_tool_calls(self, openai_parallel_base):
        """Test extraction when tool_calls is None."""
        chunk = MockOpenAIChunk(
            choices=[MockOpenAIChoice(delta=MockOpenAIDelta(tool_calls=None))]
        )

        deltas = openai_parallel_base._extract_tool_deltas(chunk, Mode.PARALLEL_TOOLS)

        assert len(deltas) == 0

    def test_extract_tool_deltas_with_missing_choices(self, openai_parallel_base):
        """Test extraction when choices is missing."""
        chunk = MockOpenAIChunk(choices=None)

        deltas = openai_parallel_base._extract_tool_deltas(chunk, Mode.PARALLEL_TOOLS)

        assert len(deltas) == 0

    def test_extract_tool_deltas_with_missing_delta(self, openai_parallel_base):
        """Test extraction when delta is missing."""
        chunk = MockOpenAIChunk(choices=[MockOpenAIChoice(delta=None)])

        deltas = openai_parallel_base._extract_tool_deltas(chunk, Mode.PARALLEL_TOOLS)

        assert len(deltas) == 0

    def test_extract_tool_deltas_with_missing_function(self, openai_parallel_base):
        """Test extraction when function is None."""
        chunk = MockOpenAIChunk(
            choices=[
                MockOpenAIChoice(
                    delta=MockOpenAIDelta(
                        tool_calls=[
                            MockOpenAIToolCall(
                                id="call_123",
                                index=0,
                                function=None,
                            )
                        ]
                    )
                )
            ]
        )

        deltas = openai_parallel_base._extract_tool_deltas(chunk, Mode.PARALLEL_TOOLS)

        assert len(deltas) == 1
        assert deltas[0].id == "call_123"
        assert deltas[0].name is None
        assert deltas[0].arguments == ""

    def test_extract_tool_deltas_with_none_values(self, openai_parallel_base):
        """Test extraction with None values for id and name (early chunks)."""
        chunk = MockOpenAIChunk(
            choices=[
                MockOpenAIChoice(
                    delta=MockOpenAIDelta(
                        tool_calls=[
                            MockOpenAIToolCall(
                                id=None,
                                index=0,
                                function=MockOpenAIFunction(
                                    name=None, arguments='{"name"'
                                ),
                            )
                        ]
                    )
                )
            ]
        )

        deltas = openai_parallel_base._extract_tool_deltas(chunk, Mode.PARALLEL_TOOLS)

        assert len(deltas) == 1
        assert deltas[0].id is None
        assert deltas[0].name is None
        assert deltas[0].arguments == '{"name"'

    def test_extract_tool_deltas_wrong_mode(self, openai_parallel_base):
        """Test extraction returns empty for non-PARALLEL_TOOLS mode."""
        chunk = MockOpenAIChunk(
            choices=[
                MockOpenAIChoice(
                    delta=MockOpenAIDelta(
                        tool_calls=[
                            MockOpenAIToolCall(
                                id="call_123",
                                function=MockOpenAIFunction(
                                    name="TestModel", arguments="{}"
                                ),
                            )
                        ]
                    )
                )
            ]
        )

        # Test with MD_JSON mode
        deltas = openai_parallel_base._extract_tool_deltas(chunk, Mode.MD_JSON)
        assert len(deltas) == 0

    def test_extract_tool_deltas_empty_chunk(self, openai_parallel_base):
        """Test extraction with completely empty chunk."""
        chunk = MockOpenAIChunk(choices=[])

        deltas = openai_parallel_base._extract_tool_deltas(chunk, Mode.PARALLEL_TOOLS)

        assert len(deltas) == 0

    @pytest.mark.parametrize(
        "expected_args",
        [
            ('{"name": "Alice"'),
            (', "age": 30}'),
            ('""'),
            ("{}"),
            (""),
        ],
    )
    def test_extract_tool_deltas_various_arguments(
        self, openai_parallel_base, expected_args
    ):
        """Test extraction with various argument strings."""
        chunk = MockOpenAIChunk(
            choices=[
                MockOpenAIChoice(
                    delta=MockOpenAIDelta(
                        tool_calls=[
                            MockOpenAIToolCall(
                                id="call_1",
                                function=MockOpenAIFunction(
                                    name="TestModel",
                                    arguments=expected_args,
                                ),
                            )
                        ]
                    )
                )
            ]
        )

        deltas = openai_parallel_base._extract_tool_deltas(chunk, Mode.PARALLEL_TOOLS)

        assert len(deltas) == 1
        assert deltas[0].arguments == expected_args

    @pytest.mark.parametrize(
        "tool_call_id,index,expected_id",
        [
            ("call_123", 0, "call_123"),
            (None, 0, None),
            ("call_456", 5, "call_456"),
        ],
    )
    def test_extract_tool_deltas_various_ids(
        self, openai_parallel_base, tool_call_id, index, expected_id
    ):
        """Test extraction with various id values."""
        chunk = MockOpenAIChunk(
            choices=[
                MockOpenAIChoice(
                    delta=MockOpenAIDelta(
                        tool_calls=[
                            MockOpenAIToolCall(
                                id=tool_call_id,
                                index=index,
                                function=MockOpenAIFunction(
                                    name="TestModel", arguments="{}"
                                ),
                            )
                        ]
                    )
                )
            ]
        )

        deltas = openai_parallel_base._extract_tool_deltas(chunk, Mode.PARALLEL_TOOLS)

        assert len(deltas) == 1
        assert deltas[0].id == expected_id
        assert deltas[0].index == index


# ============================================================================
# TestOpenAIStreamingIntegration Tests
# ============================================================================


class TestOpenAIStreamingIntegration:
    """Integration tests for OpenAI streaming with real-world scenarios."""

    @pytest.fixture
    def openai_parallel_base(self):
        """OpenAI ParallelBase fixture."""
        from instructor.dsl.parallel import OpenAIParallelBase

        return OpenAIParallelBase(TestModel, NestedModel, ComplexModel)

    def test_openai_streaming_single_tool_call(self, openai_parallel_base):
        """Test streaming with single complete tool call."""
        chunks = [
            MockOpenAIChunk(
                choices=[
                    MockOpenAIChoice(
                        delta=MockOpenAIDelta(
                            tool_calls=[
                                MockOpenAIToolCall(
                                    id="call_1",
                                    index=0,
                                    function=MockOpenAIFunction(
                                        name="TestModel",
                                        arguments='{"name": "Alice", "age": 30}',
                                    ),
                                )
                            ]
                        )
                    )
                ]
            )
        ]

        results = list(
            openai_parallel_base.from_streaming_response(chunks, Mode.PARALLEL_TOOLS)
        )

        assert len(results) == 1
        assert "call_1" in results[0].partial
        model = results[0].partial["call_1"]
        assert model.name == "Alice"
        assert model.age == 30

    def test_openai_streaming_multiple_tool_calls(self, openai_parallel_base):
        """Test streaming with multiple complete tool calls."""
        chunks = [
            MockOpenAIChunk(
                choices=[
                    MockOpenAIChoice(
                        delta=MockOpenAIDelta(
                            tool_calls=[
                                MockOpenAIToolCall(
                                    id="call_1",
                                    index=0,
                                    function=MockOpenAIFunction(
                                        name="TestModel",
                                        arguments='{"name": "Alice", "age": 30}',
                                    ),
                                ),
                                MockOpenAIToolCall(
                                    id="call_2",
                                    index=1,
                                    function=MockOpenAIFunction(
                                        name="NestedModel",
                                        arguments='{"title": "Task 1", "description": "A valid description"}',
                                    ),
                                ),
                            ]
                        )
                    )
                ]
            )
        ]

        results = list(
            openai_parallel_base.from_streaming_response(chunks, Mode.PARALLEL_TOOLS)
        )

        # Each tool_call in a chunk generates a separate ParallelResult
        assert len(results) == 2
        # First result has call_1
        assert "call_1" in results[0].partial
        # Second result has both calls (accumulated)
        assert "call_1" in results[1].partial
        assert "call_2" in results[1].partial

    def test_openai_streaming_partial_json(self, openai_parallel_base):
        """Test streaming with partial JSON across multiple chunks."""
        chunks = [
            MockOpenAIChunk(
                choices=[
                    MockOpenAIChoice(
                        delta=MockOpenAIDelta(
                            tool_calls=[
                                MockOpenAIToolCall(
                                    id="call_1",
                                    index=0,
                                    function=MockOpenAIFunction(
                                        name="TestModel",
                                        arguments='{"n',
                                    ),
                                )
                            ]
                        )
                    )
                ]
            ),
            MockOpenAIChunk(
                choices=[
                    MockOpenAIChoice(
                        delta=MockOpenAIDelta(
                            tool_calls=[
                                MockOpenAIToolCall(
                                    id="call_1",
                                    index=0,
                                    function=MockOpenAIFunction(
                                        name="TestModel",
                                        arguments='ame": "Alice"',
                                    ),
                                )
                            ]
                        )
                    )
                ]
            ),
            MockOpenAIChunk(
                choices=[
                    MockOpenAIChoice(
                        delta=MockOpenAIDelta(
                            tool_calls=[
                                MockOpenAIToolCall(
                                    id="call_1",
                                    index=0,
                                    function=MockOpenAIFunction(
                                        name="TestModel",
                                        arguments=', "age": 3',
                                    ),
                                )
                            ]
                        )
                    )
                ]
            ),
            MockOpenAIChunk(
                choices=[
                    MockOpenAIChoice(
                        delta=MockOpenAIDelta(
                            tool_calls=[
                                MockOpenAIToolCall(
                                    id="call_1",
                                    index=0,
                                    function=MockOpenAIFunction(
                                        name="TestModel",
                                        arguments="0}",
                                    ),
                                )
                            ]
                        )
                    )
                ]
            ),
        ]

        results = list(
            openai_parallel_base.from_streaming_response(chunks, Mode.PARALLEL_TOOLS)
        )

        assert len(results) == 4
        # Final result should have complete model
        final_result = results[-1]
        assert "call_1" in final_result.partial
        model = final_result.partial["call_1"]
        assert model.name == "Alice"
        assert model.age == 30

    def test_openai_streaming_complete_json(self, openai_parallel_base):
        """Test streaming with complete JSON in single chunk."""
        chunks = [
            MockOpenAIChunk(
                choices=[
                    MockOpenAIChoice(
                        delta=MockOpenAIDelta(
                            tool_calls=[
                                MockOpenAIToolCall(
                                    id="call_complete",
                                    index=0,
                                    function=MockOpenAIFunction(
                                        name="TestModel",
                                        arguments='{"name": "Bob", "age": 25, "email": "bob@example.com"}',
                                    ),
                                )
                            ]
                        )
                    )
                ]
            )
        ]

        results = list(
            openai_parallel_base.from_streaming_response(chunks, Mode.PARALLEL_TOOLS)
        )

        assert len(results) == 1
        assert "call_complete" in results[0].partial
        model = results[0].partial["call_complete"]
        assert model.name == "Bob"
        assert model.age == 25
        assert model.email == "bob@example.com"

    def test_openai_streaming_mixed_complete_partial(self, openai_parallel_base):
        """Test streaming with mix of complete and partial models."""
        chunks = [
            # First call - complete early
            MockOpenAIChunk(
                choices=[
                    MockOpenAIChoice(
                        delta=MockOpenAIDelta(
                            tool_calls=[
                                MockOpenAIToolCall(
                                    id="call_complete",
                                    index=0,
                                    function=MockOpenAIFunction(
                                        name="TestModel",
                                        arguments='{"name": "Complete", "age": 100}',
                                    ),
                                )
                            ]
                        )
                    )
                ]
            ),
            # Second call - partial
            MockOpenAIChunk(
                choices=[
                    MockOpenAIChoice(
                        delta=MockOpenAIDelta(
                            tool_calls=[
                                MockOpenAIToolCall(
                                    id="call_partial",
                                    index=1,
                                    function=MockOpenAIFunction(
                                        name="NestedModel",
                                        arguments='{"title": "Partial"',
                                    ),
                                )
                            ]
                        )
                    )
                ]
            ),
            MockOpenAIChunk(
                choices=[
                    MockOpenAIChoice(
                        delta=MockOpenAIDelta(
                            tool_calls=[
                                MockOpenAIToolCall(
                                    id="call_partial",
                                    index=1,
                                    function=MockOpenAIFunction(
                                        name="NestedModel",
                                        arguments=', "description": "In progress"}',
                                    ),
                                )
                            ]
                        )
                    )
                ]
            ),
        ]

        results = list(
            openai_parallel_base.from_streaming_response(chunks, Mode.PARALLEL_TOOLS)
        )

        # Should have 3 results (first chunk, second chunk, third chunk)
        assert len(results) >= 2
        final_result = results[-1]
        # Should have both calls tracked
        assert len(final_result.partial) >= 1

    @pytest.mark.asyncio
    async def test_openai_streaming_async_single_tool_call(self, openai_parallel_base):
        """Test async streaming with single complete tool call."""
        chunks = [
            MockOpenAIChunk(
                choices=[
                    MockOpenAIChoice(
                        delta=MockOpenAIDelta(
                            tool_calls=[
                                MockOpenAIToolCall(
                                    id="call_1",
                                    index=0,
                                    function=MockOpenAIFunction(
                                        name="TestModel",
                                        arguments='{"name": "Alice", "age": 30}',
                                    ),
                                )
                            ]
                        )
                    )
                ]
            )
        ]

        results = []
        async for result in openai_parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks), Mode.PARALLEL_TOOLS
        ):
            results.append(result)

        assert len(results) == 1
        assert "call_1" in results[0].partial
        model = results[0].partial["call_1"]
        assert model.name == "Alice"
        assert model.age == 30

    @pytest.mark.asyncio
    async def test_openai_streaming_async_multiple_calls(self, openai_parallel_base):
        """Test async streaming with multiple concurrent tool calls."""
        chunks = [
            MockOpenAIChunk(
                choices=[
                    MockOpenAIChoice(
                        delta=MockOpenAIDelta(
                            tool_calls=[
                                MockOpenAIToolCall(
                                    id="call_1",
                                    index=0,
                                    function=MockOpenAIFunction(
                                        name="TestModel",
                                        arguments='{"name": "Alice"',
                                    ),
                                )
                            ]
                        )
                    )
                ]
            ),
            MockOpenAIChunk(
                choices=[
                    MockOpenAIChoice(
                        delta=MockOpenAIDelta(
                            tool_calls=[
                                MockOpenAIToolCall(
                                    id="call_2",
                                    index=1,
                                    function=MockOpenAIFunction(
                                        name="NestedModel",
                                        arguments='{"title": "Task"',
                                    ),
                                )
                            ]
                        )
                    )
                ]
            ),
            MockOpenAIChunk(
                choices=[
                    MockOpenAIChoice(
                        delta=MockOpenAIDelta(
                            tool_calls=[
                                MockOpenAIToolCall(
                                    id="call_1",
                                    index=0,
                                    function=MockOpenAIFunction(
                                        name="TestModel",
                                        arguments=', "age": 30}',
                                    ),
                                ),
                                MockOpenAIToolCall(
                                    id="call_2",
                                    index=1,
                                    function=MockOpenAIFunction(
                                        name="NestedModel",
                                        arguments=', "description": "Valid desc"}',
                                    ),
                                ),
                            ]
                        )
                    )
                ]
            ),
        ]

        results = []
        async for result in openai_parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks), Mode.PARALLEL_TOOLS
        ):
            results.append(result)

        # Chunk 1: call_1 (partial, yields result)
        # Chunk 2: call_2 (partial, yields result)
        # Chunk 3: call_1 update (yields result) + call_2 update (yields result)
        assert len(results) == 4
        final_result = results[-1]
        assert len(final_result.partial) == 2
        assert "call_1" in final_result.partial
        assert "call_2" in final_result.partial

    @pytest.mark.parametrize(
        "json_string",
        [
            '{"name": "Alice", "age": 30}',
            '{"title": "Task", "description": "Description"}',
            '{"number": 42, "text": "Hello", "optional_text": null}',
        ],
    )
    def test_openai_streaming_various_model_types(
        self, openai_parallel_base, json_string
    ):
        """Test streaming with various valid JSON structures."""
        import json

        data = json.loads(json_string)

        if "age" in data:
            model_name = "TestModel"
        elif "title" in data:
            model_name = "NestedModel"
        else:
            model_name = "ComplexModel"

        chunks = [
            MockOpenAIChunk(
                choices=[
                    MockOpenAIChoice(
                        delta=MockOpenAIDelta(
                            tool_calls=[
                                MockOpenAIToolCall(
                                    id="call_1",
                                    index=0,
                                    function=MockOpenAIFunction(
                                        name=model_name,
                                        arguments=json_string,
                                    ),
                                )
                            ]
                        )
                    )
                ]
            )
        ]

        results = list(
            openai_parallel_base.from_streaming_response(chunks, Mode.PARALLEL_TOOLS)
        )

        assert len(results) == 1
        assert "call_1" in results[0].partial

    def test_openai_streaming_validation_context(self, openai_parallel_base):
        """Test streaming with validation context."""
        chunks = [
            MockOpenAIChunk(
                choices=[
                    MockOpenAIChoice(
                        delta=MockOpenAIDelta(
                            tool_calls=[
                                MockOpenAIToolCall(
                                    id="call_1",
                                    index=0,
                                    function=MockOpenAIFunction(
                                        name="TestModel",
                                        arguments='{"name": "Alice", "age": 30}',
                                    ),
                                )
                            ]
                        )
                    )
                ]
            )
        ]

        validation_context = {"user_id": "123", "request_id": "abc"}
        results = list(
            openai_parallel_base.from_streaming_response(
                chunks, Mode.PARALLEL_TOOLS, validation_context=validation_context
            )
        )

        assert len(results) == 1
        assert "call_1" in results[0].partial

    def test_openai_streaming_strict_mode(self, openai_parallel_base):
        """Test streaming with strict mode."""
        chunks = [
            MockOpenAIChunk(
                choices=[
                    MockOpenAIChoice(
                        delta=MockOpenAIDelta(
                            tool_calls=[
                                MockOpenAIToolCall(
                                    id="call_1",
                                    index=0,
                                    function=MockOpenAIFunction(
                                        name="TestModel",
                                        arguments='{"name": "Alice", "age": 30}',
                                    ),
                                )
                            ]
                        )
                    )
                ]
            )
        ]

        # Test with strict=False
        results = list(
            openai_parallel_base.from_streaming_response(
                chunks, Mode.PARALLEL_TOOLS, strict=False
            )
        )

        assert len(results) == 1

    def test_openai_streaming_interleaved_chunks(self, openai_parallel_base):
        """Test streaming with interleaved chunks for different calls."""
        chunks = [
            # First call partial
            MockOpenAIChunk(
                choices=[
                    MockOpenAIChoice(
                        delta=MockOpenAIDelta(
                            tool_calls=[
                                MockOpenAIToolCall(
                                    id="call_1",
                                    index=0,
                                    function=MockOpenAIFunction(
                                        name="TestModel",
                                        arguments='{"n',
                                    ),
                                )
                            ]
                        )
                    )
                ]
            ),
            # Second call partial
            MockOpenAIChunk(
                choices=[
                    MockOpenAIChoice(
                        delta=MockOpenAIDelta(
                            tool_calls=[
                                MockOpenAIToolCall(
                                    id="call_2",
                                    index=1,
                                    function=MockOpenAIFunction(
                                        name="NestedModel",
                                        arguments='{"ti',
                                    ),
                                )
                            ]
                        )
                    )
                ]
            ),
            # Both calls complete
            MockOpenAIChunk(
                choices=[
                    MockOpenAIChoice(
                        delta=MockOpenAIDelta(
                            tool_calls=[
                                MockOpenAIToolCall(
                                    id="call_1",
                                    index=0,
                                    function=MockOpenAIFunction(
                                        name="TestModel",
                                        arguments='ame": "Alice", "age": 30}',
                                    ),
                                ),
                                MockOpenAIToolCall(
                                    id="call_2",
                                    index=1,
                                    function=MockOpenAIFunction(
                                        name="NestedModel",
                                        arguments='tle": "Task", "description": "Valid desc"}',
                                    ),
                                ),
                            ]
                        )
                    )
                ]
            ),
        ]

        results = list(
            openai_parallel_base.from_streaming_response(chunks, Mode.PARALLEL_TOOLS)
        )

        # Chunk 1: call_1 (1 result)
        # Chunk 2: call_2 (1 result)
        # Chunk 3: call_1 update + call_2 update (2 results)
        assert len(results) == 4
        final_result = results[-1]
        assert len(final_result.partial) == 2

    @pytest.mark.asyncio
    async def test_edge_case_async_unicode_json(self):
        """Test handling of unicode characters in JSON in async streaming."""
        parallel_base = AsyncParallelBase(TestModel)

        json_data = '{"name": "中文日本語한글", "age": 30}'

        chunks = [
            MockStreamingChunk(
                [
                    ToolCallDelta(
                        id="call_1", index=0, name="TestModel", arguments=json_data
                    )
                ]
            ),
        ]

        results = []
        async for result in parallel_base.from_streaming_response_async(
            MockAsyncStreamingGenerator(chunks), mode=Mode.PARALLEL_TOOLS
        ):
            results.append(result)

        assert len(results) == 1
