"""
Comprehensive unit tests for parallel streaming data structures.

This module tests the core data structures from instructor.dsl.parallel:
- ToolCallDelta
- ParallelResult  
- ToolCallState
- ParallelStreamHelper

Tests focus on streaming scenarios, edge cases, error handling, and coverage.
"""
from typing import Optional
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field, field_validator

from instructor.dsl.parallel import (
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
        id="call_abc123",
        index=0,
        name="TestModel",
        arguments='{"name": "Alice"'
    )


@pytest.fixture
def tool_call_delta_full():
    """ToolCallDelta with complete JSON arguments."""
    return ToolCallDelta(
        id="call_xyz789",
        index=1,
        name="TestModel",
        arguments='{"name": "Bob", "age": 25}'
    )


@pytest.fixture
def tool_call_delta_no_id():
    """ToolCallDelta without ID (using index-based generation)."""
    return ToolCallDelta(
        id=None,
        index=2,
        name="TestModel",
        arguments='{"name": "Charlie"'
    )


@pytest.fixture
def tool_call_delta_no_name():
    """ToolCallDelta without name (early chunk)."""
    return ToolCallDelta(
        id="call_early",
        index=0,
        name=None,
        arguments=''
    )


@pytest.fixture
def tool_call_state():
    """Basic ToolCallState fixture."""
    return ToolCallState(
        tool_call_id="call_123",
        model_class=TestModel,
        accumulated_json='{"name": "Alice"',
        partial_model=None,
        is_complete=False
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
        is_complete=False
    )


@pytest.fixture
def parallel_result():
    """Basic ParallelResult fixture."""
    completed = {
        "call_done": TestModel(name="Complete", age=100, email="done@test.com")
    }
    partial = {
        "call_partial": TestModel.model_validate({"name": "Partial", "age": 50})
    }
    return ParallelResult(
        completed=completed,
        partial=partial,
        tool_call_id="call_partial"
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
    mock_tool_call2.function.arguments = '{"title": "Task 1", "description": "A description"}'
    
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
            arguments='{"number": 42, "text": "hello"}'
        )
        assert delta.id == "test_id"
        assert delta.index == 5
        assert delta.name == "ComplexModel"
        assert delta.arguments == '{"number": 42, "text": "hello"}'

    @pytest.mark.parametrize("delta_id,expected_id", [
        ("call_1", "call_1"),
        (None, None),
        ("", ""),
    ])
    def test_tool_call_delta_various_ids(self, delta_id, expected_id):
        """Test ToolCallDelta with various ID values."""
        delta = ToolCallDelta(
            id=delta_id,
            index=0,
            name="TestModel",
            arguments='{"name": "test"}'
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
        state = ToolCallState(
            tool_call_id="call_new",
            model_class=NestedModel
        )
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
            is_complete=True
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
                accumulated_json=json_str
            )
            assert state.model_class == model_class
            assert state.accumulated_json == json_str

    def test_tool_call_state_empty_accumulated_json(self):
        """Test ToolCallState with empty accumulated JSON."""
        state = ToolCallState(
            tool_call_id="call_empty",
            model_class=TestModel,
            accumulated_json=""
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
        result = ParallelResult(
            completed={},
            partial={},
            tool_call_id=""
        )
        assert result.get_all_models() == []

    def test_parallel_result_only_completed(self):
        """Test ParallelResult with only completed models."""
        completed = {
            "call_1": TestModel(name="Alice", age=30),
            "call_2": TestModel(name="Bob", age=25)
        }
        result = ParallelResult(
            completed=completed,
            partial={},
            tool_call_id="call_2"
        )
        all_models = result.get_all_models()
        assert len(all_models) == 2
        assert all(m.name in ["Alice", "Bob"] for m in all_models)

    def test_parallel_result_only_partial(self):
        """Test ParallelResult with only partial models."""
        partial = {
            "call_1": TestModel.model_validate({"name": "Charlie", "age": 25}),
            "call_2": NestedModel.model_validate({"title": "Task"})
        }
        result = ParallelResult(
            completed={},
            partial=partial,
            tool_call_id="call_1"
        )
        all_models = result.get_all_models()
        assert len(all_models) == 2

    def test_parallel_result_mixed_model_types(self):
        """Test ParallelResult with different model types."""
        completed = {
            "call_test": TestModel(name="Test", age=20)
        }
        partial = {
            "call_nested": NestedModel.model_validate({"title": "Nested", "description": "Description"}),
            "call_complex": ComplexModel.model_validate({"number": 42, "text": "Complex"})
        }
        result = ParallelResult(
            completed=completed,
            partial=partial,
            tool_call_id="call_complex"
        )
        all_models = result.get_all_models()
        assert len(all_models) == 3
        assert any(m.name == "Test" for m in all_models)
        assert any(hasattr(m, 'title') for m in all_models)

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
            completed=completed,
            partial=partial,
            tool_call_id="call_2"
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
                "call_b": NestedModel(title="B", description="Description B")
            },
            partial={},
            tool_call_id="call_a"
        )
        models1 = result1.get_all_models()
        assert len(models1) == 2
        
        # Case 2: All partial
        result2 = ParallelResult(
            completed={},
            partial={
                "call_c": TestModel(name="C", age=35, email=None),
                "call_d": ComplexModel(number=42, text="D", optional_text=None, nested=None)
            },
            tool_call_id="call_c"
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

    def test_process_delta_first_chunk_creates_state(self, parallel_base, tool_call_delta):
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
            id="call_abc123",
            index=0,
            name="TestModel",
            arguments=', "age": 30}'
        )
        helper.process_delta(delta2)
        
        assert helper.tool_call_states["call_abc123"].accumulated_json == (
            '{"name": "Alice", "age": 30}'
        )

    def test_process_delta_partial_json_parsing(self, parallel_base):
        """Test that process_delta handles partial JSON."""
        helper = ParallelStreamHelper(parallel_base)
        
        # Send partial JSON chunks
        delta1 = ToolCallDelta(
            id="call_1",
            index=0,
            name="TestModel",
            arguments='{"na'
        )
        result1 = helper.process_delta(delta1)
        
        delta2 = ToolCallDelta(
            id="call_1",
            index=0,
            name="TestModel",
            arguments='me": "Alice", "age": 30}'
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
            arguments='{"name": "Alice", "age": 30, "email": "alice@example.com"}'
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
            arguments='{"name": "Alice", "age": 30}'
        )
        result = helper.process_delta(delta)
        
        assert isinstance(result, ParallelResult)
        assert result.tool_call_id == "call_1"

    def test_process_delta_no_name_returns_early(self, parallel_base, tool_call_delta_no_name):
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
            arguments='{"test": "data"}'
        )
        
        with pytest.raises(KeyError) as exc_info:
            helper.process_delta(delta)
        
        assert "Unknown model name 'InvalidModel'" in str(exc_info.value)

    def test_process_delta_tool_call_id_missing_uses_index(self, parallel_base, tool_call_delta_no_id):
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
            id="call_1",
            index=0,
            name="TestModel",
            arguments='{"name": "Alice"'
        )
        helper.process_delta(delta1)
        
        # Second tool call
        delta2 = ToolCallDelta(
            id="call_2",
            index=1,
            name="NestedModel",
            arguments='{"title": "Task}'
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
            arguments='{"name": "Alice", "age": 30}'
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
            arguments='{"name": "Alice", "age": 30}'
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
            arguments='{"name": "Alice", "age": invalid'  # Invalid JSON
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
            arguments='{"text": "partial text'
        )
        
        result = helper.process_delta(delta)
        assert "call_1" in helper.tool_call_states
        assert helper.tool_call_states["call_1"].accumulated_json == '{"text": "partial text'

    def test_process_delta_empty_arguments(self, parallel_base):
        """Test that empty arguments are handled correctly."""
        helper = ParallelStreamHelper(parallel_base)
        
        delta = ToolCallDelta(
            id="call_1",
            index=0,
            name="TestModel",
            arguments=''
        )
        
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
            arguments='{"name": "Alice", "age": 30}'
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
            arguments='{"name": "Alice", "age": 30}'
        )
        helper.process_delta(delta1)
        
        delta2 = ToolCallDelta(
            id="call_2",
            index=1,
            name="TestModel",
            arguments='{"name": "Bob", "age": 25}'
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
            arguments='{"name": "Alice", "age": 30}'
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
            arguments='{"name": "Alice", "age": 30}'
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
            ToolCallDelta(id="call_1", index=0, name="TestModel", arguments='{"name": "Alice"'),
            ToolCallDelta(id="call_1", index=0, name="TestModel", arguments=', "age": 30}'),
            ToolCallDelta(id="call_2", index=1, name="TestModel", arguments='{"name": "Bob"'),
            ToolCallDelta(id="call_2", index=1, name="TestModel", arguments=', "age": 25}'),
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
        
        assert hasattr(helper, 'tool_call_states')
        assert hasattr(helper, '_provided_models')
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
            ToolCallDelta(id="call_1", index=0, name="TestModel", arguments='{"name": "Alice"'),
            ToolCallDelta(id="call_1", index=0, name="TestModel", arguments=', "age": 30}'),
            ToolCallDelta(id="call_2", index=1, name="NestedModel", arguments='{"title": "Task 1"'),
            ToolCallDelta(id="call_2", index=1, name="NestedModel", arguments=', "description": "A valid description"}'),
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
                arguments=f'{{"name": "User{i}", "age": {20 + i}}}'
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
            arguments='{"name": "Alice", "age": 30}'
        )
        helper.process_delta(delta1)
        helper.mark_complete("call_1")
        
        # Add second call, keep partial - need more to create a valid partial model
        delta2 = ToolCallDelta(
            id="call_2",
            index=1,
            name="TestModel",
            arguments='{"name": "Bob",'
        )
        helper.process_delta(delta2)
        
        # Add enough to create a parseable partial model
        delta2b = ToolCallDelta(
            id="call_2",
            index=1,
            name="TestModel",
            arguments=' "age": 25'
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
                id=f"call_{model_name}",
                index=0,
                name=model_name,
                arguments='{}'
            )
            helper.process_delta(delta)
            
            state = helper.tool_call_states[f"call_{model_name}"]
            assert state.model_class == model_class

    # Edge case tests

    def test_edge_case_whitespace_only_arguments(self, parallel_base):
        """Test handling of whitespace-only arguments."""
        helper = ParallelStreamHelper(parallel_base)
        
        delta = ToolCallDelta(
            id="call_1",
            index=0,
            name="TestModel",
            arguments='   '
        )
        result = helper.process_delta(delta)
        
        state = helper.tool_call_states["call_1"]
        assert state.accumulated_json == '   '

    def test_edge_case_empty_string_after_strip(self, parallel_base):
        """Test handling of empty string after strip."""
        helper = ParallelStreamHelper(parallel_base)
        
        delta = ToolCallDelta(
            id="call_1",
            index=0,
            name="TestModel",
            arguments=''
        )
        
        # First delta sets state
        helper.process_delta(delta)
        state = helper.tool_call_states["call_1"]
        assert state.accumulated_json == ''

    def test_edge_case_special_characters_in_json(self, parallel_base):
        """Test handling of special characters in JSON."""
        helper = ParallelStreamHelper(parallel_base)
        
        delta = ToolCallDelta(
            id="call_1",
            index=0,
            name="TestModel",
            arguments='{"name": "Test\\"Quote\\nNewline", "age": 30}'
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
            arguments='{"name": "中文日本語한글", "age": 30}'
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
            arguments=f'{{"name": "{long_str}"}}'
        )
        result = helper.process_delta(delta)
        
        state = helper.tool_call_states["call_1"]
        assert len(state.accumulated_json) > 10000


# ============================================================================
# Parameterized Tests
# ============================================================================

class TestParallelStreamingParameterized:
    """Parameterized tests for parallel streaming scenarios."""

    @pytest.mark.parametrize("model_class,valid_json", [
        (TestModel, '{"name": "Alice", "age": 30}'),
        (NestedModel, '{"title": "Task", "description": "Valid description"}'),
        (ComplexModel, '{"number": 42, "text": "Hello", "optional_text": null}'),
    ])
    def test_parameterized_valid_json_parsing(self, parallel_base, model_class, valid_json):
        """Test that different model types handle valid JSON correctly."""
        helper = ParallelStreamHelper(parallel_base)
        
        delta = ToolCallDelta(
            id="call_1",
            index=0,
            name=model_class.__name__,
            arguments=valid_json
        )
        
        result = helper.process_delta(delta)
        assert result.tool_call_id == "call_1"

    @pytest.mark.parametrize("partial_json,expected_accumulated", [
        ('{"name":', '{"name":'),
        ('{"name": "Alice",', '{"name": "Alice",'),
        ('{"name": "Alice", "age":', '{"name": "Alice", "age":'),
        ('{"name": "Alice", "age": 3', '{"name": "Alice", "age": 3'),
    ])
    def test_parameterized_partial_json_accumulation(self, parallel_base, partial_json, expected_accumulated):
        """Test that partial JSON chunks accumulate correctly."""
        helper = ParallelStreamHelper(parallel_base)
        
        delta = ToolCallDelta(
            id="call_1",
            index=0,
            name="TestModel",
            arguments=partial_json
        )
        
        helper.process_delta(delta)
        assert helper.tool_call_states["call_1"].accumulated_json == expected_accumulated

    @pytest.mark.parametrize("invalid_json", [
        '{"name": "Alice", "age": invalid}',
        '{"name": "Alice", "age": }',
        '{"name": "Alice", "age":',
        'not json at all',
        '{"incomplete"',
    ])
    def test_parameterized_invalid_json_handling(self, parallel_base, invalid_json):
        """Test that various invalid JSON formats are handled gracefully."""
        helper = ParallelStreamHelper(parallel_base)
        
        delta = ToolCallDelta(
            id="call_1",
            index=0,
            name="TestModel",
            arguments=invalid_json
        )
        
        # Should not raise error
        result = helper.process_delta(delta)
        assert result.tool_call_id == "call_1"

    @pytest.mark.parametrize("tool_call_id,index", [
        ("call_123", 0),
        ("call_456", 1),
        ("call_789", 2),
    ])
    def test_parameterized_various_tool_call_ids(self, parallel_base, tool_call_id, index):
        """Test that various tool call IDs work correctly."""
        helper = ParallelStreamHelper(parallel_base)
        
        delta = ToolCallDelta(
            id=tool_call_id,
            index=index,
            name="TestModel",
            arguments='{"name": "Test", "age": 30}'
        )
        
        result = helper.process_delta(delta)
        assert tool_call_id in helper.tool_call_states

    @pytest.mark.parametrize("delta_id,index,expected_state_id", [
        (None, 0, "partial_call_0"),
        (None, 5, "partial_call_5"),
        (None, 10, "partial_call_10"),
    ])
    def test_parameterized_id_generation_from_index(self, parallel_base, delta_id, index, expected_state_id):
        """Test that ID generation from index works correctly."""
        helper = ParallelStreamHelper(parallel_base)
        
        delta = ToolCallDelta(
            id=delta_id,
            index=index,
            name="TestModel",
            arguments='{"name": "Test"}'
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
                arguments=f'{{"name": "User{i}", "age": {20 + i}}}'
            )
            helper.process_delta(delta)
        
        assert len(helper.tool_call_states) == num_calls

    @pytest.mark.parametrize("chunk_sequence,expected_final", [
        (
            [
                ('{', ''),
                ('"name"', ''),
                (': ', ''),
                ('"Alice"', ''),
                (', ', ''),
                ('"age"', ''),
                (': ', ''),
                ('30', ''),
                ('}', ''),
            ],
            '{"name": "Alice", "age": 30}'
        ),
        (
            [
                ('{"name": "', ''),
                ('Alice', ''),
                ('", "age": ', ''),
                ('30', ''),
                ('}', ''),
            ],
            '{"name": "Alice", "age": 30}'
        ),
    ])
    def test_parameterized_chunk_accumulation_sequences(self, parallel_base, chunk_sequence, expected_final):
        """Test various chunk accumulation sequences."""
        helper = ParallelStreamHelper(parallel_base)
        
        for chunk, _ in chunk_sequence:
            delta = ToolCallDelta(
                id="call_1",
                index=0,
                name="TestModel",
                arguments=chunk
            )
            helper.process_delta(delta)
        
        assert helper.tool_call_states["call_1"].accumulated_json == expected_final

    @pytest.mark.parametrize("complete_after_n_chunks", [1, 2, 5, 10])
    def test_parameterized_completion_after_chunks(self, parallel_base, complete_after_n_chunks):
        """Test completion after different numbers of chunks."""
        helper = ParallelStreamHelper(parallel_base)
        
        full_json = '{"name": "Alice", "age": 30, "email": "alice@example.com"}'
        
        # Split into chunks
        chunk_size = max(1, len(full_json) // complete_after_n_chunks)
        chunks = [full_json[i:i+chunk_size] for i in range(0, len(full_json), chunk_size)]
        
        for i, chunk in enumerate(chunks):
            delta = ToolCallDelta(
                id="call_1",
                index=0,
                name="TestModel",
                arguments=chunk
            )
            helper.process_delta(delta)
            
            if i == len(chunks) - 1:
                helper.mark_complete("call_1")
        
        assert helper.tool_call_states["call_1"].is_complete is True