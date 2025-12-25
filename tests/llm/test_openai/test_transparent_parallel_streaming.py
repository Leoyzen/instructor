"""
Tests for transparent parallel streaming support.

This module tests that Iterable[Union[...]] is automatically converted to ParallelBase instance
when stream=True is used with parallel modes, without requiring users to manually create ParallelBase.
"""

from collections.abc import Iterable
from typing import Union

import pytest
from pydantic import BaseModel

import instructor
from instructor.dsl.parallel import OpenAIParallelModel, ParallelBase


class User(BaseModel):
    """User information model."""
    name: str
    email: str


class Task(BaseModel):
    """Task information model."""
    title: str
    status: str


class Organization(BaseModel):
    """Organization information model."""
    name: str
    domain: str


class TestTransparentParallelStreaming:
    """Test transparent parallel streaming conversion."""

    def test_response_model_auto_conversion(self):
        """Test that Iterable[Union[...]] is auto-converted to ParallelBase."""
        # This simulates what handle_response_model does with streaming
        from instructor.processing.response import handle_response_model

        response_model_type = Iterable[Union[User, Task, Organization]]
        new_kwargs = {"model": "test", "stream": True}

        processed_model, _ = handle_response_model(
            response_model=response_model_type,
            mode=instructor.Mode.PARALLEL_TOOLS,
            **new_kwargs,
        )

        # Verify conversion happened
        assert isinstance(processed_model, ParallelBase), (
            f"Expected ParallelBase instance, got {type(processed_model)}"
        )

    def test_response_model_no_conversion_without_stream(self):
        """Test that Iterable[Union[...]] is NOT converted when stream=False."""
        from instructor.processing.response import handle_response_model

        response_model_type = Iterable[Union[User, Task, Organization]]
        new_kwargs = {"model": "test", "stream": False}

        processed_model, _ = handle_response_model(
            response_model=response_model_type,
            mode=instructor.Mode.PARALLEL_TOOLS,
            **new_kwargs,
        )

        # Verify NO conversion happened (still a ParallelModel function, not instance)
        # handle_parallel_tools returns ParallelModel function (a callable), not an instance
        # This is expected behavior for non-streaming
        assert callable(processed_model), (
            f"Expected ParallelModel callable, got {type(processed_model)}"
        )

    def test_response_model_parallel_base_instance_passthrough(self):
        """Test that ParallelBase instance is passed through unchanged."""
        from instructor.processing.response import handle_response_model

        # User already created a ParallelBase instance
        parallel_instance = OpenAIParallelModel(Iterable[Union[User, Task]])
        new_kwargs = {"model": "test", "stream": True}

        processed_model, _ = handle_response_model(
            response_model=parallel_instance,
            mode=instructor.Mode.PARALLEL_TOOLS,
            **new_kwargs,
        )

        # Verify instance is passed through unchanged
        assert processed_model is parallel_instance

    def test_different_modes(self):
        """Test streaming conversion works for all three parallel modes."""
        from instructor.processing.response import handle_response_model

        response_model_type = Iterable[Union[User, Task]]

        # Test OpenAI PARALLEL_TOOLS
        processed, _ = handle_response_model(
            response_model=response_model_type,
            mode=instructor.Mode.PARALLEL_TOOLS,
            stream=True,
        )
        assert isinstance(processed, OpenAIParallelModel)

        # Test VERTEXAI_PARALLEL_TOOLS
        from instructor.dsl.parallel import VertexAIParallelModel

        processed, _ = handle_response_model(
            response_model=response_model_type,
            mode=instructor.Mode.VERTEXAI_PARALLEL_TOOLS,
            stream=True,
        )
        assert isinstance(processed, VertexAIParallelModel)

        # Test ANTHROPIC_PARALLEL_TOOLS
        from instructor.dsl.parallel import AnthropicParallelModel

        processed, _ = handle_response_model(
            response_model=response_model_type,
            mode=instructor.Mode.ANTHROPIC_PARALLEL_TOOLS,
            stream=True,
        )
        assert isinstance(processed, AnthropicParallelModel)

    def test_non_iterable_type_passthrough(self):
        """Test that non-Iterable types are not converted."""
        from instructor.processing.response import handle_response_model

        response_model_type = User  # Not an Iterable type
        new_kwargs = {"model": "test", "stream": True}

        processed_model, _ = handle_response_model(
            response_model=response_model_type,
            mode=instructor.Mode.PARALLEL_TOOLS,
            **new_kwargs,
        )

        # Verify no conversion happened
        assert processed_model is response_model_type

    def test_kwargs_preserved(self):
        """Test that kwargs are preserved after conversion."""
        from instructor.processing.response import handle_response_model

        response_model_type = Iterable[Union[User, Task]]
        original_kwargs = {
            "model": "gpt-4",
            "stream": True,
            "temperature": 0.7,
            "custom_param": "test",
        }

        processed_model, new_kwargs = handle_response_model(
            response_model=response_model_type,
            mode=instructor.Mode.PARALLEL_TOOLS,
            **original_kwargs,
        )

        # Verify all kwargs are preserved
        assert new_kwargs["model"] == "gpt-4"
        assert new_kwargs["stream"] is True
        assert new_kwargs["temperature"] == 0.7
        assert new_kwargs["custom_param"] == "test"


class TestParallelResultStructure:
    """Test ParallelResult data structure."""

    def test_parallel_result_attributes(self):
        """Test that ParallelResult has all required attributes."""
        from instructor.dsl.parallel import ParallelResult

        completed = {"id_1": User(name="Alice", email="alice@example.com")}
        partial = {"id_2": Task(title="Draft", status="in-progress")}

        result = ParallelResult(
            completed=completed,
            partial=partial,
            tool_call_id="id_2",
        )

        assert result.completed is completed
        assert result.partial is partial
        assert result.tool_call_id == "id_2"

    def test_parallel_result_get_all_models(self):
        """Test get_all_models returns all models."""
        from instructor.dsl.parallel import ParallelResult

        user = User(name="Alice", email="alice@example.com")
        task = Task(title="Review", status="pending")
        organization = Organization(name="Acme Corp", domain="acme.com")

        completed = {"user_id": user}
        partial = {"task_id": task, "org_id": organization}

        result = ParallelResult(
            completed=completed,
            partial=partial,
            tool_call_id="task_id",
        )

        all_models = result.get_all_models()

        assert len(all_models) == 3
        assert user in all_models
        assert task in all_models
        assert organization in all_models


class TestBackwardCompatibility:
    """Test backward compatibility with existing code patterns."""

    def test_non_streaming_unchanged(self):
        """Test that non-streaming parallel mode works as before."""
        from instructor.processing.response import handle_response_model

        response_model_type = Iterable[Union[User, Task]]
        new_kwargs = {"model": "test"}  # No stream=True

        processed_model, processed_kwargs = handle_response_model(
            response_model=response_model_type,
            mode=instructor.Mode.PARALLEL_TOOLS,
            **new_kwargs,
        )

        # Should use original logic (not convert to ParallelBase instance)
        # The handler returns a ParallelModel callable function
        assert callable(processed_model)

    def test_manual_parallel_base_instance_still_works(self):
        """Test that manually created ParallelBase instances still work."""
        from instructor.processing.response import handle_response_model

        # User manually creates ParallelBase instance (old pattern)
        parallel_instance = OpenAIParallelModel(Iterable[Union[User, Task]])
        new_kwargs = {"model": "test", "stream": True}

        processed_model, _ = handle_response_model(
            response_model=parallel_instance,
            mode=instructor.Mode.PARALLEL_TOOLS,
            **new_kwargs,
        )

        # Should pass through unchanged
        assert processed_model is parallel_instance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])