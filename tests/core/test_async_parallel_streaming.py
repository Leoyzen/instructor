"""
Tests for AsyncInstructor.create() with parallel streaming support.

This module tests that AsyncInstructor.create() properly handles
ParallelBase instances with stream=True in parallel modes.
"""

import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel

from instructor import Mode, from_openai
from instructor.dsl.parallel import ParallelBase, ParallelResult


class User(BaseModel):
    """Simple user model for testing."""

    name: str
    email: str


class Task(BaseModel):
    """Simple task model for testing."""

    title: str
    status: str


@pytest.mark.llm
class TestAsyncParallelStreamingCreate:
    """Test cases for AsyncInstructor.create() with parallel streaming."""

    async def test_create_with_parallel_base_stream(
        self,
        async_openai_client: AsyncOpenAI,
    ):
        """Test that AsyncInstructor.create() handles ParallelBase with stream=True."""
        client = from_openai(async_openai_client, mode=Mode.PARALLEL_TOOLS)

        # Create a ParallelBase instance
        parallel_base = ParallelBase(User, Task)

        # Call create() with stream=True and ParallelBase instance
        results = []
        async for result in await client.create(
            messages=[
                {
                    "role": "user",
                    "content": "Extract one user and one task",
                }
            ],
            response_model=parallel_base,
            stream=True,
        ):
            # Should receive ParallelResult objects
            assert isinstance(result, ParallelResult)
            results.append(result)

        # Should have received at least one result
        assert len(results) > 0

        # Verify we get ParallelResult with correct structure
        assert hasattr(result, "completed")
        assert hasattr(result, "partial")
        assert hasattr(result, "tool_call_id")

    async def test_create_with_parallel_base_no_stream(
        self,
        async_openai_client: AsyncOpenAI,
    ):
        """Test that AsyncInstructor.create() handles ParallelBase without stream (non-streaming)."""
        client = from_openai(async_openai_client, mode=Mode.PARALLEL_TOOLS)

        # Create a ParallelBase instance
        parallel_base = ParallelBase(User, Task)

        # Call create() without stream=True (default stream=False)
        response = await client.create(
            messages=[
                {
                    "role": "user",
                    "content": "Extract one user and one task",
                }
            ],
            response_model=parallel_base,
        )

        # Should return the ParallelBase instance (generator-like)
        # When stream is False, ParallelBase.from_response() should be called
        assert isinstance(response, ParallelBase)

    async def test_create_with_validation_context_and_stream(
        self,
        async_openai_client: AsyncOpenAI,
    ):
        """Test that AsyncInstructor.create() passes validation_context correctly."""
        client = from_openai(async_openai_client, mode=Mode.PARALLEL_TOOLS)

        parallel_base = ParallelBase(User)

        validation_context = {"user_id": "123"}

        async for result in await client.create(
            messages=[{"role": "user", "content": "Extract a user"}],
            response_model=parallel_base,
            stream=True,
            validation_context=validation_context,
        ):
            assert isinstance(result, ParallelResult)

    async def test_create_with_strict_and_stream(
        self,
        async_openai_client: AsyncOpenAI,
    ):
        """Test that AsyncInstructor.create() passes strict parameter correctly."""
        client = from_openai(async_openai_client, mode=Mode.PARALLEL_TOOLS)

        parallel_base = ParallelBase(User)

        async for result in await client.create(
            messages=[{"role": "user", "content": "Extract a user"}],
            response_model=parallel_base,
            stream=True,
            strict=False,
        ):
            assert isinstance(result, ParallelResult)

    async def test_parallel_base_structure(
        self,
        async_openai_client: AsyncOpenAI,
    ):
        """Test that the parallel_base.py streaming infrastructure is properly implemented."""
        from instructor.dsl.parallel import (
            ParallelBase,
            ParallelResult,
            ParallelStreamHelper,
            ToolCallDelta,
            ToolCallState,
        )

        # Verify the data structures exist
        assert ParallelBase is not None
        assert ParallelResult is not None
        assert ParallelStreamHelper is not None
        assert ToolCallDelta is not None
        assert ToolCallState is not None

        client = from_openai(async_openai_client, mode=Mode.PARALLEL_TOOLS)

        # Test basic parallel base initialization
        parallel_base = ParallelBase(User, Task)

        assert len(parallel_base.models) == 2
        assert "User" in parallel_base.registry
        assert "Task" in parallel_base.registry
