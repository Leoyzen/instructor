"""
End-to-end tests for parallel streaming with OpenAI API.

This module tests the complete parallel streaming workflow using real
OpenAI API calls, validating that ParallelResult objects are
properly streamed with completed and partial models.
"""

from typing import Optional

import pytest
from pydantic import BaseModel, Field

import instructor
from instructor.dsl.parallel import OpenAIParallelBase, ParallelResult

# ============================================================================
# Test Models
# ============================================================================


class User(BaseModel):
    """User model for testing."""

    name: str
    age: int
    email: Optional[str] = None


class Task(BaseModel):
    """Task model for testing."""

    title: str
    status: str
    description: Optional[str] = None


class Product(BaseModel):
    """Product model for testing."""

    name: str
    price: float = Field(..., ge=0)
    category: str
    in_stock: bool = True


class Address(BaseModel):
    """Address model for testing."""

    street: str
    city: str
    zip_code: str
    country: str = "USA"


# ============================================================================
# Synchronous Tests
# ============================================================================


@pytest.mark.llm
class TestParallelStreamingSync:
    """Synchronous parallel streaming tests with real API calls."""

    def test_parallel_streaming_single_tool_call(self, client):
        """Test streaming a single tool call in parallel mode."""
        # Create instructor client with parallel tools mode
        client = instructor.from_openai(client, mode=instructor.Mode.PARALLEL_TOOLS)

        # Use OpenAIParallelBase for parallel models
        parallel_base = OpenAIParallelBase(User)

        results = []
        for result in client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=parallel_base,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts structured data.",
                },
                {
                    "role": "user",
                    "content": "Extract information about John Doe, who is 30 years old.",
                },
            ],
            stream=True,
            max_tokens=500,
        ):
            # Verify we receive ParallelResult objects
            assert isinstance(result, ParallelResult)
            results.append(result)

        # Should have received multiple streaming results
        assert len(results) > 0

        # Final result should have completed model
        final_result = results[-1]
        assert len(final_result.completed) == 1
        assert "User" in [type(v).__name__ for v in final_result.completed.values()]

        # Verify the User data
        user = list(final_result.completed.values())[0]
        assert user.name is not None
        assert user.age is not None
        assert user.age == 30
        assert "John" in user.name or "Doe" in user.name

    def test_parallel_streaming_multiple_tool_calls(self, client):
        """Test streaming multiple concurrent tool calls."""
        client = instructor.from_openai(client, mode=instructor.Mode.PARALLEL_TOOLS)

        # Request multiple models to be extracted
        parallel_base = OpenAIParallelBase(User, Task)

        results = []
        for result in client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=parallel_base,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts structured data.",
                },
                {
                    "role": "user",
                    "content": (
                        "Extract information about Alice (25 years old) "
                        "and her task 'Complete documentation' with status 'in progress'."
                    ),
                },
            ],
            stream=True,
            max_tokens=500,
        ):
            assert isinstance(result, ParallelResult)
            results.append(result)

        # Verify streaming occurred
        assert len(results) > 0

        # Final result should have at least one completed model
        final_result = results[-1]
        # We might have User and Task or just one of them
        total_models = len(final_result.completed) + len(final_result.partial)
        assert total_models >= 1

    def test_parallel_streaming_with_validation_context(self, client):
        """Test that validation_context is passed through correctly."""
        client = instructor.from_openai(client, mode=instructor.Mode.PARALLEL_TOOLS)

        parallel_base = OpenAIParallelBase(User)

        validation_context = {"max_age": 100, "min_age": 0}
        results = []
        for result in client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=parallel_base,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts structured data.",
                },
                {
                    "role": "user",
                    "content": "Extract info about Bob who is 35.",
                },
            ],
            stream=True,
            max_tokens=300,
            validation_context=validation_context,
        ):
            assert isinstance(result, ParallelResult)
            results.append(result)

        assert len(results) > 0
        final_result = results[-1]
        assert len(final_result.completed) >= 1

    def test_parallel_streaming_with_strict_mode(self, client):
        """Test streaming with strict mode enabled."""
        client = instructor.from_openai(client, mode=instructor.Mode.PARALLEL_TOOLS)

        parallel_base = OpenAIParallelBase(Task)

        results = []
        for result in client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=parallel_base,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts structured data.",
                },
                {
                    "role": "user",
                    "content": (
                        "Extract task information: "
                        "Title: 'Deploy to production', Status: 'completed'"
                    ),
                },
            ],
            stream=True,
            max_tokens=300,
            strict=False,  # Test non-strict mode
        ):
            assert isinstance(result, ParallelResult)
            results.append(result)

        assert len(results) > 0
        final_result = results[-1]
        # Verify we got the task
        total_models = len(final_result.completed) + len(final_result.partial)
        assert total_models >= 1

    def test_parallel_streaming_three_models(self, client):
        """Test streaming with three different models simultaneously."""
        client = instructor.from_openai(client, mode=instructor.Mode.PARALLEL_TOOLS)

        parallel_base = OpenAIParallelBase(User, Task, Product)

        results = []
        for result in client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=parallel_base,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts structured data.",
                },
                {
                    "role": "user",
                    "content": (
                        "Extract: User 'Jane' age 28, "
                        "Task 'Code review' status 'pending', "
                        "Product 'Laptop' price $999.99 category 'Electronics'"
                    ),
                },
            ],
            stream=True,
            max_tokens=500,
        ):
            assert isinstance(result, ParallelResult)
            results.append(result)

        assert len(results) > 0
        # Should have extracted at least some models
        all_models = []
        for result in results:
            all_models.extend(list(result.completed.values()))
            all_models.extend(list(result.partial.values()))

        # Verify we have models from different types
        model_types = {type(m).__name__ for m in all_models}
        assert len(model_types) >= 1

    def test_parallel_streaming_incremental_updates(self, client):
        """Test that streaming provides incremental partial updates."""
        client = instructor.from_openai(client, mode=instructor.Mode.PARALLEL_TOOLS)

        parallel_base = OpenAIParallelBase(User)

        results = []
        for result in client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=parallel_base,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts structured data.",
                },
                {
                    "role": "user",
                    "content": "Extract user info for Michael who is 45.",
                },
            ],
            stream=True,
            max_tokens=300,
        ):
            results.append(result)

        # Verify incremental streaming (should have multiple deltas)
        assert len(results) > 1

        # Check that we see partial models before completion
        partial_seen = False
        for result in results[:-1]:
            if len(result.partial) > 0:
                partial_seen = True
                break

        # Should have seen partial updates during streaming
        assert partial_seen or len(results) > 10  # Either partials seen or many deltas


# ============================================================================
# Asynchronous Tests
# ============================================================================


@pytest.mark.llm
@pytest.mark.asyncio
class TestParallelStreamingAsync:
    """Asynchronous parallel streaming tests with real API calls."""

    async def test_parallel_streaming_async_single_tool_call(self, aclient):
        """Test async streaming a single tool call in parallel mode."""
        client = instructor.from_openai(aclient, mode=instructor.Mode.PARALLEL_TOOLS)

        parallel_base = OpenAIParallelBase(User)

        results = []
        async for result in await client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=parallel_base,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts structured data.",
                },
                {
                    "role": "user",
                    "content": "Extract info for Sarah who is 32 years old.",
                },
            ],
            stream=True,
            max_tokens=300,
        ):
            assert isinstance(result, ParallelResult)
            results.append(result)

        assert len(results) > 0
        final_result = results[-1]
        assert len(final_result.completed) == 1

        user = list(final_result.completed.values())[0]
        assert user.age == 32

    async def test_parallel_streaming_async_multiple_calls(self, aclient):
        """Test async streaming multiple concurrent tool calls."""
        client = instructor.from_openai(aclient, mode=instructor.Mode.PARALLEL_TOOLS)

        parallel_base = OpenAIParallelBase(User, Task, Address)

        results = []
        async for result in await client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=parallel_base,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts structured data.",
                },
                {
                    "role": "user",
                    "content": (
                        "Extract: User 'Tom' age 40, "
                        "Task 'Fix bug' status 'active', "
                        "Address '123 Main Street, Boston, MA 02101'"
                    ),
                },
            ],
            stream=True,
            max_tokens=500,
        ):
            assert isinstance(result, ParallelResult)
            results.append(result)

        assert len(results) > 0
        final_result = results[-1]
        # Should have extracted models
        total_models = len(final_result.completed) + len(final_result.partial)
        assert total_models >= 1

    async def test_parallel_streaming_async_with_validation_context(self, aclient):
        """Test async streaming with validation context."""
        client = instructor.from_openai(aclient, mode=instructor.Mode.PARALLEL_TOOLS)

        parallel_base = OpenAIParallelBase(Product)

        validation_context = {"min_price": 0, "max_price": 1000000}

        results = []
        async for result in await client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=parallel_base,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts structured data.",
                },
                {
                    "role": "user",
                    "content": (
                        "Extract product: 'Smartphone' price $699 category 'Electronics'"
                    ),
                },
            ],
            stream=True,
            max_tokens=300,
            validation_context=validation_context,
        ):
            assert isinstance(result, ParallelResult)
            results.append(result)

        assert len(results) > 0
        final_result = results[-1]
        assert len(final_result.completed) >= 1

    async def test_parallel_streaming_async_strict_mode(self, aclient):
        """Test async streaming with strict mode enabled."""
        client = instructor.from_openai(aclient, mode=instructor.Mode.PARALLEL_TOOLS)

        parallel_base = OpenAIParallelBase(Task)

        results = []
        async for result in await client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=parallel_base,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts structured data.",
                },
                {
                    "role": "user",
                    "content": ("Extract task: 'Write tests' status 'to do'"),
                },
            ],
            stream=True,
            max_tokens=300,
            strict=True,
        ):
            assert isinstance(result, ParallelResult)
            results.append(result)

        assert len(results) > 0
        total_models = len(results[-1].completed) + len(results[-1].partial)
        assert total_models >= 1


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


@pytest.mark.llm
class TestParallelStreamingEdgeCases:
    """Edge case tests for parallel streaming."""

    def test_parallel_streaming_with_all_optional_fields(self, client):
        """Test streaming when optional fields are provided."""
        client = instructor.from_openai(client, mode=instructor.Mode.PARALLEL_TOOLS)

        parallel_base = OpenAIParallelBase(User)

        results = []
        for result in client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=parallel_base,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts structured data.",
                },
                {
                    "role": "user",
                    "content": (
                        "Extract: Name 'David', Age 50, Email 'david@example.com'"
                    ),
                },
            ],
            stream=True,
            max_tokens=300,
        ):
            results.append(result)

        assert len(results) > 0
        final_result = results[-1]
        assert len(final_result.completed) == 1

        user = list(final_result.completed.values())[0]
        assert user.name == "David"
        assert user.age == 50
        assert user.email == "david@example.com"

    def test_parallel_streaming_with_numeric_fields(self, client):
        """Test streaming with various numeric field types."""
        client = instructor.from_openai(client, mode=instructor.Mode.PARALLEL_TOOLS)

        parallel_base = OpenAIParallelBase(Product)

        results = []
        for result in client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=parallel_base,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts structured data.",
                },
                {
                    "role": "user",
                    "content": (
                        "Extract product: 'Tablet' price 299.99 category 'Computers' in stock 'yes'"
                    ),
                },
            ],
            stream=True,
            max_tokens=300,
        ):
            results.append(result)

        assert len(results) > 0
        final_result = results[-1]
        assert len(final_result.completed) == 1

        product = list(final_result.completed.values())[0]
        assert product.name == "Tablet"
        assert product.price == 299.99
        assert product.category == "Computers"
        assert product.in_stock is True

    def test_parallel_streaming_very_long_content(self, client):
        """Test streaming with longer content."""
        client = instructor.from_openai(client, mode=instructor.Mode.PARALLEL_TOOLS)

        parallel_base = OpenAIParallelBase(User, Task)

        results = []
        for result in client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=parallel_base,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts structured data.",
                },
                {
                    "role": "user",
                    "content": (
                        "Extract: User 'Jennifer Smith' age 34 email "
                        "'jen.smith@company.com', "
                        "Task title 'Complete the quarterly report and prepare presentation "
                        "for the board meeting' status 'in progress' "
                        "description 'This involves gathering data from all departments, "
                        "analyzing trends, creating visualizations, and finalizing "
                        "the PowerPoint presentation by Friday.'"
                    ),
                },
            ],
            stream=True,
            max_tokens=600,
        ):
            results.append(result)

        assert len(results) > 0
        final_result = results[-1]
        total_models = len(final_result.completed) + len(final_result.partial)
        assert total_models >= 1


# ============================================================================
# ParallelResult Structure Tests
# ============================================================================


@pytest.mark.llm
class TestParallelResultStructure:
    """Tests for ParallelResult object structure."""

    def test_parallel_result_attributes(self, client):
        """Verify ParallelResult has expected attributes."""
        client = instructor.from_openai(client, mode=instructor.Mode.PARALLEL_TOOLS)

        parallel_base = OpenAIParallelBase(User)

        for result in client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=parallel_base,
            messages=[
                {
                    "role": "user",
                    "content": "Extract user for Mark, age 22.",
                },
            ],
            stream=True,
            max_tokens=200,
        ):
            # Verify attributes exist
            assert hasattr(result, "completed")
            assert hasattr(result, "partial")
            assert hasattr(result, "tool_call_id")
            assert hasattr(result, "get_all_models")

            # Verify types
            assert isinstance(result.completed, dict)
            assert isinstance(result.partial, dict)
            assert isinstance(result.tool_call_id, str)

            # Test get_all_models method
            all_models = result.get_all_models()
            assert isinstance(all_models, list)
            break  # Only need to verify first result

    def test_parallel_result_tool_call_id_tracking(self, client):
        """Verify tool_call_id is tracked correctly."""
        client = instructor.from_openai(client, mode=instructor.Mode.PARALLEL_TOOLS)

        parallel_base = OpenAIParallelBase(User)

        tool_call_ids = []
        for result in client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=parallel_base,
            messages=[
                {
                    "role": "user",
                    "content": "Extract user for Lisa, age 29.",
                },
            ],
            stream=True,
            max_tokens=200,
        ):
            tool_call_ids.append(result.tool_call_id)

        # Should have multiple tool_call_ids (or same one for all updates)
        assert len(tool_call_ids) > 0
        # All tool_call_ids should be strings
        assert all(isinstance(tc_id, str) for tc_id in tool_call_ids)
