"""
Parallel Streaming Example for Instructor

This example demonstrates how to use parallel streaming with Instructor
to extract multiple structured objects in real-time as they are being generated.

Key features demonstrated:
- Basic parallel streaming with multiple Pydantic models
- Real-time progress tracking with ParallelResult
- Synchronous and asynchronous usage
- Validation context support
- Error handling and best practices
- Different scenarios: single model, multiple models, complex nested data
"""

import argparse
import asyncio
from typing import Optional

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

import instructor
from instructor.dsl.parallel import OpenAIParallelBase, ParallelResult

# ============================================================================
# Pydantic Models for Extraction
# ============================================================================


class User(BaseModel):
    """User information model."""

    name: str
    age: int = Field(..., ge=0, description="User's age in years")
    email: Optional[str] = None
    role: str = "User"


class Task(BaseModel):
    """Task information model."""

    title: str
    status: str = Field(..., description="Task status: todo, in-progress, completed")
    description: Optional[str] = None
    priority: int = Field(
        default=1, ge=1, le=5, description="Priority from 1 (low) to 5 (high)"
    )


class Product(BaseModel):
    """Product information model."""

    name: str
    price: float = Field(..., ge=0, description="Price in USD")
    category: str
    in_stock: bool = True
    quantity: Optional[int] = Field(default=None, ge=0)


class Address(BaseModel):
    """Address information model."""

    street: str
    city: str
    state: Optional[str] = None
    zip_code: str
    country: str = "USA"


# ============================================================================
# Example 1: Basic Parallel Streaming (Single Model)
# ============================================================================


def example_1_basic_single_model(client: OpenAI) -> None:
    """
    Basic example: Stream a single User model extraction.

    This demonstrates the simplest use case of parallel streaming
    with a single model type.
    """
    print("\n" + "=" * 70)
    print("Example 1: Basic Parallel Streaming (Single Model)")
    print("=" * 70)

    # Create parallel base with User model
    parallel_base = OpenAIParallelBase(User)

    print("\nExtracting user information...\n")

    # Stream the parallel tool calls
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
                "content": "Extract information about John Doe, who is 30 years old and works as a Software Engineer.",
            },
        ],
        stream=True,
        max_tokens=300,
    ):
        # Display progress
        completed_count = len(result.completed)
        partial_count = len(result.partial)

        # Show completed models
        for tool_call_id, model in result.completed.items():
            print(f"✓ Completed [{tool_call_id}]: {model}")

        # Show partial models (in progress)
        for tool_call_id, model in result.partial.items():
            print(f"⏳ Partial [{tool_call_id}]: {model}")

        if completed_count > 0:
            print(
                f"\nProgress: {completed_count} completed, {partial_count} in progress"
            )


# ============================================================================
# Example 2: Multiple Models Streaming
# ============================================================================


def example_2_multiple_models(client: OpenAI) -> None:
    """
    Example: Stream extraction of multiple different model types.

    This shows how to extract User and Task information simultaneously,
    demonstrating the true power of parallel streaming.
    """
    print("\n" + "=" * 70)
    print("Example 2: Multiple Models Streaming")
    print("=" * 70)

    # Create parallel base with multiple models
    parallel_base = OpenAIParallelBase(User, Task)

    print("\nExtracting user and task information simultaneously...\n")

    # Track which models we've already displayed
    completed_tool_calls = set()

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
                    "Extract information about Alice Smith (25 years old, Software Developer) "
                    "and her task 'Complete API Documentation' with status 'in-progress' "
                    "and priority 4."
                ),
            },
        ],
        stream=True,
        max_tokens=400,
    ):
        # Display newly completed models
        for tool_call_id, model in result.completed.items():
            if tool_call_id not in completed_tool_calls:
                model_type = type(model).__name__
                print(f"\n✓ Completed [{model_type}]:")
                print(f"  {model.model_dump_json(indent=4)}")
                completed_tool_calls.add(tool_call_id)

        # Show partial progress
        if result.partial:
            partial_summary = [
                f"{type(m).__name__}: {m}" for m in result.partial.values()
            ]
            if partial_summary:
                print(f"⏳ Streaming progress: {', '.join(partial_summary)}")


# ============================================================================
# Example 3: Complex Multi-Model Extraction
# ============================================================================


def example_3_complex_extraction(client: OpenAI) -> None:
    """
    Example: Complex extraction with multiple models of different types.

    This demonstrates extracting User, Task, Product, and Address information
    all in a single streaming request.
    """
    print("\n" + "=" * 70)
    print("Example 3: Complex Multi-Model Extraction")
    print("=" * 70)

    # Create parallel base with four different models
    parallel_base = OpenAIParallelBase(User, Task, Product, Address)

    print("\nExtracting complex information (User, Task, Product, Address)...\n")

    # Track all models we've seen
    completed_tool_calls = set()

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
                    "Extract all the following information:\n"
                    "User: Bob Johnson, age 35, email bob@company.com, Manager\n"
                    "Task: 'Review quarterly report', status 'completed', priority 3\n"
                    "Product: 'Wireless Mouse', price $29.99, category 'Electronics', in stock yes\n"
                    "Address: '123 Business Ave', San Francisco, CA, '94102', USA"
                ),
            },
        ],
        stream=True,
        max_tokens=600,
    ):
        # Display newly completed models
        newly_completed = []
        for tool_call_id, model in result.completed.items():
            if tool_call_id not in completed_tool_calls:
                model_type = type(model).__name__
                print(f"\n✓ Completed [{model_type}]:")
                for field_name, field_value in model.model_dump().items():
                    print(f"  {field_name}: {field_value}")
                completed_tool_calls.add(tool_call_id)
                newly_completed.append(model_type)

        # Update progress
        total_completed = len(completed_tool_calls)
        total_partial = len(result.partial)
        if newly_completed or total_partial > 0:
            print(f"\nProgress: {total_completed} completed, {total_partial} streaming")


# ============================================================================
# Example 4: Real-Time Progress Display
# ============================================================================


def example_4_progress_display(client: OpenAI) -> None:
    """
    Example: Enhanced progress display with detailed streaming information.

    This shows how to implement a rich progress display that updates
    in real-time as models are being generated.
    """
    print("\n" + "=" * 70)
    print("Example 4: Real-Time Progress Display")
    print("=" * 70)

    parallel_base = OpenAIParallelBase(User, Task, Product)

    print("\nStreaming with enhanced progress display...\n")

    result_count = 0
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
                    "Extract: User 'Emily' age 28, "
                    "Task 'Design database schema' status 'todo', "
                    "Product 'USB-C Cable' price $12.99 category 'Accessories'"
                ),
            },
        ],
        stream=True,
        max_tokens=400,
    ):
        result_count += 1

        # Build progress bar
        total_items = len(result.completed) + len(result.partial)
        progress_bar = "█" * len(result.completed) + "░" * len(result.partial)

        print(f"\n[{result_count}] Update | Progress: [{progress_bar}]")
        print(f"Completed: {len(result.completed)} | Streaming: {len(result.partial)}")

        # Show details
        if result.completed:
            print("\nCompleted models:")
            for model in result.completed.values():
                print(f"  {type(model).__name__}: {model}")

        if result.partial:
            print("\nPartial models:")
            for model in result.partial.values():
                print(f"  {type(model).__name__}: {model}")


# ============================================================================
# Example 5: Streaming with Validation Context
# ============================================================================


def example_5_validation_context(client: OpenAI) -> None:
    """
    Example: Parallel streaming with validation context.

    This demonstrates how to pass validation context to influence
    the validation process for extracted models.
    """
    print("\n" + "=" * 70)
    print("Example 5: Streaming with Validation Context")
    print("=" * 70)

    parallel_base = OpenAIParallelBase(User, Task)

    # Define validation context
    validation_context = {
        "min_age": 18,
        "max_age": 100,
        "allowed_statuses": ["todo", "in-progress", "completed"],
    }

    print("\nStreaming with validation context...\n")
    print(f"Validation context: {validation_context}\n")

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
                    "Extract: User 'Sarah' age 29, "
                    "Task 'Deploy to production' status 'in-progress'"
                ),
            },
        ],
        stream=True,
        max_tokens=300,
        validation_context=validation_context,
    ):
        for model in result.completed.values():
            print(f"✓ Validated [{type(model).__name__}]: {model}")


# ============================================================================
# Example 6: Strict Mode Streaming
# ============================================================================


def example_6_strict_mode(client: OpenAI, strict: bool) -> None:
    """
    Example: Parallel streaming with strict mode enabled/disabled.

    Strict mode affects how strictly the JSON parsing is handled.
    """
    print("\n" + "=" * 70)
    print(f"Example 6: Streaming with Strict Mode ({strict})")
    print("=" * 70)

    parallel_base = OpenAIParallelBase(User)

    print(f"\nStreaming with strict={strict}...\n")

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
                "content": "Extract user: 'Michael' age 42",
            },
        ],
        stream=True,
        max_tokens=200,
        strict=strict,
    ):
        if result.completed:
            user = list(result.completed.values())[0]
            print(f"✓ User extracted (strict={strict}): {user}")


# ============================================================================
# Example 7: Async Parallel Streaming
# ============================================================================


async def example_7_async_streaming() -> None:
    """
    Example: Asynchronous parallel streaming.

    This demonstrates the same functionality but using async/await,
    which is useful for applications that need to handle multiple
    concurrent requests or IO-bound operations.
    """
    print("\n" + "=" * 70)
    print("Example 7: Asynchronous Parallel Streaming")
    print("=" * 70)

    # Create async client
    async_client = instructor.from_openai(
        AsyncOpenAI(), mode=instructor.Mode.PARALLEL_TOOLS
    )

    parallel_base = OpenAIParallelBase(User, Task)

    print("\nAsync streaming in progress...\n")

    result_count = 0
    async for result in await async_client.chat.completions.create(
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
                    "Extract: User 'Jessica' age 31, "
                    "Task 'Optimize database queries' status 'completed'"
                ),
            },
        ],
        stream=True,
        max_tokens=300,
    ):
        result_count += 1
        print(
            f"[Async {result_count}] Completed: {len(result.completed)}, Partial: {len(result.partial)}"
        )

        for model in result.completed.values():
            print(f"  ✓ {type(model).__name__}: {model}")


# ============================================================================
# Example 8: Error Handling and Best Practices
# ============================================================================


def example_8_error_handling(client: OpenAI) -> None:
    """
    Example: Error handling and best practices for parallel streaming.

    This demonstrates proper error handling, validation, and
    defensive programming techniques.
    """
    print("\n" + "=" * 70)
    print("Example 8: Error Handling and Best Practices")
    print("=" * 70)

    parallel_base = OpenAIParallelBase(User, Task)

    print("\nDemonstrating robust error handling...\n")

    try:
        results = []
        completed_tool_calls = set()

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
                    "content": "Extract: User 'David' age 45, Task 'Fix bug' status 'active'",
                },
            ],
            stream=True,
            max_tokens=300,
        ):
            results.append(result)

            # Validate result structure
            if not isinstance(result, ParallelResult):
                print(f"⚠️ Warning: Unexpected result type: {type(result)}")
                continue

            # Check for completed models
            for tool_call_id, model in result.completed.items():
                if tool_call_id not in completed_tool_calls:
                    # Validate the model
                    try:
                        # Re-validate the model (defensive programming)
                        validated = type(model).model_validate(model.model_dump())
                        print(f"✓ Validated and complete: {type(model).__name__}")
                        completed_tool_calls.add(tool_call_id)
                    except Exception as e:
                        print(f"⚠️ Validation error: {e}")

            # Track progress
            if len(results) % 5 == 0:
                print(
                    f"Progress: {len(results)} chunks received, {len(completed_tool_calls)} models completed"
                )

        # Final validation
        print(f"\n✓ Streaming completed: {len(completed_tool_calls)} models extracted")

    except Exception as e:
        print(f"\n❌ Error during streaming: {type(e).__name__}: {e}")
        print("This demonstrates proper error handling.")


# ============================================================================
# Example 9: Filtering and Processing Results
# ============================================================================


def example_9_filtering_and_processing(client: OpenAI) -> None:
    """
    Example: Filtering and processing streaming results in real-time.

    This shows how to filter, transform, or process models
    as they complete during streaming.
    """
    print("\n" + "=" * 70)
    print("Example 9: Filtering and Processing Results")
    print("=" * 70)

    parallel_base = OpenAIParallelBase(User, Task, Product)

    print("\nStreaming and processing results...\n")

    # Track processed models
    processed = {"User": 0, "Task": 0, "Product": 0}

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
                    "Extract: User 'Maria' age 27, "
                    "Task 'Write unit tests' status 'todo', "
                    "Product 'Mouse Pad' price $8.99 category 'Accessories'"
                ),
            },
        ],
        stream=True,
        max_tokens=400,
    ):
        # Process completed models
        for model in result.completed.values():
            model_type = type(model).__name__

            # Different processing based on model type
            if model_type == "User":
                processed["User"] += 1
                print(f"👤 User processed: {model.name} (age: {model.age})")

            elif model_type == "Task":
                processed["Task"] += 1
                # Check if task is high priority
                status = model.status
                icon = (
                    "🔴"
                    if status == "todo"
                    else "🟡"
                    if status == "in-progress"
                    else "🟢"
                )
                print(f"{icon} Task processed: {model.title} [{status}]")

            elif model_type == "Product":
                processed["Product"] += 1
                # Check if in stock
                stock_status = "✅ In stock" if model.in_stock else "❌ Out of stock"
                print(
                    f"📦 Product processed: {model.name} - ${model.price:.2f} ({stock_status})"
                )

            print(f"   Summary: {processed}")


# ============================================================================
# Example 10: Progressive UI Updates Simulation
# ============================================================================


def example_10_progressive_ui_updates(client: OpenAI) -> None:
    """
    Example: Simulate progressive UI updates during streaming.

    This demonstrates how parallel streaming can be used to update
    a user interface progressively as results arrive.
    """
    print("\n" + "=" * 70)
    print("Example 10: Progressive UI Updates Simulation")
    print("=" * 70)

    parallel_base = OpenAIParallelBase(User, Task, Product, Address)

    print("\nSimulating progressive UI updates...\n")

    # Simulate UI state
    ui_components = {
        "user_card": False,
        "task_card": False,
        "product_card": False,
        "address_card": False,
    }

    def render_ui(components: dict) -> str:
        """Simulate UI rendering based on available components."""
        active = [name for name, visible in components.items() if visible]
        return f"UI Components Loaded: {len(active)}/4 {active}"

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
                    "Extract: User 'Laura' age 33, "
                    "Task 'Create API documentation' status 'in-progress', "
                    "Product 'Notebook' price $5.99 category 'Stationery', "
                    "Address '456 Design Street, Portland, OR, 97201'"
                ),
            },
        ],
        stream=True,
        max_tokens=500,
    ):
        # Update UI components as models complete
        for model in result.completed.values():
            model_type = type(model).__name__.lower()

            # Create/update UI component
            if (
                f"{model_type}_card" in ui_components
                and not ui_components[f"{model_type}_card"]
            ):
                ui_components[f"{model_type}_card"] = True

                # Simulate UI update
                print(f"\n🎨 UI Update: {model_type.upper()} card added!")
                print(f"   Data: {model}")
                print(f"   {render_ui(ui_components)}")

        # Show loading for partial models
        if result.partial:
            loading = [
                type(m).__name__.lower() + "_card" for m in result.partial.values()
            ]
            print(f"⏳ Loading components: {loading}")


# ============================================================================
# Main Execution
# ============================================================================


def run_all_examples(client: OpenAI) -> None:
    """Run all synchronous examples."""
    examples = [
        ("Example 1: Basic Single Model", example_1_basic_single_model),
        ("Example 2: Multiple Models", example_2_multiple_models),
        ("Example 3: Complex Extraction", example_3_complex_extraction),
        ("Example 4: Progress Display", example_4_progress_display),
        ("Example 5: Validation Context", example_5_validation_context),
        (
            "Example 6: Strict Mode (True)",
            lambda c: example_6_strict_mode(c, strict=True),
        ),
        (
            "Example 6b: Strict Mode (False)",
            lambda c: example_6_strict_mode(c, strict=False),
        ),
        ("Example 8: Error Handling", example_8_error_handling),
        ("Example 9: Filtering & Processing", example_9_filtering_and_processing),
        ("Example 10: Progressive UI", example_10_progressive_ui_updates),
    ]

    for name, example_fn in examples:
        try:
            example_fn(client)
        except Exception as e:
            print(f"\n❌ Error in {name}: {type(e).__name__}: {e}")


async def run_async_examples() -> None:
    """Run all asynchronous examples."""
    try:
        await example_7_async_streaming()
    except Exception as e:
        print(f"\n❌ Error in async example: {type(e).__name__}: {e}")


def main():
    """Main entry point with command-line argument support."""
    parser = argparse.ArgumentParser(
        description="Parallel Streaming Examples for Instructor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    # Run all examples
  python run.py --example 1        # Run specific example 1
  python run.py --async            # Run async examples only
  python run.py --model gpt-4       # Use specific model
        """,
    )
    parser.add_argument(
        "--example",
        type=int,
        choices=range(1, 11),
        help="Run a specific example (1-10)",
    )
    parser.add_argument(
        "--async",
        action="store_true",
        dest="async_mode",
        help="Run async examples only",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "🚀" * 35)
    print("  Instructor Parallel Streaming Examples")
    print("🚀" * 35)
    print(f"\nUsing model: {args.model}")

    # Create client
    client = instructor.from_openai(OpenAI(), mode=instructor.Mode.PARALLEL_TOOLS)

    if args.async_mode:
        # Run async examples only
        print("\n" + "⚡" * 35)
        print("  Running Async Examples")
        print("⚡" * 35)
        asyncio.run(run_async_examples())
    elif args.example:
        # Run specific example
        # Note: example_6 is handled specially below to avoid unused lambda parameter
        examples_map = {
            1: example_1_basic_single_model,
            2: example_2_multiple_models,
            3: example_3_complex_extraction,
            4: example_4_progress_display,
            5: example_5_validation_context,
            7: lambda _: (
                asyncio.run(example_7_async_streaming()),
                print("\n⚠️ Note: Example 7 is async-only"),
            )[0]
            if args.example == 7
            else None,
            8: example_8_error_handling,
            9: example_9_filtering_and_processing,
            10: example_10_progressive_ui_updates,
        }

        example_fn = examples_map.get(args.example)
        if example_fn:
            if args.example == 7:
                # Special handling for async example
                asyncio.run(example_7_async_streaming())
            else:
                example_fn(client)
        else:
            print(f"\n❌ Example {args.example} not found")
    else:
        # Run all examples
        run_all_examples(client)

        # Also run async examples
        print("\n" + "⚡" * 35)
        print("  Running Async Examples")
        print("⚡" * 35)
        asyncio.run(run_async_examples())

    # Print footer
    print("\n" + "=" * 70)
    print("✓ All examples completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
