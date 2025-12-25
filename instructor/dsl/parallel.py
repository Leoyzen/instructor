import json
import sys
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Iterable as ABCIterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union, get_args, get_origin

# Import jiter for progressive JSON parsing
from jiter import from_json
from pydantic import BaseModel

from ..mode import Mode

if TYPE_CHECKING:
    from ..processing.function_calls import OpenAISchema

    T = TypeVar("T", bound=OpenAISchema)
else:
    # At runtime, we'll bind to BaseModel instead to avoid circular import
    T = TypeVar("T", bound=BaseModel)


@dataclass
class ToolCallDelta:
    """
    Represents a streaming delta for a single tool call.

    This data structure captures incremental updates received during
    streaming of parallel tool calls from LLM providers.

    Attributes:
        id: The unique identifier for this tool call. May be None in
            early chunks before the ID is assigned.
        index: The index position of this tool call in the parallel set.
        name: The tool/function name being called. May be None in early chunks.
        arguments: The portion of function arguments received in this chunk.

    Example:
        >>> delta = ToolCallDelta(
        ...     id="call_abc123",
        ...     index=0,
        ...     name="User",
        ...     arguments='{"name": "Alice"'
        ... )
    """

    id: Optional[str]
    index: int
    name: Optional[str]
    arguments: str


@dataclass
class ParallelResult:
    """
    Represents the state of parallel tool calls during streaming.

    This data structure provides a snapshot of all parallel tool calls
    at a given point in the streaming process, distinguishing between
    completed and partially completed models.

    Attributes:
        completed: Dictionary mapping tool_call_id to fully completed
            BaseModel instances.
        partial: Dictionary mapping tool_call_id to Partial[Model]
            instances that are still streaming.
        tool_call_id: The tool_call_id that was updated in this iteration.
            This helps identify which delta triggered the current state.

    Example:
        >>> result = ParallelResult(
        ...     completed={"call_123": User(name="Alice", email="alice@example.com")},
        ...     partial={"call_456": PartialTask(title="Create", description="...")},
        ...     tool_call_id="call_456"
        ... )
        >>> result.get_all_models()
        [User(name='Alice', ...), PartialTask(title='Create', ...)]
    """

    completed: dict[str, BaseModel]
    partial: dict[str, BaseModel]
    tool_call_id: str

    def get_all_models(self) -> list[BaseModel]:
        """
        Get all available models (completed + partial).

        Returns:
            A list of all currently available model instances, regardless
            of whether they are complete or partial.
        """
        return list(self.completed.values()) + list(self.partial.values())


@dataclass
class ToolCallState:
    """
    Tracks the streaming state of a single tool call.

    This internal data structure maintains the state needed to progressively
    build up and validate a single model during streaming.

    Attributes:
        tool_call_id: Unique identifier for this tool call.
        model_class: The Pydantic model class to instantiate.
        accumulated_json: The accumulated JSON string built from all chunks.
        partial_model: The most recent Partial[Model] instance from parsing.
            This represents the current best-effort parse of the data.
        is_complete: Whether this tool call has finished streaming.

    Example:
        >>> state = ToolCallState(
        ...     tool_call_id="call_123",
        ...     model_class=User,
        ...     accumulated_json='{"name": "Alice"',
        ...     partial_model=PartialUser(name="Alice"),
        ...     is_complete=False
        ... )
    """

    tool_call_id: str
    model_class: type[BaseModel]
    accumulated_json: str = ""
    partial_model: Optional[BaseModel] = None
    is_complete: bool = False


class ParallelStreamHelper:
    """
    Tracks streaming state for multiple parallel tool calls.

    This helper class manages the complex state of multiple concurrent tool
    calls during streaming. It maintains a mapping of tool_call_id to
    ToolCallState and processes incoming deltas to update partial and
    completed models.

    The helper is responsible for:
    - Creating new state for each unique tool call
    - Accumulating JSON chunks for each tool call
    - Parsing partial JSON using jiter.from_json() with graceful degradation
    - Distinguishing between partial and completed models
    - Handling errors (missing tool_call_id, invalid JSON, etc.)

    Attributes:
        parallel_base: The ParallelBase instance providing the model registry
            and mode information.
        tool_call_states: Mapping from tool_call_id to ToolCallState objects.
        _provided_models: Set of tool_call_ids that have been yielded as completed.

    Example:
        >>> parallel_base = ParallelBase(User, Task, Organization)
        >>> helper = ParallelStreamHelper(parallel_base)
        >>> for delta in deltas:
        ...     result = helper.process_delta(
        ...         delta,
        ...         validation_context=None,
        ...         strict=None
        ...     )
        ...     print(result)
    """

    def __init__(self, parallel_base: "ParallelBase"):
        """
        Initialize the ParallelStreamHelper.

        Args:
            parallel_base: The ParallelBase instance containing the model
                registry and configuration for this parallel operation.
        """
        self.parallel_base = parallel_base
        self.tool_call_states: dict[str, ToolCallState] = {}
        self._provided_models = set()

    def process_delta(
        self,
        delta: ToolCallDelta,
        validation_context: Optional[dict[str, Any]] = None,
        strict: Optional[bool] = None,
    ) -> ParallelResult:
        """
        Process a streaming delta and return updated partial results.

        This method handles both new and existing tool calls. For new tool
        calls, it creates a new ToolCallState. For existing ones, it updates
        the accumulated JSON and attempts to parse it.

        Args:
            delta: The ToolCallDelta containing the chunk data.
            validation_context: Optional context for Pydantic validation.
            strict: Optional strict mode for JSON parsing.

        Returns:
            A ParallelResult containing the current state of all tool calls,
            including both completed and partial models.

        Raises:
            KeyError: If a delta references an unknown model name that's not
                in the parallel_base registry.

        Example:
            >>> delta = ToolCallDelta(
            ...     id="call_123",
            ...     index=0,
            ...     name="User",
            ...     arguments='{"name": "Alice"'
            ... )
            >>> result = helper.process_delta(delta)
            >>> print(f"Partial: {result.partial['call_123']}")
        """
        # Determine the tool_call_id to use
        tool_call_id = delta.id
        if tool_call_id is None:
            # Generate local ID based on index if ID not provided
            tool_call_id = f"partial_call_{delta.index}"

        # Get or create state for this tool call
        if tool_call_id not in self.tool_call_states:
            # New tool call - create initial state
            if delta.name is None:
                # Skip delta without name (will be updated in subsequent chunks)
                return self._get_result(tool_call_id)

            if delta.name not in self.parallel_base.registry:
                raise KeyError(
                    f"Unknown model name '{delta.name}' in parallel tool call. "
                    f"Available models: {list(self.parallel_base.registry.keys())}"
                )

            # Create new state
            self.tool_call_states[tool_call_id] = ToolCallState(
                tool_call_id=tool_call_id,
                model_class=self.parallel_base.registry[delta.name],
                accumulated_json="",
                partial_model=None,
                is_complete=False,
            )

        # Get the state
        state = self.tool_call_states[tool_call_id]

        # Append arguments to accumulated JSON
        if delta.arguments:
            state.accumulated_json += delta.arguments

        # Attempt to parse accumulated JSON
        try:
            # Use jiter.from_json for progressive JSON parsing with graceful degradation
            parsed = from_json(
                (state.accumulated_json.strip() or "{}").encode(),
                partial_mode="trailing-strings",
            )

            # Create partial model instance for validation
            # We use the original model_class directly to validate partial JSON
            try:
                state.partial_model = state.model_class.model_validate(
                    parsed, context=validation_context, strict=strict
                )
            except Exception:
                # Validation errors are expected for partial JSON
                # We keep the previous partial_model if available
                pass

        except Exception:
            # JSON parsing errors are expected for partial strings
            # The error is silently handled as we'll retry with more data
            pass

        return self._get_result(tool_call_id)

    def mark_complete(self, tool_call_id: str) -> ParallelResult:
        """
        Mark a tool call as complete.

        This method should be called when all chunks for a tool call have
        been received. It performs final validation and moves the model
        from partial to completed.

        Args:
            tool_call_id: The ID of the tool call to mark as complete.

        Returns:
            A ParallelResult with the completed model in the completed dictionary.

        Example:
            >>> result = helper.mark_complete("call_123")
            >>> assert "call_123" in result.completed
        """
        if tool_call_id not in self.tool_call_states:
            return self._get_result(tool_call_id)

        state = self.tool_call_states[tool_call_id]
        state.is_complete = True

        return self._get_result(tool_call_id)

    def get_result(self) -> ParallelResult:
        """
        Get the current state without processing a new delta.

        Returns:
            A ParallelResult representing the current state of all tool calls.
        """
        # Return result without updating specific tool_call_id
        return self._get_result(None)

    def _get_result(self, tool_call_id: Optional[str]) -> ParallelResult:
        """
        Internal method to build a ParallelResult from current states.

        Args:
            tool_call_id: The tool_call_id that triggered this result (if any).

        Returns:
            A ParallelResult containing completed and partial models.
        """
        completed = {}
        partial = {}

        for tc_id, state in self.tool_call_states.items():
            if state.is_complete and state.partial_model is not None:
                completed[tc_id] = state.partial_model
            elif not state.is_complete and state.partial_model is not None:
                partial[tc_id] = state.partial_model

        return ParallelResult(
            completed=completed,
            partial=partial,
            tool_call_id=tool_call_id or "",
        )


class ParallelBase:
    def __init__(self, *models: type[BaseModel]):
        # Note that for everything else we've created a class, but for parallel base it is an instance
        assert len(models) > 0, "At least one model is required"
        self.models = models
        self.registry = {
            model.__name__ if hasattr(model, "__name__") else str(model): model
            for model in models
        }

    def from_response(
        self,
        response: Any,
        mode: Mode,
        validation_context: Optional[Any] = None,
        strict: Optional[bool] = None,
    ) -> Generator[BaseModel, None, None]:
        #! We expect this from the OpenAISchema class, We should address
        #! this with a protocol or an abstract class... @jxnlco
        assert mode == Mode.PARALLEL_TOOLS, "Mode must be PARALLEL_TOOLS"
        for tool_call in response.choices[0].message.tool_calls:
            name = tool_call.function.name
            arguments = tool_call.function.arguments
            yield self.registry[name].model_validate_json(
                arguments, context=validation_context, strict=strict
            )


class VertexAIParallelBase(ParallelBase):
    def from_response(
        self,
        response: Any,
        mode: Mode,
        validation_context: Optional[Any] = None,
        strict: Optional[bool] = None,
    ) -> Generator[BaseModel, None, None]:
        assert mode == Mode.VERTEXAI_PARALLEL_TOOLS, (
            "Mode must be VERTEXAI_PARALLEL_TOOLS"
        )

        if not response or not response.candidates:
            return

        for candidate in response.candidates:
            if not candidate.content or not candidate.content.parts:
                continue

            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call is not None:
                    name = part.function_call.name
                    arguments = part.function_call.args

                    if name in self.registry:
                        # Convert dict to JSON string before validation
                        json_str = json.dumps(arguments)
                        yield self.registry[name].model_validate_json(
                            json_str, context=validation_context, strict=strict
                        )


if sys.version_info >= (3, 10):
    from types import UnionType

    def is_union_type(typehint: type[Iterable]) -> bool:
        return get_origin(get_args(typehint)[0]) in (Union, UnionType)

else:

    def is_union_type(typehint: type[Iterable]) -> bool:
        return get_origin(get_args(typehint)[0]) is Union


def get_types_array(typehint: type[Iterable]) -> tuple[type[T], ...]:
    should_be_iterable = get_origin(typehint)

    if should_be_iterable is not ABCIterable:
        raise TypeError(f"Model should be with Iterable instead of {typehint}")

    if is_union_type(typehint):
        # works for Iterable[Union[int, str]], Iterable[int | str]
        the_types = get_args(get_args(typehint)[0])
        return the_types

    # works for Iterable[int]
    return get_args(typehint)


def handle_parallel_model(
    typehint: type[Iterable],
) -> list[dict[str, Any]]:
    # Import at runtime to avoid circular import
    from ..processing.function_calls import openai_schema

    the_types = get_types_array(typehint)
    return [
        {"type": "function", "function": openai_schema(model).openai_schema}
        for model in the_types
    ]


def handle_anthropic_parallel_model(
    typehint: type[Iterable],
) -> list[dict[str, Any]]:
    # Import at runtime to avoid circular import
    from ..processing.function_calls import openai_schema

    the_types = get_types_array(typehint)
    return [openai_schema(model).anthropic_schema for model in the_types]


def ParallelModel(typehint: type[Iterable]) -> ParallelBase:
    the_types = get_types_array(typehint)
    return ParallelBase(*[model for model in the_types])


def VertexAIParallelModel(typehint: type[Iterable]) -> VertexAIParallelBase:
    the_types = get_types_array(typehint)
    return VertexAIParallelBase(*[model for model in the_types])


class AnthropicParallelBase(ParallelBase):
    def from_response(
        self,
        response: Any,
        mode: Mode,
        validation_context: Optional[Any] = None,
        strict: Optional[bool] = None,
    ) -> Generator[BaseModel, None, None]:
        assert mode == Mode.ANTHROPIC_PARALLEL_TOOLS, (
            "Mode must be ANTHROPIC_PARALLEL_TOOLS"
        )

        if not response or not hasattr(response, "content"):
            return

        for content in response.content:
            if getattr(content, "type", None) == "tool_use":
                name = content.name
                arguments = content.input
                if name in self.registry:
                    json_str = json.dumps(arguments)
                    yield self.registry[name].model_validate_json(
                        json_str, context=validation_context, strict=strict
                    )


def AnthropicParallelModel(typehint: type[Iterable]) -> AnthropicParallelBase:
    the_types = get_types_array(typehint)
    return AnthropicParallelBase(*[model for model in the_types])
