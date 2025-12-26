"""Test to verify Iterable[Model] is handled correctly and not converted to ParallelBase."""

from collections.abc import Iterable

import pytest
from pydantic import BaseModel

from instructor.dsl.parallel import ParallelBase, is_union_type
from instructor.mode import Mode


class TermConcept(BaseModel):
    name: str
    description: str


def test_iterable_single_type_with_parallel_tools_mode():
    """
    Test that Iterable[Model] in PARALLEL_TOOLS mode is NOT converted to ParallelBase.
    
    This is the key fix: Iterable[Model] should use IterableBase,
    not ParallelBase. Only Iterable[Union[ModelA, ModelB, ...]]
    should convert to ParallelBase.
    """
    from instructor.processing.response import handle_response_model
    
    response_model = Iterable[TermConcept]
    
    # Process with stream=True in PARALLEL_TOOLS mode
    response_model_processed, kwargs = handle_response_model(
        response_model=response_model,
        mode=Mode.PARALLEL_TOOLS,
        stream=True,
        messages=[],
    )
    
    # Verify response_model was NOT converted to ParallelBase
    assert not isinstance(response_model_processed, ParallelBase), (
        "Iterable[Model] should NOT be converted to ParallelBase"
    )
    
    # Verify it's still an Iterable type
    from typing import get_origin
    assert get_origin(response_model_processed) is Iterable, (
        "Response model should still be Iterable type"
    )


def test_iterable_union_type_with_parallel_tools_mode():
    """
    Test that Iterable[Union[ModelA, ModelB]] in PARALLEL_TOOLS mode IS converted to ParallelBase.
    
    This ensures backward compatibility: Iterable[Union[...]] should still work
    as parallel mode.
    """
    from typing import Union

    from instructor.processing.response import handle_response_model
    
    class ModelA(BaseModel):
        name: str
        
    class ModelB(BaseModel):
        value: int
    
    response_model = Iterable[Union[ModelA, ModelB]]
    
    # Process with stream=True in PARALLEL_TOOLS mode
    response_model_processed, kwargs = handle_response_model(
        response_model=response_model,
        mode=Mode.PARALLEL_TOOLS,
        stream=True,
        messages=[],
    )
    
    # Verify response_model WAS converted to ParallelBase
    assert isinstance(response_model_processed, ParallelBase), (
        "Iterable[Union[...]] should be converted to ParallelBase"
    )


def test_iterable_single_type_with_tools_mode():
    """
    Test that Iterable[Model] in regular TOOLS mode is handled correctly.
    
    This should work normally without any special parallel handling.
    """
    from instructor.processing.response import handle_response_model
    
    response_model = Iterable[TermConcept]
    
    # Process with stream=True in TOOLS mode
    response_model_processed, kwargs = handle_response_model(
        response_model=response_model,
        mode=Mode.TOOLS,
        stream=True,
        messages=[],
    )
    
    # Verify response_model was NOT converted to ParallelBase
    assert not isinstance(response_model_processed, ParallelBase), (
        "Iterable[Model] in TOOLS mode should NOT be converted to ParallelBase"
    )
    
    # Verify stream=True is preserved
    assert kwargs.get("stream") is True, (
        "stream=True should be preserved for IterableBase processing"
    )


def test_is_union_type_helper():
    """
    Test the is_union_type helper function from parallel module.
    """
    assert is_union_type(Iterable[Union[str, int]]) is True
    assert is_union_type(Iterable[Union[int, str, float]]) is True
    assert is_union_type(Iterable[str]) is False
    assert is_union_type(Iterable[int]) is False
    assert is_union_type(Iterable[TermConcept]) is False


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])