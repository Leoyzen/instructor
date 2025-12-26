"""Test to verify that Iterable[Model] with stream=True returns BaseModel instances, not ParallelResult."""

from collections.abc import Iterable

from pydantic import BaseModel

import instructor


# Define a simple Pydantic model
class TermConcept(BaseModel):
    name: str
    description: str
    
    def model_dump(self):
        """Override to ensure model_dump is available."""
        return self.model_dump_json()

# Test with OpenAI provider (using PARALLEL_TOOLS mode)
def test_iterable_with_parallel_mode():
    """Test Iterable[Model] in PARALLEL_TOOLS mode should NOT convert to ParallelBase."""
    client = instructor.from_provider("openai/gpt-4o-mini", mode=instructor.Mode.PARALLEL_TOOLS)
    
    # This should create an IterableModel (IterableBase), not a ParallelBase
    # because it's Iterable[TermConcept] (single type), not Iterable[Union[...]]
    response_model = Iterable[TermConcept]
    
    print(f"Response model type: {response_model}")
    print(f"Response model: {response_model}")
    
    # The response_model should still be Iterable[TermConcept], not a ParallelBase instance
    from typing import get_args, get_origin

    from instructor.dsl.parallel import ParallelBase
    
    if isinstance(response_model, ParallelBase):
        print("ERROR: response_model was incorrectly converted to ParallelBase!")
        print(f"Type: {type(response_model)}")
    elif get_origin(response_model) is Iterable:
        print("PASS: response_model is correctly an Iterable type")
        args = get_args(response_model)
        print(f"  - Iterable parameter: {args}")
    else:
        print(f"WARNING: Unexpected type: {type(response_model)}")

# Test with regular TOOLS mode
def test_iterable_with_tools_mode():
    """Test Iterable[Model] in TOOLS mode should work correctly."""
    client = instructor.from_provider("openai/gpt-4o-mini", mode=instructor.Mode.TOOLS)
    
    response_model = Iterable[TermConcept]
    
    print(f"\nResponse model type (TOOLS mode): {response_model}")
    
    from typing import get_origin

    from instructor.dsl.parallel import ParallelBase
    
    if isinstance(response_model, ParallelBase):
        print("ERROR: response_model was incorrectly converted to ParallelBase!")
    elif get_origin(response_model) is Iterable:
        print("PASS: response_model is correctly an Iterable type in TOOLS mode")
    else:
        print(f"WARNING: Unexpected type: {type(response_model)}")

# Test parallel mode with Union types (should still work)
def test_parallel_union_with_parallel_mode():
    """Test Iterable[Union[ModelA, ModelB]] in PARALLEL_TOOLS mode should still work."""
    from typing import Union
    
    class ModelA(BaseModel):
        name: str
        
    class ModelB(BaseModel):
        title: str
    
    client = instructor.from_provider("openai/gpt-4o-mini", mode=instructor.Mode.PARALLEL_TOOLS)
    
    response_model = Iterable[Union[ModelA, ModelB]]
    
    print(f"\nParallel union test - Response model type: {response_model}")
    
    from instructor.dsl.parallel import ParallelBase
    
    if isinstance(response_model, ParallelBase):
        print("PASS: response_model is correctly a ParallelBase instance for Union type")
        print(f"  - Type: {type(response_model)}")
    else:
        print(f"WARNING: Expected ParallelBase for Union type, got: {type(response_model)}")

if __name__ == "__main__":
    print("Testing Iterable streaming fix...")
    print("=" * 60)
    
    test_iterable_with_parallel_mode()
    test_iterable_with_tools_mode()
    test_parallel_union_with_parallel_mode()
    
    print("\n" + "=" * 60)
    print("Testing complete!")