"""Test the exact user scenario that was failing."""

from collections.abc import Iterable

from pydantic import BaseModel

import instructor


# Simulate user's term_concept model
class TermConcept(BaseModel):
    name: str
    description: str

# Simulate user's function
def extract_user(chn_text: str, en_text: str):
    # Simplified prompt
    turn_prompt = f"Process: {chn_text} -> {en_text}"
    system_prompt = "You are a helpful assistant."
    
    return client.create(
        model="gpt-4o-mini",
        response_model=Iterable[TermConcept],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": turn_prompt}
        ],
        stream=True,
        max_retries=3,
    )

# Test the scenario
if __name__ == "__main__":
    print("Testing user scenario with Iterable[TermConcept] and stream=True...\n")
    
    # Create client using from_provider
    client = instructor.from_provider("openai/gpt-4o-mini", mode=instructor.Mode.PARALLEL_TOOLS)
    
    # Check if response_model is correctly handled
    from instructor.dsl.parallel import ParallelBase
    
    response_model = Iterable[TermConcept]
    
    print(f"Response model before create: {response_model}")
    print(f"Type: {type(response_model)}")
    print(f"Is ParallelBase: {isinstance(response_model, ParallelBase)}")
    
    if isinstance(response_model, ParallelBase):
        print("\n❌ ERROR: Iterable[TermConcept] was incorrectly converted to ParallelBase!")
        print("This is the bug we're trying to fix.")
    else:
        print("\n✓ PASS: Iterable[TermConcept] is NOT a ParallelBase instance (correct)")
    
    # Now test what happens during create
    # We need to simulate the handle_response_model processing
    from instructor.mode import Mode
    from instructor.processing.response import handle_response_model
    
    processed_model, kwargs = handle_response_model(
        response_model=response_model,
        mode=Mode.PARALLEL_TOOLS,
        stream=True,
        messages=[]
    )
    
    print(f"\nAfter handle_response_model:")
    print(f"  - Processed model type: {type(processed_model)}")
    print(f"  - Is ParallelBase: {isinstance(processed_model, ParallelBase)}")
    
    if isinstance(processed_model, ParallelBase):
        print("\n❌ ERROR: Still converting to ParallelBase after fix!")
        print("The fix did not work correctly.")
    else:
        print("\n✓ PASS: Not converted to ParallelBase (fix is working!)")
    
    print("\nTest complete!")        print("\n✓ PASS: Not converted to ParallelBase (fix is working!)")
    
    print("\nTest complete!")