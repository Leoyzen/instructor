"""
Test cases for get_types_array function to verify it works with both
typing.Iterable and collections.abc.Iterable.

This tests the fix for the bug where typing.Iterable was not recognized
due to identity comparison (is) instead of name-based comparison.
"""

import pytest
from pydantic import BaseModel

from instructor.dsl.parallel import get_types_array


# Test models
class User(BaseModel):
    name: str
    age: int


class Task(BaseModel):
    title: str
    description: str


class Organization(BaseModel):
    company: str
    employees: int


class TestGetTypesArrayTypingIterable:
    """Test get_types_array with typing.Iterable (Python 3.9+ recommended way)."""

    def test_typing_iterable_single_type(self):
        """Test typing.Iterable with single type."""
        from collections.abc import Iterable

        typehint = Iterable[User]
        result = get_types_array(typehint)

        assert result == (User,)
        assert len(result) == 1
        assert result[0] == User

    def test_typing_iterable_union(self):
        """Test typing.Iterable with Union type."""
        from collections.abc import Iterable
        from typing import Union

        typehint = Iterable[Union[User, Task]]
        result = get_types_array(typehint)

        # Union types are extracted
        assert User in result
        assert Task in result
        assert len(result) == 2

    def test_typing_iterable_union_pipe_syntax(self):
        """Test typing.Iterable with pipe syntax (Python 3.10+)."""
        # This test only works on Python 3.10+
        import sys

        if sys.version_info >= (3, 10):
            from collections.abc import Iterable

            typehint = Iterable[User | Task]
            result = get_types_array(typehint)

            assert User in result
            assert Task in result

    def test_typing_iterable_multiple_union(self):
        """Test typing.Iterable with multiple types in Union."""
        from collections.abc import Iterable
        from typing import Union

        typehint = Iterable[Union[User, Task, Organization]]
        result = get_types_array(typehint)

        assert len(result) == 3
        assert User in result
        assert Task in result
        assert Organization in result


class TestGetTypesArrayAbcIterable:
    """Test get_types_array with collections.abc.Iterable (old way)."""

    def test_abc_iterable_single_type(self):
        """Test collections.abc.Iterable with single type."""
        from collections.abc import Iterable

        typehint = Iterable[User]
        result = get_types_array(typehint)

        assert result == (User,)
        assert len(result) == 1
        assert result[0] == User

    def test_abc_iterable_union(self):
        """Test collections.abc.Iterable with Union type."""
        from collections.abc import Iterable
        from typing import Union

        typehint = Iterable[Union[User, Task]]
        result = get_types_array(typehint)

        assert User in result
        assert Task in result
        assert len(result) == 2

    def test_abc_iterable_multiple_union(self):
        """Test collections.abc.Iterable with multiple types in Union."""
        from collections.abc import Iterable
        from typing import Union

        typehint = Iterable[Union[User, Task, Organization]]
        result = get_types_array(typehint)

        assert len(result) == 3
        assert User in result
        assert Task in result
        assert Organization in result


class TestGetTypesArrayErrorCases:
    """Test error cases for get_types_array."""

    def test_non_iterable_type_raises_error(self):
        """Test that non-Iterable type raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            get_types_array(User)

        assert "Model should be with Iterable" in str(exc_info.value)

    def test_list_type_raises_error(self):
        """Test that List type (not Iterable) raises TypeError."""
        from typing import List

        with pytest.raises(TypeError) as exc_info:
            get_types_array(List[User])

        assert "Model should be with Iterable" in str(exc_info.value)

    def test_none_type_raises_error(self):
        """Test that None type raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            get_types_array(None)

        # The error message will mention that typehint is None
        assert "Model should be with Iterable" in str(exc_info.value)

    def test_simple_type_raises_error(self):
        """Test that simple type (not Iterable) raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            get_types_array(str)

        assert "Model should be with Iterable" in str(exc_info.value)


class TestGetTypesArrayCompatibility:
    """Test that both typing.Iterable and collections.abc.Iterable work identically."""

    def test_both_iterable_types_produce_same_result(self):
        """Test that typing.Iterable and collections.abc.Iterable produce same result."""
        from collections.abc import Iterable as AbcIterable
        from collections.abc import Iterable as TypingIterable
        from typing import Union

        # With Union
        typing_typehint = TypingIterable[Union[User, Task]]
        abc_typehint = AbcIterable[Union[User, Task]]

        typing_result = get_types_array(typing_typehint)
        abc_result = get_types_array(abc_typehint)

        # Both should produce the same result
        assert typing_result == abc_result
        assert User in typing_result
        assert Task in typing_result

    def test_single_type_both_variants(self):
        """Test single type with both Iterable variants."""
        from collections.abc import Iterable as AbcIterable
        from collections.abc import Iterable as TypingIterable

        typing_typehint = TypingIterable[User]
        abc_typehint = AbcIterable[User]

        typing_result = get_types_array(typing_typehint)
        abc_result = get_types_array(abc_typehint)

        assert typing_result == abc_result
        assert abc_result == (User,)