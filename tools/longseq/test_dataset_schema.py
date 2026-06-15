"""
Property-based tests for dataset_schema.py.

These tests verify the correctness of the three dataset schemas (EGTEA, Ego4D,
EgoExo4D) using property-based testing with the hypothesis library.

The tests cover:
- is_valid_tracked() for boundary gaze_type values
- is_fixation() for boundary gaze_type values
- is_saccade() for boundary gaze_type values
- EgoExo4DSchema.is_fixation() raising NotImplementedError
- get_schema() factory for valid and invalid dataset names
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st
from hypothesis.strategies import composite

from tools.longseq.dataset_schema import (
    DatasetSchema,
    EGTEASchema,
    Ego4DSchema,
    EgoExo4DSchema,
    get_schema,
)


# ---------------------------------------------------------------------------
# Mock FrameRecord for testing
# ---------------------------------------------------------------------------

class MockFrameRecord:
    """Minimal mock of FrameRecord for testing schema methods.

    The real FrameRecord is defined in loader.py (task 3.1).  This mock
    provides only the gaze_type attribute needed for schema testing.
    """

    def __init__(self, gaze_type: int | float):
        self.gaze_type = gaze_type


# ---------------------------------------------------------------------------
# Test strategies
# ---------------------------------------------------------------------------

@composite
def gaze_type_int(draw):
    """Generate integer gaze_type values covering the full range."""
    return draw(st.integers(min_value=-10, max_value=10))


@composite
def gaze_type_float(draw):
    """Generate float gaze_type values covering the full range."""
    return draw(st.floats(min_value=-1.0, max_value=2.0, allow_nan=False, allow_infinity=False))


@composite
def boundary_gaze_type(draw):
    """Generate gaze_type values at and around key boundaries.

    Key boundaries:
    - EGTEA: 0, 1
    - Ego4D: 0, 1, 2
    - EgoExo4D: 0.5
    """
    boundaries = [-1, 0, 0.4, 0.49, 0.5, 0.51, 0.6, 1, 1.5, 2, 3]
    return draw(st.sampled_from(boundaries))


# ---------------------------------------------------------------------------
# EGTEASchema tests
# ---------------------------------------------------------------------------

class TestEGTEASchema:
    """Property-based tests for EGTEASchema."""

    @given(gaze_type=gaze_type_int())
    def test_is_valid_tracked(self, gaze_type: int):
        """EGTEA: is_valid_tracked returns True iff gaze_type == 1."""
        schema = EGTEASchema()
        record = MockFrameRecord(gaze_type)
        assert schema.is_valid_tracked(record) == (gaze_type == 1)

    @given(gaze_type=gaze_type_int())
    def test_is_fixation(self, gaze_type: int):
        """EGTEA: is_fixation returns True iff gaze_type == 1."""
        schema = EGTEASchema()
        record = MockFrameRecord(gaze_type)
        assert schema.is_fixation(record) == (gaze_type == 1)

    @given(gaze_type=gaze_type_int())
    def test_is_saccade_always_false(self, gaze_type: int):
        """EGTEA: is_saccade always returns False (no saccade annotations)."""
        schema = EGTEASchema()
        record = MockFrameRecord(gaze_type)
        assert schema.is_saccade(record) is False

    def test_has_saccade_annotations_false(self):
        """EGTEA: has_saccade_annotations returns False."""
        schema = EGTEASchema()
        assert schema.has_saccade_annotations() is False

    @pytest.mark.parametrize("gaze_type", [0, 1, -1, 2, 100])
    def test_boundary_values(self, gaze_type: int):
        """Test specific boundary values for EGTEA."""
        schema = EGTEASchema()
        record = MockFrameRecord(gaze_type)

        expected_valid = (gaze_type == 1)
        expected_fixation = (gaze_type == 1)

        assert schema.is_valid_tracked(record) == expected_valid
        assert schema.is_fixation(record) == expected_fixation
        assert schema.is_saccade(record) is False


# ---------------------------------------------------------------------------
# Ego4DSchema tests
# ---------------------------------------------------------------------------

class TestEgo4DSchema:
    """Property-based tests for Ego4DSchema."""

    @given(gaze_type=gaze_type_int())
    def test_is_valid_tracked(self, gaze_type: int):
        """Ego4D: is_valid_tracked returns True iff gaze_type in (0, 1)."""
        schema = Ego4DSchema()
        record = MockFrameRecord(gaze_type)
        assert schema.is_valid_tracked(record) == (gaze_type in (0, 1))

    @given(gaze_type=gaze_type_int())
    def test_is_fixation(self, gaze_type: int):
        """Ego4D: is_fixation returns True iff gaze_type == 0."""
        schema = Ego4DSchema()
        record = MockFrameRecord(gaze_type)
        assert schema.is_fixation(record) == (gaze_type == 0)

    @given(gaze_type=gaze_type_int())
    def test_is_saccade(self, gaze_type: int):
        """Ego4D: is_saccade returns True iff gaze_type == 1."""
        schema = Ego4DSchema()
        record = MockFrameRecord(gaze_type)
        assert schema.is_saccade(record) == (gaze_type == 1)

    def test_has_saccade_annotations_true(self):
        """Ego4D: has_saccade_annotations returns True."""
        schema = Ego4DSchema()
        assert schema.has_saccade_annotations() is True

    @pytest.mark.parametrize("gaze_type", [0, 1, 2, -1, 3])
    def test_boundary_values(self, gaze_type: int):
        """Test specific boundary values for Ego4D."""
        schema = Ego4DSchema()
        record = MockFrameRecord(gaze_type)

        expected_valid = (gaze_type in (0, 1))
        expected_fixation = (gaze_type == 0)
        expected_saccade = (gaze_type == 1)

        assert schema.is_valid_tracked(record) == expected_valid
        assert schema.is_fixation(record) == expected_fixation
        assert schema.is_saccade(record) == expected_saccade

    def test_fixation_and_saccade_mutually_exclusive(self):
        """Ego4D: fixation and saccade are mutually exclusive for valid frames."""
        schema = Ego4DSchema()

        # For gaze_type 0: fixation=True, saccade=False
        record_fixation = MockFrameRecord(0)
        assert schema.is_fixation(record_fixation) is True
        assert schema.is_saccade(record_fixation) is False

        # For gaze_type 1: fixation=False, saccade=True
        record_saccade = MockFrameRecord(1)
        assert schema.is_fixation(record_saccade) is False
        assert schema.is_saccade(record_saccade) is True


# ---------------------------------------------------------------------------
# EgoExo4DSchema tests
# ---------------------------------------------------------------------------

class TestEgoExo4DSchema:
    """Property-based tests for EgoExo4DSchema."""

    @given(gaze_type=gaze_type_float())
    def test_is_valid_tracked(self, gaze_type: float):
        """EgoExo4D: is_valid_tracked returns True iff gaze_type >= 0.5."""
        schema = EgoExo4DSchema()
        record = MockFrameRecord(gaze_type)
        assert schema.is_valid_tracked(record) == (gaze_type >= 0.5)

    def test_is_fixation_raises_not_implemented(self):
        """EgoExo4D: is_fixation raises NotImplementedError with correct message."""
        schema = EgoExo4DSchema()
        record = MockFrameRecord(0.7)

        with pytest.raises(NotImplementedError) as exc_info:
            schema.is_fixation(record)

        assert "EgoExo4D has no fixation annotations" in str(exc_info.value)
        assert "jitter/stability metrics are forbidden" in str(exc_info.value)

    @given(gaze_type=gaze_type_float())
    def test_is_saccade_always_false(self, gaze_type: float):
        """EgoExo4D: is_saccade always returns False (no saccade annotations)."""
        schema = EgoExo4DSchema()
        record = MockFrameRecord(gaze_type)
        assert schema.is_saccade(record) is False

    def test_has_saccade_annotations_false(self):
        """EgoExo4D: has_saccade_annotations returns False."""
        schema = EgoExo4DSchema()
        assert schema.has_saccade_annotations() is False

    @pytest.mark.parametrize("gaze_type", [0.0, 0.49, 0.5, 0.51, 1.0])
    def test_boundary_values(self, gaze_type: float):
        """Test specific boundary values for EgoExo4D."""
        schema = EgoExo4DSchema()
        record = MockFrameRecord(gaze_type)

        expected_valid = (gaze_type >= 0.5)

        assert schema.is_valid_tracked(record) == expected_valid
        assert schema.is_saccade(record) is False

        # is_fixation should always raise NotImplementedError
        with pytest.raises(NotImplementedError):
            schema.is_fixation(record)

    def test_boundary_exactly_0_5(self):
        """EgoExo4D: gaze_type == 0.5 is considered tracked."""
        schema = EgoExo4DSchema()
        record = MockFrameRecord(0.5)
        assert schema.is_valid_tracked(record) is True

    def test_just_below_threshold(self):
        """EgoExo4D: gaze_type < 0.5 is not tracked."""
        schema = EgoExo4DSchema()
        record = MockFrameRecord(0.499)
        assert schema.is_valid_tracked(record) is False


# ---------------------------------------------------------------------------
# get_schema factory tests
# ---------------------------------------------------------------------------

class TestGetSchema:
    """Tests for the get_schema factory function."""

    @pytest.mark.parametrize("dataset_name", ["egtea", "EGTEA", "EgTeA"])
    def test_get_schema_egtea(self, dataset_name: str):
        """get_schema returns EGTEASchema for 'egtea' (case-insensitive)."""
        schema = get_schema(dataset_name)
        assert isinstance(schema, EGTEASchema)

    @pytest.mark.parametrize("dataset_name", ["ego4d", "EGO4D", "Ego4d"])
    def test_get_schema_ego4d(self, dataset_name: str):
        """get_schema returns Ego4DSchema for 'ego4d' (case-insensitive)."""
        schema = get_schema(dataset_name)
        assert isinstance(schema, Ego4DSchema)

    @pytest.mark.parametrize("dataset_name", ["egoexo4d", "EGOEXO4D", "EgoExo4D"])
    def test_get_schema_egoexo4d(self, dataset_name: str):
        """get_schema returns EgoExo4DSchema for 'egoexo4d' (case-insensitive)."""
        schema = get_schema(dataset_name)
        assert isinstance(schema, EgoExo4DSchema)

    @pytest.mark.parametrize("invalid_name", ["unknown", "invalid", "EGTEA_GAZE", "ego", ""])
    def test_get_schema_invalid_raises_value_error(self, invalid_name: str):
        """get_schema raises ValueError for unknown dataset names."""
        with pytest.raises(ValueError) as exc_info:
            get_schema(invalid_name)

        assert "Unknown dataset" in str(exc_info.value)
        assert invalid_name in str(exc_info.value)

    def test_get_schema_returns_new_instance(self):
        """get_schema returns a new instance on each call."""
        schema1 = get_schema("egtea")
        schema2 = get_schema("egtea")
        assert schema1 is not schema2


# ---------------------------------------------------------------------------
# Cross-schema consistency tests
# ---------------------------------------------------------------------------

class TestCrossSchemaConsistency:
    """Tests for consistency across schemas."""

    @given(gaze_type=boundary_gaze_type())
    def test_ego4d_fixation_saccade_exclusive(self, gaze_type: int | float):
        """Ego4D: fixation and saccade are mutually exclusive for all values."""
        schema = Ego4DSchema()
        record = MockFrameRecord(gaze_type)

        # A frame cannot be both fixation and saccade
        assert not (schema.is_fixation(record) and schema.is_saccade(record))

    @given(gaze_type=boundary_gaze_type())
    def test_egtea_valid_implies_fixation(self, gaze_type: int | float):
        """EGTEA: if is_valid_tracked is True, then is_fixation is also True."""
        schema = EGTEASchema()
        record = MockFrameRecord(gaze_type)

        if schema.is_valid_tracked(record):
            assert schema.is_fixation(record) is True

    @given(gaze_type=boundary_gaze_type())
    def test_ego4d_valid_implies_fixation_or_saccade(self, gaze_type: int | float):
        """Ego4D: if is_valid_tracked is True, frame is either fixation or saccade."""
        schema = Ego4DSchema()
        record = MockFrameRecord(gaze_type)

        if schema.is_valid_tracked(record):
            assert schema.is_fixation(record) or schema.is_saccade(record)
