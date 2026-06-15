"""
tools/longseq/dataset_schema.py

Dataset-specific gaze_type interpretation for EGTEA, Ego4D, and EgoExo4D.

This module encodes the dataset-specific semantics of the ``gaze_type`` column
from Requirements 7 and 12:

- EGTEA:   gaze_type == 1 → fixation; gaze_type == 0 → untracked
- Ego4D:   gaze_type == 0 → fixation; gaze_type == 1 → saccade; gaze_type == 2 → OOB
- EgoExo4D: gaze_type >= 0.5 → tracked; no fixation/saccade distinction

The ``FrameRecord`` dataclass is defined in ``loader.py`` (task 3.1).  This module
uses a forward reference to avoid a circular import.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .loader import FrameRecord


class DatasetSchema(ABC):
    """Abstract base class for dataset-specific gaze_type interpretation.

    Subclasses encode the semantics of the ``gaze_type`` column for a specific
    dataset.  All methods receive a :class:`FrameRecord` instance and return
    a boolean indicating whether the frame satisfies the queried condition.

    The centralised valid-frame logic in ``loader.is_valid_frame()`` calls
    ``is_valid_tracked()`` as part of its four-condition check.  Metric modules
    must use ``is_valid_frame()`` rather than calling schema methods directly.
    """

    @abstractmethod
    def is_valid_tracked(self, record: FrameRecord) -> bool:
        """Return ``True`` if the frame is a valid tracked frame.

        A tracked frame is one where the ground-truth gaze annotation is
        reliable and within the image bounds.  The exact interpretation of
        ``gaze_type`` values is dataset-specific.

        Parameters
        ----------
        record:
            A :class:`FrameRecord` instance with a populated ``gaze_type`` field.

        Returns
        -------
        bool
            ``True`` if the frame is a valid tracked frame for this dataset.
        """
        ...

    @abstractmethod
    def is_fixation(self, record: FrameRecord) -> bool:
        """Return ``True`` if the frame is annotated as a fixation.

        Parameters
        ----------
        record:
            A :class:`FrameRecord` instance with a populated ``gaze_type`` field.

        Returns
        -------
        bool
            ``True`` if the frame is a fixation frame for this dataset.

        Raises
        ------
        NotImplementedError
            For datasets that do not provide fixation annotations (EgoExo4D).
        """
        ...

    @abstractmethod
    def is_saccade(self, record: FrameRecord) -> bool:
        """Return ``True`` if the frame is annotated as a saccade.

        Parameters
        ----------
        record:
            A :class:`FrameRecord` instance with a populated ``gaze_type`` field.

        Returns
        -------
        bool
            ``True`` if the frame is a saccade frame for this dataset.
        """
        ...

    @abstractmethod
    def has_saccade_annotations(self) -> bool:
        """Return ``True`` if the dataset provides saccade annotations.

        Returns
        -------
        bool
            ``True`` if saccade transition metrics can be computed.
        """
        ...


class EGTEASchema(DatasetSchema):
    """Schema for the EGTEA dataset.

    EGTEA encodes ``gaze_type`` as:

    - ``1`` → fixation (valid tracked frame)
    - ``0`` → untracked

    EGTEA does not provide saccade annotations, so
    :meth:`has_saccade_annotations` returns ``False`` and :meth:`is_saccade`
    always returns ``False``.

    Metrics computable (Req 12.1):
        All metrics except Saccade Transition Accuracy.
    """

    def is_valid_tracked(self, record: FrameRecord) -> bool:
        # Req 7.2: gaze_type == 1 → fixation (valid tracked)
        return record.gaze_type == 1

    def is_fixation(self, record: FrameRecord) -> bool:
        # Req 7.2: gaze_type == 1 → fixation
        return record.gaze_type == 1

    def is_saccade(self, record: FrameRecord) -> bool:
        # EGTEA has no saccade annotations
        return False

    def has_saccade_annotations(self) -> bool:
        # Req 12.1: no saccade annotations
        return False


class Ego4DSchema(DatasetSchema):
    """Schema for the Ego4D dataset.

    Ego4D encodes ``gaze_type`` as:

    - ``0`` → fixation
    - ``1`` → saccade
    - ``2`` → out-of-bounds (OOB)

    Both fixation (``0``) and saccade (``1``) are considered valid tracked
    frames.  Ego4D provides saccade annotations, so
    :meth:`has_saccade_annotations` returns ``True``.

    Metrics computable (Req 12.2):
        All metrics including Saccade Transition Accuracy.
    """

    def is_valid_tracked(self, record: FrameRecord) -> bool:
        # Req 7.3: gaze_type in (0, 1) → tracked (fixation or saccade)
        return record.gaze_type in (0, 1)

    def is_fixation(self, record: FrameRecord) -> bool:
        # Req 7.3: gaze_type == 0 → fixation
        return record.gaze_type == 0

    def is_saccade(self, record: FrameRecord) -> bool:
        # Req 7.3: gaze_type == 1 → saccade
        return record.gaze_type == 1

    def has_saccade_annotations(self) -> bool:
        # Req 12.2: saccade annotations available
        return True


class EgoExo4DSchema(DatasetSchema):
    """Schema for the EgoExo4D dataset.

    EgoExo4D encodes ``gaze_type`` as a continuous value in ``[0, 1]``:

    - ``gaze_type >= 0.5`` → tracked frame
    - ``gaze_type < 0.5`` → untracked frame

    EgoExo4D does **not** provide fixation or saccade annotations.  Calling
    :meth:`is_fixation` raises :class:`NotImplementedError` because pseudo-
    fixation inference is forbidden (design doc §Dataset Schema).

    Metrics computable (Req 12.3):
        All metrics except Saccade Transition Accuracy.
        Jitter and Fixation Stability metrics are **forbidden**, not merely
        skipped, because ``is_fixation()`` raises ``NotImplementedError``.
    """

    def is_valid_tracked(self, record: FrameRecord) -> bool:
        # Req 7.4: gaze_type >= 0.5 → tracked
        return record.gaze_type >= 0.5

    def is_fixation(self, record: FrameRecord) -> bool:
        # EgoExo4D has no fixation annotations.
        # Pseudo-fixation inference is FORBIDDEN (design doc).
        raise NotImplementedError(
            "EgoExo4D has no fixation annotations; jitter/stability metrics are forbidden"
        )

    def is_saccade(self, record: FrameRecord) -> bool:
        # EgoExo4D has no saccade annotations
        return False

    def has_saccade_annotations(self) -> bool:
        # Req 12.3: no saccade annotations
        return False


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

# Registry mapping dataset names (lower-case) to schema classes.
_SCHEMA_REGISTRY: dict[str, type[DatasetSchema]] = {
    "egtea": EGTEASchema,
    "ego4d": Ego4DSchema,
    "egoexo4d": EgoExo4DSchema,
}


def get_schema(dataset: str) -> DatasetSchema:
    """Return a :class:`DatasetSchema` instance for the named dataset.

    Parameters
    ----------
    dataset:
        Dataset name, one of ``"egtea"``, ``"ego4d"``, or ``"egoexo4d"``.
        Case-insensitive.

    Returns
    -------
    DatasetSchema
        An instance of the appropriate schema class.

    Raises
    ------
    ValueError
        If ``dataset`` is not a recognised dataset name.
    """
    key = dataset.lower()
    if key not in _SCHEMA_REGISTRY:
        valid = sorted(_SCHEMA_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset {dataset!r}. Must be one of: {valid}."
        )
    return _SCHEMA_REGISTRY[key]()
