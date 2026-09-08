"""Abstract base adapter interfaces for COMPASS.

Defines the contract for ingesting IMU and GNSS sensor feeds across
heterogeneous platforms (Android SensorManager, external serial/Bluetooth/USB,
file replays, socket feeds).

Downstream navigation pipelines interact ONLY with these abstract interfaces,
guaranteeing platform-agnostic operation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from navigation.schemas.gnss import GNSSSample
from navigation.schemas.imu import RawIMUSample


class SensorAdapter(ABC):
    """Abstract interface for raw IMU sensor ingestion streams."""

    @abstractmethod
    def open(self) -> None:
        """Initialize and open the underlying hardware sensor feed or file stream."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Terminate stream, release hardware resources, or close file handles."""
        raise NotImplementedError

    @abstractmethod
    def read(self) -> Optional[RawIMUSample]:
        """Read the next available raw IMU measurement in device body frame.

        Returns:
            RawIMUSample if data is available, or None if the buffer is currently
            empty or the stream has concluded.
        """
        raise NotImplementedError

    @abstractmethod
    def is_open(self) -> bool:
        """Check whether the adapter is actively connected and streaming."""
        raise NotImplementedError


class GnssAdapter(ABC):
    """Abstract interface for GNSS fix ingestion streams."""

    @abstractmethod
    def open(self) -> None:
        """Initialize and open the GNSS receiver feed or location listener."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Terminate GNSS receiver connection and release resources."""
        raise NotImplementedError

    @abstractmethod
    def read(self) -> Optional[GNSSSample]:
        """Read the latest available GNSS fix sample.

        Returns:
            GNSSSample if a fix has arrived, or None if no new fix is pending.
        """
        raise NotImplementedError

    @abstractmethod
    def is_open(self) -> bool:
        """Check whether the GNSS adapter is active."""
        raise NotImplementedError
