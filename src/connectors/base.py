from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class DatabaseConnector(ABC):
    """Abstract base class for database connectors."""

    @abstractmethod
    def connect(self) -> Any:
        """Return a connection object."""

    @abstractmethod
    def execute(self, query: str) -> Tuple[List[str], List[Tuple]]:
        """Execute a query and return (column_names, rows)."""

    @abstractmethod
    def get_schema(self) -> str:
        """Return a string representation of the database schema."""

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
