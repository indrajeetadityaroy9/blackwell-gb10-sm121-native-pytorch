"""JSON I/O for pydantic models."""

from pathlib import Path
from typing import Type, TypeVar, Union

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def save_json_file(obj: BaseModel, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(obj.model_dump_json(indent=2, exclude_unset=True))


def load_json_file(model_cls: Type[T], path: Union[str, Path]) -> T:
    return model_cls.model_validate_json(Path(path).read_text())
