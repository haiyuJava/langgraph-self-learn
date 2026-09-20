import operator
from typing import Annotated

from langchain_core.messages import AnyMessage
from typing_extensions import NotRequired, TypedDict


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: NotRequired[int]
