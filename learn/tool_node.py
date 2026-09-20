from langchain_core.messages import ToolMessage

if __package__:
    from .state_self_defined import MessagesState
    from .tool_self_defined import tools_by_name
else:
    from state_self_defined import MessagesState
    from tool_self_defined import tools_by_name


def tool_node(state: MessagesState):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )
    return {"messages": result}
