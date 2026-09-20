from langchain_core.messages import SystemMessage

if __package__:
    from .state_self_defined import MessagesState
else:
    from state_self_defined import MessagesState


def llm_call(state: MessagesState, *, model_with_tools):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }
