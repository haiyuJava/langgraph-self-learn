import argparse
import os
from functools import partial
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

if __package__:
    from .end_logic import should_continue
    from .model_node_self_defined import llm_call
    from .state_self_defined import MessagesState
    from .tool_node import tool_node
    from .tool_self_defined import tools
else:
    from end_logic import should_continue
    from model_node_self_defined import llm_call
    from state_self_defined import MessagesState
    from tool_node import tool_node
    from tool_self_defined import tools


def build_agent(model=None):
    """Build the workflow, optionally using an injected chat model."""
    if model is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("Set ANTHROPIC_API_KEY before running the Claude agent.")
        model = init_chat_model(
            "claude-sonnet-4-6", model_provider="anthropic", temperature=0
        )
    model_with_tools = model.bind_tools(tools)

    agent_builder = StateGraph(MessagesState)
    agent_builder.add_node(
        "llm_call", partial(llm_call, model_with_tools=model_with_tools)
    )
    agent_builder.add_node("tool_node", tool_node)
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call", should_continue, ["tool_node", END]
    )
    agent_builder.add_edge("tool_node", "llm_call")
    return agent_builder.compile()


def main():
    parser = argparse.ArgumentParser(description="Run the arithmetic agent.")
    parser.add_argument("prompt", nargs="?", default="Add 3 and 4.")
    parser.add_argument("--draw", metavar="PNG", help="Save a graph PNG (uses network).")
    args = parser.parse_args()
    agent = build_agent()
    messages = agent.invoke({"messages": [HumanMessage(content=args.prompt)]})
    for message in messages["messages"]:
        message.pretty_print()
    if args.draw:
        Path(args.draw).write_bytes(agent.get_graph().draw_mermaid_png())


if __name__ == "__main__":
    main()
