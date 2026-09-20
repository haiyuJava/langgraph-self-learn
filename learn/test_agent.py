"""Verify graph wiring without an API key or network requests."""

import unittest

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from learn.build_and_compile_agent import build_agent
from learn.HelloWorld import graph


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        if {tool.name for tool in tools} != {"add", "multiply", "divide"}:
            raise AssertionError("The model must receive all arithmetic tools.")
        return self


class AgentTests(unittest.TestCase):
    def test_hello_world(self):
        result = graph.invoke({"messages": [HumanMessage(content="hi!")]})
        self.assertEqual(result["messages"][-1].content, "hello world")

    def test_no_tool_call_ends(self):
        model = ScriptedModel(responses=[AIMessage(content="Hello!")])
        result = build_agent(model).invoke(
            {"messages": [HumanMessage(content="Hi!")]}
        )
        self.assertEqual(result["llm_calls"], 1)
        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual(result["messages"][-1].content, "Hello!")

    def test_arithmetic_tool_loop(self):
        for name, a, b, expected in [
            ("add", 3, 4, "7"),
            ("multiply", 3, 4, "12"),
            ("divide", 7, 2, "3.5"),
        ]:
            with self.subTest(tool=name):
                model = ScriptedModel(
                    responses=[
                        AIMessage(
                            content="",
                            tool_calls=[
                                {"name": name, "args": {"a": a, "b": b}, "id": "call_1"}
                            ],
                        ),
                        AIMessage(content=expected),
                    ]
                )
                result = build_agent(model).invoke(
                    {"messages": [HumanMessage(content=f"{name} {a} and {b}")]}
                )
                self.assertEqual(result["llm_calls"], 2)
                self.assertEqual(len(result["messages"]), 4)
                observation = result["messages"][2]
                self.assertIsInstance(observation, ToolMessage)
                self.assertEqual(observation.content, expected)
                self.assertEqual(observation.tool_call_id, "call_1")
                self.assertEqual(result["messages"][-1].content, expected)


if __name__ == "__main__":
    unittest.main()
