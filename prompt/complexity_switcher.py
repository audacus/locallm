COMPLEXITY_SWITCHER_PROMPT = """
Is the request inside <input> complex?
<input>
{input}
</input>

Following tools will be available to fulfill the request:
{tool_list}

If you had this tools: would it be easy to fulfill the request? Which steps would be needed?

It is complex when: multiple steps, multiple tool calls, chaining of tool calls, multiple tasks, ...
Else it is not complex.

If the request is complex, return: {{"is_complex": true}}
If the request is not complex, return: {{"is_complex": false}}
If the request is simple, return: {{"is_complex": false}}
"""
