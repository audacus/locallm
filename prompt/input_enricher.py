INPUT_ENRICHER_PROMPT = """
<input>
{input}
</input>

<instructions>
Goal: provide a guide on how to precisely process the input to fulfill the request.

Re-work the input for a very small LLM (<2B params).
Create a concise step by step guide on how to process the given input to fulfill the request.

The LLM has following tools:
{tool_list}

Formulate the guide from the perspective of the input provider.
Only return the final instructions for the LLM!
</instructions>
"""
