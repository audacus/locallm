INPUT_ENRICHER_PROMPT = """
<input>
{input}
</input>

<instructions_complex>
If the request in the <input> is complex (not straight forward, multistep, ...), follow these instructions.
If the request is simple, follow the instructions in <instructions_simple>.

Goal: provide a precise guide on how to fulfill the given request.

Re-work the <input> for a very small LLM (<2B params).
Create a concise step by step guide that leads exactly leads to the fulfillment of the request in <input>.

The LLM will have following tools available:
{tool_list}

Formulate the guide from the perspective of the input provider.
Only return the final instructions for the LLM!
</instructions_complex>

<instructions_simple>
If the request in <input> is simple and can be solved in a straight forward way, return without any output.
</instructions_simple>
"""
