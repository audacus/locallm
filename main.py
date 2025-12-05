from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Orchestrator-8B-4bit")


def do_stuff(prompt: str) -> str:
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
        )

    return generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        verbose=True,
    )


if __name__ == "__main__":
    do_stuff("Hello, world!")
