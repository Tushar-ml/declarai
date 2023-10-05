import declarai

gpt_35 = declarai.huggingface("gpt2")

@gpt_35.task
def generate_poem(a: int, b: int) -> int:
    """
    Given two number, find sum of two number
    """


res = generate_poem(
    a = 1, b = 2
)
print(res)
