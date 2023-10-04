import declarai

gpt_35 = declarai.openai(model="gpt-3.5-turbo")

@gpt_35.task
def generate_poem(a: int, b: int) -> int:
    """
    Given two number, find sum of two number
    """


res = generate_poem(
    a = 1, b = 2
)
print(res)
