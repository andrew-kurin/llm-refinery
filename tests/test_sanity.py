from llama_tune.utils.sanity import has_reasoning_tags


def test_has_reasoning_tags_detects_think_tags():
    assert has_reasoning_tags("<think>\n\n</think>\n\nHello")
    assert has_reasoning_tags("<thinking>hidden</thinking>visible")
    assert not has_reasoning_tags("Hello without hidden reasoning markers")
