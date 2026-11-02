from types import SimpleNamespace

from src.llm.qa_chain import QAChain, is_basic_conversation


def test_basic_conversation_classifier_is_narrow():
    assert is_basic_conversation("Hi")
    assert is_basic_conversation("How are you?")
    assert is_basic_conversation("Thank you!")
    assert is_basic_conversation("Who are you?")
    assert not is_basic_conversation("Hi, explain Grover's algorithm")
    assert not is_basic_conversation("Explain quantum entanglement")


def test_answer_bypasses_retriever_for_greeting():
    class Retriever:
        def retrieve(self, *args, **kwargs):
            raise AssertionError("RAG retrieval must not run for a greeting")

    chain = QAChain(Retriever())
    chain._generate = lambda system, prompt: "Hello! How can I help you today?"
    response = chain.answer("Hello")
    assert response.answer.startswith("Hello!")
    assert response.has_context is False
    assert response.retrieval_results == []


def test_stream_answer_bypasses_retriever_for_greeting():
    class Retriever:
        def retrieve(self, *args, **kwargs):
            raise AssertionError("RAG retrieval must not run for a greeting")

    class FakeStream:
        def __iter__(self):
            return iter([
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello"))]),
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="!"))]),
            ])
        def close(self):
            pass

    chain = QAChain(Retriever())
    chain._llm_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: FakeStream())))
    assert "".join(chain.stream_answer("Hi")) == "Hello!"


def test_attachment_answer_does_not_mix_unrelated_rag_results():
    class Retriever:
        def retrieve(self, *args, **kwargs):
            raise AssertionError("Attachment-focused requests must not retrieve unrelated KB chunks")
        def format_context(self, results):
            assert results == []
            return ""

    chain = QAChain(Retriever())
    captured = {}
    def fake_generate(system, prompt):
        captured["system"] = system
        captured["prompt"] = prompt
        return "The image describes three generations of quantum computing."
    chain._generate = fake_generate
    response = chain.answer("Summarize the text in this image", attachments=[
        ("timeline.png", "THREE FUTURE GENERATIONS OF QUANTUM COMPUTING\n1st generation 2018-2028\n2nd generation 2028-2039\n3rd generation 2031-2042")
    ])
    assert "three generations" in response.answer
    assert "timeline.png" in captured["prompt"]
    assert "unrelated" in captured["system"]
    assert response.sources == ["timeline.png"]


def test_document_preserves_ocr_line_breaks():
    from src.ingestion.load_documents import Document
    doc = Document("Heading   One\n\n  First   point\nSecond point", "image.png")
    assert doc.content == "Heading One\n\nFirst point\nSecond point"


def test_stream_answer_requests_continuation_after_length_finish():
    class FakeStream:
        def __init__(self, text, reason):
            self.text, self.reason = text, reason
        def __iter__(self):
            return iter([SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content=self.text), finish_reason=self.reason
            )])])
        def close(self):
            pass

    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return FakeStream("First half", "length") if len(calls) == 1 else FakeStream(" and ending.", "stop")
    class Retriever:
        def retrieve(self, *args, **kwargs):
            return [SimpleNamespace(source="kb.txt", text="Supported context", score=0.9)]
        def format_context(self, results):
            return "Supported context"
    chain = QAChain(Retriever())
    chain._llm_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    answer = "".join(chain.stream_answer("Explain a supported topic"))
    assert answer == "First half and ending."
    assert len(calls) == 2
    assert calls[1]["max_tokens"] >= 8192
