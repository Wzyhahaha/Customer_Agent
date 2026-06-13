from pathlib import Path

from model import factory


class FakeSentenceTransformer:
    created_with: tuple[str, bool] | None = None

    def __init__(self, model_path: str, trust_remote_code: bool = False):
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        FakeSentenceTransformer.created_with = (model_path, trust_remote_code)

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, str):
            return FakeVector([1.0, 0.0])
        return FakeVector([[1.0, 0.0] for _ in texts])


class FakeVector(list):
    def tolist(self):
        return list(self)


def test_embeddings_factory_uses_configured_local_bge_m3(monkeypatch):
    model_path = Path(__file__).resolve().parent

    monkeypatch.setitem(factory.rag_conf, "embedding_model_provider", "local_bge_m3")
    monkeypatch.setitem(factory.rag_conf, "embedding_model_path", str(model_path))
    monkeypatch.setattr(factory, "SentenceTransformer", FakeSentenceTransformer)

    embedding = factory.EmbeddingsFactory().generator()

    assert isinstance(embedding, factory.LocalBgeM3Embeddings)
    assert FakeSentenceTransformer.created_with == (str(model_path.resolve()), True)
    assert embedding.embed_query("保修多久") == [1.0, 0.0]
