from bert_api.config import Settings


def test_defaults_and_env(monkeypatch):
    monkeypatch.delenv("TF_SERVING_URL", raising=False)
    assert Settings.from_env().tf_serving_url == "http://localhost:8501"
    monkeypatch.setenv("TF_SERVING_URL", "http://tfserving:8501/")
    monkeypatch.setenv("TF_SERVING_READ_TIMEOUT_S", "3.5")
    monkeypatch.setenv("MAX_SEQ_LENGTH", "128")
    s = Settings.from_env()
    assert s.tf_serving_url == "http://tfserving:8501"
    assert s.read_timeout_s == 3.5
    assert s.max_seq_length == 128
