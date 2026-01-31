import sys
sys.path.insert(0, 'src')

from sigmak import peer_selection as ps


def test_compute_similarity_tfidf():
    # Use TF-IDF backend for deterministic, lightweight test
    a = "The quick brown fox jumps over the lazy dog."
    b = "A quick brown fox jumped over the lazy dog!"
    s = ps.compute_semantic_similarity(a, b, backend='tfidf')
    assert s > 0.4


def test_validate_peer_group_order_of_magnitude(monkeypatch):
    # Mock yfinance.Ticker.info responses
    class Dummy:
        def __init__(self, info):
            self.info = info

    def fake_Ticker(ticker):
        mapping = {
            'TGT': {'marketCap': 1_000_000_000, 'industry': 'Software'},
            'A': {'marketCap': 1_200_000_000, 'industry': 'Software'},
            'B': {'marketCap': 50_000_000, 'industry': 'Software'},
            'C': {'marketCap': 5_000_000_000, 'industry': 'Hardware'},
            'D': {'marketCap': None, 'industry': None},
        }
        return Dummy(mapping.get(ticker, {}))

    # Create a fake yfinance-like module with Ticker attribute
    import types

    fake_yf = types.SimpleNamespace(Ticker=fake_Ticker)
    monkeypatch.setattr('sigmak.peer_selection._try_import_yfinance', lambda: fake_yf)

    peers = ['A', 'B', 'C', 'D']
    validated = ps.validate_peer_group('TGT', peers)
    # Only A and C fall within 0.1x-10x range (B is too small)
    assert 'A' in validated
    assert 'C' in validated
    assert 'B' not in validated
