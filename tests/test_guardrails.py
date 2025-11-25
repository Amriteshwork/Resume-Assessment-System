from app.tools import mask_pii


def test_mask_pii_email_and_phone():
    text = "Contact me at john.doe@example.com or +1 555 123 4567."
    out = mask_pii(text)
    assert "example.com" not in out
    assert "[REDACTED_EMAIL]" in out
    assert "[REDACTED_PHONE]" in out


def test_mask_pii_empty():
    assert mask_pii("") == ""
    assert mask_pii(None) == ""  # type: ignore[arg-type]
