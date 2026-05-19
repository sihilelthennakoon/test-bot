from ragbot.safety.pii import PIIMasker


def test_pii_masker_redacts_email_and_phone() -> None:
    masker = PIIMasker()
    result = masker.mask("Email alice@example.com or call 555-123-4567.")

    assert result.was_masked is True
    assert "alice@example.com" not in result.text
    assert "555-123-4567" not in result.text
    assert "[EMAIL_1]" in result.text
    assert "[PHONE_1]" in result.text
