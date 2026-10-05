import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.downloader import stream_games
from app import config

@pytest.mark.asyncio
async def test_downloader_404_raises():
    """Test that HTTP 404 raises ValueError with user not found message."""
    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch("httpx.AsyncClient.stream") as mock_stream:
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_response
        mock_stream.return_value = mock_cm

        with pytest.raises(ValueError, match="User not found"):
            async for _ in stream_games("nonexistent_user_xyz"):
                pass

@pytest.mark.asyncio
async def test_downloader_403_raises():
    """Test that HTTP 403 raises ValueError with access forbidden message."""
    mock_response = MagicMock()
    mock_response.status_code = 403

    with patch("httpx.AsyncClient.stream") as mock_stream:
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_response
        mock_stream.return_value = mock_cm

        with pytest.raises(ValueError, match="Access forbidden"):
            async for _ in stream_games("forbidden_user"):
                pass

@pytest.mark.asyncio
async def test_downloader_429_retries_and_fails():
    """Test that HTTP 429 retries 5 times before raising HTTPStatusError."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Too Many Requests", request=MagicMock(), response=mock_response
    )

    with patch("httpx.AsyncClient.stream") as mock_stream, patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_response
        mock_stream.return_value = mock_cm

        with pytest.raises(httpx.HTTPStatusError):
            async for _ in stream_games("rate_limited_user"):
                pass

        assert mock_stream.call_count == 5
        assert mock_sleep.call_count == 4
