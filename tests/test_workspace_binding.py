"""Tests for workspace binding service."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanobot.workspace_binding import WorkspaceBinding, WorkspaceBindingService


@pytest.fixture
def binding_service(tmp_path: Path):
    """Provide a workspace binding service with temp storage."""
    store_path = tmp_path / "bindings.json"
    return WorkspaceBindingService(store_path)


class TestWorkspaceBindingService:
    """Tests for WorkspaceBindingService."""

    def test_bind_success(self, binding_service):
        """Test successful binding."""
        success, msg = binding_service.bind(
            chat_id="-123456789",
            workspace_name="test-coding",
            channel="telegram",
        )
        assert success is True
        assert "test-coding" in msg
        assert binding_service.is_bound("-123456789")
        assert binding_service.get_workspace("-123456789") == "test-coding"

    def test_bind_requires_coding_suffix(self, binding_service):
        """Test binding requires -coding suffix."""
        success, msg = binding_service.bind(
            chat_id="-123456789",
            workspace_name="test",  # Missing -coding suffix
            channel="telegram",
        )
        assert success is False
        assert "-coding" in msg
        assert not binding_service.is_bound("-123456789")

    def test_bind_requires_group_chat(self, binding_service):
        """Test binding requires group chat (negative chat_id)."""
        success, msg = binding_service.bind(
            chat_id="123456789",  # Positive = private chat
            workspace_name="test-coding",
            channel="telegram",
        )
        assert success is False
        assert "group chats" in msg
        assert not binding_service.is_bound("123456789")

    def test_bind_unsupported_channel(self, binding_service):
        """Test binding fails for unsupported channels."""
        success, msg = binding_service.bind(
            chat_id="-123456789",
            workspace_name="test-coding",
            channel="whatsapp",  # Not supported
        )
        assert success is False
        assert "whatsapp" in msg

    def test_unbind_success(self, binding_service):
        """Test successful unbinding."""
        # First bind
        binding_service.bind("-123456789", "test-coding", "telegram")
        assert binding_service.is_bound("-123456789")

        # Then unbind
        success, msg = binding_service.unbind("-123456789")
        assert success is True
        assert "test-coding" in msg
        assert not binding_service.is_bound("-123456789")

    def test_unbind_not_bound(self, binding_service):
        """Test unbinding when not bound."""
        success, msg = binding_service.unbind("-123456789")
        assert success is False
        assert "not bound" in msg

    def test_get_binding(self, binding_service):
        """Test getting binding details."""
        binding_service.bind("-123456789", "test-coding", "telegram")

        binding = binding_service.get_binding("-123456789")
        assert binding is not None
        assert binding.chat_id == "-123456789"
        assert binding.workspace_name == "test-coding"
        assert binding.channel == "telegram"

    def test_get_binding_not_found(self, binding_service):
        """Test getting binding when not bound."""
        binding = binding_service.get_binding("-123456789")
        assert binding is None

    def test_list_bindings(self, binding_service):
        """Test listing all bindings."""
        binding_service.bind("-123456789", "test1-coding", "telegram")
        binding_service.bind("-987654321", "test2-coding", "telegram")

        bindings = binding_service.list_bindings()
        assert len(bindings) == 2
        assert {b.workspace_name for b in bindings} == {"test1-coding", "test2-coding"}

    def test_get_bound_chats(self, binding_service):
        """Test getting chats bound to a specific workspace."""
        binding_service.bind("-111111111", "shared-coding", "telegram")
        binding_service.bind("-222222222", "shared-coding", "telegram")
        binding_service.bind("-333333333", "other-coding", "telegram")

        chats = binding_service.get_bound_chats("shared-coding")
        assert set(chats) == {"-111111111", "-222222222"}

    def test_persistence(self, tmp_path: Path):
        """Test bindings are persisted to disk."""
        store_path = tmp_path / "bindings.json"

        # Create and bind
        service1 = WorkspaceBindingService(store_path)
        service1.bind("-123456789", "persist-coding", "telegram")

        # Create new instance with same path
        service2 = WorkspaceBindingService(store_path)
        assert service2.is_bound("-123456789")
        assert service2.get_workspace("-123456789") == "persist-coding"

    def test_invalid_chat_id(self, binding_service):
        """Test binding with invalid chat_id."""
        success, msg = binding_service.bind(
            chat_id="not-a-number",
            workspace_name="test-coding",
            channel="telegram",
        )
        assert success is False
        assert "Invalid" in msg


class TestWorkspaceBindingDataclass:
    """Tests for WorkspaceBinding dataclass."""

    def test_binding_creation(self):
        """Test creating a binding."""
        binding = WorkspaceBinding(
            chat_id="-123",
            workspace_name="test-coding",
            channel="telegram",
            metadata={"key": "value"},
        )
        assert binding.chat_id == "-123"
        assert binding.workspace_name == "test-coding"
        assert binding.channel == "telegram"
        assert binding.metadata == {"key": "value"}

    def test_binding_defaults(self):
        """Test binding defaults."""
        binding = WorkspaceBinding(
            chat_id="-123",
            workspace_name="test-coding",
        )
        assert binding.channel == "telegram"
        assert binding.metadata == {}
