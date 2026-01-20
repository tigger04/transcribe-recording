#!/usr/bin/env python3
# ABOUTME: Unit tests for the diarize.py script.
# ABOUTME: Tests device detection and argument parsing.

"""
Unit tests for speaker diarization script.

Usage: python3 test_diarize.py
"""

import sys
import unittest
from unittest.mock import patch, MagicMock


class TestGetDevice(unittest.TestCase):
    """Tests for the get_device function."""

    def setUp(self):
        """Import the diarize module fresh for each test."""
        # Clear any cached imports
        if 'diarize' in sys.modules:
            del sys.modules['diarize']

    @patch('torch.backends.mps.is_available')
    @patch('torch.cuda.is_available')
    def test_get_device_auto_prefers_mps(self, mock_cuda, mock_mps):
        """Auto mode should prefer MPS when available."""
        mock_mps.return_value = True
        mock_cuda.return_value = False

        import torch
        from diarize import get_device

        device = get_device("auto")
        self.assertEqual(device.type, "mps")

    @patch('torch.backends.mps.is_available')
    @patch('torch.cuda.is_available')
    def test_get_device_auto_falls_back_to_cuda(self, mock_cuda, mock_mps):
        """Auto mode should use CUDA when MPS unavailable."""
        mock_mps.return_value = False
        mock_cuda.return_value = True

        import torch
        from diarize import get_device

        device = get_device("auto")
        self.assertEqual(device.type, "cuda")

    @patch('torch.backends.mps.is_available')
    @patch('torch.cuda.is_available')
    def test_get_device_auto_falls_back_to_cpu(self, mock_cuda, mock_mps):
        """Auto mode should fall back to CPU when no GPU available."""
        mock_mps.return_value = False
        mock_cuda.return_value = False

        import torch
        from diarize import get_device

        device = get_device("auto")
        self.assertEqual(device.type, "cpu")

    def test_get_device_cpu_explicit(self):
        """Explicit CPU mode should always return CPU."""
        from diarize import get_device

        device = get_device("cpu")
        self.assertEqual(device.type, "cpu")

    @patch('torch.backends.mps.is_available')
    def test_get_device_mps_when_available(self, mock_mps):
        """Explicit MPS mode should return MPS when available."""
        mock_mps.return_value = True

        from diarize import get_device

        device = get_device("mps")
        self.assertEqual(device.type, "mps")

    @patch('torch.backends.mps.is_available')
    def test_get_device_mps_raises_when_unavailable(self, mock_mps):
        """Explicit MPS mode should raise error when unavailable."""
        mock_mps.return_value = False

        from diarize import get_device

        with self.assertRaises(RuntimeError) as context:
            get_device("mps")
        self.assertIn("MPS requested but not available", str(context.exception))

    @patch('torch.cuda.is_available')
    def test_get_device_cuda_when_available(self, mock_cuda):
        """Explicit CUDA mode should return CUDA when available."""
        mock_cuda.return_value = True

        from diarize import get_device

        device = get_device("cuda")
        self.assertEqual(device.type, "cuda")

    @patch('torch.cuda.is_available')
    def test_get_device_cuda_raises_when_unavailable(self, mock_cuda):
        """Explicit CUDA mode should raise error when unavailable."""
        mock_cuda.return_value = False

        from diarize import get_device

        with self.assertRaises(RuntimeError) as context:
            get_device("cuda")
        self.assertIn("CUDA requested but not available", str(context.exception))


if __name__ == '__main__':
    # Add scripts directory to path so we can import diarize
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    unittest.main()
