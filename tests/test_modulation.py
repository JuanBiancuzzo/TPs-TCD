import numpy as np
import pytest
from src.modulation import Modulation, Scheme
from src.report import Reporter

class TestModulation:
    @pytest.fixture
    def reporter(self):
        return Reporter()

    def test_bpsk_complete_symbols(self, reporter):
        """Test BPSK with complete symbols (multiple of k bits)"""
        mod = Modulation(scheme=Scheme.PSK, M=2)
        bits = np.array([1, 0, 1, 1])  # 4 bits = 4 símbolos BPSK completos
        
        # Encode
        symbols = mod.encode(bits, reporter)
        assert symbols.shape == (4, 1)  # 4 símbolos de dimensión 1
        # En LSB-first: 1->-1, 0->1
        assert np.allclose(symbols, np.array([[-1], [1], [-1], [-1]]))
        
        # Decode
        decoded = mod.decode(symbols)
        np.testing.assert_array_equal(decoded, bits)

    def test_qpsk_complete_symbols(self, reporter):
        """Test QPSK with complete symbols (multiple of k=2 bits)"""
        mod = Modulation(scheme=Scheme.PSK, M=4)
        # Input bits in LSB-first
        bits = np.array([0,0, 1,0, 1,1, 0,1])  # 00,10,11,01 in LSB-first
        
        symbols = mod.encode(bits, reporter)
        assert symbols.shape == (4, 2)  # 4 símbolos de dimensión 2
        
        # Verify each encoded symbol matches the modulator's constellation
        # Compute expected symbol index per LSB-first + Gray mapping
        num_symbols = symbols.shape[0]
        for pos in range(num_symbols):
            # reconstruct integer from LSB-first bits
            b0 = bits[pos * 2]
            b1 = bits[pos * 2 + 1]
            num = int(b0) | (int(b1) << 1)
            gray_idx = mod._bin_to_gray(num)
            expected_sym = mod.symbols[gray_idx]
            assert np.allclose(symbols[pos], expected_sym)
        
        decoded = mod.decode(symbols)
        np.testing.assert_array_equal(decoded, bits)

    def test_incomplete_symbol(self, reporter):
        """Test PSK with incomplete last symbol"""
        mod = Modulation(scheme=Scheme.PSK, M=4)
        # 3 bits (no múltiplo de k=2)
        bits = np.array([1, 1, 0])
        
        symbols = mod.encode(bits, reporter)
        assert symbols.shape == (2, 2)  # 2 símbolos QPSK
        
        decoded = mod.decode(symbols)
        np.testing.assert_array_equal(decoded, bits)


    def test_8psk(self, reporter):
        """Test 8-PSK modulation"""
        mod = Modulation(scheme=Scheme.PSK, M=8)
        # Test con 6 bits (2 símbolos completos)
        bits = np.array([1,0,1, 0,0,0])
        
        symbols = mod.encode(bits, reporter)
        assert symbols.shape == (2, 2)
        
        decoded = mod.decode(symbols)
        np.testing.assert_array_equal(decoded, bits)