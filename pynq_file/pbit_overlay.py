"""
    0x00 CTRL slv_reg0                       [0]=en  [1]=soft_rst           (W)
    0x04 BETA slv_reg1                       Q8.8 signed                    (W)
    0x08 SEED slv_reg2                       32-bit PRNG seed (nonzero)     (W)
    0x0C STATUS                              [0]=sweep_done                     (R)
    0x10 M                                   current p-bit states, bit i        (R)
    0x14 CNT                                 sweeps completed                   (R)
    0x18 CLAMP_MASK slv_reg6                 1=>hold p-bit i                (W)
    0x1C CLAMP_VAL slv_reg7                  held value when clamped        (W)
    0x40 J[0..N*N-1] slv_reg16..24           Q8.8 signed                    (W)
    0x40+4*N*N  H[0..N-1] slv_reg25..27      Q8.8 signed                    (W)
"""
import time
import numpy as np
from pynq import Overlay, MMIO

Q = 256  # Q8.8 scale


def to_q88(x):
    return int(round(float(x) * Q)) & 0xFFFF


def from_q88(u):
    u &= 0xFFFF
    return (u - 0x10000 if u & 0x8000 else u) / Q


class PbitEngine:
    CTRL, BETA, SEED, STATUS, M, CNT = 0x00, 0x04, 0x08, 0x0C, 0x10, 0x14
    CLAMP_MASK, CLAMP_VAL = 0x18, 0x1C
    JBASE = 0x40

    def __init__(self, bitfile="mainDesign.bit", ip="pbitIP", N=3):
        self.ol = Overlay(bitfile)
        base = self.ol.ip_dict[ip]["phys_addr"]
        addr_range = self.ol.ip_dict[ip]["addr_range"]
        self.mmio = MMIO(base, addr_range)
        self.N = N
        self.HBASE = self.JBASE + 4 * N * N #byte-based memory

    # ---- configuration ----------------------------------------------------
    def load_gate(self, J, h, beta, seed=0x1234_5678):
        J = np.asarray(J); h = np.asarray(h); N = self.N
        assert J.shape == (N, N) and h.shape == (N,)
        for i in range(N):
            for j in range(N):
                self.mmio.write(self.JBASE + 4 * (i * N + j), to_q88(J[i, j]))
        for i in range(N):
            self.mmio.write(self.HBASE + 4 * i, to_q88(h[i]))
        self.mmio.write(self.BETA, to_q88(beta))
        self.mmio.write(self.SEED, seed & 0xFFFFFFFF)

    def set_beta(self, beta):
        self.mmio.write(self.BETA, to_q88(beta))

    def reset(self):
        self.mmio.write(self.CTRL, 0b10); self.mmio.write(self.CTRL, 0b00)

    def enable(self, on=True):
        self.mmio.write(self.CTRL, 0b01 if on else 0b00)

    def read_state(self):
        return self.mmio.read(self.M) & ((1 << self.N) - 1)

    def clamp(self, mask=0, val=0):
        """Hold p-bits in `mask` at bits of `val` (invertible / conditional mode)."""
        self.mmio.write(self.CLAMP_MASK, mask & ((1 << self.N) - 1))
        self.mmio.write(self.CLAMP_VAL, val & ((1 << self.N) - 1))

    def invertible_and(self, y, betas=None, n_samples=20000):
        """Run the AND gate backward: clamp output Y=y, anneal, histogram inputs (A,B)."""
        if betas is None:
            betas = np.linspace(0.1, 3.0, 150)
        self.clamp(mask=0b100, val=(0b100 if y else 0b000))   # Y is bit 2
        p = self.histogram_annealed(betas, n_samples)
        ab = np.zeros(4)
        for s, pr in enumerate(p):
            ab[s & 0b011] += pr          
        self.clamp(0, 0)
        return {(a, b): ab[b * 2 + a] for a in (0, 1) for b in (0, 1)}

    # ---- SnapShot ------------------------------------------------------

    def histogram_annealed(self, betas, n_samples=20000, sweeps_per_step=200, settle_us=5):
        """Sample the state distribution while ramping beta hot->cold; clamp untouched."""
        counts = np.zeros(1 << self.N, dtype=np.int64)
        self.reset(); self.enable(True)
        for b in betas:
            self.set_beta(b)
            time.sleep(sweeps_per_step * settle_us * 1e-6)
        for _ in range(n_samples):
            time.sleep(settle_us * 1e-6)
            counts[self.read_state()] += 1
        self.enable(False)
        return counts / counts.sum()

    def anneal(self, betas, sweeps_per_step=200, settle_us=5):
        """Ramp beta hot->cold; return the final settled state (A,B,Y,...)."""
        self.reset(); self.enable(True)
        for b in betas:
            self.set_beta(b)
            time.sleep(sweeps_per_step * settle_us * 1e-6)
        st = self.read_state(); self.enable(False)
        return [(st >> i) & 1 for i in range(self.N)]



if __name__ == "__main__":
    # AND gate [A,B,Y]
    J = np.array([[0, -1, 2], [-1, 0, 2], [2, 2, 0]], float)
    h = np.array([1, 1, -2], float)

    eng = PbitEngine(N=3)
    eng.load_gate(J, h, beta=2.0)

    p = eng.histogram_annealed(np.linspace(0.1, 3.0, 150), n_samples=20000)
    for s, pr in enumerate(p): #captures index,value
        print(f"A B Y = {s&1} {(s>>1)&1} {(s>>2)&1}   prob={pr:.3f}")

    # invertible AND: clamp the output
    print("clamp Y=1 ->", eng.invertible_and(1))  
    print("clamp Y=0 ->", eng.invertible_and(0))   
    eng.clamp(0, 0)  # release