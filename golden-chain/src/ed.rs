//! Exact diagonalization oracle in the fusion-path basis.
//!
//! Deliberately independent of symtn: the F-symbol of the bond projector is
//! the closed form in φ, not fusion-symbols' table, so agreement with the
//! tensor code is evidence rather than a tautology.

/// A Fibonacci label: the vacuum `1` or `τ`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Fib {
    One,
    Tau,
}

/// Boundary condition of the fusion path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    /// `x_0 = 1`, `x_L = total`.
    Open(Fib),
    /// `x_L = x_0`.
    Periodic,
}

/// The golden ratio φ.
pub fn phi() -> f64 {
    (1.0 + 5.0_f64.sqrt()) / 2.0
}

/// Fusion-path basis of `len` τ anyons.
#[derive(Debug)]
pub struct Chain {
    len: usize,
    boundary: Boundary,
    /// Open: `x_0..=x_L`. Periodic: `x_0..x_L` (cyclic). Sorted.
    paths: Vec<Vec<Fib>>,
}

impl Chain {
    pub fn new(len: usize, boundary: Boundary) -> Self {
        assert!(len >= 2, "a chain needs at least two anyons");
        // x_{i+1} ∈ x_i ⊗ τ: after a 1 only τ may follow.
        let next = |x: Fib| -> &'static [Fib] {
            match x {
                Fib::One => &[Fib::Tau],
                Fib::Tau => &[Fib::One, Fib::Tau],
            }
        };
        let mut paths: Vec<Vec<Fib>> = match boundary {
            Boundary::Open(_) => vec![vec![Fib::One]],
            Boundary::Periodic => vec![vec![Fib::One], vec![Fib::Tau]],
        };
        let steps = match boundary {
            Boundary::Open(_) => len,
            Boundary::Periodic => len - 1,
        };
        for _ in 0..steps {
            paths = paths
                .into_iter()
                .flat_map(|p| {
                    next(*p.last().unwrap()).iter().map(move |&x| {
                        let mut q = p.clone();
                        q.push(x);
                        q
                    })
                })
                .collect();
        }
        paths.retain(|p| match boundary {
            Boundary::Open(total) => *p.last().unwrap() == total,
            Boundary::Periodic => next(*p.last().unwrap()).contains(&p[0]),
        });
        paths.sort();
        Self {
            len,
            boundary,
            paths,
        }
    }

    pub fn dim(&self) -> usize {
        self.paths.len()
    }

    pub fn paths(&self) -> &[Vec<Fib>] {
        &self.paths
    }

    /// Index of a path as stored by [`Self::paths`].
    pub fn index(&self, path: &[Fib]) -> Option<usize> {
        self.paths.binary_search_by(|p| p.as_slice().cmp(path)).ok()
    }

    /// Number of nearest-neighbour bonds: `L-1` open, `L` periodic.
    pub fn bonds(&self) -> usize {
        match self.boundary {
            Boundary::Open(_) => self.len - 1,
            Boundary::Periodic => self.len,
        }
    }

    /// Dense row-major `P^(1)` on anyons `bond` and `bond + 1` (1-based).
    ///
    /// The pair acts on `(x_{i-1}, x_i, x_{i+1})` and changes only `x_i`. With
    /// `x_{i±1} = (τ, τ)` it is `F e₁e₁ᵀ F†` for
    /// `F^{τττ}_τ = [[φ⁻¹, φ^{-1/2}], [φ^{-1/2}, -φ⁻¹]]` in the basis
    /// `x_i ∈ (1, τ)`; `(1, 1)` forces the trivial channel, and a single
    /// neighbouring `1` forces `τ`.
    pub fn projector(&self, bond: usize) -> Vec<f64> {
        assert!((1..=self.bonds()).contains(&bond));
        let n = self.dim();
        let phi = phi();
        let mut m = vec![0.0; n * n];
        for (col, path) in self.paths.iter().enumerate() {
            let at = |k: usize| match self.boundary {
                Boundary::Open(_) => k,
                Boundary::Periodic => k % self.len,
            };
            let (a, x, b) = (path[at(bond - 1)], path[at(bond)], path[at(bond + 1)]);
            match (a, b) {
                (Fib::One, Fib::One) => m[col * n + col] = 1.0,
                (Fib::Tau, Fib::Tau) => {
                    let p = |x: Fib, y: Fib| match (x, y) {
                        (Fib::One, Fib::One) => phi.powi(-2),
                        (Fib::Tau, Fib::Tau) => phi.recip(),
                        _ => phi.powf(-1.5),
                    };
                    for y in [Fib::One, Fib::Tau] {
                        let mut target = path.clone();
                        target[at(bond)] = y;
                        let row = self.index(&target).expect("τ τ neighbours admit both");
                        m[row * n + col] = p(y, x);
                    }
                }
                _ => {}
            }
        }
        m
    }

    /// Dense row-major `H = -J Σ_bonds P^(1)`.
    pub fn hamiltonian(&self, j: f64) -> Vec<f64> {
        let n = self.dim();
        let mut h = vec![0.0; n * n];
        for bond in 1..=self.bonds() {
            for (h, p) in h.iter_mut().zip(self.projector(bond)) {
                *h -= j * p;
            }
        }
        h
    }
}
