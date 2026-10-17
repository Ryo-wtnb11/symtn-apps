//! The fusion-path ED oracle against a TensorKit value and the
//! Temperley–Lieb algebra its projectors must satisfy.

use golden_chain::ed::{Boundary, Chain, Fib, phi};

fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            for j in 0..n {
                c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
    c
}

fn assert_close(a: &[f64], b: &[f64], what: &str) {
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= 1e-13, "{what}[{k}]: {x} != {y}");
    }
}

/// T1. Open L = 3, total τ: the two paths (1,τ,1,τ), (1,τ,τ,τ) give
/// H = -(P_12 + P_23), the matrix symtn pins against TensorKit 0.17
/// (`symtn/tests/table_planar_workflow.rs`,
/// `fibonacci_public_planar_workflow_matches_tensorkit`).
#[test]
fn three_anyons_match_the_tensorkit_golden_chain_matrix() {
    let chain = Chain::new(3, Boundary::Open(Fib::Tau));
    assert_eq!(
        chain.paths(),
        [
            vec![Fib::One, Fib::Tau, Fib::One, Fib::Tau],
            vec![Fib::One, Fib::Tau, Fib::Tau, Fib::Tau]
        ]
    );
    let tensorkit = [
        -1.381966011250105,
        -0.48586827175664565,
        -0.48586827175664565,
        -0.6180339887498948,
    ];
    assert_close(&chain.hamiltonian(1.0), &tensorkit, "H");
}

/// T2. Path counts are Fibonacci (open) and Lucas (periodic) numbers.
#[test]
fn dimensions_are_fibonacci_and_lucas_numbers() {
    let fib = |n: usize| (0..n).fold((0usize, 1usize), |(a, b), _| (b, a + b)).0;
    for len in 2..=12 {
        assert_eq!(Chain::new(len, Boundary::Open(Fib::Tau)).dim(), fib(len));
        assert_eq!(
            Chain::new(len, Boundary::Open(Fib::One)).dim(),
            fib(len - 1)
        );
        assert_eq!(
            Chain::new(len, Boundary::Periodic).dim(),
            fib(len - 1) + fib(len + 1)
        );
    }
}

/// T2. `e_i = φ P_i` generate Temperley–Lieb with loop value φ:
/// `P² = P`, `P_i P_{i±1} P_i = φ⁻² P_i`, distant projectors commute, and
/// each is symmetric. This pins every F entry up to the basis gauge.
#[test]
fn projectors_satisfy_temperley_lieb() {
    for boundary in [
        Boundary::Open(Fib::One),
        Boundary::Open(Fib::Tau),
        Boundary::Periodic,
    ] {
        let chain = Chain::new(7, boundary);
        let n = chain.dim();
        let bonds = chain.bonds();
        let p: Vec<_> = (1..=bonds).map(|i| chain.projector(i)).collect();
        let neighbours = |i: usize, j: usize| {
            let d = i.abs_diff(j);
            d == 1 || (boundary == Boundary::Periodic && d == bonds - 1)
        };
        for i in 0..bonds {
            let transpose: Vec<f64> = (0..n * n).map(|k| p[i][(k % n) * n + k / n]).collect();
            assert_close(&transpose, &p[i], "P symmetric");
            assert_close(&matmul(&p[i], &p[i], n), &p[i], "P^2");
            for j in 0..bonds {
                if neighbours(i, j) {
                    let pij = matmul(&matmul(&p[i], &p[j], n), &p[i], n);
                    let expected: Vec<f64> = p[i].iter().map(|x| x / (phi() * phi())).collect();
                    assert_close(&pij, &expected, "P_i P_j P_i");
                } else if i != j {
                    assert_close(
                        &matmul(&p[i], &p[j], n),
                        &matmul(&p[j], &p[i], n),
                        "commute",
                    );
                }
            }
        }
    }
}
