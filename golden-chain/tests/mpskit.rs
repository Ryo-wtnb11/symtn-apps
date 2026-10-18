//! T8. DMRG against MPSKit 0.13.13's `DMRG2` on the same chain: ground
//! energy, entanglement spectra and entropies per bond. Only the generator's
//! output is in git; its header records the versions and the exact MPSKit
//! calls. CI never runs Julia.

mod common;

use golden_chain::dmrg::find_groundstate;
use golden_chain::ed::Fib;

const FIXTURE: &str = include_str!("../../reference/mpskit/fixtures/golden_chain.txt");

struct Case {
    len: usize,
    j: f64,
    total: Fib,
    rank: usize,
    energy: f64,
    entropy: Vec<f64>,
    /// `spectra[bond - 1]`: `(sector, values)`.
    spectra: Vec<Vec<(Fib, Vec<f64>)>>,
}

fn fib(label: &str) -> Fib {
    match label {
        "1" => Fib::One,
        "tau" => Fib::Tau,
        other => panic!("unknown sector {other}"),
    }
}

fn cases() -> Vec<Case> {
    let mut cases: Vec<Case> = Vec::new();
    for line in FIXTURE.lines().filter(|l| !l.starts_with('#')) {
        let mut words = line.split_whitespace();
        let key = words.next().expect("nonempty line");
        let mut num = || words.next().expect("field").parse::<f64>().expect("number");
        if key == "case" {
            let (len, j) = (num() as usize, num());
            let rest: Vec<&str> = line.split_whitespace().skip(3).collect();
            cases.push(Case {
                len,
                j,
                total: fib(rest[0]),
                rank: rest[1].parse().expect("D"),
                energy: f64::NAN,
                entropy: Vec::new(),
                spectra: vec![Vec::new(); len - 1],
            });
            continue;
        }
        let case = cases.last_mut().expect("case header first");
        match key {
            "energy" => case.energy = num(),
            "epsilon" => {}
            "entropy" => {
                let _bond = num();
                case.entropy.push(num());
            }
            "spectrum" => {
                let rest: Vec<&str> = line.split_whitespace().skip(1).collect();
                let bond: usize = rest[0].parse().expect("bond");
                let values = rest[2..]
                    .iter()
                    .map(|v| v.parse().expect("value"))
                    .collect();
                case.spectra[bond - 1].push((fib(rest[1]), values));
            }
            other => panic!("unknown fixture key {other}"),
        }
    }
    cases
}

fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| b.total_cmp(a));
    v
}

/// Our spectrum minus theirs, per sector and position after sorting, a
/// missing value counting as 0 (a sector or state one side dropped).
fn spectrum_distance(ours: &[(Fib, Vec<f64>)], theirs: &[(Fib, Vec<f64>)]) -> f64 {
    let get = |s: &[(Fib, Vec<f64>)], x: Fib| {
        sorted(
            s.iter()
                .find(|(y, _)| *y == x)
                .map(|(_, v)| v.clone())
                .unwrap_or_default(),
        )
    };
    [Fib::One, Fib::Tau]
        .into_iter()
        .flat_map(|x| {
            let (a, b) = (get(ours, x), get(theirs, x));
            (0..a.len().max(b.len()))
                .map(move |k| (a.get(k).unwrap_or(&0.0) - b.get(k).unwrap_or(&0.0)).abs())
        })
        .fold(0.0, f64::max)
}

/// At D = 128 both runs discard at most ~1e-13 of weight in their last sweep
/// (asserted below for ours; the centre bond is kept whole up to L = 20), so both are the exact
/// ground state up to the 1e-12 eigensolver residual, and E, every Schmidt
/// value and every entropy must agree far inside 1e-10 (observed: ≤ 3e-13).
/// A different gauge, sector weighting or normalization convention would
/// show up at O(1e-2) or more.
#[test]
fn dmrg_matches_mpskit_fixture() {
    const TOL: f64 = 1e-10;
    let cases = cases();
    assert_eq!(cases.len(), 16);
    for case in &cases {
        let what = format!("L {} J {} total {:?}", case.len, case.j, case.total);
        let gs = find_groundstate(
            case.len,
            case.j,
            case.total,
            &common::mpskit_dmrg2(vec![case.rank]),
        )
        .unwrap();
        assert!(gs.converged, "{what}");
        let discarded = gs.sweeps.last().expect("one sweep").max_truncation_error;
        assert!(discarded <= 1e-12, "{what}: discarded {discarded}");
        assert!(
            (gs.energy - case.energy).abs() <= TOL,
            "{what}: E {} != MPSKit {}",
            gs.energy,
            case.energy
        );
        let ours = gs.spectra.iter().zip(gs.entropies());
        let theirs = case.spectra.iter().zip(&case.entropy);
        for (bond, ((ours, s), (theirs, t))) in (1..).zip(ours.zip(theirs)) {
            let d = spectrum_distance(ours, theirs);
            assert!(d <= TOL, "{what} bond {bond}: spectra differ by {d}");
            assert!(
                (s - t).abs() <= TOL,
                "{what} bond {bond}: S {s} != MPSKit {t}"
            );
        }
    }
}
