# Intuition Protocol — Findings Review

**Date:** 2026-03-06
**Repository:** https://github.com/code-423n4/2026-03-intuition
**Scope:** TrustBonding.sol, ProgressiveCurve.sol, OffsetProgressiveCurve.sol, AtomWallet.sol, TrustSwapAndBridgeRouter.sol

## Summary

| Finding | Verdict | Report? |
|---------|---------|---------|
| MED-1: Rollover re-init on zero | **VALID** | Yes |
| MED-2: Epoch-boundary timing | **INVALID** | No |
| MED-3: deposit_for griefing | **OUT OF SCOPE** | No |
| MED-4: Bridge fee quoting | **Previously known / Speculative** | No |

---

## MED-1: `totalUtilization` epoch rollover re-initializes when value returns to zero — VALID

**Location:** `MultiVault.sol:1529-1534`

The `_rollover()` function uses `totalUtilization[currentEpochLocal] == 0` as a sentinel for "first action in epoch." But `totalUtilization` is an `int256` that is incremented by deposits and decremented by withdrawals, so it can legitimately return to zero after real activity within the same epoch. When this happens, the next action re-triggers the rollover branch, recopying the previous epoch's value and injecting phantom utilization.

The same pattern affects `personalUtilization` at line 1545.

**Downstream impact:** Corrupted utilization deltas flow into `TrustBonding._getSystemUtilizationRatio()` and `_emissionsForEpoch()`, causing misallocation of token emissions.

**Fix:** Use a separate `mapping(uint256 => bool) totalUtilizationInitialized` flag instead of inferring initialization from `value == 0`.

---

## MED-2: Epoch-boundary utilization timing can inflate ratios — INVALID

The finding misunderstands the system:

1. Rewards derive from **bonded balance** (`userBondedBalanceAtEpochEnd`), not MultiVault utilization. Depositing into MultiVault alone doesn't earn rewards.
2. Utilization ratios are **capped at 100%** (`BASIS_POINTS_DIVISOR`) — they're scaling multipliers, not unbounded.
3. Increasing utilization to earn a higher multiplier is **intended behavior** — the design incentivizes protocol usage.
4. Rollover properly bridges epochs — withdrawing in e+1 correctly reduces from the rolled-over value.

---

## MED-3: `deposit_for` pulls tokens from `_addr` — OUT OF SCOPE

The technical claim is accurate (deposit_for is permissionless, transfers from target), but:

- `VotingEscrow.sol` is **explicitly listed in `out_of_scope.txt`**
- This is the **original Curve Finance design** — intentional upstream behavior documented in NatSpec

---

## MED-4: Router pays exactly `quoteTransferRemote()` — Previously Known / Speculative

Two sub-claims:

1. **Fee quoted on `minTrustOut` vs actual `amountOut`:** Real bug, but already identified and patched in V12 audit (ineligible per README).
2. **Quote itself may be inaccurate:** Speculative. The router uses the standard quoting interface. If `transferRemote` receives insufficient fee, it reverts atomically (no stranded transfers). MetaERC20Hub is an external dependency outside router control.
