# Intuition Protocol — Findings Review

**Date:** 2026-03-06
**Repository:** https://github.com/code-423n4/2026-03-intuition
**Scope:** TrustBonding.sol, ProgressiveCurve.sol, OffsetProgressiveCurve.sol, AtomWallet.sol, TrustSwapAndBridgeRouter.sol

## Summary

| Finding | Verdict | Report? |
|---------|---------|---------|
| MED-1: Rollover re-init on zero | **Low/QA** (code quality, not exploitable) | No |
| MED-2: Epoch-boundary timing | **INVALID** | No |
| MED-3: deposit_for griefing | **OUT OF SCOPE** | No |
| MED-4: Bridge fee quoting | **Previously known / Speculative** | No |

---

## MED-1: `totalUtilization` epoch rollover re-initializes when value returns to zero — Low/QA (downgraded)

**Location:** `MultiVault.sol:1529-1534`

**Code pattern is technically wrong** — using `value == 0` as an initialization sentinel is fragile. However, the claimed attack path is not viable.

### Why totalUtilization cannot reach 0

The finding's "concrete trace" assumes utilization can freely go to zero through deposits and withdrawals. This ignores three structural invariants:

1. **Protocol fees on entry create a permanent gap.** Deposits add full `msg.value` to utilization (line 676), but protocol fees (`atomCreationProtocolFee`, `protocolFee`) are extracted before assets enter the vault. Redemptions remove `rawAssetsBeforeFees` (line 838), which can only draw from what's IN the vault — structurally less than what was deposited.

2. **MinShares are permanently locked.** On vault creation (line 1596), `minShare` is minted to `BURN_ADDRESS`. These shares' underlying assets can never be redeemed, creating a permanent positive utilization floor.

3. **Exit fees further widen the gap.** Each redemption cycle leaves a positive residual in utilization.

**Net effect:** `totalUtilization` is monotonically biased upward. Once any deposit occurs, it cannot return to 0 through normal protocol operations.

### personalUtilization is double-guarded

The finding claims `personalUtilization` is also affected, but it has a second guard the finding ignores. At line 1538:
```solidity
if (userLastEpoch == currentEpochLocal) {
    return; // already up to date; no rollover needed
}
```
Once a user acts in the current epoch, `userEpochHistory[user][0]` is set to `currentEpochLocal`. All subsequent calls for that user in the same epoch skip the entire personal rollover block. Even if personalUtilization reached 0, re-rollover cannot happen.

### C4 severity assessment

- **Not Medium.** C4 Medium requires "assets not at direct risk, but function of protocol could be impacted, or leak value with hypothetical attack path with stated assumptions, but external requirements." Here the external requirements are effectively impossible (entire protocol utilization reaching exactly zero wei).
- **Low/QA.** The code pattern is incorrect as a matter of defensive programming, but has no practical impact. Recommend fixing the sentinel pattern as good practice, but this does not warrant a Medium payout.

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
