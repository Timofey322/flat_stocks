from __future__ import annotations

import pandas as pd

from trading_system.types import StrategyParameters


def range_entry_vote(signals: pd.DataFrame, params: StrategyParameters) -> pd.Series:
    return signals["range_position"] <= params.range_entry_percentile


def range_exit_vote(signals: pd.DataFrame, params: StrategyParameters) -> pd.Series:
    return signals["range_position"] >= params.range_exit_percentile


def rsi_entry_vote(signals: pd.DataFrame, params: StrategyParameters) -> pd.Series:
    return signals["rsi"] <= params.rsi_entry_threshold


def rsi_exit_vote(signals: pd.DataFrame, params: StrategyParameters) -> pd.Series:
    return signals["rsi"] >= params.rsi_exit_threshold


def nw_entry_vote(signals: pd.DataFrame, params: StrategyParameters) -> pd.Series:
    if not params.use_nw_envelope or "nw_position" not in signals.columns:
        return pd.Series(True, index=signals.index)
    at_lower = signals["nw_position"] <= params.nw_entry_position_max
    below_band = signals["close"] <= signals["nw_lower"]
    return at_lower | below_band


def nw_exit_vote(signals: pd.DataFrame, params: StrategyParameters) -> pd.Series:
    if not params.use_nw_envelope or "nw_position" not in signals.columns:
        return pd.Series(False, index=signals.index)
    at_upper = signals["nw_position"] >= params.nw_exit_position_min
    above_band = signals["close"] >= signals["nw_upper"]
    return at_upper | above_band


def vote_count(votes: list[pd.Series]) -> pd.Series:
    stacked = pd.concat(votes, axis=1)
    return stacked.fillna(False).astype(int).sum(axis=1)


def synergy_entry_mask(
    signals: pd.DataFrame,
    params: StrategyParameters,
    fundamental_gate: pd.Series,
) -> pd.Series:
    votes = [
        fundamental_gate,
        range_entry_vote(signals, params),
        rsi_entry_vote(signals, params),
    ]
    if params.use_nw_envelope:
        votes.append(nw_entry_vote(signals, params))
    min_votes = min(params.synergy_min_votes_entry, len(votes))
    return vote_count(votes) >= min_votes


def synergy_exit_mask(
    signals: pd.DataFrame,
    params: StrategyParameters,
    fair_value_exit: pd.Series,
) -> pd.Series:
    votes = [
        range_exit_vote(signals, params),
        rsi_exit_vote(signals, params),
    ]
    if params.use_nw_envelope:
        votes.append(nw_exit_vote(signals, params))
    min_votes = min(params.synergy_min_votes_exit, len(votes))
    synergy = vote_count(votes) >= min_votes
    return synergy | fair_value_exit
