"""The standard error that the whole research leans on."""

import numpy as np

from trading_system.research.stats import hac_mean


def test_the_mean_is_the_mean_of_the_days_that_count() -> None:
    values = np.array([1.0, 2.0, np.nan, 100.0, 3.0])
    mask = np.array([True, True, True, False, True])

    result = hac_mean(values, mask, lag=0)

    assert result.mean == 2.0 and result.n == 3  # the NaN and the masked 100 are left out


def test_a_real_effect_in_plenty_of_data_has_a_large_t() -> None:
    rng = np.random.default_rng(0)
    values = 0.5 + rng.normal(0.0, 1.0, 2000)

    assert hac_mean(values, np.ones(2000, bool), lag=0).t > 20


def test_too_few_observations_give_no_answer_rather_than_a_number() -> None:
    result = hac_mean(np.array([1.0, 2.0]), np.array([True, False]), lag=3)

    assert result.n == 1 and np.isnan(result.t) and np.isnan(result.mean)


def test_a_lag_longer_than_the_series_does_not_break() -> None:
    result = hac_mean(np.array([1.0, -1.0, 2.0]), np.ones(3, bool), lag=50)

    assert result.n == 3 and result.se >= 0  # it answers; no IndexError


def test_overlapping_windows_inflate_the_naive_t_and_the_newey_west_t_does_not() -> None:
    """Zero true effect. Each observation is the sum of 10 daily shocks and neighbours share 9 of
    them, as a 10-day forward return checked every day does. Counting them as independent gives a
    t with a spread near sqrt(10) = 3.2; accounting for the overlap brings it back near 1."""
    horizon, days = 10, 3000
    naive, adjusted = [], []
    for seed in range(150):
        shocks = np.random.default_rng(seed).normal(0.0, 1.0, days + horizon - 1)
        overlapping = np.convolve(shocks, np.ones(horizon), mode="valid")  # `days` long
        everything = np.ones(days, bool)
        naive.append(hac_mean(overlapping, everything, lag=0).t)
        adjusted.append(hac_mean(overlapping, everything, lag=horizon - 1).t)

    assert np.std(naive) > 2.5
    assert 0.7 < np.std(adjusted) < 1.5


def test_a_day_left_out_keeps_its_place_in_time() -> None:
    """Two observations three days apart are not neighbours; dropping the days between them must
    not make them look adjacent. With the gap kept, a lag of 1 sees no overlap between them."""
    values = np.array([1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0])
    mask = np.array([True, False, False, True, True, False, False, True])

    kept = hac_mean(values, mask, lag=1)
    squeezed = hac_mean(values[mask], np.ones(4, bool), lag=1)

    assert kept.mean == squeezed.mean == 0.0
    assert kept.se != squeezed.se
