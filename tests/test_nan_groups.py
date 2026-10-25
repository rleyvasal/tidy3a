"""Missing group keys (plan milestone 10, decision 11).

All NaN (or NaT) values of a key form one group, as in pandas. By default
``group_by`` leaves those positions out of every group and says so in a
note; ``dropna=False`` keeps them as a group labelled NaN, sorted last.
"""

import numpy as np
import pytest
import xarray as xr

from tidy3a import (
    GroupError,
    bins,
    col,
    filter,
    group_by,
    mean,
    min,
    mutate,
    n,
    sum,
    summarise,
    tidy,
)

NAN = float("nan")


def ring_data():
    ds = xr.Dataset({
        "v": ("point", [1.0, 2.0, 3.0, 4.0, 5.0]),
        "ring": ("point", [1.0, NAN, 2.0, NAN, 1.0]),
    })
    return tidy(ds)


def dropped_note(frame):
    return [note for note in frame._notes if "dropped" in note]


def test_missing_keys_are_dropped_by_default_with_a_note():
    out = ring_data() >> group_by("ring") >> summarise(m=mean(col("v")), k=n())
    data = out.collect()
    assert data.ring.values.tolist() == [1.0, 2.0]
    assert data["m"].values.tolist() == [3.0, 3.0]
    assert data["k"].values.tolist() == [2, 1]
    assert dropped_note(out) == [
        "changed: group_by(ring) dropped 2 positions with a missing ring; dropna=False keeps them as a group"
    ]


def test_the_note_counts_one_position_in_the_singular():
    ds = xr.Dataset({"v": ("point", [1.0, 2.0]), "ring": ("point", [1.0, NAN])})
    out = tidy(ds) >> group_by("ring")
    assert dropped_note(out) == [
        "changed: group_by(ring) dropped 1 position with a missing ring; dropna=False keeps them as a group"
    ]


def test_keys_without_missing_values_have_no_note():
    ds = xr.Dataset({"v": ("point", [1.0, 2.0]), "ring": ("point", [1.0, 2.0])})
    assert dropped_note(tidy(ds) >> group_by("ring") >> summarise(m=mean(col("v")))) == []


def test_dropna_false_keeps_the_missing_group_last():
    out = ring_data() >> group_by("ring", dropna=False) >> summarise(m=mean(col("v")), k=n())
    data = out.collect()
    labels = data.ring.values
    assert labels[:2].tolist() == [1.0, 2.0] and np.isnan(labels[2])
    assert data["m"].values.tolist() == [3.0, 3.0, 3.0]
    assert data["k"].values.tolist() == [2, 1, 2]
    assert dropped_note(out) == []


def test_every_nan_is_one_group_and_signed_zeros_are_one_value():
    negative_nan = -np.float64("nan")
    ds = xr.Dataset({
        "v": ("point", [1.0, 2.0, 3.0, 4.0]),
        "ring": ("point", [0.0, NAN, -0.0, negative_nan]),
    })
    out = (tidy(ds) >> group_by("ring", dropna=False) >> summarise(s=sum(col("v")), k=n())).collect()
    assert len(out.ring) == 2 and out.ring.values[0] == 0.0 and np.isnan(out.ring.values[1])
    assert out["s"].values.tolist() == [4.0, 6.0]
    assert out["k"].values.tolist() == [2, 2]


def test_float32_keys():
    ds = xr.Dataset({
        "v": ("point", np.array([1.0, 2.0, 3.0], dtype="float32")),
        "ring": ("point", np.array([NAN, 1.5, NAN], dtype="float32")),
    })
    kept = (tidy(ds) >> group_by("ring", dropna=False) >> summarise(s=sum(col("v")))).collect()
    assert kept.ring.values[0] == 1.5 and np.isnan(kept.ring.values[1])
    assert kept["s"].values.tolist() == [2.0, 4.0]
    dropped = (tidy(ds) >> group_by("ring") >> summarise(s=sum(col("v")))).collect()
    assert dropped.ring.values.tolist() == [1.5] and dropped["s"].values.tolist() == [2.0]


def times_data():
    times = np.array(["2026-01-02", "NaT", "2026-01-01", "NaT"], dtype="datetime64[ns]")
    return tidy(xr.Dataset({"v": ("point", [1.0, 2.0, 3.0, 4.0]), "t": ("point", times)}))


def test_missing_times_are_dropped_by_default():
    out = times_data() >> group_by("t") >> summarise(m=mean(col("v")))
    data = out.collect()
    assert data.t.values.astype("datetime64[D]").astype(str).tolist() == ["2026-01-01", "2026-01-02"]
    assert data["m"].values.tolist() == [3.0, 1.0]
    assert dropped_note(out) == [
        "changed: group_by(t) dropped 2 positions with a missing t; dropna=False keeps them as a group"
    ]


def test_missing_times_are_one_group_last_with_dropna_false():
    out = (times_data() >> group_by("t", dropna=False) >> summarise(m=mean(col("v")))).collect()
    assert out["m"].values.tolist() == [3.0, 1.0, 3.0]
    assert np.isnat(out.t.values[-1]) and not np.isnat(out.t.values[:2]).any()


def test_grouped_mutate_gives_dropped_positions_no_group_statistics():
    out = (ring_data() >> group_by("ring") >> mutate(d=col("v") - mean(col("v")), k=n(), w=col("v") * 2)).collect()
    d, k = out["d"].values, out["k"].values
    assert d[[0, 2, 4]].tolist() == [-2.0, 0.0, 2.0] and np.isnan(d[[1, 3]]).all()
    assert k[[0, 2, 4]].tolist() == [2.0, 1.0, 2.0] and np.isnan(k[[1, 3]]).all()
    assert out["w"].values.tolist() == [2.0, 4.0, 6.0, 8.0, 10.0]  # not a group statistic: unaffected


def test_grouped_mutate_with_dropna_false_reduces_the_missing_group_too():
    out = (ring_data() >> group_by("ring", dropna=False) >> mutate(d=col("v") - mean(col("v")), k=n())).collect()
    assert out["d"].values.tolist() == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert out["k"].values.tolist() == [2, 2, 1, 2, 2]


def test_grouped_mutate_of_times_gives_dropped_positions_nat():
    out = (times_data() >> group_by("t") >> mutate(first=min(col("t")))).collect()
    assert np.isnat(out["first"].values[[1, 3]]).all()
    assert not np.isnat(out["first"].values[[0, 2]]).any()


def all_missing():
    return tidy(xr.Dataset({"v": ("point", [1.0, 2.0, 3.0]), "ring": ("point", [NAN, NAN, NAN])}))


def test_all_missing_keys_leave_no_groups_by_default():
    out = all_missing() >> group_by("ring") >> summarise(m=mean(col("v")), k=n(), c=1)
    data = out.collect()
    assert data.sizes["ring"] == 0 and data["m"].size == 0 and data["k"].size == 0 and data["c"].size == 0
    assert dropped_note(out) == [
        "changed: group_by(ring) dropped 3 positions with a missing ring; dropna=False keeps them as a group"
    ]
    mutated = (all_missing() >> group_by("ring") >> mutate(m=mean(col("v")))).collect()
    assert np.isnan(mutated["m"].values).all()


def test_all_missing_keys_are_one_group_with_dropna_false():
    out = (all_missing() >> group_by("ring", dropna=False) >> summarise(m=mean(col("v")), c=1)).collect()
    assert np.isnan(out.ring.values).tolist() == [True]
    assert out["m"].values.tolist() == [2.0] and out["c"].values.tolist() == [1]


def test_scalar_summaries_have_one_value_per_kept_group():
    out = (ring_data() >> group_by("ring") >> summarise(c=1)).collect()
    assert out.ring.values.tolist() == [1.0, 2.0] and out["c"].values.tolist() == [1, 1]


def test_groups_found_again_keep_dropna_and_the_note():
    # filter changes the data, so summarise finds the groups again from the key.
    out = ring_data() >> group_by("ring", dropna=False) >> filter(col("v") > 1) >> summarise(k=n())
    data = out.collect()
    assert data.ring.values[:2].tolist() == [1.0, 2.0] and np.isnan(data.ring.values[2])
    assert data["k"].values.tolist() == [1, 1, 2]
    dropped = ring_data() >> group_by("ring") >> filter(col("v") > 2) >> summarise(k=n())
    assert dropped.collect()["k"].values.tolist() == [1, 1]
    assert dropped_note(dropped) == [
        "changed: group_by(ring) dropped 2 positions with a missing ring; dropna=False keeps them as a group",
        "changed: group_by(ring) dropped 1 position with a missing ring; dropna=False keeps them as a group",
    ]


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("dropna", [True, False])
def test_many_groups_with_missing_keys_match_each_group_reduced_alone(dtype, dropna):
    rng = np.random.default_rng(3)
    size = 3_000
    keys = rng.integers(0, 60, size).astype(float)
    keys[rng.random(size) < 0.1] = NAN
    values = rng.normal(size=(2, size)).astype(dtype)
    ds = xr.Dataset({"v": (("sensor", "point"), values), "key": ("point", keys)})
    out = (tidy(ds) >> group_by("key", dropna=dropna) >> summarise(
        s=sum(col("v")), m=mean(col("v")), k=n())).collect()
    labels = np.unique(keys[~np.isnan(keys)])
    if not dropna:
        labels = np.append(labels, NAN)
    np.testing.assert_array_equal(out.key.values, labels)
    for i, label in enumerate(labels):
        mask = np.isnan(keys) if np.isnan(label) else keys == label
        group = (tidy(ds.isel(point=mask)) >> summarise(s=sum(col("v")), m=mean(col("v")), k=n())).collect()
        np.testing.assert_array_equal(out["s"].isel(key=i).values, group["s"].values)
        np.testing.assert_array_equal(out["m"].isel(key=i).values, group["m"].values)
        assert out["k"].isel(key=i).item() == group["k"].item()
    centred = (tidy(ds) >> group_by("key", dropna=dropna) >> mutate(d=col("v") - mean(col("v")))).collect()
    missing = np.isnan(centred["d"].transpose("sensor", "point").values)
    assert missing[:, np.isnan(keys)].all() == dropna


def intensities(values):
    return tidy(xr.Dataset({"intensity": ("point", np.array(values, dtype=float))}))


def test_bins_drop_missing_values_by_default_with_a_note():
    out = intensities([10.0, NAN, 20.0, 30.0]) >> group_by(bins(col("intensity"), [0, 15, 35])) >> summarise(k=n())
    data = out.collect()
    assert data.intensity_bin.values.tolist() == ["[0, 15)", "[15, 35)"]
    assert data["k"].values.tolist() == [1, 2]
    assert dropped_note(out) == [(
        "changed: group_by(bins(intensity)) dropped 1 position with a missing intensity; "
        "dropna=False keeps them as a group"
    )]


def test_bins_keep_missing_values_as_a_last_group_with_dropna_false():
    out = (intensities([10.0, NAN, 20.0, 30.0])
           >> group_by(bins(col("intensity"), [0, 15, 35]), dropna=False) >> summarise(k=n())).collect()
    # Bin labels are text, so the missing group's label is text too: one type of label.
    assert out.intensity_bin.values.tolist() == ["[0, 15)", "[15, 35)", "NaN"]
    assert out["k"].values.tolist() == [1, 2, 1]


@pytest.mark.parametrize("dropna", [True, False])
def test_bins_still_refuse_values_outside_every_bin(dropna):
    with pytest.raises(GroupError, match="1 positions of intensity fall outside every bin"):
        intensities([10.0, NAN, 99.0]) >> group_by(bins(col("intensity"), [0, 15, 35]), dropna=dropna)


def test_bins_by_duration_ignore_missing_times_when_placing_the_edges():
    times = np.array(["2026-06-01", "NaT", "2026-06-02", "2026-06-04"], dtype="datetime64[ns]")
    ds = xr.Dataset({"v": ("point", [1.0, 2.0, 3.0, 4.0]), "t": ("point", times)})
    out = tidy(ds) >> group_by(bins(col("t"), "2 days")) >> summarise(m=mean(col("v")))
    data = out.collect()
    assert data.t_bin.values.tolist() == ["[2026-06-01, 2026-06-03)", "[2026-06-03, 2026-06-05)"]
    assert data["m"].values.tolist() == [2.0, 4.0]
    assert len(dropped_note(out)) == 1
    kept = (tidy(ds) >> group_by(bins(col("t"), "2 days"), dropna=False) >> summarise(m=mean(col("v")))).collect()
    assert kept.t_bin.values.tolist() == ["[2026-06-01, 2026-06-03)", "[2026-06-03, 2026-06-05)", "NaT"]
    assert kept["m"].values.tolist() == [2.0, 4.0, 2.0]


def test_bins_by_duration_of_only_missing_times_have_no_edges():
    times = np.array(["NaT", "NaT"], dtype="datetime64[ns]")
    ds = xr.Dataset({"v": ("point", [1.0, 2.0]), "t": ("point", times)})
    with pytest.raises(GroupError, match="there are no times to bin"):
        tidy(ds) >> group_by(bins(col("t"), "2 days"))


def test_dropna_must_be_true_or_false():
    with pytest.raises(GroupError, match="dropna must be True or False"):
        ring_data() >> group_by("ring", dropna="no")


def test_dropna_false_shows_in_the_verb_text():
    assert str(group_by("ring", dropna=False)) == "<verb group_by(\"ring\", dropna = False)>"
