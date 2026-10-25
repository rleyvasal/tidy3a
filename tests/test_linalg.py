import numpy as np
import pytest
import xarray as xr

from tidy3a import (
    AlignError,
    MutateError,
    assign,
    col,
    conj,
    dot,
    fields,
    filter,
    infer,
    mutate,
    pack,
    pull,
    read_array,
    rename_axes,
    sqrt,
    sum,
    tidy,
)

x, y, z, intensity, ring, distance, p = fields("x", "y", "z", "intensity", "ring", "distance", "p")
AXES = ["x", "y", "z"]


def rotation_from_wxyz(q):
    w, a, b, c = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (b * b + c * c), 2 * (a * b - c * w), 2 * (a * c + b * w)],
        [2 * (a * b + c * w), 1 - 2 * (a * a + c * c), 2 * (b * c - a * w)],
        [2 * (a * c - b * w), 2 * (b * c + a * w), 1 - 2 * (a * a + b * b)],
    ])


def operator(matrix, rows="i", columns="feature"):
    return xr.DataArray(matrix, dims=(rows, columns), coords={rows: AXES, columns: AXES})


def random_cloud(rng, count=200):
    values = rng.normal(scale=30.0, size=(count, 5)).astype("float32")
    return read_array(
        values, dims={"point": infer, "feature": 5},
        labels={"feature": ["x", "y", "z", "intensity", "ring"]}, field_axis="feature",
    ), values


# ── the point-cloud rotation against an independent NumPy calculation ──────


def test_rotation_and_offset_match_numpy():
    rng = np.random.default_rng(7)
    cloud, values = random_cloud(rng)
    quaternion = rng.normal(size=4)
    R = rotation_from_wxyz(quaternion)
    t = np.array([0.94, 0.0, 1.84])
    offset = xr.DataArray(t, dims="i", coords={"i": AXES})

    out = cloud >> mutate(p = dot(operator(R), [x, y, z], over="feature") + offset)
    expected = values[:, :3].astype(float) @ R.T + t
    got = out.collect()["p"].transpose("point", "i").values
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-4)


def test_distance_filter_keeps_points_at_the_boundary():
    # Sensor points that land 49, 50, 51, and 3 m away after a 90° turn and 1 m offset.
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    values = np.array([
        [0, -48, 0, 10, 1], [0, -49, 0, 20, 2], [51, 1, 0, 30, 1], [0, 1, 3, 40, 2],
    ], dtype="float32")
    cloud = read_array(values, dims={"point": infer, "feature": 5},
                       labels={"feature": ["x", "y", "z", "intensity", "ring"]}, field_axis="feature")
    offset = xr.DataArray([1.0, 0.0, 0.0], dims="i", coords={"i": AXES})
    nearby = (
        cloud
        >> mutate(p = dot(operator(R), [x, y, z], over="feature") + offset)
        >> mutate(distance = sqrt(sum(p ** 2, over="i")))
        >> filter(distance <= 50)
    )
    assert (nearby >> pull("point")).tolist() == [1, 2, 4]
    assert (nearby >> pull(intensity)).tolist() == [10.0, 20.0, 40.0]
    np.testing.assert_allclose(nearby >> pull(distance), [49.0, 50.0, 3.0])


def test_rotation_works_the_same_after_variables_from():
    rng = np.random.default_rng(3)
    values = rng.normal(size=(50, 5)).astype("float32")
    labels = {"feature": ["x", "y", "z", "intensity", "ring"]}
    field = read_array(values, dims={"point": infer, "feature": 5}, labels=labels, field_axis="feature")
    split = read_array(values, dims={"point": infer, "feature": 5}, labels=labels, variables_from="feature")
    R = operator(rotation_from_wxyz([0.9, 0.1, -0.3, 0.2]))
    a = (field >> mutate(p = dot(R, [x, y, z], over="feature"))).collect()["p"]
    b = (split >> mutate(p = dot(R, [x, y, z], over="feature"))).collect()["p"]
    xr.testing.assert_allclose(a.transpose("point", "i"), b.transpose("point", "i"))


def test_vector_write_back_rotates_in_place():
    R = operator(np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    values = np.array([[1, 2, 3, 10, 1], [4, 5, 6, 20, 2]], dtype="float32")
    cloud = read_array(values, dims={"point": infer, "feature": 5},
                       labels={"feature": ["x", "y", "z", "intensity", "ring"]}, field_axis="feature")
    out =(cloud >> mutate(assign([x, y, z], dot(R, [x, y, z], over="feature")))).collect()["values"]
    # Every component comes from the original vector: x' = -y, y' = x.
    assert out.sel(feature="x").values.tolist() == [-2.0, -5.0]
    assert out.sel(feature="y").values.tolist() == [1.0, 4.0]
    assert out.sel(feature="intensity").values.tolist() == [10.0, 20.0]


# ── dot rules ───────────────────────────────────────────────────────────────


def test_dot_over_keeps_batch_axes_and_default_needs_one_shared_axis():
    A = xr.DataArray(np.ones((2, 3)), dims=("time", "feature"), coords={"feature": AXES})
    B = xr.DataArray(np.ones((2, 3)), dims=("time", "feature"), coords={"feature": AXES})
    frame = tidy(xr.Dataset({"a": A, "b": B}))
    out = (frame >> mutate(c = dot(col("a"), col("b"), over="feature"))).collect()["c"]
    assert out.dims == ("time",) and out.values.tolist() == [3.0, 3.0]
    with pytest.raises(Exception, match="share 2 axes"):
        frame >> mutate(c = dot(col("a"), col("b")))


def test_dot_aligns_by_label_not_by_position():
    R = operator(np.diag([1.0, 2.0, 3.0]), columns="feature")
    v = xr.DataArray([3.0, 1.0, 2.0], dims="feature", coords={"feature": ["z", "x", "y"]})
    frame = tidy(xr.Dataset({"v": v}))
    out = frame >> mutate(w = dot(R, col("v"), over="feature"))
    assert out.collect()["w"].sel(i="z").item() == 9.0
    assert "feature reordered to match" in repr(out)


def test_dot_rejects_mismatched_labels():
    R = operator(np.eye(3))
    v = xr.DataArray([1.0, 2.0, 3.0], dims="feature", coords={"feature": ["x", "y", "w"]})
    with pytest.raises(AlignError, match="feature labels differ"):
        tidy(xr.Dataset({"v": v})) >> mutate(w = dot(R, col("v"), over="feature"))


def test_dot_does_not_align_axes_by_length():
    R = operator(np.eye(3))
    v = xr.DataArray([1.0, 2.0, 3.0], dims="component")
    with pytest.raises(Exception, match="component is not an axis"):
        tidy(xr.Dataset({"v": v})) >> mutate(w = dot(R, col("v"), over="component"))


@pytest.mark.parametrize("seed", range(5))
def test_dot_matches_einsum_at_any_rank_and_axis_order(seed):
    rng = np.random.default_rng(seed)
    sizes = {"a": 2, "b": 3, "c": 4, "d": 2}
    left_dims = list(rng.permutation(["a", "b", "c"]))
    right_dims = list(rng.permutation(["b", "c", "d"]))
    left = xr.DataArray(rng.normal(size=[sizes[d] for d in left_dims]), dims=left_dims)
    right = xr.DataArray(rng.normal(size=[sizes[d] for d in right_dims]), dims=right_dims)
    frame = tidy(xr.Dataset({"l": left, "r": right}))
    got = (frame >> mutate(o = dot(col("l"), col("r"), over=["b", "c"]))).collect()["o"]
    expected = np.einsum("abc,bcd->ad", left.transpose("a", "b", "c").values, right.transpose("b", "c", "d").values)
    np.testing.assert_allclose(got.transpose("a", "d").values, expected)


# ── complex arrays ──────────────────────────────────────────────────────────


def gate(matrix):
    return xr.DataArray(np.asarray(matrix, dtype=complex), dims=("output", "input"),
                        coords={"output": [0, 1], "input": [0, 1]})


def state(amplitudes):
    return tidy(xr.Dataset({"amplitude": xr.DataArray(np.asarray(amplitudes, dtype=complex),
                                                      dims="input", coords={"input": [0, 1]})}))


def test_gate_example_from_the_plan():
    out = state([1, 0]) >> mutate(amplitude_out = dot(gate([[0, 1], [1, 0]]), col("amplitude"), over="input"))
    assert out.collect()["amplitude_out"].values.tolist() == [0, 1]


def test_complex_input_is_not_conjugated_and_norm_is_preserved():
    hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    phase = np.diag([1, 1j])
    amplitudes = np.array([0.6, 0.8j])
    out = (state(amplitudes)
           >> mutate(a = dot(gate(phase @ hadamard), col("amplitude"), over="input"))).collect()["a"]
    np.testing.assert_allclose(out.values, phase @ hadamard @ amplitudes)
    assert np.isclose(np.linalg.norm(out.values), 1.0)


def test_inner_product_needs_explicit_conj():
    frame = state([1j, 0])
    a = col("amplitude")
    plain = (frame >> mutate(v = dot(a, a, over="input"))).collect()["v"].item()
    inner = (frame >> mutate(v = dot(conj(a), a, over="input"))).collect()["v"].item()
    assert plain == -1 and inner == 1


def test_gate_on_one_axis_of_a_two_qubit_state():
    psi = np.array([[1, 0], [0, 0]], dtype=complex)  # |00>
    da = xr.DataArray(psi, dims=("q0", "q1"), coords={"q0": [0, 1], "q1": [0, 1]})
    flip = xr.DataArray([[0, 1], [1, 0]], dims=("q1_out", "q1"), coords={"q1_out": [0, 1], "q1": [0, 1]})
    out = (tidy(da) >> mutate(o = dot(flip, col("values"), over="q1"))).collect()["o"]
    assert set(out.dims) == {"q0", "q1_out"}
    assert out.transpose("q0", "q1_out").values.tolist() == [[0, 1], [0, 0]]  # |01>


def test_chaining_gates_with_rename_axes():
    x_gate = gate([[0, 1], [1, 0]])
    out = (state([1, 0])
           >> mutate(a1 = dot(x_gate, col("amplitude"), over="input"))
           >> mutate(a2 = dot(x_gate, rename_axes(col("a1"), output="input"), over="input"))).collect()
    assert out["a2"].values.tolist() == [1, 0]


# ── helpers ─────────────────────────────────────────────────────────────────


def test_rename_axes_lets_a_vector_view_be_stored():
    rng = np.random.default_rng(0)
    cloud, _ = random_cloud(rng, 4)
    with pytest.raises(MutateError, match="rename_axes"):
        cloud >> mutate(v = [x, y, z])
    out = (cloud >> mutate(v = rename_axes([x, y, z], feature="component"))).collect()["v"]
    assert out.dims == ("point", "component") and out.component.values.tolist() == AXES


def test_rename_axes_errors():
    rng = np.random.default_rng(0)
    cloud, _ = random_cloud(rng, 4)
    with pytest.raises(Exception, match="not an axis"):
        cloud >> mutate(v = rename_axes(x, time="t"))
    with pytest.raises(TypeError, match="old=\"new\""):
        rename_axes(x)


def test_pack_names_its_axis_and_labels():
    ds = xr.Dataset({"vx": ("t", [1.0, 2.0]), "vy": ("t", [3.0, 4.0]), "vz": ("t", [5.0, 6.0])})
    vx, vy, vz = fields("vx", "vy", "vz")
    out = (tidy(ds) >> mutate(v = pack([vx, vy, vz], along="feature"))).collect()["v"]
    assert out.dims == ("feature", "t") and out.feature.values.tolist() == ["vx", "vy", "vz"]
    with pytest.raises(TypeError, match="never named for you"):
        pack([vx, vy], along="")


@pytest.mark.parametrize("scalar_first", [False, True])
def test_pack_matches_reordered_labels_when_first_value_can_be_scalar(scalar_first):
    frame = tidy(xr.Dataset({"vx": ("point", [1, 2])}, coords={"point": ["a", "b"]}))
    vy = xr.DataArray([20, 10], dims="point", coords={"point": ["b", "a"]}, name="vy")
    items = [0, col("vx"), vy] if scalar_first else [col("vx"), vy]
    out = frame >> mutate(v=pack(items, along="component"))
    expected = [[0, 0], [1, 2], [10, 20]] if scalar_first else [[1, 2], [10, 20]]
    assert out.collect()["v"].transpose("component", "point").values.tolist() == expected
    assert "point reordered to match" in repr(out)


@pytest.mark.parametrize("labels, message", [(["a", "other"], "labels differ"), (None, "unlabelled")])
def test_pack_checks_axes_shared_by_later_values(labels, message):
    frame = tidy(xr.Dataset({"vx": ("point", [1, 2])}, coords={"point": ["a", "b"]}))
    vy = xr.DataArray([10, 20], dims="point", coords={} if labels is None else {"point": labels}, name="vy")
    with pytest.raises(AlignError, match=message):
        frame >> mutate(v=pack([0, col("vx"), vy], along="component"))


def test_numeric_functions_work_on_data_and_plain_values():
    rng = np.random.default_rng(1)
    cloud, values = random_cloud(rng, 10)
    out = (cloud >> mutate(r = sqrt(x ** 2 + y ** 2))).collect()["r"]
    np.testing.assert_allclose(out.values, np.hypot(values[:, 0], values[:, 1]), rtol=1e-6)
    assert sqrt(16.0) == 4.0


def test_dot_does_not_depend_on_optional_einsum_packages():
    # xarray switches dot to opt_einsum when it is installed, which changes results in the
    # last digits and lays them out column-major, making later steps twice as slow.
    pytest.importorskip("opt_einsum")
    rng = np.random.default_rng(5)
    labels = ["x", "y", "z"]
    cloud = tidy(xr.DataArray(rng.normal(size=(1000, 3)).astype("float32"), dims=("point", "feature"),
                              coords={"feature": labels}), field_axis="feature")
    rotation = xr.DataArray(rng.normal(size=(3, 3)).astype("float32"), dims=("i", "feature"),
                            coords={"i": labels, "feature": labels})
    x, y, z = fields("x", "y", "z")
    results = {}
    for allowed in (True, False):
        with xr.set_options(use_opt_einsum=allowed):
            results[allowed] = (cloud >> mutate(p=dot(rotation, [x, y, z], over="feature"))).collect()["p"]
    assert results[True].values.tobytes() == results[False].values.tobytes()
    assert results[True].values.flags.c_contiguous
