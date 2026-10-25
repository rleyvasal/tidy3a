"""labelarray: labelled N-D arrays in NumPy, becoming a library of its own (plan milestone 10, track A).

Nothing here imports the rest of tidy3a. So far: the errors, the exact label
rule (``labels``), the label indexes and their factory (``index``), exact
alignment of one axis (``align``), key transforms for grouping and
``factorize`` (``transforms``), float32 sums and means (``sums``), and
the contraction behind ``dot`` (``contract``). The grouped kernels and the
array and collection move in as the extraction proceeds, and the package
leaves tidy3a once its API has settled.
"""
