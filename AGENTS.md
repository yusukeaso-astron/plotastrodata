# Runtime type policy

When reviewing or changing modules, return Python `float`, `int`, and `bool`
for computed scalar results and expose numeric scalar constants as Python
scalars. Convert at the public output boundary and keep annotations and
docstrings consistent with the actual types.

Keep NumPy array outputs and their dtypes unchanged, including zero-dimensional
arrays. Do not add conversions throughout intermediate numerical calculations
or recursively convert user-provided values in generic container/pass-through
helpers. Preserve integer constants as integers. Preserve extended-precision NumPy floating scalars (such as `np.longdouble`)
as an explicit exception to Python scalar normalization. Include this
exception in return annotations and docstrings. This policy also applies
retroactively to previously reviewed modules.

Apply this policy to subsequent module-by-module annotation reviews as well.
