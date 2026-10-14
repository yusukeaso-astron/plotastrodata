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

Internal calculations and private callbacks need no scalar normalization.
Normalize computed values when returning public results or storing public
attributes. Treat serialization separately: narrow unsupported extended-precision
values only in an export payload, preserving the in-memory results and arrays.


# Plotting angle policy

Angles are measured in degrees from the original data +y axis toward +x.
Regions, beams, lines, arrows, and polarization segments must remain aligned
with the data. `swapxy` exchanges direction components, equivalent to
`90° - angle`. `xflip` and `yflip` affect only the display and require no
angle correction. Apply any additional rotation in the original coordinates
before `swapxy`.

Relative numeric overlay positions, including `beampos`, refer to the
displayed axes and are not exchanged by `swapxy`.

Keep method docstrings consistent with this policy.

# Swapped data policy

`AstroFrame(swapxy=True).read(d)` transposes the data
and exchanges the processed x/y coordinates. Transform the stored beam angle
to `90 - bpa` as well, keeping both beam lengths unchanged, including for PV
data. Construct an effective PV beam using the actual velocity spacing before
transforming its angle. Preserve original sky-beam metadata in `beam_org`.

AstroData does not track or undo swaps. Its analysis and FITS-writing methods
continue to use their usual axis assumptions on the current data. In PV mode,
v/dv follow the processed y axis, even when that axis now contains positions.
Report once per swapped read that subsequent operations use the current axis
order, and document this behavior.

Plotting must not apply a second swap to an already transformed beam.
add_beam always takes angles in the processed data frame, including explicitly
supplied beams. Other user-supplied overlay angles retain the original-data
convention above.
