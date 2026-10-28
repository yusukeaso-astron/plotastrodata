"""Compose PNG images or first PDF pages using the same grid geometry.

Sizes are rounded to whole layout units (pixels for PNG, points for PDF).
PDF composition uses the required pypdf dependency, imported only when used.
"""
from contextlib import ExitStack
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from PIL import Image

from plotastrodata._type_utils import RealScalar
from plotastrodata._validation_utils import (
    _is_finite_real_scalar,
    _is_positive_real_scalar,
    _require_integer,
    _require_positive_real_scalar,
)


if TYPE_CHECKING:
    from pypdf import PageObject, PdfWriter
    from pypdf.generic import IndirectObject, StreamObject


PathInput = str | PathLike[str]
ScaleInput = Sequence[RealScalar] | Sequence[Sequence[RealScalar]] | None


def _inputs(file_list: Sequence[PathInput], output_file: PathInput,
            columns: int, padding: RealScalar, scale_factors: ScaleInput,
            suffix: str
            ) -> tuple[list[Path], Path, int, int, list[RealScalar]]:
    """Validate shared options without opening files and normalize their types.

    Return input Paths, the output Path, integer columns and padding, and
    one real scalar scale per input (preserving extended precision).
    Flatten nested scales in row-major order, filling missing entries with
    1.0 and validating unused entries as well.
    Reject output paths that resolve to an input or have the wrong suffix.
    """
    if isinstance(file_list, (str, PathLike)):
        raise TypeError('file_list must be a sequence of paths')
    files = [Path(f) for f in file_list]
    if not files:
        raise ValueError('file_list is empty')
    _require_integer(columns, 'columns')
    if not _is_finite_real_scalar(padding) or padding < 0:
        raise ValueError('padding must be a finite nonnegative real scalar')
    integer_padding = int(padding)
    if padding != integer_padding:
        raise ValueError('padding must be a whole number of layout units')
    output = Path(output_file)
    if output.suffix.lower() != suffix:
        raise ValueError(f'output_file must end in {suffix}')
    resolved_output = output.resolve()
    if any(resolved_output == f.resolve() for f in files):
        raise ValueError('output_file must differ from every input file')
    columns = int(columns)
    scales = _scale_factors(scale_factors, len(files), columns)
    return files, output, columns, integer_padding, scales


def _grid_scales(values: Sequence[Sequence[RealScalar]], item_count: int,
                 columns: int) -> list[RealScalar]:
    """Flatten a partial scale grid, validating even unused final-row cells.

    Missing entries default to 1.0; supplied real scalar precision is retained.
    """
    rows = (item_count + columns - 1) // columns
    if len(values) > rows:
        raise ValueError('too many scale-factor rows')
    scales: list[RealScalar] = [1.0] * item_count
    for r, row in enumerate(values):
        if isinstance(row, (str, bytes)) or isinstance(row, RealScalar):
            raise ValueError('scale_factors must be flat numbers'
                             + ' or a nested grid')
        row = list(row)
        if len(row) > columns:
            raise ValueError('too many scale-factor columns')
        for c, value in enumerate(row):
            _require_positive_real_scalar(value, 'scale factor')
            index = r * columns + c
            if index < item_count:
                scales[index] = value
    return scales


def _scale_factors(scale_factors: ScaleInput, item_count: int,
                   columns: int) -> list[RealScalar]:
    """Validate flat or nested factors and return one scale per input.

    None gives unit scales. Flat sequences require exactly item_count values;
    nested grids may omit entries. Preserve supplied scalar precision.
    """
    if scale_factors is None:
        return [1.0] * item_count
    values = list(scale_factors)
    if not values:
        raise ValueError('scale_factors must not be empty')
    if all(isinstance(v, RealScalar) for v in values):
        if len(values) != item_count:
            raise ValueError('flat scale_factors must have one value per input')
        for value in values:
            _require_positive_real_scalar(value, 'scale factor')
        return values
    return _grid_scales(values, item_count, columns)


def _layout(sizes: Sequence[tuple[RealScalar, RealScalar]], columns: int,
            padding: int, scales: Sequence[RealScalar]
            ) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]]:
    """Return canvas (width, height) and row-major (x, y, width, height) items.

    Scale source sizes and round to whole units, with a minimum of one.
    Size each column/row to its largest item and omit wholly unused columns.
    Coordinates start at the top left; integer centering leaves odd spare
    units on the right/bottom. Padding separates neighboring cells only.
    """
    columns = min(columns, len(sizes))
    rows = (len(sizes) + columns - 1) // columns
    widths, heights = [0] * columns, [0] * rows
    drawn = []
    for i, ((w, h), scale) in enumerate(zip(sizes, scales)):
        dw, dh = w * scale, h * scale
        if not all(_is_positive_real_scalar(v)
                   for v in (w, h, dw, dh)):
            raise ValueError('input and scaled dimensions must be'
                             + ' positive and finite')
        cw, ch = max(1, round(dw)), max(1, round(dh))
        drawn.append((cw, ch))
        widths[i % columns] = max(widths[i % columns], cw)
        heights[i // columns] = max(heights[i // columns], ch)
    xs, ys = [], []
    for sizes_, positions in ((widths, xs), (heights, ys)):
        position = 0
        for size in sizes_:
            positions.append(position)
            position += size + padding
    cells = []
    for i, (w, h) in enumerate(drawn):
        r, c = divmod(i, columns)
        cells.append((xs[c] + (widths[c] - w) // 2,
                      ys[r] + (heights[r] - h) // 2, w, h))
    return (sum(widths) + padding * (columns - 1),
            sum(heights) + padding * (rows - 1)), cells


def concat_pngs(file_list: Sequence[PathInput], output_file: PathInput,
                columns: int, padding: RealScalar = 0, *,
                scale_factors: ScaleInput = None,
                background: tuple[int, int, int, int]
                | tuple[int, int, int] | str = 'white',
                resample: Image.Resampling | int = Image.Resampling.LANCZOS
                ) -> None:
    """Concatenate PNG images into one grid PNG.

    Images fill rows from left to right. Each input is scaled directly and
    rounded to whole pixels, with a minimum size of one pixel per dimension.
    Rows and columns fit their largest scaled item. Images are centered,
    leaving any odd spare pixel on the right/bottom. Unused cells are blank;
    entirely unused columns are omitted. No input content is cropped.
    All opened images are closed, including on failure.

    Args:
        file_list (Sequence[PathInput]): Input PNG paths in row-major order.
        output_file (PathInput): Output path ending in .png. Must differ
            from every input path.
        columns (int): Positive number of grid columns.
        padding (RealScalar, optional): Finite nonnegative whole-number
            space between neighboring cells, in pixels. Defaults to 0.
        scale_factors (ScaleInput, optional): Positive finite multipliers,
            given as a flat sequence with one value per image or a nested
            grid. Missing grid entries default to 1.0. Unused final-row
            entries are validated but have no effect. Defaults to None,
            which uses 1.0 for every image.
        background (tuple[int, int, int, int] | tuple[int, int, int] | str,
            optional): Canvas color. Use (255, 255, 255, 0) for transparent
            output. Defaults to 'white'.
        resample (Image.Resampling | int, optional): Resizing filter. Defaults
            to Image.Resampling.LANCZOS.

    Returns:
        None: The composed image is saved to output_file.
    """
    files, output, columns, padding, scales = _inputs(
        file_list, output_file, columns, padding, scale_factors, '.png')
    with ExitStack() as stack:
        images = []
        for filename in files:
            source = stack.enter_context(Image.open(filename))
            if source.format != 'PNG':
                raise ValueError(f'Input is not PNG: {filename}')
            image = (source if source.mode == 'RGBA' else
                     stack.enter_context(source.convert('RGBA')))
            images.append(image)
        sizes = [im.size for im in images]
        size, cells = _layout(sizes, columns, padding, scales)
        canvas = stack.enter_context(Image.new('RGBA', size, background))
        for image, (x, y, w, h) in zip(images, cells):
            if image.size == (w, h):
                canvas.alpha_composite(image, dest=(x, y))
            else:
                with image.resize((w, h), resample=resample) as resized:
                    canvas.alpha_composite(resized, dest=(x, y))
        canvas.save(output, format='PNG')


def _pdf_form_stream(page: "PageObject", writer: "PdfWriter") -> "StreamObject":
    """Copy a single content stream with its original encoding untouched.

    Separate copies allow pages sharing a stream to have different form
    geometry and resources. Multiple streams must be joined in sequence to
    preserve graphics state; that fallback still uses pypdf's decode limits.
    """
    from pypdf.generic import ArrayObject, DecodedStreamObject, NullObject

    contents = page.get('/Contents')
    contents = contents.get_object() if contents is not None else None
    entries = contents if isinstance(contents, ArrayObject) else [contents]
    streams = []
    for entry in entries:
        stream = entry.get_object() if entry is not None else None
        if stream is not None and not isinstance(stream, NullObject):
            streams.append(stream)
    if len(streams) == 1:
        return streams[0].clone(writer, force_duplicate=True)
    form = DecodedStreamObject()
    if not streams:
        form.set_data(b'')
        return form
    form.set_data(b'\n'.join(stream.get_data() for stream in streams))
    return form.flate_encode()


def _pdf_form(page: "PageObject", writer: "PdfWriter"
              ) -> tuple["IndirectObject", tuple[float, float]]:
    """Copy a page into a clipped Form XObject without parsing its operators.

    Keep resources and the page transparency group local to the form. Its
    matrix normalizes crop offsets and clockwise page rotation. Return the
    writer-owned form reference and its visible width/height in layout units.
    Single content streams retain their encoding without decompression.
    Multiple streams and resource cloning still use pypdf's normal limits.
    """
    from pypdf.generic import (
        ArrayObject, DictionaryObject, FloatObject,
        NameObject, NumberObject, RectangleObject,
    )

    box = page.cropbox
    left, bottom, right, top = map(float, box)
    if not all(_is_finite_real_scalar(v) for v in box):
        raise ValueError('PDF crop box must contain finite coordinates')
    if right <= left or top <= bottom:
        raise ValueError('PDF crop box must have positive width and height')
    # Exact right-angle coefficients avoid numerical drift from trigonometry.
    matrices = {0: (1, 0, 0, 1, -left, -bottom),
                90: (0, -1, 1, 0, -bottom, right),
                180: (-1, 0, 0, -1, right, top),
                270: (0, 1, -1, 0, top, -left)}
    rotation = page.rotation % 360
    if rotation not in matrices:
        raise ValueError('PDF page rotation must be a multiple of 90 degrees')
    size = (right - left, top - bottom)
    if rotation in (90, 270):
        size = size[::-1]

    form = _pdf_form_stream(page, writer)
    form.update({NameObject('/Type'): NameObject('/XObject'),
                 NameObject('/Subtype'): NameObject('/Form'),
                 NameObject('/FormType'): NumberObject(1),
                 NameObject('/BBox'): RectangleObject(box),
                 NameObject('/Matrix'): ArrayObject(
                     [FloatObject(v) for v in matrices[rotation]]),
                 NameObject('/Resources'): page.get(
                     '/Resources', DictionaryObject()).clone(writer)})
    if '/Group' in page:
        form[NameObject('/Group')] = page['/Group'].clone(writer)
    # pypdf has no public writer method to register a new arbitrary stream.
    return writer._add_object(form), size


def _read_pdf_form(filename: Path, writer: "PdfWriter"
                   ) -> tuple["IndirectObject", tuple[float, float]]:
    """Read and clone the first page, closing its input before returning.

    Add the source path to failures while preserving their exception type.
    """
    from pypdf import PdfReader

    try:
        with filename.open('rb') as stream:
            reader = PdfReader(stream)
            try:
                if reader.is_encrypted:
                    raise ValueError('Encrypted PDF is unsupported')
                if not reader.pages:
                    raise ValueError('PDF has no pages')
                return _pdf_form(reader.pages[0], writer)
            finally:
                reader.close()
    except Exception as exc:
        exc.add_note(f'PDF input: {filename}')
        raise


def _place_pdf_forms(writer: "PdfWriter", forms: Sequence["IndirectObject"],
                     sizes: Sequence[tuple[float, float]],
                     canvas_size: tuple[int, int],
                     cells: Sequence[tuple[int, int, int, int]]) -> None:
    """Place forms with short drawing commands, retaining their aspect ratios."""
    from pypdf.generic import (
        DecodedStreamObject, DictionaryObject, FloatObject, NameObject,
    )

    canvas = writer.add_blank_page(width=canvas_size[0], height=canvas_size[1])
    xobjects = DictionaryObject()
    commands = []
    for i, (form, (width, height), (x, y, w, h)) in enumerate(
            zip(forms, sizes, cells)):
        scale = min(w / width, h / height)
        dx = x + (w - width * scale) / 2
        dy = canvas_size[1] - y - h + (h - height * scale) / 2
        name = NameObject(f'/Figure{i}')
        xobjects[name] = form
        # Use pypdf's PDF number formatting, including small nonzero values.
        numbers = [FloatObject(v).myrepr() for v in (scale, dx, dy)]
        commands.append(f'q {numbers[0]} 0 0 {numbers[0]} '
                        f'{numbers[1]} {numbers[2]} cm {name} Do Q\n')
    canvas[NameObject('/Resources')] = DictionaryObject(
        {NameObject('/XObject'): xobjects})
    contents = DecodedStreamObject()
    contents.set_data(''.join(commands).encode('ascii'))
    canvas[NameObject('/Contents')] = writer._add_object(contents)


def concat_pdfs(file_list: Sequence[PathInput], output_file: PathInput,
                columns: int, padding: RealScalar = 0, *,
                scale_factors: ScaleInput = None) -> None:
    """Concatenate the first page of each PDF into one grid PDF page.

    Uses the same direct scaling, whole-unit rounding, row/column sizing,
    and centering rules as concat_pngs, with PDF points replacing pixels.
    Matching numeric source dimensions give matching grid geometry; PNG DPI
    metadata is ignored. PDF content preserves its aspect ratio within the
    rounded cells. Text and vectors remain PDF content. Page rotations and
    crop-box offsets are handled by form matrices. Page content is embedded
    without parsing individual drawing operators. Single content streams
    retain their original encoding without decompression; multiple streams
    are decoded and joined while preserving graphics state. The background is
    unpainted, and interactive annotations are omitted.
    All readers and streams are closed, including on failure. Input read
    and embedding errors include the source path in an exception note.

    Args:
        file_list (Sequence[PathInput]): Input paths to unencrypted PDFs
            with at least one page, in row-major order. Only the first page
            of each file is used.
        output_file (PathInput): Output path ending in .pdf. Must differ
            from every input path.
        columns (int): Positive number of grid columns.
        padding (RealScalar, optional): Finite nonnegative whole-number
            space between neighboring cells, in PDF points. Defaults to 0.
        scale_factors (ScaleInput, optional): Positive finite multipliers,
            given as a flat sequence with one value per PDF or a nested
            grid. Missing grid entries default to 1.0. Unused final-row
            entries are validated but have no effect. Defaults to None,
            which uses 1.0 for every PDF.

    Returns:
        None: The composed page is saved to output_file.
    """
    files, output, columns, padding, scales = _inputs(
        file_list, output_file, columns, padding, scale_factors, '.pdf')
    try:
        from pypdf import PdfWriter
    except ImportError as exc:
        raise ImportError('concat_pdfs() requires pypdf.') from exc
    with PdfWriter() as writer:
        forms, sizes = [], []
        for filename in files:
            form, size = _read_pdf_form(filename, writer)
            forms.append(form)
            sizes.append(size)
        canvas_size, cells = _layout(sizes, columns, padding, scales)
        _place_pdf_forms(writer, forms, sizes, canvas_size, cells)
        with output.open('wb') as stream:
            writer.write(stream)
