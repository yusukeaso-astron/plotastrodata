"""Compose PNG images or first PDF pages using the same grid geometry.

Sizes are rounded to whole layout units (pixels for PNG, points for PDF).
PDF composition uses the required pypdf dependency, imported only when used.
"""
from contextlib import ExitStack
from os import PathLike
from pathlib import Path
from typing import Sequence

from PIL import Image

from plotastrodata._type_utils import RealScalar
from plotastrodata._validation_utils import (
    _is_finite_real_scalar,
    _is_positive_real_scalar,
    _require_integer,
    _require_positive_real_scalar,
)


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
    if padding != int(padding):
        raise ValueError('padding must be a whole number of layout units')
    output = Path(output_file)
    if output.suffix.lower() != suffix:
        raise ValueError(f'output_file must end in {suffix}')
    if any(output.resolve() == f.resolve() for f in files):
        raise ValueError('output_file must differ from every input file')
    scales = _scale_factors(scale_factors, len(files), int(columns))
    return files, output, int(columns), int(padding), scales


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
        if not all(_is_positive_real_scalar(v)
                   for v in (w, h, w * scale, h * scale)):
            raise ValueError('input and scaled dimensions must be'
                             + ' positive and finite')
        dw, dh = w * scale, h * scale
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


def concat_pdfs(file_list: Sequence[PathInput], output_file: PathInput,
                columns: int, padding: RealScalar = 0, *,
                scale_factors: ScaleInput = None,
                parsing_budget: int | None = None) -> None:
    """Concatenate the first page of each PDF into one grid PDF page.

    Uses the same direct scaling, whole-unit rounding, row/column sizing,
    and centering rules as concat_pngs, with PDF points replacing pixels.
    Matching numeric source dimensions give matching grid geometry; PNG DPI
    metadata is ignored. PDF content preserves its aspect ratio within the
    rounded cells. Text and vectors remain PDF content. Page rotations and
    crop-box offsets are normalized before placement. The background is
    unpainted, and interactive annotations are omitted.
    All readers and streams are closed, including on failure.

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
        parsing_budget (int | None, optional): Positive maximum parsing
            work per PDF content stream, counting operators, operands, and
            nested elements. Increase this for dense vector plots that raise
            pypdf.errors.LimitReachedError, for example to 10000000.
            Larger budgets can require more memory and processing time.
            Defaults to None, preserving the current pypdf configuration.
            An explicit budget applies only during this call; the previous
            configuration is restored on success or failure.

    Returns:
        None: The composed page is saved to output_file.
    """
    files, output, columns, padding, scales = _inputs(
        file_list, output_file, columns, padding, scale_factors, '.pdf')
    if parsing_budget is not None:
        _require_integer(parsing_budget, 'parsing_budget')
    try:
        from pypdf import (PdfReader, PdfWriter, Transformation,
                           apply_configuration)
    except ImportError as exc:
        raise ImportError('concat_pdfs() requires pypdf.') from exc
    with ExitStack() as stack:
        if parsing_budget is not None:
            stack.enter_context(apply_configuration(
                stream_parsing_work_maximum_count=int(parsing_budget)))
        pages = []
        for filename in files:
            stream = stack.enter_context(filename.open('rb'))
            reader = PdfReader(stream)
            stack.callback(reader.close)
            if reader.is_encrypted:
                raise ValueError(f'Encrypted PDF is unsupported: {filename}')
            if not reader.pages:
                raise ValueError(f'PDF has no pages: {filename}')
            page = reader.pages[0]
            page.pop('/Annots', None)
            if page.rotation:
                # Only rotation rewriting needs a separate writer-owned page.
                source_writer = stack.enter_context(PdfWriter())
                page = source_writer.add_page(page)
                page.transfer_rotation_to_content()
            pages.append(page)
        sizes = [(float(p.cropbox.width), float(p.cropbox.height))
                 for p in pages]
        size, cells = _layout(sizes, columns, padding, scales)
        writer = PdfWriter()
        stack.callback(writer.close)
        canvas = writer.add_blank_page(width=size[0], height=size[1])
        for page, (x, y, w, h) in zip(pages, cells):
            box = page.cropbox
            # Fit uniformly into the rounded rectangle; retain vector aspect ratio.
            scale = min(w / float(box.width), h / float(box.height))
            dx = (w - float(box.width) * scale) / 2
            dy = (h - float(box.height) * scale) / 2
            transform = (Transformation().translate(-float(box.left), -float(box.bottom))
                         .scale(scale).translate(x + dx, size[1] - y - h + dy))
            canvas.merge_transformed_page(page, transform, expand=False)
        with output.open('wb') as stream:
            writer.write(stream)
