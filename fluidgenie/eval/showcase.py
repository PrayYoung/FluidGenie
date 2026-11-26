"""Render physical arrays with fixed shared scales and bounded GIF sizes."""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps


def vorticity(fields):
    u, v = fields[..., 0], fields[..., 1]
    return (np.roll(v, -1, axis=-1) - np.roll(v, 1, axis=-1)) * fields.shape[-2] / 2 - (
        np.roll(u, -1, axis=-2) - np.roll(u, 1, axis=-2)
    ) * fields.shape[-3] / 2


def font(size=14):
    for name in ["DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def panel(field, view, limit, size):
    values = field[..., 2] if view == "density" else vorticity(field)
    normalized = (
        np.clip(values / limit, 0, 1)
        if view == "density"
        else np.clip(0.5 + 0.5 * values / limit, 0, 1)
    )
    rgb = colormaps["inferno" if view == "density" else "RdBu_r"](
        normalized, bytes=True
    )[..., :3]
    return Image.fromarray(np.ascontiguousarray(rgb[::-1])).resize(
        (size, size), Image.Resampling.BICUBIC
    )


def grid_frame(rows, columns, row_labels, title, *, view, limit, size=128):
    left = 100 if any(row_labels) else 0
    gap = 6
    top = 64
    canvas = Image.new(
        "RGB",
        (
            left + len(columns) * (size + gap) + gap,
            top + len(rows) * (size + gap) + gap,
        ),
        (9, 18, 31),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 9), title, font=font(15), fill=(240, 246, 255))
    for col, label in enumerate(columns):
        draw.text(
            (left + gap + col * (size + gap) + size / 2, 45),
            label,
            font=font(13),
            anchor="mm",
            fill=(194, 215, 236),
        )
    for row, (fields, label) in enumerate(zip(rows, row_labels)):
        if label:
            draw.multiline_text(
                (8, top + row * (size + gap) + size / 2 - 16),
                label,
                font=font(13),
                fill=(210, 225, 241),
            )
        for col, field in enumerate(fields):
            canvas.paste(
                panel(field, view, limit, size),
                (left + gap + col * (size + gap), top + row * (size + gap)),
            )
    return canvas


def square_frame(field, step, action, hold, density_limit, vorticity_limit):
    size = 224
    gap = 8
    image = Image.new("RGB", (size * 2 + gap * 3, size + 88), (9, 18, 31))
    draw = ImageDraw.Draw(image)
    label = ["No-op", "Up / W", "Down / S", "Right / D", "Left / A"][action]
    draw.text(
        (10, 10), f"Saved FM + LAM | step {step} | {label}", font=font(16), fill="white"
    )
    for i, (name, view, limit) in enumerate(
        [
            ("Smoke density", "density", density_limit),
            ("Vorticity", "vorticity", vorticity_limit),
        ]
    ):
        draw.text(
            (gap + i * (size + gap), 43), name, font=font(14), fill=(194, 215, 236)
        )
        image.paste(panel(field, view, limit, size), (gap + i * (size + gap), 68))
    return image


def save_gif(frames, path, *, duration_ms=100, max_bytes=2_500_000):
    """Use adaptive shared palettes; reduce rendering size before exceeding budget."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No actual frames supplied")
    for stride in (1, 2):
        selected_indices = list(range(0, len(frames), stride))
        if selected_indices[-1] != len(frames) - 1:
            selected_indices.append(len(frames) - 1)
        selected = [frames[i] for i in selected_indices]
        for scale in (1.0, 0.85, 0.7):
            size = tuple(max(1, round(x * scale)) for x in selected[0].size)
            rgb = [
                im.resize(size, Image.Resampling.LANCZOS) if scale != 1 else im
                for im in selected
            ]
            thumbs = [
                im.resize((96, max(1, round(im.height * 96 / im.width)))) for im in rgb
            ]
            # Reserve visible HUD colors even when tiny thumbnail text averages
            # into the dark background during shared-palette selection.
            hud = np.repeat(
                np.asarray(
                    [(240, 246, 255), (194, 215, 236), (9, 18, 31)], dtype=np.uint8
                ),
                16,
                axis=0,
            )[:, None, :]
            hud = np.repeat(hud, 96, axis=1)
            sample = Image.fromarray(
                np.concatenate([*[np.asarray(im) for im in thumbs], hud], axis=0)
            )
            for colors in (256, 128, 64):
                palette = sample.quantize(
                    colors=colors, method=Image.Quantize.MEDIANCUT
                )
                encoded = [
                    im.quantize(palette=palette, dither=Image.Dither.NONE) for im in rgb
                ]
                encoded[0].save(
                    path,
                    save_all=True,
                    append_images=encoded[1:],
                    duration=duration_ms * stride,
                    loop=0,
                    optimize=True,
                    disposal=2,
                )
                if path.stat().st_size < max_bytes:
                    return {
                        "bytes": path.stat().st_size,
                        "frames": len(encoded),
                        "width": size[0],
                        "height": size[1],
                        "palette_colors": colors,
                        "render_scale": scale,
                        "frame_stride": stride,
                        "duration_ms": duration_ms * stride,
                        "source_frame_indices": selected_indices,
                    }
    raise ValueError("Could not fit real-data animation within the GIF budget")
