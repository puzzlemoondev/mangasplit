#!/usr/bin/env python3

import argparse
import mimetypes
import shutil
import subprocess
import textwrap
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import total_ordering
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional


@dataclass
class Dimension:
    width: int
    height: int

    def is_double_page(self) -> bool:
        return self.width > self.height


@dataclass
class ImageMagickSplitItem:
    original_path: Path
    split_path: tuple[Path, Path]


class ImageMagick:
    @staticmethod
    def get_dimension(path: Path) -> Dimension:
        result = subprocess.check_output(
            ["identify", "-format", "%w %h", str(path)], text=True, encoding="utf-8"
        )
        width, height = map(int, result.split())
        return Dimension(width, height)

    @staticmethod
    def trim(image: Path, cwd: str) -> Path:
        output_dir = Path(cwd).joinpath(f"cover_{int(time.time())}")
        output_dir.mkdir()
        output_file = output_dir.joinpath("000.jpg")
        subprocess.check_call(
            ["magick", image, "-trim", "+repage", output_file], cwd=output_dir
        )
        return output_file

    @staticmethod
    def split_images(
        paths: list[Path], cwd: str, is_ltr: bool
    ) -> Iterable[ImageMagickSplitItem]:
        output_dir = Path(cwd).joinpath(f"splits_{int(time.time())}")
        output_dir.mkdir()
        subprocess.check_call(
            [
                "magick",
                *paths,
                "-scene",
                "1",
                "-crop",
                "2x1@",
                "+repage",
                "%09d.jpg",
            ],
            cwd=output_dir,
        )

        results = iter(ImageFinder.list_image_files(output_dir))
        for path in paths:
            left, right = next(results), next(results)
            item = ImageMagickSplitItem(
                original_path=path,
                split_path=(left, right) if is_ltr else (right, left),
            )
            yield item


@dataclass
class ImageFinderResultCover:
    path: Path
    dimension: Dimension


@dataclass
class ImageFinderResult:
    single_pages: list[Path]
    double_pages: list[Path]
    cover: Optional[ImageFinderResultCover] = field(default=None)


class ImageFinder:
    def __init__(self, directory: Path, has_cover: bool):
        self.directory = directory
        self.has_cover = has_cover

    def find(self) -> ImageFinderResult:
        cover = None
        double_pages = []
        single_pages = []
        files = self.list_image_files(self.directory)
        if len(files) != 0:
            if self.has_cover:
                cover_file = files.pop(0)
                cover = ImageFinderResultCover(
                    path=cover_file, dimension=ImageMagick.get_dimension(cover_file)
                )
            for file in files:
                dimension = ImageMagick.get_dimension(file)
                if dimension.is_double_page():
                    double_pages.append(file)
                else:
                    single_pages.append(file)
        return ImageFinderResult(
            single_pages=single_pages, double_pages=double_pages, cover=cover
        )

    @staticmethod
    def list_image_files(directory: Path) -> list[Path]:
        paths = []
        for path in directory.iterdir():
            if path.is_file() and ImageFinder._is_image_file(path):
                paths.append(path)
        return sorted(paths, key=lambda x: x.stem)

    @staticmethod
    def _is_image_file(file: Path) -> bool:
        file_type, _ = mimetypes.guess_type(file)
        return file_type and file_type.startswith("image")


@total_ordering
@dataclass
class ProcessedImagePath:
    original_path: Path
    processed_path: Path
    part: Optional[int] = field(default=None)

    def __eq__(self, other):
        if not isinstance(other, ProcessedImagePath):
            return NotImplemented
        return (self.original_path.stem, self.part) == (
            other.original_path.stem,
            other.part,
        )

    def __lt__(self, other):
        if not isinstance(other, ProcessedImagePath):
            return NotImplemented
        if self.original_path.stem != other.original_path.stem:
            return self.original_path.stem < other.original_path.stem
        return (self.part or 0) < (other.part or 0)


class Splitter:
    def __init__(
        self,
        input_dir: Path,
        tmpdir: str,
        is_ltr: bool,
        keep_double: bool,
        has_cover: bool,
    ):
        self.finder = ImageFinder(directory=input_dir, has_cover=has_cover)
        self.tmpdir = tmpdir
        self.is_ltr = is_ltr
        self.keep_double = keep_double

    def split(self) -> list[ProcessedImagePath]:
        images = self.finder.find()
        pages = [
            self._process_single_pages(images.single_pages),
            self._process_double_pages(images.double_pages),
        ]
        if self.keep_double:
            pages.append(self._process_single_pages(images.double_pages))
        if images.cover is not None:
            processed_cover = self._process_cover(images.cover)
            pages.append((processed_cover,))

        return sorted(chain.from_iterable(pages))

    @staticmethod
    def _process_single_pages(
        images: list[Path],
    ) -> Iterable[ProcessedImagePath]:
        if len(images) == 0:
            return

        for image in images:
            yield ProcessedImagePath(original_path=image, processed_path=image)

    def _process_double_pages(
        self,
        images: list[Path],
    ) -> Iterable[ProcessedImagePath]:
        if len(images) == 0:
            return

        splits = ImageMagick.split_images(images, cwd=self.tmpdir, is_ltr=self.is_ltr)
        for split in splits:
            for part, processed_path in enumerate(split.split_path, start=1):
                yield ProcessedImagePath(
                    original_path=split.original_path,
                    processed_path=processed_path,
                    part=part,
                )

    def _process_cover(self, cover: ImageFinderResultCover) -> ProcessedImagePath:
        if cover.dimension.is_double_page():
            trimmed = ImageMagick.trim(cover.path, self.tmpdir)
            return ProcessedImagePath(original_path=cover.path, processed_path=trimmed)
        return ProcessedImagePath(original_path=cover.path, processed_path=cover.path)


class Saver:
    def __init__(self, output_dir: Path, overwrite=False):
        self.output_dir = output_dir
        self.overwrite = overwrite

    def save(self, images: list[ProcessedImagePath]):
        width = self._get_zero_padding_width(images)
        for index, image in enumerate(images):
            self.output_dir.mkdir(parents=True, exist_ok=True)
            dest_path = self.output_dir.joinpath(
                f"{index + 1:0{width}}{image.processed_path.suffix}"
            )
            if dest_path.exists():
                if not self.overwrite:
                    raise FileExistsError(f"{dest_path} exists")
                else:
                    print(f"{dest_path} exists, overwriting...")
            shutil.copyfile(image.processed_path, dest_path)

    @staticmethod
    def _get_zero_padding_width(images: list[ProcessedImagePath]) -> int:
        count = len(images)
        width = len(str(count))
        return max(width, 3)


def dirpath(value: str, strict=True):
    path = Path(value).resolve()
    assert path.is_dir() if strict else not path.suffix, "provided path is not a dir"
    return path


def parse_args():
    prog = "mangasplit"
    parser = argparse.ArgumentParser(
        prog=prog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="split manga images vertically",
        epilog=textwrap.dedent(
            f"""
        examples:
          {prog} # process current directory
          {prog} -i input_dir/ -o output_dir/
          {prog} -i input_dir/ -o output_dir/ -k # keep double pages
          {prog} -i input_dir/ -o output_dir/ -l # left to right mode
          {prog} -i input_dir/ -o output_dir/ -n # treat first page as normal content
          {prog} -i input_dir/ -o output_dir/ -f # overwrites existing files
            """
        ),
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        default=Path.cwd(),
        type=dirpath,
        help="Directory containing images with format that magick supports",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=Path.cwd(),
        type=lambda x: dirpath(x, strict=False),
        help="Output directory",
    )
    parser.add_argument(
        "-l",
        "--ltr",
        action="store_true",
        default=False,
        help="Input images are left to right",
    )
    parser.add_argument(
        "-k",
        "--keep_double",
        action="store_true",
        default=False,
        help="Keep double pages",
    )
    parser.add_argument(
        "-n",
        "--no-cover",
        action="store_true",
        default=False,
        help="Treat first page as content",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Force overwrite existing files",
    )
    return parser.parse_args()


def main():
    if not shutil.which("magick"):
        raise FileNotFoundError("magick executable not found")
    args = parse_args()

    with TemporaryDirectory() as tmpdir:
        splitter = Splitter(
            input_dir=args.input_dir,
            tmpdir=tmpdir,
            is_ltr=args.ltr,
            keep_double=args.keep_double,
            has_cover=not args.no_cover,
        )
        images = splitter.split()
        saver = Saver(output_dir=args.output_dir, overwrite=args.force)
        saver.save(images)


if __name__ == "__main__":
    main()
