# mangasplit

python script to split manga images vertically using image magick

## Requirements

- `python3` (>=3.7) somewhere in your PATH
- `magick` somewhere in your PATH

## Install

```shell
curl -o ~/.local/bin/mangasplit https://raw.githubusercontent.com/puzzlemoondev/mangasplit/main/mangasplit.py && chmod +x ~/.local/bin/mangasplit
```

Or clone the repo and run `make install`

## Usage

```shell
usage: mangasplit [-h] [-i INPUT_DIR] [-o OUTPUT_DIR] [-l] [-k] [-n] [-f]

split manga images vertically

options:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input_dir INPUT_DIR
                        Directory containing images with format that magick
                        supports
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Output directory
  -l, --ltr             Input images are left to right
  -k, --keep_double     Keep double pages
  -n, --no-cover        Treat first page as content
  -f, --force           Force overwrite existing files

examples:
  mangasplit # process current directory
  mangasplit -i input_dir/ -o output_dir/
  mangasplit -i input_dir/ -o output_dir/ -k # keep double pages
  mangasplit -i input_dir/ -o output_dir/ -l # left to right mode
  mangasplit -i input_dir/ -o output_dir/ -n # treat first page as normal content
  mangasplit -i input_dir/ -o output_dir/ -f # overwrites existing files
```
