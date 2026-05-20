Bundled subtitle font for FFmpeg video rendering.

- `noto-cjk/NotoSansCJKsc-Regular.otf` is Noto Sans CJK SC Regular from the Noto CJK project.
- It is used by `fastapi_webui_v2.py` when burning Chinese/CJK subtitles into rendered videos.
- The font is licensed under the SIL Open Font License 1.1; see `LICENSE-NOTO-CJK.txt`.

You can override the bundled font with:

- `VIDEO_SUBTITLE_FONT_FILE=/path/to/font.ttf`
- `VIDEO_SUBTITLE_FONT="Font Family Name"`
- `VIDEO_SUBTITLE_FONTS_DIR=/path/to/fonts`
