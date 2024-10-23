.PHONY: add-quartodoc build interlinks preview all

add-quartodoc:
	uv run quarto add machow/quartodoc

build:
	uv run quartodoc build

interlinks:
	uv run quartodoc interlinks

render:
	uv run quarto render

all: add-quartodoc build interlinks render
