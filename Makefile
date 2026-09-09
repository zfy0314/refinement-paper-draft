.PHONY: all clean

all:
	latexmk main.tex

clean:
	latexmk -C main.tex
