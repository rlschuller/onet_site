all:
	pandoc index.md -s --toc --mathjax -o index.html --css pandoc.css
