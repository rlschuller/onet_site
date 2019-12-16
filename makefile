all:
	pandoc -s index.md --toc --mathjax="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" -o index.html --css pandoc.css --number-sections
	pandoc -s index.md --toc --mathjax="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" -o index_no_css.html --number-sections
