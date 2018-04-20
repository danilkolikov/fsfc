init:
	pip3 install -r requirements.txt

test:
	python3 -m unittest

html:
	make html -C docs
