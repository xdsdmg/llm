freeze:
	pip freeze > requirements.txt

test:
	venv/bin/python3 tokenizer/test.py
