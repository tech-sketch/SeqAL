init:
	poetry install

lint:
	poetry run flake8 seqal
	poetry run flake8 examples
	poetry run flake8 tests 

	poetry run isort seqal
	poetry run isort examples
	poetry run isort tests

	poetry run black seqal
	poetry run black examples
	poetry run black tests

test:
	poetry run pytest tests
