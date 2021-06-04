init:
	poetry install

lint:
	poetry run flake8
	poetry run isort

test:
	poetry run test

runserver:
	poetry run server
