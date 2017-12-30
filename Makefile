setup: setup-venv extract-zip

clean: clean-venv clean-zip

setup-venv: init-venv deps
	PYTHONPATH=venv ; . venv/bin/activate

init-venv:
	if [ ! -e "venv/bin/activate_this.py" ] ; then PYTHONPATH=venv ; python3 -m venv venv/ ; fi

check-venv:
	if [ "${VIRTUAL_ENV}" == "" ] ; then echo "Error: Must be run inside virtual environment." && exit 1 ; fi

check-not-venv:
	if [ "${VIRTUAL_ENV}" != "" ] ; then echo "Error: Must deactivate virtual environment." && exit 1 ; fi

deps:
	PYTHONPATH=venv ; . venv/bin/activate && venv/bin/pip3 install -U -r requirements.txt

freeze: check-venv
	. venv/bin/activate && venv/bin/pip freeze > requirements.txt

clean-venv: check-not-venv
	rm -rf venv

lint:
	flake8 .

extract-zip:
	python3 read_zip_data.py extract

clean-zip:
	python3 read_zip_data.py clean

read:
	python3 read.py

test:
	python3 controller.py

generate:
	python3 Generator.py Entertainment

run:
	python3 top_comment.py
