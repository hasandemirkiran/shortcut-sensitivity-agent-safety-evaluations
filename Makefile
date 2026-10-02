.PHONY: verify paper submission

verify:
	python3 scripts/verify_numbers.py

paper: verify
	mkdir -p output/pdf
	tectonic main.tex --outdir output/pdf --keep-logs

submission: verify
	mkdir -p output/pdf
	tectonic submission.tex --outdir output/pdf --keep-logs
